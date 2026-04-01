"""
Task 3: Subtle & Compound Bugs (Hard)

These scenarios involve multiple interacting bugs, subtle numerical issues,
or require deep understanding of PyTorch internals to diagnose.
"""

try:
    from . import BugScenario, register_scenario
except ImportError:
    from scenarios import BugScenario, register_scenario

# =============================================================================
# Scenario 1: DDP + gradient accumulation interaction bug
# =============================================================================
register_scenario(BugScenario(
    scenario_id="hard_ddp_grad_accum",
    title="DDP + Gradient Accumulation Sync Bug",
    difficulty="hard",
    task_id="subtle_bugs",
    description=(
        "Training a large language model with DistributedDataParallel across 4 GPUs "
        "using gradient accumulation (4 steps). The training runs but is 3x slower than "
        "expected AND produces worse results than single-GPU training. There are two "
        "interacting issues to identify."
    ),
    training_config={
        "model": "GPT-2 Small (124M params)",
        "dataset": "OpenWebText subset",
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_gpus": 4,
        "framework": "torch.nn.parallel.DistributedDataParallel",
        "epochs": 5,
        "max_seq_length": 1024,
    },
    code_snippet="""import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

model = GPT2Model(config).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

gradient_accumulation_steps = 4

for epoch in range(5):
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].cuda(local_rank)
        labels = batch['labels'].cuda(local_rank)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()  # BUG 1: DDP syncs gradients on EVERY backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # BUG 2: Learning rate is not scaled for effective batch size
    # effective_batch = batch_size * grad_accum * num_gpus = 8 * 4 * 4 = 128
    # but lr=3e-4 was tuned for batch_size=32
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 5.2, "val_loss": 5.3, "throughput": 450.0, "gpu_memory_mb": 28000, "learning_rate": 3e-4,
         "extra_metrics": {"expected_throughput": 1400.0, "communication_overhead": "67%"}},
        {"epoch": 1, "train_loss": 4.8, "val_loss": 4.9, "throughput": 440.0, "gpu_memory_mb": 28000, "learning_rate": 3e-4},
        {"epoch": 3, "train_loss": 4.2, "val_loss": 4.5, "throughput": 435.0, "gpu_memory_mb": 28000, "learning_rate": 3e-4,
         "extra_metrics": {"single_gpu_val_loss": 3.8, "expected_val_loss": 3.5}},
        {"epoch": 5, "train_loss": 3.9, "val_loss": 4.3, "throughput": 430.0, "gpu_memory_mb": 28000, "learning_rate": 3e-4},
    ],
    root_cause_category="ddp_grad_accumulation",
    root_cause_description="Two interacting bugs: (1) DDP synchronizes (all-reduce) gradients on every backward() call, but with gradient accumulation, synchronization should only happen on the accumulation boundary. Use model.no_sync() context manager for non-sync steps. This causes 3x unnecessary communication overhead. (2) The learning rate is not scaled for the effective batch size (128 = 8*4*4). Linear scaling rule suggests lr should be ~4x smaller or the lr was tuned for a different batch size.",
    correct_fix_description="(1) Wrap non-sync backward passes in model.no_sync() context manager. (2) Scale learning rate appropriately for the effective batch size, or use learning rate warmup.",
    correct_fix_code="""for step, batch in enumerate(train_loader):
    input_ids = batch['input_ids'].cuda(local_rank)
    labels = batch['labels'].cuda(local_rank)

    # FIX 1: Only sync gradients on accumulation boundary
    is_accumulating = (step + 1) % gradient_accumulation_steps != 0
    context = model.no_sync() if is_accumulating else nullcontext()

    with context:
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()

# FIX 2: Scale learning rate for effective batch size
# effective_batch = 8 * 4 * 4 = 128, original was tuned for 32
# Scale: lr = base_lr * sqrt(128/32) = 3e-4 * 2 = 6e-4, or use warmup""",
    diagnosis_keywords=["DDP", "distributed", "no_sync", "gradient accumulation", "sync", "all-reduce", "communication", "throughput", "slow", "effective batch"],
    fix_keywords=["no_sync", "model.no_sync()", "sync boundary", "scale learning rate", "effective batch size", "linear scaling"],
    relevant_inspections=["analyze_logs", "inspect_gradients", "inspect_model_architecture"],
    secondary_bugs=[
        {"category": "lr_not_scaled", "description": "Learning rate not adjusted for 4x larger effective batch size from gradient accumulation * DDP"}
    ],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "Performance issues:\n1. THROUGHPUT: 430-450 tokens/s vs expected 1400 tokens/s (3.1x slower)\n   - Communication overhead accounts for ~67% of training time\n   - 4 GPUs should provide ~3.5x speedup, but actual is 0.9x\n2. QUALITY: Val loss 4.3 after 5 epochs vs 3.8 for single-GPU\n   - Multi-GPU training is producing WORSE results than single GPU\n   - This suggests both a communication inefficiency and a convergence issue",
            "data": {"speedup": 0.9, "expected_speedup": 3.5, "quality_gap": 0.5}
        },
        "inspect_gradients": {
            "inspection_type": "inspect_gradients",
            "findings": "Gradient synchronization analysis:\n- DDP all-reduce is triggered 4x per accumulation cycle (every backward())\n- Only the LAST sync is needed; first 3 are wasted\n- Each all-reduce takes ~25ms for this model size\n- Per accumulation cycle: 100ms wasted communication\n\nAdditionally, effective batch size = 8 × 4 (accum) × 4 (GPUs) = 128\nBut learning rate 3e-4 was tuned for batch_size=32.\nGradients are 4x smaller than expected due to averaging over more samples.",
            "data": {"redundant_syncs": 3, "sync_time_ms": 25, "wasted_time_per_cycle_ms": 75, "effective_batch": 128, "lr_tuned_for": 32}
        },
    },
))

# =============================================================================
# Scenario 2: Numerical instability in custom loss + mixed precision
# =============================================================================
register_scenario(BugScenario(
    scenario_id="hard_mixed_precision_instability",
    title="Mixed Precision + Custom Loss Numerical Instability",
    difficulty="hard",
    task_id="subtle_bugs",
    description=(
        "Training a segmentation model with automatic mixed precision (AMP) and a "
        "custom Dice loss function. Training occasionally produces NaN losses, "
        "happening randomly every few hundred batches. The NaN is intermittent and "
        "difficult to reproduce."
    ),
    training_config={
        "model": "U-Net",
        "dataset": "Medical Image Segmentation",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "batch_size": 8,
        "epochs": 50,
        "mixed_precision": "fp16 (torch.cuda.amp)",
        "loss_function": "Custom Dice Loss",
    },
    code_snippet="""import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    # BUG 1: In fp16, small values get flushed to zero
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # BUG 2: smooth=1e-6 is below fp16 precision (min ~6e-8, but loses precision)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

model = UNet(in_channels=1, out_channels=1).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scaler = GradScaler()

for epoch in range(50):
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        optimizer.zero_grad()

        with autocast():  # BUG 3: dice_loss inside autocast causes fp16 computation
            outputs = model(images)
            loss = dice_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 0.82, "val_loss": 0.80, "learning_rate": 0.001, "extra_metrics": {"nan_batches": 0}},
        {"epoch": 5, "train_loss": 0.45, "val_loss": 0.48, "learning_rate": 0.001, "extra_metrics": {"nan_batches": 2}},
        {"epoch": 10, "train_loss": 0.32, "val_loss": 0.38, "learning_rate": 0.001, "extra_metrics": {"nan_batches": 5, "nan_example": "batch 347"}},
        {"epoch": 20, "train_loss": float('nan'), "val_loss": float('nan'), "learning_rate": 0.001,
         "extra_metrics": {"nan_batches": 15, "note": "Training collapsed at epoch 20 due to accumulated NaN updates"}},
    ],
    error_message="RuntimeWarning: NaN detected in loss at batch 347 (epoch 10). Intermittent NaN; occurs ~5 times per epoch.",
    root_cause_category="mixed_precision_numerical",
    root_cause_description="Three interacting issues: (1) The custom Dice loss performs division and subtraction operations that are numerically unstable in fp16 — small intersection values get flushed to zero. (2) The smoothing constant (1e-6) is near fp16 precision limits. (3) The dice_loss is computed inside autocast(), which casts intermediate results to fp16. The loss computation should use fp32 for numerical stability.",
    correct_fix_description="(1) Move the Dice loss computation to fp32 by either computing it outside autocast or explicitly casting to float32. (2) Increase the smoothing constant to 1.0 (standard for Dice loss). (3) Use torch.clamp to prevent extreme values.",
    correct_fix_code="""def dice_loss(pred, target, smooth=1.0):
    # FIX: Cast to fp32 for numerical stability
    pred = pred.float()
    target = target.float()
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Alternative: compute loss outside autocast
with autocast():
    outputs = model(images)
# FIX: Loss computation in fp32
loss = dice_loss(outputs.float(), masks)""",
    diagnosis_keywords=["mixed precision", "fp16", "half precision", "AMP", "autocast", "numerical", "NaN", "dice loss", "precision", "instability"],
    fix_keywords=["float32", "fp32", ".float()", "outside autocast", "cast to float", "smooth=1.0", "clamp", "numerical stability"],
    relevant_inspections=["analyze_logs", "inspect_model_architecture", "inspect_gradients"],
    secondary_bugs=[
        {"category": "small_smooth", "description": "Smoothing constant 1e-6 is too small for fp16 precision"},
        {"category": "loss_inside_autocast", "description": "Custom loss computed inside autocast uses fp16 arithmetic"}
    ],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "NaN occurrences are intermittent but increasing:\n- Epoch 0: 0 NaN batches\n- Epoch 5: 2 NaN batches\n- Epoch 10: 5 NaN batches\n- Epoch 20: Training collapse (15 NaN batches, model diverged)\n\nNaN tends to occur on batches with small or sparse target masks (low intersection). This is consistent with numerical instability in division operations under fp16 precision. The GradScaler tries to recover but eventually fails.",
            "data": {"nan_frequency": "increasing", "nan_pattern": "sparse_masks", "collapse_epoch": 20}
        },
        "inspect_model_architecture": {
            "inspection_type": "inspect_model_architecture",
            "findings": "Loss function analysis (dice_loss):\n- smooth=1e-6 — fp16 min subnormal is ~6e-8, so smooth provides almost no protection\n- Division: (2*intersection + 1e-6) / (union + 1e-6)\n- When intersection → 0 in fp16: numerator can become exactly 0 or denormalized\n- sigmoid output in fp16 has reduced precision for small values\n\n⚠️ The entire dice_loss runs in fp16 because it's inside autocast(). Custom loss functions with divisions, small constants, or subtraction should run in fp32.\n\nAMP autocast only automatically promotes certain ops (like matmul). Custom functions remain in the input dtype (fp16).",
            "data": {"loss_dtype": "float16", "smooth": 1e-6, "fp16_safe": False, "risky_ops": ["division", "sigmoid", "subtraction"]}
        },
    },
))

# =============================================================================
# Scenario 3: Incorrect weight loading + frozen layers
# =============================================================================
register_scenario(BugScenario(
    scenario_id="hard_weight_loading_frozen",
    title="Misaligned Weight Loading + Incorrectly Frozen Layers",
    difficulty="hard",
    task_id="subtle_bugs",
    description=(
        "Fine-tuning a Vision Transformer (ViT) for medical imaging. The model was "
        "pretrained on ImageNet. Despite using a proven architecture and hyperparameters, "
        "the model trains poorly — loss decreases very slowly and accuracy plateaus "
        "at 45% (expected: 85%+). No errors are raised."
    ),
    training_config={
        "model": "ViT-B/16 (pretrained on ImageNet)",
        "dataset": "Retinal OCT (4 classes)",
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "batch_size": 16,
        "epochs": 30,
        "frozen_layers": "first 8 transformer blocks (intended)",
        "weight_decay": 0.05,
    },
    code_snippet="""import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Load pretrained ViT
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# Replace classifier head
model.heads = nn.Linear(768, 4)  # 4-class OCT classification

# Freeze first 8 transformer blocks (of 12)
for i, block in enumerate(model.encoder.layers):
    if i < 8:
        for param in block.parameters():
            param.requires_grad = False

# BUG 1: model.heads was replaced AFTER loading weights — new head has random init
# but the weight loading happened correctly. However...

# BUG 2: The positional embedding and patch embedding are NOT frozen
# but they're also not included in the optimizer param groups with different lr
# They get the same lr as the head, which is wrong for pretrained params

# BUG 3: weight_decay is applied to bias and LayerNorm parameters
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.05  # BUG: No parameter groups — decay on bias/norm too
)

model = model.cuda()

for epoch in range(30):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 1.39, "val_loss": 1.40, "train_accuracy": 0.25, "val_accuracy": 0.25, "grad_norm": 0.8, "learning_rate": 1e-4},
        {"epoch": 5, "train_loss": 1.20, "val_loss": 1.25, "train_accuracy": 0.35, "val_accuracy": 0.33, "grad_norm": 0.5, "learning_rate": 1e-4},
        {"epoch": 10, "train_loss": 1.05, "val_loss": 1.18, "train_accuracy": 0.42, "val_accuracy": 0.38, "grad_norm": 0.3, "learning_rate": 1e-4},
        {"epoch": 20, "train_loss": 0.95, "val_loss": 1.15, "train_accuracy": 0.48, "val_accuracy": 0.42, "grad_norm": 0.15, "learning_rate": 1e-4},
        {"epoch": 30, "train_loss": 0.90, "val_loss": 1.15, "train_accuracy": 0.50, "val_accuracy": 0.45, "grad_norm": 0.08, "learning_rate": 1e-4},
    ],
    root_cause_category="weight_loading_frozen_layers",
    root_cause_description="Three interacting issues: (1) The patch embedding and positional embedding from the pretrained model are unfrozen and being updated with the same high learning rate as the new classification head, destroying the learned spatial representations. (2) Weight decay (0.05) is applied uniformly to all parameters including bias and LayerNorm, which is known to hurt fine-tuning (biases and norm params should not be decayed). (3) No differential learning rates — the unfrozen transformer blocks 8-11 should use a much lower lr than the new head.",
    correct_fix_description="(1) Freeze patch and positional embeddings OR use a very low learning rate for them. (2) Create parameter groups that exclude bias and LayerNorm from weight decay. (3) Use differential learning rates — lower for pretrained layers, higher for the new head.",
    correct_fix_code="""# FIX: Parameter groups with differential learning rates and proper weight decay
# Freeze embeddings
for param in model.conv_proj.parameters():
    param.requires_grad = False
model.encoder.pos_embedding.requires_grad = False

# Separate parameters for proper weight decay
pretrained_decay = []
pretrained_no_decay = []
head_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'heads' in name:
        head_params.append(param)
    elif 'bias' in name or 'norm' in name or 'ln' in name:
        pretrained_no_decay.append(param)
    else:
        pretrained_decay.append(param)

optimizer = torch.optim.AdamW([
    {'params': pretrained_decay, 'lr': 1e-5, 'weight_decay': 0.05},
    {'params': pretrained_no_decay, 'lr': 1e-5, 'weight_decay': 0.0},
    {'params': head_params, 'lr': 1e-3, 'weight_decay': 0.0},
])""",
    diagnosis_keywords=["weight loading", "frozen layers", "fine-tuning", "pretrained", "embedding", "weight decay", "learning rate", "param groups", "bias decay"],
    fix_keywords=["parameter groups", "differential lr", "freeze embedding", "no weight decay bias", "lr schedule", "discriminative lr"],
    relevant_inspections=["inspect_model_architecture", "inspect_gradients", "analyze_logs"],
    secondary_bugs=[
        {"category": "weight_decay_on_bias", "description": "weight_decay=0.05 applied to bias and LayerNorm parameters"},
        {"category": "no_differential_lr", "description": "All unfrozen parameters use same learning rate — pretrained backbone and new head"}
    ],
    inspection_data={
        "inspect_model_architecture": {
            "inspection_type": "inspect_model_architecture",
            "findings": "Layer freezing analysis:\n- Frozen: Transformer blocks 0-7 ✓\n- Unfrozen: Transformer blocks 8-11 ✓\n- model.heads (new classifier): Unfrozen ✓\n\n⚠️ ISSUES FOUND:\n- model.conv_proj (patch embedding): UNFROZEN — being updated with lr=1e-4\n- model.encoder.pos_embedding: UNFROZEN — being updated with lr=1e-4\n- All unfrozen params use SAME learning rate (1e-4)\n- weight_decay=0.05 applied to ALL params including 48 bias tensors and 24 LayerNorm layers\n\nThe patch/positional embeddings are critical pretrained representations. Updating them at the same rate as the new head destroys learned spatial features.",
            "data": {"patch_embed_frozen": False, "pos_embed_frozen": False, "num_lr_groups": 1, "bias_decayed": True}
        },
        "inspect_gradients": {
            "inspection_type": "inspect_gradients",
            "findings": "Gradient analysis by component:\n- model.heads: grad_norm=0.45 (healthy for new head)\n- transformer block 11: grad_norm=0.12 (reasonable)\n- transformer block 8: grad_norm=0.05 (low but ok)\n- model.conv_proj: grad_norm=0.25 (DESTROYING pretrained patch embedding)\n- pos_embedding: grad_norm=0.18 (MODIFYING pretrained position encoding)\n\nThe patch embedding and positional embedding gradients are large enough to significantly alter these pretrained representations, especially over 30 epochs. These should be frozen or use lr < 1e-6.",
            "data": {"head_grad": 0.45, "conv_proj_grad": 0.25, "pos_embed_grad": 0.18, "impact": "high"}
        },
    },
))

# =============================================================================
# Scenario 4: Tokenizer/model mismatch + wrong padding
# =============================================================================
register_scenario(BugScenario(
    scenario_id="hard_tokenizer_mismatch",
    title="Tokenizer-Model Mismatch + Padding Bug",
    difficulty="hard",
    task_id="subtle_bugs",
    description=(
        "Fine-tuning a BERT model for text classification. The model trains and loss "
        "decreases, but performance is much worse than expected (76% vs 92% expected). "
        "The model seems to partially work but has a persistent performance gap."
    ),
    training_config={
        "model": "bert-base-uncased",
        "dataset": "SST-2 Sentiment Analysis",
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 5,
        "max_sequence_length": 128,
        "tokenizer": "bert-base-cased",
    },
    code_snippet="""import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# BUG 1: Tokenizer cased vs model uncased mismatch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # CASED tokenizer
model = BertModel.from_pretrained('bert-base-uncased')  # UNCASED model

classifier = nn.Linear(768, 2)
model = model.cuda()
classifier = classifier.cuda()

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(classifier.parameters()),
    lr=2e-5
)

def collate_fn(batch):
    texts, labels = zip(*batch)
    encoding = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    # BUG 2: Not passing attention_mask to model — padding tokens affect output
    return encoding['input_ids'], torch.tensor(labels)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)

for epoch in range(5):
    model.train()
    for input_ids, labels in train_loader:
        input_ids, labels = input_ids.cuda(), labels.cuda()
        optimizer.zero_grad()
        # BUG 2 continued: No attention_mask passed to model
        outputs = model(input_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_output)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 0.69, "val_loss": 0.68, "train_accuracy": 0.55, "val_accuracy": 0.56, "learning_rate": 2e-5},
        {"epoch": 1, "train_loss": 0.52, "val_loss": 0.55, "train_accuracy": 0.68, "val_accuracy": 0.65, "learning_rate": 2e-5},
        {"epoch": 2, "train_loss": 0.40, "val_loss": 0.48, "train_accuracy": 0.75, "val_accuracy": 0.72, "learning_rate": 2e-5},
        {"epoch": 3, "train_loss": 0.32, "val_loss": 0.45, "train_accuracy": 0.80, "val_accuracy": 0.76, "learning_rate": 2e-5},
        {"epoch": 5, "train_loss": 0.25, "val_loss": 0.44, "train_accuracy": 0.84, "val_accuracy": 0.76, "learning_rate": 2e-5},
    ],
    root_cause_category="tokenizer_model_mismatch",
    root_cause_description="Two interacting bugs: (1) The tokenizer (bert-base-cased) doesn't match the model (bert-base-uncased). The cased tokenizer has a different vocabulary, so token IDs map to wrong embeddings in the uncased model. For example, 'The' and 'the' get different IDs in cased tokenizer, but uncased model only has lowercase embeddings. (2) The attention_mask is not passed to the model, so self-attention attends equally to PAD tokens, corrupting the [CLS] representation used for classification.",
    correct_fix_description="(1) Use matching tokenizer: BertTokenizer.from_pretrained('bert-base-uncased'). (2) Pass attention_mask to the model in collate_fn and forward pass.",
    correct_fix_code="""# FIX 1: Match tokenizer to model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# FIX 2: Pass attention_mask through collate_fn and to model
def collate_fn(batch):
    texts, labels = zip(*batch)
    encoding = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
    return encoding['input_ids'], encoding['attention_mask'], torch.tensor(labels)

for input_ids, attention_mask, labels in train_loader:
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    labels = labels.cuda()
    outputs = model(input_ids, attention_mask=attention_mask)
    cls_output = outputs.last_hidden_state[:, 0, :]
    logits = classifier(cls_output)""",
    diagnosis_keywords=["tokenizer", "cased", "uncased", "mismatch", "vocabulary", "attention mask", "padding", "PAD tokens", "vocab mismatch"],
    fix_keywords=["match tokenizer", "bert-base-uncased tokenizer", "attention_mask", "pass mask", "same vocabulary"],
    relevant_inspections=["inspect_data_pipeline", "inspect_model_architecture", "analyze_logs"],
    secondary_bugs=[
        {"category": "missing_attention_mask", "description": "Attention mask not passed to model — PAD tokens pollute attention"}
    ],
    inspection_data={
        "inspect_data_pipeline": {
            "inspection_type": "inspect_data_pipeline",
            "findings": "Tokenizer-Model compatibility check:\n- Tokenizer: bert-base-cased (vocab_size=28,996)\n- Model: bert-base-uncased (vocab_size=30,522)\n\n⚠️ VOCABULARY MISMATCH DETECTED:\n- Tokenizer vocabulary is CASED (preserves capitalization)\n- Model vocabulary is UNCASED (all lowercase)\n- Token ID mapping is DIFFERENT between the two\n- Example: 'Hello' → cased ID 8667, but uncased model expects ID 7592\n- ~30% of tokens map to incorrect embeddings\n\nAdditionally:\n- attention_mask is created by tokenizer but NOT passed to model\n- Padding token (ID 0) attends in self-attention, corrupting representations\n- Average padding per batch: 42% of sequence length",
            "data": {"tokenizer": "bert-base-cased", "model": "bert-base-uncased", "vocab_mismatch": True, "attention_mask_used": False}
        },
    },
))

# =============================================================================
# Scenario 5: FSDP + activation checkpointing + gradient clipping interaction
# =============================================================================
register_scenario(BugScenario(
    scenario_id="hard_fsdp_checkpoint",
    title="FSDP + Activation Checkpointing + Grad Clipping Conflict",
    difficulty="hard",
    task_id="subtle_bugs",
    description=(
        "Training a 7B parameter LLM with Fully Sharded Data Parallel (FSDP), "
        "activation checkpointing, and gradient clipping. Training is unstable — "
        "intermittent loss spikes every ~200 steps and the model fails to converge "
        "to the quality of a reference implementation."
    ),
    training_config={
        "model": "LLaMA-7B architecture",
        "dataset": "RedPajama subset",
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "batch_size": 4,
        "gradient_accumulation": 8,
        "num_gpus": 8,
        "framework": "FSDP",
        "activation_checkpointing": True,
        "gradient_clipping": 1.0,
        "warmup_steps": 1000,
    },
    code_snippet="""import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

# Mixed precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,  # BUG 1: Should be float32 for stable all-reduce
    buffer_dtype=torch.float16,
)

model = LLaMA7B(config)
model = FSDP(model, mixed_precision=mp_policy)

# Apply activation checkpointing
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda m: isinstance(m, TransformerBlock)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

for step, batch in enumerate(train_loader):
    loss = model(batch['input_ids'], batch['labels'])
    loss = loss / 8  # gradient accumulation

    loss.backward()

    if (step + 1) % 8 == 0:
        # BUG 2: grad clip AFTER FSDP unshard - need to use FSDP-aware clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # BUG 3: Missing gradient scaler for fp16 training
        optimizer.step()
        optimizer.zero_grad()
""",
    training_logs_data=[
        {"epoch": 0, "step": 100, "train_loss": 8.5, "grad_norm": 2.5, "learning_rate": 3e-5, "gpu_memory_mb": 35000},
        {"epoch": 0, "step": 200, "train_loss": 6.2, "grad_norm": 1.8, "learning_rate": 6e-5, "gpu_memory_mb": 35000},
        {"epoch": 0, "step": 400, "train_loss": 18.5, "grad_norm": 45.2, "learning_rate": 1.2e-4, "gpu_memory_mb": 35000,
         "extra_metrics": {"note": "LOSS SPIKE — recovered after 20 steps"}},
        {"epoch": 0, "step": 600, "train_loss": 4.8, "grad_norm": 1.5, "learning_rate": 1.8e-4, "gpu_memory_mb": 35000},
        {"epoch": 0, "step": 800, "train_loss": 15.2, "grad_norm": 38.7, "learning_rate": 2.4e-4, "gpu_memory_mb": 35000,
         "extra_metrics": {"note": "Second LOSS SPIKE"}},
        {"epoch": 0, "step": 1000, "train_loss": 4.2, "grad_norm": 1.2, "learning_rate": 3e-4, "gpu_memory_mb": 35000},
    ],
    root_cause_category="fsdp_checkpoint_interaction",
    root_cause_description="Three interacting issues: (1) reduce_dtype=float16 causes precision loss during gradient all-reduce operations, leading to intermittent gradient corruption on certain batches. Should be float32. (2) Standard clip_grad_norm_ doesn't work correctly with FSDP's sharded parameters — it only sees the local shard, not the full gradient. Must use FSDP's clip_grad_norm_ method. (3) No GradScaler is used despite fp16 parameter dtype, meaning very small gradients get flushed to zero randomly.",
    correct_fix_description="(1) Set reduce_dtype=torch.float32 for stable gradient reduction. (2) Use model.clip_grad_norm_(1.0) for FSDP-aware clipping. (3) Use FSDP's native mixed precision or add GradScaler for fp16 stability.",
    correct_fix_code="""# FIX 1: Use float32 for gradient reduction
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float32,  # FIX: Stable gradient reduction
    buffer_dtype=torch.float16,
)

# FIX 2: FSDP-aware gradient clipping
if (step + 1) % 8 == 0:
    model.clip_grad_norm_(max_norm=1.0)  # FIX: FSDP method, not torch.nn.utils
    optimizer.step()
    optimizer.zero_grad()

# FIX 3: Consider using bfloat16 instead of float16 to avoid need for GradScaler
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,  # bfloat16 has same range as float32
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)""",
    diagnosis_keywords=["FSDP", "activation checkpointing", "gradient clipping", "loss spike", "reduce_dtype", "float16", "all-reduce", "sharded", "mixed precision"],
    fix_keywords=["reduce_dtype=float32", "model.clip_grad_norm_", "FSDP clip", "bfloat16", "GradScaler", "float32 reduction"],
    relevant_inspections=["analyze_logs", "inspect_gradients", "inspect_model_architecture"],
    secondary_bugs=[
        {"category": "wrong_clip_method", "description": "Using torch.nn.utils.clip_grad_norm_ instead of FSDP's method"},
        {"category": "missing_grad_scaler", "description": "No GradScaler for fp16 training — small gradients flushed to zero"}
    ],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "Training shows intermittent loss spikes:\n- Step 400: loss=18.5 (grad_norm=45.2) — 4x expected loss\n- Step 800: loss=15.2 (grad_norm=38.7) — similar spike\n- Spikes occur every ~200-400 steps, then training recovers\n- Between spikes, loss decreases normally (8.5 → 6.2 → 4.8 → 4.2)\n\nThis pattern suggests intermittent gradient corruption, NOT a data issue. The gradient clipping (max_norm=1.0) is not effectively preventing the spikes (grad_norm reaches 45+), suggesting the clipping mechanism is not working correctly with FSDP.",
            "data": {"spike_frequency": "every ~200-400 steps", "max_spike_loss": 18.5, "clip_effective": False}
        },
        "inspect_gradients": {
            "inspection_type": "inspect_gradients",
            "findings": "Gradient analysis:\n- clip_grad_norm_ reports max_norm=1.0 but actual post-clip norm is often >10\n- This is because torch.nn.utils.clip_grad_norm_ only sees the LOCAL SHARD with FSDP\n- Each rank clips its own shard to norm 1.0, but the FULL gradient norm is √(num_shards) × local_norm\n- With 8 GPUs: effective max norm = √8 × 1.0 = 2.83, NOT 1.0\n\nAdditionally, reduce_dtype=float16 causes precision loss during gradient all-reduce:\n- Gradient values < 6e-8 are flushed to zero in fp16\n- ~2% of gradient elements are affected per step\n- On some batches, this accumulated error triggers instability",
            "data": {"clip_works_correctly": False, "effective_max_norm": 2.83, "reduce_precision_loss": "2% of grad elements"}
        },
    },
))
