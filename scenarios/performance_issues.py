"""
Task 2: Performance Degradations (Medium)

Subtle bugs that don't crash training but cause poor performance.
These require deeper analysis of metrics, code, and data pipeline.
"""

try:
    from . import BugScenario, register_scenario
except ImportError:
    from scenarios import BugScenario, register_scenario

# =============================================================================
# Scenario 1: Data leakage between train/val
# =============================================================================
register_scenario(BugScenario(
    scenario_id="med_data_leakage",
    title="Data Leakage Between Train/Validation Sets",
    difficulty="medium",
    task_id="performance_issues",
    description=(
        "Training a medical image classifier. The model achieves 98% validation accuracy "
        "during training but only 62% accuracy on the held-out test set. There is a "
        "suspiciously large gap between validation and test performance."
    ),
    training_config={
        "model": "DenseNet-121",
        "dataset": "Chest X-Ray (Pneumonia Detection)",
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 25,
        "train_size": 4000,
        "val_size": 1000,
        "test_size": 1000,
        "data_augmentation": "RandomHorizontalFlip, RandomRotation(10)",
    },
    code_snippet="""import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# Load full dataset
full_dataset = datasets.ImageFolder('chest_xray/all', transform=transform)

# BUG: Data augmentation applied BEFORE splitting, AND split is random
# without controlling for patient IDs. Same patients appear in train AND val.
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# BUG: val_dataset still uses augmentation transforms!

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.densenet121(num_classes=2).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(25):
    # Training loop...
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            # ...
            pass
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 0.65, "val_loss": 0.62, "train_accuracy": 0.72, "val_accuracy": 0.74, "learning_rate": 0.0001},
        {"epoch": 5, "train_loss": 0.22, "val_loss": 0.18, "train_accuracy": 0.92, "val_accuracy": 0.94, "learning_rate": 0.0001},
        {"epoch": 10, "train_loss": 0.08, "val_loss": 0.05, "train_accuracy": 0.97, "val_accuracy": 0.98, "learning_rate": 0.0001},
        {"epoch": 15, "train_loss": 0.04, "val_loss": 0.03, "train_accuracy": 0.98, "val_accuracy": 0.99, "learning_rate": 0.0001},
        {"epoch": 25, "train_loss": 0.02, "val_loss": 0.02, "train_accuracy": 0.99, "val_accuracy": 0.99, "learning_rate": 0.0001,
         "extra_metrics": {"test_accuracy": 0.62, "test_loss": 1.15}},
    ],
    root_cause_category="data_leakage",
    root_cause_description="Two issues cause data leakage: (1) The dataset is split randomly by image rather than by patient, so augmented versions of the same patient's images appear in both train and validation sets. (2) Data augmentation transforms are applied to the validation set as well, meaning the model may see near-identical augmented versions of the same images during validation. This makes validation accuracy unrealistically high.",
    correct_fix_description="(1) Split data by patient ID, not by individual image, to prevent the same patient appearing in both splits. (2) Use separate transforms for train (with augmentation) and val (without augmentation). (3) Consider using a pre-defined train/val/test split.",
    correct_fix_code="""# Fix: Separate transforms and split by patient
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
# Use pre-split directories to avoid patient leakage
train_dataset = datasets.ImageFolder('chest_xray/train', transform=train_transform)
val_dataset = datasets.ImageFolder('chest_xray/val', transform=val_transform)""",
    diagnosis_keywords=["data leakage", "leakage", "val accuracy too high", "test accuracy low", "same patient", "overfitting", "augmentation on val", "split"],
    fix_keywords=["patient split", "separate transforms", "no augmentation val", "pre-split", "group split", "stratified"],
    relevant_inspections=["analyze_logs", "inspect_data_pipeline"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "ANOMALY DETECTED: Validation accuracy (99%) is HIGHER than training accuracy (99%) and nearly perfect, but test accuracy is only 62%. This 37-point gap between validation and test performance is a strong indicator of data leakage. The validation set is not representative of unseen data. Val loss (0.02) tracks training loss almost identically, suggesting the val set is not independent.",
            "data": {"val_accuracy": 0.99, "test_accuracy": 0.62, "gap": 0.37, "leakage_suspected": True}
        },
        "inspect_data_pipeline": {
            "inspection_type": "inspect_data_pipeline",
            "findings": "Data pipeline analysis:\n1. ISSUE: Same transform (with augmentation) used for both train and val\n2. ISSUE: random_split() splits by image index, not patient ID\n   - Found 312 patient IDs in training set that also appear in validation set\n   - This means 31% of validation images are from patients the model trained on\n3. ISSUE: Augmentation applied to validation (RandomHorizontalFlip, RandomRotation)\n\nThese issues cause severe data leakage, inflating validation metrics.",
            "data": {"shared_patients": 312, "augmented_val": True, "split_method": "random_image"}
        },
    },
))

# =============================================================================
# Scenario 2: BatchNorm in eval mode during training
# =============================================================================
register_scenario(BugScenario(
    scenario_id="med_batchnorm_eval",
    title="BatchNorm in Eval Mode During Training",
    difficulty="medium",
    task_id="performance_issues",
    description=(
        "Fine-tuning a pretrained ResNet-50 on a custom dataset. "
        "The loss barely decreases and the model seems to not train at all. "
        "Accuracy stays around random chance. However, the same code worked before."
    ),
    training_config={
        "model": "ResNet-50 (pretrained)",
        "dataset": "Custom Product Images (50 classes)",
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 30,
        "pretrained": True,
        "fine_tuning": "full model",
    },
    code_snippet="""import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 50)  # Replace classifier for 50 classes
model = model.cuda()

# Freeze batch norm layers (common practice for fine-tuning)
# BUG: This sets the ENTIRE model to eval mode, not just batchnorm
model.eval()  # BUG: Should only freeze BN, not entire model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(30):
    # BUG: model.train() is NEVER called, so model stays in eval mode
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Validation...
        pass
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 3.91, "val_loss": 3.92, "train_accuracy": 0.02, "val_accuracy": 0.02, "grad_norm": 0.01, "learning_rate": 0.01},
        {"epoch": 5, "train_loss": 3.88, "val_loss": 3.90, "train_accuracy": 0.03, "val_accuracy": 0.02, "grad_norm": 0.008, "learning_rate": 0.01},
        {"epoch": 10, "train_loss": 3.85, "val_loss": 3.88, "train_accuracy": 0.04, "val_accuracy": 0.03, "grad_norm": 0.005, "learning_rate": 0.01},
        {"epoch": 20, "train_loss": 3.82, "val_loss": 3.87, "train_accuracy": 0.05, "val_accuracy": 0.03, "grad_norm": 0.003, "learning_rate": 0.01},
        {"epoch": 30, "train_loss": 3.80, "val_loss": 3.86, "train_accuracy": 0.05, "val_accuracy": 0.04, "grad_norm": 0.002, "learning_rate": 0.01},
    ],
    root_cause_category="batchnorm_eval_mode",
    root_cause_description="model.eval() is called before training and model.train() is never called before the training loop. This keeps all BatchNorm layers in eval mode, which uses running statistics instead of batch statistics. With the pretrained model's running statistics (from ImageNet), the normalization doesn't match the new dataset's distribution. Additionally, dropout is disabled, reducing regularization.",
    correct_fix_description="Call model.train() at the start of each training epoch. If you want to freeze only BatchNorm, iterate through modules and set only BN layers to eval mode.",
    correct_fix_code="""for epoch in range(30):
    model.train()  # FIX: Set model to training mode each epoch
    for images, labels in train_loader:
        # ... training loop

    model.eval()  # Set to eval for validation
    with torch.no_grad():
        # ... validation""",
    diagnosis_keywords=["eval mode", "model.eval", "train mode", "batchnorm", "batch norm", "batch normalization", "running statistics", "not training"],
    fix_keywords=["model.train()", "train mode", "set train", "eval to train", "training mode"],
    relevant_inspections=["analyze_logs", "inspect_model_architecture", "inspect_gradients"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "Loss barely decreases over 30 epochs (3.91 → 3.80). For a pretrained ResNet-50 with lr=0.01, we'd expect rapid convergence on a small custom dataset. Accuracy is ~5% (vs 2% random chance for 50 classes). Gradient norms are extremely small (0.01 → 0.002) and decreasing. The model appears to be barely learning.",
            "data": {"loss_change": 0.11, "expected_loss_change": 3.0, "learning_rate_effective": "very_low"}
        },
        "inspect_gradients": {
            "inspection_type": "inspect_gradients",
            "findings": "Gradient norms are extremely small (0.01) and continue to decrease. A frozen BatchNorm normalizes activations using pre-trained running statistics, which may not match the new data distribution. This causes the network to compute activations in a different range, resulting in tiny loss gradients.",
            "data": {"avg_grad_norm": 0.005, "expected_range": "0.1-1.0", "gradient_flow": "severely_weak"}
        },
        "inspect_model_architecture": {
            "inspection_type": "inspect_model_architecture",
            "findings": "Model inspection:\n- model.training = False (MODEL IS IN EVAL MODE)\n- All BatchNorm layers: training=False (using running_mean/running_var from ImageNet)\n- Dropout layers: disabled (training=False)\n\n⚠️ The model was set to eval mode with model.eval() but model.train() is never called before the training loop. The model trains the entire time in eval mode, meaning:\n1. BatchNorm uses ImageNet statistics (not current batch)\n2. Dropout is disabled\n3. This drastically affects gradient flow and normalization",
            "data": {"model_training_flag": False, "bn_training": False, "dropout_active": False}
        },
    },
))

# =============================================================================
# Scenario 3: Memory leak from storing computation graph
# =============================================================================
register_scenario(BugScenario(
    scenario_id="med_memory_leak",
    title="Memory Leak - Storing Computation Graph",
    difficulty="medium",
    task_id="performance_issues",
    description=(
        "Training a Transformer model for NLP. Training becomes progressively slower "
        "and eventually crashes with CUDA out-of-memory (OOM) error after a few epochs, "
        "even though the batch size seems reasonable for the GPU."
    ),
    training_config={
        "model": "Transformer (6 layers, d=512)",
        "dataset": "WikiText-103",
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 20,
        "seq_length": 512,
        "gradient_accumulation_steps": 1,
    },
    code_snippet="""import torch
import torch.nn as nn

model = TransformerLM(vocab_size=30000, d_model=512, nhead=8, num_layers=6).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

running_loss = 0.0
loss_history = []

for epoch in range(20):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, 30000), targets.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss  # BUG: Storing tensor with grad, not loss.item()
        loss_history.append(loss)  # BUG: Appending tensor keeps computation graph

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: loss={running_loss / (batch_idx + 1)}")
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 8.5, "gpu_memory_mb": 4200, "throughput": 120.0, "learning_rate": 0.0001},
        {"epoch": 0, "step": 500, "train_loss": 7.2, "gpu_memory_mb": 8500, "throughput": 95.0, "learning_rate": 0.0001},
        {"epoch": 0, "step": 1000, "train_loss": 6.8, "gpu_memory_mb": 14200, "throughput": 65.0, "learning_rate": 0.0001},
        {"epoch": 0, "step": 1500, "train_loss": 6.5, "gpu_memory_mb": 22000, "throughput": 40.0, "learning_rate": 0.0001},
        {"epoch": 0, "step": 2000, "train_loss": 6.3, "gpu_memory_mb": 31500, "throughput": 20.0, "learning_rate": 0.0001},
    ],
    error_message="RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 40.00 GiB total capacity; 38.21 GiB already allocated)",
    root_cause_category="memory_leak_computation_graph",
    root_cause_description="Two bugs: (1) `running_loss += loss` accumulates the loss tensor (with its computation graph) instead of the detached scalar `loss.item()`. (2) `loss_history.append(loss)` stores full loss tensors, keeping the entire computation graph alive in memory. Each batch's full backward graph persists, causing GPU memory to grow linearly with the number of batches until OOM.",
    correct_fix_description="Use loss.item() to extract the scalar value before accumulating, and store .item() values in the loss history instead of full tensors.",
    correct_fix_code="""running_loss += loss.item()  # FIX: .item() detaches and converts to Python float
loss_history.append(loss.item())  # FIX: Store scalar, not tensor""",
    diagnosis_keywords=["memory leak", "computation graph", ".item()", "tensor not detached", "OOM", "out of memory", "graph retained", "loss.item"],
    fix_keywords=["loss.item()", ".item()", "detach", "scalar", "float", "detach()"],
    relevant_inspections=["analyze_logs", "check_device_placement"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "GPU memory usage grows linearly and unboundedly:\n- Step 0: 4,200 MB\n- Step 500: 8,500 MB (+4,300 MB)\n- Step 1000: 14,200 MB (+5,700 MB)\n- Step 1500: 22,000 MB (+7,800 MB)\n- Step 2000: 31,500 MB (+9,500 MB → OOM)\n\nThroughput degrades from 120 → 20 samples/sec. This is a classic memory leak pattern where computation graphs are being retained.",
            "data": {"memory_trend": "linear_growth", "leak_rate_mb_per_step": 15, "oom_step": 2000}
        },
        "check_device_placement": {
            "inspection_type": "check_device_placement",
            "findings": "GPU memory analysis:\n- Model parameters: ~150 MB (fixed)\n- Optimizer states: ~450 MB (fixed)\n- Activations (per batch): ~800 MB (should be freed after backward)\n\n⚠️ LEAKED TENSORS DETECTED:\n- `running_loss`: Tensor with requires_grad=True, retains full backward graph\n- `loss_history`: List of 2000+ loss tensors, each retaining its backward graph\n- Each retained graph holds ~15 MB of intermediate activations\n\nFix: Use loss.item() to detach the scalar from the computation graph.",
            "data": {"leaked_tensors": 2000, "leak_source": ["running_loss", "loss_history"], "fix": "use .item()"}
        },
    },
))

# =============================================================================
# Scenario 4: Severe class imbalance without compensation
# =============================================================================
register_scenario(BugScenario(
    scenario_id="med_class_imbalance",
    title="Severe Class Imbalance Without Compensation",
    difficulty="medium",
    task_id="performance_issues",
    description=(
        "Training a fraud detection model on transaction data. The model achieves 99.5% "
        "overall accuracy but fails almost completely on the minority class (fraud). "
        "Precision and recall for fraud detection are very low."
    ),
    training_config={
        "model": "3-layer MLP",
        "dataset": "Transaction Data (fraud detection)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 20,
        "class_distribution": {"normal": 99500, "fraud": 500},
        "loss_function": "CrossEntropyLoss (no weights)",
    },
    code_snippet="""import torch
import torch.nn as nn

class FraudDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

model = FraudDetector(input_dim=30).cuda()
criterion = nn.CrossEntropyLoss()  # BUG: No class weights for 200:1 imbalance
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# BUG: No oversampling or undersampling strategy

for epoch in range(20):
    model.train()
    for features, labels in train_loader:
        features, labels = features.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 0.15, "val_loss": 0.12, "train_accuracy": 0.95, "val_accuracy": 0.96, "learning_rate": 0.001,
         "extra_metrics": {"fraud_precision": 0.10, "fraud_recall": 0.15, "fraud_f1": 0.12}},
        {"epoch": 5, "train_loss": 0.02, "val_loss": 0.02, "train_accuracy": 0.995, "val_accuracy": 0.994, "learning_rate": 0.001,
         "extra_metrics": {"fraud_precision": 0.30, "fraud_recall": 0.08, "fraud_f1": 0.13}},
        {"epoch": 10, "train_loss": 0.008, "val_loss": 0.009, "train_accuracy": 0.998, "val_accuracy": 0.996, "learning_rate": 0.001,
         "extra_metrics": {"fraud_precision": 0.50, "fraud_recall": 0.04, "fraud_f1": 0.07}},
        {"epoch": 20, "train_loss": 0.003, "val_loss": 0.005, "train_accuracy": 0.999, "val_accuracy": 0.997, "learning_rate": 0.001,
         "extra_metrics": {"fraud_precision": 0.67, "fraud_recall": 0.02, "fraud_f1": 0.04}},
    ],
    root_cause_category="class_imbalance",
    root_cause_description="The dataset has a 200:1 class imbalance (99,500 normal vs 500 fraud). Using standard CrossEntropyLoss without class weights, the model learns to predict 'normal' for everything, achieving 99.5%+ accuracy while detecting almost no fraud. No sampling strategy (oversampling/undersampling) is used, so the model rarely sees fraud examples.",
    correct_fix_description="Apply one or more compensating strategies: (1) Use weighted CrossEntropyLoss with class weights inversely proportional to frequency, (2) Use oversampling (e.g., WeightedRandomSampler), (3) Use focal loss, or (4) SMOTE for synthetic minority samples.",
    correct_fix_code="""# Fix option 1: Weighted loss
class_weights = torch.tensor([1.0, 200.0]).cuda()  # Inverse frequency
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Fix option 2: Weighted sampler
from torch.utils.data import WeightedRandomSampler
sample_weights = [200.0 if label == 1 else 1.0 for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)""",
    diagnosis_keywords=["class imbalance", "imbalance", "minority class", "fraud recall", "unbalanced", "skewed", "predicts majority"],
    fix_keywords=["class weights", "weighted loss", "oversampling", "WeightedRandomSampler", "focal loss", "SMOTE", "undersample"],
    relevant_inspections=["analyze_logs", "inspect_data_pipeline"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "Overall accuracy is 99.7% but per-class metrics reveal a critical issue:\n- Fraud precision: 0.67 (few predictions, some correct)\n- Fraud recall: 0.02 (model misses 98% of fraud!)\n- Fraud F1: 0.04\n\nThe model has essentially learned to always predict 'normal', achieving high overall accuracy due to the class distribution (99.5% normal). This is a classic class imbalance problem.",
            "data": {"overall_accuracy": 0.997, "minority_recall": 0.02, "majority_ratio": 0.995}
        },
        "inspect_data_pipeline": {
            "inspection_type": "inspect_data_pipeline",
            "findings": "Dataset distribution analysis:\n- Class 0 (Normal): 99,500 samples (99.5%)\n- Class 1 (Fraud): 500 samples (0.5%)\n- Imbalance ratio: 199:1\n\nBatch analysis (batch_size=256):\n- Expected fraud samples per batch: 1.28\n- ~22% of batches contain ZERO fraud samples\n\nNo compensating strategy detected:\n- CrossEntropyLoss: weight=None (no class weights)\n- Sampler: default (no oversampling)\n- No focal loss or other imbalance-aware loss",
            "data": {"imbalance_ratio": 199, "minority_per_batch": 1.28, "compensation": "none"}
        },
    },
))

# =============================================================================
# Scenario 5: Gradient vanishing in deep network
# =============================================================================
register_scenario(BugScenario(
    scenario_id="med_vanishing_gradients",
    title="Vanishing Gradients in Deep Network",
    difficulty="medium",
    task_id="performance_issues",
    description=(
        "Training a deep 50-layer MLP (no residual connections) for tabular data. "
        "The model trains incredibly slowly and barely improves over many epochs. "
        "The first few layers seem to not update at all."
    ),
    training_config={
        "model": "50-layer MLP (no skip connections)",
        "dataset": "Tabular regression",
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 100,
        "hidden_size": 256,
        "activation": "Sigmoid",
        "initialization": "default (PyTorch)",
    },
    code_snippet="""import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=50):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = DeepMLP(input_dim=100, hidden_dim=256, output_dim=1, num_layers=50).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for features, targets in train_loader:
        features, targets = features.cuda(), targets.cuda()
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 1.25, "val_loss": 1.24, "grad_norm": 0.0001, "learning_rate": 0.01},
        {"epoch": 10, "train_loss": 1.24, "val_loss": 1.24, "grad_norm": 0.00008, "learning_rate": 0.01},
        {"epoch": 30, "train_loss": 1.22, "val_loss": 1.23, "grad_norm": 0.00005, "learning_rate": 0.01},
        {"epoch": 50, "train_loss": 1.20, "val_loss": 1.22, "grad_norm": 0.00003, "learning_rate": 0.01},
        {"epoch": 100, "train_loss": 1.18, "val_loss": 1.21, "grad_norm": 0.00002, "learning_rate": 0.01},
    ],
    root_cause_category="vanishing_gradients",
    root_cause_description="Three factors combine to cause vanishing gradients: (1) Sigmoid activation saturates, with derivatives < 0.25 everywhere, so 50 layers multiply: 0.25^50 ≈ 0. (2) No skip/residual connections to allow gradient flow. (3) Default initialization doesn't account for the deep architecture. The early layers receive near-zero gradients and cannot learn.",
    correct_fix_description="Multiple fixes needed: (1) Replace Sigmoid with ReLU or LeakyReLU (gradients are 0 or 1, not bounded). (2) Add residual/skip connections. (3) Use He initialization for ReLU activations. (4) Consider reducing depth or using BatchNorm.",
    correct_fix_code="""# Fix: Use ReLU, residual connections, and proper init
class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=50):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers - 2)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        for layer, norm in zip(self.layers, self.norms):
            x = x + F.relu(norm(layer(x)))  # Residual connection
        return self.output_proj(x)""",
    diagnosis_keywords=["vanishing gradient", "sigmoid", "deep network", "gradient vanish", "too deep", "early layers", "no update", "residual"],
    fix_keywords=["ReLU", "residual", "skip connection", "He init", "kaiming", "LayerNorm", "BatchNorm", "reduce depth"],
    relevant_inspections=["inspect_gradients", "inspect_model_architecture", "analyze_logs"],
    inspection_data={
        "inspect_gradients": {
            "inspection_type": "inspect_gradients",
            "findings": "Per-layer gradient analysis (epoch 0):\n- Layer 50 (output): grad norm = 0.05\n- Layer 45: grad norm = 0.001\n- Layer 40: grad norm = 0.00002\n- Layer 30: grad norm = 1.2e-8\n- Layer 20: grad norm = 5.4e-15\n- Layer 10: grad norm = 2.1e-22\n- Layer 1 (input): grad norm = 8.7e-30\n\nGradients decay exponentially through layers. The first 30 layers have essentially zero gradient and cannot learn. This is classic vanishing gradient caused by Sigmoid activation in a deep network.",
            "data": {"gradient_decay": "exponential", "effective_layers": 5, "dead_layers": 45, "activation": "Sigmoid"}
        },
        "inspect_model_architecture": {
            "inspection_type": "inspect_model_architecture",
            "findings": "Architecture:\n- 50-layer MLP with Sigmoid activations\n- No skip/residual connections\n- No normalization layers (BatchNorm/LayerNorm)\n- Default PyTorch initialization\n\n⚠️ CRITICAL ISSUES:\n1. Sigmoid max derivative = 0.25 → after 50 layers: 0.25^50 ≈ 0\n2. No residual connections to bypass gradient bottleneck\n3. Network is far too deep for a plain MLP architecture",
            "data": {"depth": 50, "activation": "Sigmoid", "has_residual": False, "has_norm": False}
        },
    },
))
