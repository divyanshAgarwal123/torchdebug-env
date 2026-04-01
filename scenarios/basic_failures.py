"""
Task 1: Basic Training Failures (Easy)

Common PyTorch mistakes that cause obvious training failures.
These are bugs that a junior ML engineer would make.
"""

try:
    from . import BugScenario, register_scenario
except ImportError:
    from scenarios import BugScenario, register_scenario

# =============================================================================
# Scenario 1: Learning rate too high → loss explodes to NaN
# =============================================================================
register_scenario(BugScenario(
    scenario_id="easy_lr_too_high",
    title="Learning Rate Too High",
    difficulty="easy",
    task_id="basic_failures",
    description=(
        "Training a ResNet-18 image classifier on CIFAR-10. "
        "The training starts but the loss quickly explodes to NaN after a few epochs. "
        "The model fails to converge and produces meaningless predictions."
    ),
    training_config={
        "model": "ResNet-18",
        "dataset": "CIFAR-10",
        "optimizer": "SGD",
        "learning_rate": 10.0,
        "momentum": 0.9,
        "batch_size": 128,
        "epochs": 50,
        "weight_decay": 1e-4,
        "lr_scheduler": "None",
    },
    code_snippet="""import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet18(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=10.0,       # BUG: Learning rate is way too high
    momentum=0.9,
    weight_decay=1e-4
)

for epoch in range(50):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 2.45, "val_loss": 2.50, "train_accuracy": 0.12, "grad_norm": 15.2, "learning_rate": 10.0},
        {"epoch": 1, "train_loss": 8.73, "val_loss": 9.10, "train_accuracy": 0.10, "grad_norm": 145.8, "learning_rate": 10.0},
        {"epoch": 2, "train_loss": 52.1, "val_loss": 55.3, "train_accuracy": 0.10, "grad_norm": 1240.5, "learning_rate": 10.0},
        {"epoch": 3, "train_loss": float('inf'), "val_loss": float('inf'), "train_accuracy": 0.10, "grad_norm": float('inf'), "learning_rate": 10.0},
        {"epoch": 4, "train_loss": float('nan'), "val_loss": float('nan'), "train_accuracy": 0.10, "grad_norm": float('nan'), "learning_rate": 10.0},
    ],
    error_message="RuntimeWarning: overflow encountered in float_scalars. Loss became NaN at epoch 4.",
    root_cause_category="learning_rate_too_high",
    root_cause_description="The learning rate is set to 10.0, which is extremely high for SGD on CIFAR-10. A typical learning rate for this setup is 0.01-0.1. The high LR causes gradient updates to overshoot, leading to exponentially growing loss and eventually NaN values.",
    correct_fix_description="Reduce the learning rate to a reasonable value like 0.01 or 0.1 for SGD with momentum on CIFAR-10.",
    correct_fix_code="optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)",
    diagnosis_keywords=["learning rate", "lr", "too high", "too large", "exploding", "nan", "overflow", "diverge", "10.0"],
    fix_keywords=["reduce", "lower", "decrease", "lr", "learning rate", "0.01", "0.1", "0.001"],
    relevant_inspections=["analyze_logs", "inspect_gradients"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "Training loss shows exponential growth: 2.45 → 8.73 → 52.1 → inf → NaN. Gradient norms also explode: 15.2 → 145.8 → 1240.5 → inf. This is a clear sign of divergent training, typically caused by a learning rate that is too high.",
            "data": {"loss_trend": "exponential_growth", "grad_norm_trend": "exponential_growth", "nan_epoch": 4}
        },
        "inspect_gradients": {
            "inspection_type": "inspect_gradients",
            "findings": "Gradient norms are extremely large and growing exponentially. At epoch 0, the max gradient norm across layers is 15.2 (already high for a well-initialized network). By epoch 2, gradients have grown to 1240.5. The parameter updates (lr * gradient) are larger than the parameters themselves, causing instability.",
            "data": {"max_grad_norm_epoch0": 15.2, "max_grad_norm_epoch2": 1240.5, "parameter_scale": 0.5, "update_scale_epoch0": 152.0}
        },
    },
))

# =============================================================================
# Scenario 2: Missing .to(device) → device mismatch
# =============================================================================
register_scenario(BugScenario(
    scenario_id="easy_device_mismatch",
    title="Device Mismatch Error",
    difficulty="easy",
    task_id="basic_failures",
    description=(
        "Training a simple CNN on MNIST using GPU. The training immediately crashes "
        "with a RuntimeError about tensors being on different devices."
    ),
    training_config={
        "model": "SimpleCNN",
        "dataset": "MNIST",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "device": "cuda:0",
    },
    code_snippet="""import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda:0")
model = SimpleCNN()
# BUG: model not moved to GPU with model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # CRASH: model on CPU, data on GPU
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
""",
    training_logs_data=[],
    error_message=(
        "RuntimeError: Expected all tensors to be on the same device, "
        "but found at least two devices, cuda:0 and cpu! "
        "(when checking argument for argument mat1 in method addmm)"
    ),
    root_cause_category="device_mismatch",
    root_cause_description="The model is not moved to GPU with model.to(device). While the input data is correctly moved to cuda:0, the model parameters remain on CPU, causing a device mismatch when the forward pass tries to multiply CPU weights with GPU input tensors.",
    correct_fix_description="Add model.to(device) or model = model.to(device) after creating the model, before training.",
    correct_fix_code="model = SimpleCNN().to(device)",
    diagnosis_keywords=["device", "mismatch", "cuda", "cpu", "gpu", ".to(", "not moved", "different devices"],
    fix_keywords=["model.to(device)", ".to(device)", "move model", "model.cuda()", "to gpu"],
    relevant_inspections=["check_device_placement"],
    inspection_data={
        "check_device_placement": {
            "inspection_type": "check_device_placement",
            "findings": "Device placement analysis:\n- Model parameters: ALL on cpu\n- Input data: moved to cuda:0 via images.to(device)\n- Labels: moved to cuda:0 via labels.to(device)\n- Criterion: on cpu (follows model)\n\nMISMATCH DETECTED: Model is on cpu but input tensors are on cuda:0. The model must be moved to the same device as the data.",
            "data": {"model_device": "cpu", "data_device": "cuda:0", "mismatch": True}
        },
    },
))

# =============================================================================
# Scenario 3: Wrong loss function for task
# =============================================================================
register_scenario(BugScenario(
    scenario_id="easy_wrong_loss",
    title="Wrong Loss Function",
    difficulty="easy",
    task_id="basic_failures",
    description=(
        "Training a classifier for 10-class image classification. "
        "The model trains without errors but accuracy stays at ~10% (random) "
        "even though the loss seems to decrease. The model outputs look wrong."
    ),
    training_config={
        "model": "ResNet-18",
        "dataset": "CIFAR-10 (10 classes)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
        "loss_function": "MSELoss",
        "num_classes": 10,
    },
    code_snippet="""import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet18(num_classes=10).cuda()
criterion = nn.MSELoss()  # BUG: MSELoss is wrong for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        # BUG: MSELoss expects same shape tensors, labels need one-hot encoding
        loss = criterion(outputs, torch.nn.functional.one_hot(labels, 10).float())
        loss.backward()
        optimizer.step()
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 0.092, "val_loss": 0.091, "train_accuracy": 0.10, "val_accuracy": 0.10, "grad_norm": 0.8, "learning_rate": 0.001},
        {"epoch": 5, "train_loss": 0.085, "val_loss": 0.088, "train_accuracy": 0.12, "val_accuracy": 0.11, "grad_norm": 0.3, "learning_rate": 0.001},
        {"epoch": 10, "train_loss": 0.081, "val_loss": 0.086, "train_accuracy": 0.15, "val_accuracy": 0.13, "grad_norm": 0.15, "learning_rate": 0.001},
        {"epoch": 15, "train_loss": 0.079, "val_loss": 0.085, "train_accuracy": 0.18, "val_accuracy": 0.14, "grad_norm": 0.08, "learning_rate": 0.001},
        {"epoch": 20, "train_loss": 0.078, "val_loss": 0.085, "train_accuracy": 0.19, "val_accuracy": 0.15, "grad_norm": 0.05, "learning_rate": 0.001},
    ],
    root_cause_category="wrong_loss_function",
    root_cause_description="MSELoss is used for a classification task. MSELoss treats the problem as regression and provides weak gradient signal for classification. CrossEntropyLoss (which includes LogSoftmax + NLLLoss) is the standard loss for multi-class classification as it directly optimizes the log-probability of the correct class.",
    correct_fix_description="Replace nn.MSELoss() with nn.CrossEntropyLoss() and pass raw logits and integer class labels (no one-hot encoding needed).",
    correct_fix_code='criterion = nn.CrossEntropyLoss()\n# ...\nloss = criterion(outputs, labels)  # CrossEntropyLoss takes raw logits and integer labels',
    diagnosis_keywords=["loss function", "MSELoss", "wrong loss", "classification", "CrossEntropyLoss", "regression loss"],
    fix_keywords=["CrossEntropyLoss", "cross entropy", "replace loss", "classification loss", "NLLLoss"],
    relevant_inspections=["analyze_logs", "inspect_model_architecture"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "Loss decreases very slowly from 0.092 to 0.078 over 20 epochs. However, accuracy barely improves from 10% to ~15-19%, which is only marginally better than random (10% for 10 classes). The gradient norms are shrinking rapidly (0.8 → 0.05), suggesting the model is settling into a poor local minimum. This pattern is characteristic of using a regression loss (MSELoss) for a classification problem.",
            "data": {"loss_trend": "slow_decrease", "accuracy_trend": "near_random", "grad_trend": "vanishing"}
        },
        "inspect_model_architecture": {
            "inspection_type": "inspect_model_architecture",
            "findings": "Model: ResNet-18 with 10 output units (correct for 10-class classification).\nLoss function: nn.MSELoss() — this is a regression loss, NOT suitable for classification.\nLabels are one-hot encoded before being passed to MSELoss. This is inefficient and provides poor gradient signal compared to CrossEntropyLoss.\n\nRecommendation: Use nn.CrossEntropyLoss() with integer labels (no one-hot encoding).",
            "data": {"model_outputs": 10, "loss_type": "MSELoss", "expected_loss": "CrossEntropyLoss"}
        },
    },
))

# =============================================================================
# Scenario 4: Missing zero_grad → gradient accumulation
# =============================================================================
register_scenario(BugScenario(
    scenario_id="easy_missing_zero_grad",
    title="Missing optimizer.zero_grad()",
    difficulty="easy",
    task_id="basic_failures",
    description=(
        "Training a text classifier using a simple LSTM. The training seems to work "
        "at first but the loss is unstable and oscillates wildly. The model achieves "
        "poor performance despite using standard hyperparameters."
    ),
    training_config={
        "model": "LSTM Classifier",
        "dataset": "IMDB Sentiment",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "hidden_size": 256,
        "num_layers": 2,
    },
    code_snippet="""import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

model = LSTMClassifier(10000, 128, 256, 2).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for texts, labels in train_loader:
        texts, labels = texts.cuda(), labels.cuda()
        # BUG: Missing optimizer.zero_grad() here!
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 0.69, "val_loss": 0.70, "train_accuracy": 0.52, "grad_norm": 2.3, "learning_rate": 0.001},
        {"epoch": 1, "train_loss": 0.72, "val_loss": 0.71, "train_accuracy": 0.50, "grad_norm": 8.5, "learning_rate": 0.001},
        {"epoch": 2, "train_loss": 0.65, "val_loss": 0.73, "train_accuracy": 0.55, "grad_norm": 15.2, "learning_rate": 0.001},
        {"epoch": 3, "train_loss": 0.78, "val_loss": 0.75, "train_accuracy": 0.48, "grad_norm": 22.7, "learning_rate": 0.001},
        {"epoch": 4, "train_loss": 0.58, "val_loss": 0.80, "train_accuracy": 0.60, "grad_norm": 31.4, "learning_rate": 0.001},
        {"epoch": 5, "train_loss": 0.82, "val_loss": 0.85, "train_accuracy": 0.45, "grad_norm": 44.8, "learning_rate": 0.001},
    ],
    root_cause_category="missing_zero_grad",
    root_cause_description="optimizer.zero_grad() is missing from the training loop. Without it, gradients from each backward() call accumulate across batches, causing the effective gradient to grow linearly with the number of batches. This leads to increasingly large and unstable parameter updates.",
    correct_fix_description="Add optimizer.zero_grad() before the forward pass in the training loop, before computing the loss and calling loss.backward().",
    correct_fix_code="# Add before forward pass:\noptimizer.zero_grad()\noutputs = model(texts)",
    diagnosis_keywords=["zero_grad", "gradient accumulation", "accumulate", "not zeroed", "not cleared", "missing zero"],
    fix_keywords=["zero_grad", "optimizer.zero_grad()", "clear gradients", "reset gradients"],
    relevant_inspections=["analyze_logs", "inspect_gradients"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "Loss oscillates wildly between epochs: 0.69 → 0.72 → 0.65 → 0.78 → 0.58 → 0.82. This instability is unusual for Adam optimizer with lr=0.001. Gradient norms grow continuously: 2.3 → 8.5 → 15.2 → 22.7 → 31.4 → 44.8. The linear growth pattern of gradient norms strongly suggests gradient accumulation — gradients are not being zeroed between batches.",
            "data": {"loss_trend": "oscillating", "grad_trend": "linear_growth", "pattern": "gradient_accumulation"}
        },
        "inspect_gradients": {
            "inspection_type": "inspect_gradients",
            "findings": "Gradient analysis reveals a clear accumulation pattern:\n- After batch 1: avg grad norm = 2.3\n- After batch 50: avg grad norm = 115 (≈ 50 × 2.3)\n- After batch 100: avg grad norm = 230\n\nThe gradients grow linearly with the number of batches, confirming that optimizer.zero_grad() is never called. Each backward() pass adds gradients to the existing .grad tensors rather than replacing them.",
            "data": {"accumulation_detected": True, "growth_rate": "linear", "batch1_norm": 2.3, "batch50_norm": 115.0}
        },
    },
))

# =============================================================================
# Scenario 5: Softmax before CrossEntropyLoss (double softmax)
# =============================================================================
register_scenario(BugScenario(
    scenario_id="easy_double_softmax",
    title="Double Softmax (Softmax + CrossEntropyLoss)",
    difficulty="easy",
    task_id="basic_failures",
    description=(
        "Training a multi-class classifier. The loss decreases but the model "
        "has significantly lower accuracy than expected. Train and val accuracy "
        "plateau around 65% when similar models achieve 90%+."
    ),
    training_config={
        "model": "Custom MLP",
        "dataset": "FashionMNIST (10 classes)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 30,
    },
    code_snippet="""import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)  # BUG: Softmax applied before CrossEntropyLoss
        return x

model = MLP().cuda()
criterion = nn.CrossEntropyLoss()  # Already includes LogSoftmax internally
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)  # outputs are already softmax probabilities
        loss = criterion(outputs, labels)  # CrossEntropy applies log(softmax()) again!
        loss.backward()
        optimizer.step()
""",
    training_logs_data=[
        {"epoch": 0, "train_loss": 1.85, "val_loss": 1.82, "train_accuracy": 0.42, "val_accuracy": 0.43, "grad_norm": 0.5, "learning_rate": 0.001},
        {"epoch": 5, "train_loss": 1.62, "val_loss": 1.65, "train_accuracy": 0.55, "val_accuracy": 0.54, "grad_norm": 0.3, "learning_rate": 0.001},
        {"epoch": 10, "train_loss": 1.55, "val_loss": 1.60, "train_accuracy": 0.60, "val_accuracy": 0.58, "grad_norm": 0.15, "learning_rate": 0.001},
        {"epoch": 20, "train_loss": 1.50, "val_loss": 1.58, "train_accuracy": 0.63, "val_accuracy": 0.61, "grad_norm": 0.05, "learning_rate": 0.001},
        {"epoch": 30, "train_loss": 1.49, "val_loss": 1.58, "train_accuracy": 0.65, "val_accuracy": 0.62, "grad_norm": 0.02, "learning_rate": 0.001},
    ],
    root_cause_category="double_softmax",
    root_cause_description="F.softmax() is applied in the model's forward() method, but nn.CrossEntropyLoss already includes LogSoftmax internally. This creates a 'double softmax' which squashes the logits and weakens the gradient signal, preventing the model from learning sharp class boundaries.",
    correct_fix_description="Remove the F.softmax() from the model's forward method. CrossEntropyLoss expects raw logits, not probabilities. Alternatively, if you want to keep softmax in forward(), use nn.NLLLoss() instead of CrossEntropyLoss and apply log_softmax instead of softmax.",
    correct_fix_code="# Remove softmax from forward():\ndef forward(self, x):\n    x = x.view(-1, 784)\n    x = F.relu(self.fc1(x))\n    x = F.relu(self.fc2(x))\n    x = self.fc3(x)  # Return raw logits\n    return x",
    diagnosis_keywords=["double softmax", "softmax", "CrossEntropyLoss", "log_softmax", "probabilities", "squash", "logits"],
    fix_keywords=["remove softmax", "raw logits", "no softmax", "NLLLoss", "remove F.softmax"],
    relevant_inspections=["analyze_logs", "inspect_model_architecture"],
    inspection_data={
        "analyze_logs": {
            "inspection_type": "analyze_logs",
            "findings": "The model shows signs of learning but plateaus at ~65% accuracy (val: 62%), far below the expected 90%+ for FashionMNIST with this architecture. The loss decreases very slowly and gradient norms become very small (0.02), suggesting gradient signal is being suppressed. The training loss (1.49) is also unusually high for 30 epochs of training.",
            "data": {"accuracy_plateau": 0.65, "expected_accuracy": 0.90, "grad_signal": "weak"}
        },
        "inspect_model_architecture": {
            "inspection_type": "inspect_model_architecture",
            "findings": "Architecture check:\n- Model forward() applies F.softmax(x, dim=1) before returning output\n- Loss function is nn.CrossEntropyLoss() which internally computes log(softmax(x))\n\n⚠️ DOUBLE SOFTMAX DETECTED: The model output is softmax(logits), which is then passed to CrossEntropyLoss which computes log(softmax(softmax(logits))). This double application of softmax compresses the output distribution and severely weakens gradient signal.\n\nFix: Either remove F.softmax from forward() or use nn.NLLLoss() with F.log_softmax().",
            "data": {"has_softmax": True, "loss_type": "CrossEntropyLoss", "double_softmax": True}
        },
    },
))
