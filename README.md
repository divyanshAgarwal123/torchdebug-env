---
title: TorchDebug OpenEnv
emoji: 🔥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# 🔥 TorchDebug — PyTorch Training Run Debugger

> **An OpenEnv environment that challenges AI agents to diagnose and fix real-world PyTorch training failures.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-BSD--3-green)](LICENSE)

## 🎯 What is TorchDebug?

TorchDebug simulates the real work of an ML engineer debugging broken training runs. A training run is presented with:

- **📊 Training logs** showing loss/accuracy/gradient progression
- **💻 Code snippets** containing one or more bugs
- **⚙️ Configuration** detailing hyperparameters and setup
- **❌ Error messages** (if the run crashed)

The agent must investigate (analyze logs, inspect gradients, check architecture), diagnose the root cause, and prescribe a fix. Performance is graded on diagnosis accuracy, fix quality, investigation efficiency, and hint usage.

## 🏗️ Architecture

```
torchdebug_env/
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── models.py              # Pydantic Action/Observation/State models
├── client.py              # MCPToolClient subclass
├── inference.py           # Baseline LLM agent script
├── __init__.py
├── server/
│   ├── app.py             # FastAPI server entry point
│   ├── torchdebug_environment.py  # Core MCPEnvironment
│   └── Dockerfile         # Container build
├── scenarios/
│   ├── __init__.py        # Scenario registry
│   ├── basic_failures.py  # Task 1: Easy scenarios
│   ├── performance_issues.py  # Task 2: Medium scenarios
│   └── subtle_bugs.py     # Task 3: Hard scenarios
└── utils/
    └── reward.py          # Grading & reward computation
```

## 📋 Tasks & Scenarios

### Task 1: Basic Failures (Easy) — 5 Scenarios
| ID | Bug | Symptom |
|---|---|---|
| `easy_lr_too_high` | Learning rate 10.0 for SGD | Loss explodes to NaN |
| `easy_device_mismatch` | Missing `model.to(device)` | RuntimeError: different devices |
| `easy_wrong_loss` | MSELoss for classification | Accuracy stuck at ~10% |
| `easy_missing_zero_grad` | No `optimizer.zero_grad()` | Unstable oscillating loss |
| `easy_double_softmax` | Softmax + CrossEntropyLoss | Accuracy plateaus at 65% |

### Task 2: Performance Issues (Medium) — 5 Scenarios
| ID | Bug | Symptom |
|---|---|---|
| `med_data_leakage` | Train/val data leakage | 99% val but 62% test accuracy |
| `med_batchnorm_eval` | BatchNorm in eval during training | Model barely learns |
| `med_memory_leak` | Storing loss tensor (not `.item()`) | GPU OOM after few epochs |
| `med_class_imbalance` | No compensation for 200:1 imbalance | 99.5% accuracy, 2% minority recall |
| `med_vanishing_gradients` | 50-layer Sigmoid MLP, no skip connections | Near-zero gradients in early layers |

### Task 3: Subtle & Compound Bugs (Hard) — 5 Scenarios
| ID | Bug | Symptom |
|---|---|---|
| `hard_ddp_grad_accum` | DDP syncs on every backward + unscaled LR | 3x slower than expected |
| `hard_mixed_precision_instability` | fp16 custom loss + tiny smoothing constant | Intermittent NaN every ~200 batches |
| `hard_weight_loading_frozen` | Unfrozen embeddings + weight decay on bias | Fine-tuning plateaus at 45% |
| `hard_tokenizer_mismatch` | Cased tokenizer + uncased model + no attention mask | Performance gap (76% vs 92%) |
| `hard_fsdp_checkpoint` | FSDP fp16 reduce + wrong clip method | Intermittent loss spikes |

## 🤖 Agent Interface

### Available Actions

| Action | Description | When to Use |
|---|---|---|
| `analyze_logs` | Analyze training log patterns | First step — understand the trajectory |
| `inspect_gradients` | Deep gradient flow analysis | When suspecting gradient issues |
| `inspect_data_pipeline` | Check data loading & splitting | When data issues suspected |
| `inspect_model_architecture` | Examine model, loss, layers | When architecture bugs suspected |
| `check_device_placement` | Analyze tensor device placement | When device errors occur |
| `diagnose` | Submit root cause diagnosis | After investigation |
| `prescribe_fix` | Submit fix (description + code) | Final action — ends episode |
| `request_hint` | Get a progressive hint | Use sparingly (score penalty) |

### Reward Structure

| Component | Weight | Description |
|---|---|---|
| Diagnosis quality | 40% | Fuzzy keyword match against ground truth |
| Fix quality | 40% | Match against correct fix description/code |
| Investigation efficiency | 10% | Using relevant inspections |
| Step efficiency | 10% | Fewer steps = higher bonus |
| Hint penalty | -5/10/15% | Increasing penalty per hint |

## 🚀 Quick Start

### 1. Start the Environment Server

```bash
# Install dependencies
pip install -e .

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -f server/Dockerfile -t torchdebug-env .
docker run -p 8000:8000 torchdebug-env
```

### 2. Run Baseline Inference

```bash
# Required variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="sk-..."
python inference.py

# With HuggingFace Inference
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

The baseline evaluates one deterministic scenario from each task (easy/medium/hard)
and writes reproducible scores to [outputs/evals/baseline_results.json](outputs/evals/baseline_results.json).

### 3. Validate

```bash
openenv validate torchdebug_env
```

### 4. Run Pre-submission Checks (Recommended)

```bash
# Core checks (validate + local reset/step smoke tests)
python presubmit.py

# Core + deterministic grader tests are run by default
# Use --skip-tests only if your environment cannot run pytest
python presubmit.py --skip-tests

# Include container checks (docker build/run + /health + /reset)
python presubmit.py --docker

# Include baseline run (requires API_BASE_URL, MODEL_NAME, HF_TOKEN)
python presubmit.py --docker --baseline
```

For HF Space style external validation, use [scripts/validate-submission.sh](scripts/validate-submission.sh).

`presubmit.py` now also writes a machine-readable report to
[outputs/evals/submission_report.json](outputs/evals/submission_report.json).

## 📊 Baseline Results

| Task | Difficulty | GPT-4 Avg Score | Llama-3-70B Avg Score |
|---|---|---|---|
| basic_failures | Easy | ~0.80 | ~0.65 |
| performance_issues | Medium | ~0.60 | ~0.45 |
| subtle_bugs | Hard | ~0.35 | ~0.20 |

## 🔧 Development

```bash
# Clone and install
git clone <repo-url>
cd torchdebug_env
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build Docker
openenv build torchdebug_env
```

## 🌐 HuggingFace Space Deployment

```bash
# Deploy to HuggingFace Spaces
openenv deploy torchdebug_env --space your-username/torchdebug-env
```

## 📜 License

BSD-3-Clause — Compatible with Meta/PyTorch licensing.

## 🏆 Hackathon Context

Built for the **Meta PyTorch OpenEnv Hackathon × Scaler School** (Round 1).

**Why TorchDebug matters:**
- 🔥 **Real-world utility**: Every ML engineer spends hours debugging training runs
- 🎯 **Sponsor alignment**: Showcases PyTorch ecosystem deeply (DDP, FSDP, AMP, transformers)
- 🧠 **Progressive difficulty**: Tests both basic knowledge and advanced distributed training skills
- 📈 **Meaningful rewards**: Partial credit for investigation — not just binary pass/fail

## 🧪 Judging Criteria Mapping (Round 1)

### 1) Real-world utility (30%)
- Environment models a real ML engineering workflow: diagnosing failed/underperforming PyTorch training jobs.
- Scenarios include production-like issues: device mismatch, data leakage, AMP instability, DDP/FSDP interactions.

### 2) Task & grader quality (25%)
- 3 difficulty tiers (easy → medium → hard) with deterministic scenario definitions.
- Programmatic scoring in [utils/reward.py](utils/reward.py) returns bounded scores in $[0,1]$.
- Grader includes anti-gaming logic and evidence-alignment incentives.

### 3) Environment design (20%)
- Clean episode lifecycle via `reset()` / `step()` / `state()` patterns.
- Typed action/observation/state models in [models.py](models.py).
- Reward shaping includes partial progress, efficiency terms, and hint penalties.

### 4) Code quality & OpenEnv compliance (15%)
- OpenEnv manifest in [openenv.yaml](openenv.yaml).
- Local validation and smoke-check automation in [presubmit.py](presubmit.py).
- External submission validator in [scripts/validate-submission.sh](scripts/validate-submission.sh).
- Dockerized runtime via [server/Dockerfile](server/Dockerfile).

### 5) Creativity & novelty (10%)
- Focuses on training-debug intelligence rather than benchmark gaming.
- Hard tasks require multi-factor reasoning (numerics + systems + architecture).

## 🚀 Final Submission Tips (High Impact)

- Use a hard scenario baseline in `inference.py` output to demonstrate non-trivial agent capability.
- Include the generated artifacts:
    - [outputs/evals/baseline_results.json](outputs/evals/baseline_results.json)
    - [outputs/evals/submission_report.json](outputs/evals/submission_report.json)
- Before submitting HF URL, run:
    - `python presubmit.py --docker --baseline`
    - `bash scripts/validate-submission.sh https://<your-space>.hf.space .`

## ✅ Submission Checklist (Practical)

- [ ] `openenv validate .` passes
- [ ] `python presubmit.py --docker` passes
- [ ] `python presubmit.py --baseline` passes with valid API credentials
- [ ] `outputs/evals/baseline_results.json` is generated and committed (or attached)
- [ ] Hugging Face Space is deployed and responds to `/health` and `/reset`
