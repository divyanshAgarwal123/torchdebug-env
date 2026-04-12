# Copyright (c) 2026, TorchDebug Environment
# OpenEnv-compatible Pydantic models for the PyTorch Training Debugger

"""
Pydantic models for the TorchDebug environment.

Defines typed Action, Observation, and State models for the OpenEnv spec.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


# =============================================================================
# Sub-models (used within Observation)
# =============================================================================

class TrainingLogEntry(BaseModel):
    """A single epoch/step of training log data."""
    model_config = {"extra": "allow"}

    epoch: int = Field(description="Epoch number")
    step: Optional[int] = Field(default=None, description="Step within epoch")
    train_loss: Optional[Any] = Field(default=None, description="Training loss (may be float, 'inf', or 'nan')")
    val_loss: Optional[Any] = Field(default=None, description="Validation loss (may be float, 'inf', or 'nan')")
    train_accuracy: Optional[float] = Field(default=None, description="Training accuracy")
    val_accuracy: Optional[float] = Field(default=None, description="Validation accuracy")
    learning_rate: Optional[float] = Field(default=None, description="Current learning rate")
    grad_norm: Optional[Any] = Field(default=None, description="Gradient norm (may be float, 'inf', or 'nan')")
    gpu_memory_mb: Optional[float] = Field(default=None, description="GPU memory usage in MB")
    throughput: Optional[float] = Field(default=None, description="Samples per second")
    extra_metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")


class SystemInfo(BaseModel):
    """System/hardware information for the training run."""
    gpu_type: str = Field(default="NVIDIA A100", description="GPU type")
    gpu_memory_total_mb: int = Field(default=40960, description="Total GPU memory")
    num_gpus: int = Field(default=1, description="Number of GPUs")
    cuda_version: str = Field(default="12.1", description="CUDA version")
    pytorch_version: str = Field(default="2.2.0", description="PyTorch version")
    python_version: str = Field(default="3.11", description="Python version")


class InspectionResult(BaseModel):
    """Result from an inspection/analysis action."""
    inspection_type: str = Field(description="Type of inspection performed")
    findings: str = Field(description="Detailed findings from the inspection")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured data from inspection")


# =============================================================================
# Observation Model
# =============================================================================

class TorchDebugObservation(Observation):
    """What the agent sees at each step of the debugging episode."""

    task_id: str = Field(description="Identifier for the current bug scenario")
    task_description: str = Field(description="Natural language description of the problem")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Task difficulty level")

    training_config: Dict[str, Any] = Field(
        description="Training configuration (lr, batch_size, optimizer, epochs, etc.)"
    )
    code_snippet: str = Field(
        description="PyTorch model/training code with potential bugs"
    )
    training_logs: List[TrainingLogEntry] = Field(
        default_factory=list,
        description="Training log entries showing loss/metric progression"
    )
    system_info: SystemInfo = Field(
        default_factory=SystemInfo,
        description="System/hardware information"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error/traceback message if the training crashed"
    )

    # Episode state
    available_actions: List[str] = Field(
        description="List of action types the agent can take"
    )
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed")
    hints_used: int = Field(default=0, description="Number of hints requested")

    # Previous inspection results
    inspection_results: List[InspectionResult] = Field(
        default_factory=list,
        description="Results from previous inspection actions"
    )

    # Feedback from previous actions
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback from the last action taken"
    )

    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.01, description="Reward from last action")

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, v: Any) -> float:
        """Ensure reward is strictly in the open interval (0, 1)."""
        v = float(v) if v is not None else 0.01
        if v <= 0.0:
            return 0.01
        if v >= 1.0:
            return 0.99
        return v


# =============================================================================
# Action Model
# =============================================================================

class TorchDebugAction(Action):
    """What the agent can do at each step."""

    action_type: Literal[
        "analyze_logs",
        "inspect_gradients",
        "inspect_data_pipeline",
        "inspect_model_architecture",
        "check_device_placement",
        "diagnose",
        "prescribe_fix",
        "request_hint",
    ] = Field(description="The type of action to perform")

    # For diagnosis action
    diagnosis: Optional[str] = Field(
        default=None,
        description="Root cause diagnosis (required for 'diagnose' action). "
        "Should identify the specific bug category."
    )

    # For fix prescription
    fix_description: Optional[str] = Field(
        default=None,
        description="Description of the fix (required for 'prescribe_fix' action). "
        "Should explain what code change resolves the issue."
    )
    fix_code: Optional[str] = Field(
        default=None,
        description="Optional corrected code snippet"
    )

    # Generic parameters for inspection actions
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the action"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# =============================================================================
# State Model
# =============================================================================

class TorchDebugState(State):
    """Internal environment state for the debugging episode."""

    task_id: Optional[str] = Field(default=None, description="Current task/scenario ID")
    difficulty: Optional[str] = Field(default=None, description="Current difficulty level")
    diagnosed: bool = Field(default=False, description="Whether a correct diagnosis was made")
    fixed: bool = Field(default=False, description="Whether a correct fix was prescribed")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated")
    hints_used: int = Field(default=0, description="Number of hints used")
    actions_taken: List[str] = Field(default_factory=list, description="History of action types")
