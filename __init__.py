"""
TorchDebug Environment — PyTorch Training Run Debugger.

An OpenEnv environment for diagnosing and fixing broken PyTorch training runs.
Agents must investigate training logs, inspect code, diagnose root causes, 
and prescribe fixes for common ML engineering failures.

Tasks:
    - basic_failures (easy): Common crashes — device mismatch, NaN loss, wrong loss
    - performance_issues (medium): Subtle issues — data leakage, vanishing gradients
    - subtle_bugs (hard): Compound bugs — DDP interactions, mixed precision, FSDP

Example:
    >>> from torchdebug_env import TorchDebugEnv
    >>>
    >>> with TorchDebugEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("get_observation")
    ...     result = env.call_tool("take_action", action_type="analyze_logs")
"""

try:
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
except ImportError:  # pragma: no cover
    CallToolAction = None
    ListToolsAction = None

try:
    from .client import TorchDebugEnv
except ImportError:  # pragma: no cover
    TorchDebugEnv = None

__all__ = ["TorchDebugEnv", "CallToolAction", "ListToolsAction"]
