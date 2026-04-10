"""TorchDebug environment package exports."""

from __future__ import annotations

from typing import Any

from .models import TorchDebugAction, TorchDebugObservation, TorchDebugState

__all__ = [
    "TorchDebugAction",
    "TorchDebugObservation",
    "TorchDebugState",
    "TorchDebugEnv",
]


def __getattr__(name: str) -> Any:
    if name == "TorchDebugEnv":
        from .client import TorchDebugEnv as _TorchDebugEnv

        return _TorchDebugEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
