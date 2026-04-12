"""
FastAPI application for the TorchDebug environment server.

This module creates an HTTP server that exposes TorchDebugEnvironment
over HTTP and WebSocket endpoints, compatible with OpenEnv EnvClient.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Via OpenEnv CLI:
    uv run --project . server
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
from typing import Any, Dict

from fastapi import Request

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core import create_app

try:
    from ..models import TorchDebugAction, TorchDebugObservation
except ImportError:
    from models import TorchDebugAction, TorchDebugObservation

try:
    from .torchdebug_environment import TorchDebugEnvironment
    from .gradio_ui import build_torchdebug_gradio_app
except ImportError:
    from server.torchdebug_environment import TorchDebugEnvironment
    from server.gradio_ui import build_torchdebug_gradio_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory function (reference pattern from REPL/Calendar envs)
# ---------------------------------------------------------------------------

def create_torchdebug_environment() -> TorchDebugEnvironment:
    """Factory that creates a TorchDebugEnvironment instance.

    This follows the OpenEnv reference pattern where create_app receives
    a callable factory rather than a class directly.
    """
    return TorchDebugEnvironment()


# Create the app with Gradio UI (matching REPL env pattern)
_sig = inspect.signature(create_app)
if "gradio_builder" in _sig.parameters:
    app = create_app(
        create_torchdebug_environment,
        TorchDebugAction,
        TorchDebugObservation,
        env_name="torchdebug_env",
        max_concurrent_envs=4,
        gradio_builder=build_torchdebug_gradio_app,
    )
else:
    logger.warning(
        "Installed openenv-core does not support gradio_builder; "
        "interactive Gradio UI will not be available."
    )
    app = create_app(
        create_torchdebug_environment,
        TorchDebugAction,
        TorchDebugObservation,
        env_name="torchdebug_env",
        max_concurrent_envs=4,
    )


# ---------------------------------------------------------------------------
# Stateful HTTP overrides for /reset and /step
# ---------------------------------------------------------------------------
# The OpenEnv framework's HTTP endpoints are stateless (each request creates
# a fresh env). For HTTP-based clients (inference.py, validators), we need
# a shared env that persists between /reset and /step calls.
#
# WebSocket sessions managed by the framework are unaffected.
# ---------------------------------------------------------------------------

_env_lock = threading.Lock()
_env_instance: TorchDebugEnvironment | None = None


def _clamp_reward(v: float) -> float:
    """Ensure every reward is strictly in (0, 1)."""
    v = float(v) if v is not None else 0.01
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.95
    return v


def _serialize_obs(obs: TorchDebugObservation) -> Dict[str, Any]:
    """Serialize observation to match OpenEnv response format."""
    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    return {
        "observation": obs_dict,
        "reward": _clamp_reward(obs.reward),
        "done": obs.done,
    }


# Remove the framework's stateless /reset and /step so ours take effect.
app.router.routes = [
    r for r in app.router.routes
    if not (hasattr(r, "path") and r.path in ("/reset", "/step"))
]


@app.post("/reset")
async def stateful_reset(request: Request):
    """Reset: create a fresh env, run reset(), keep instance for /step."""
    global _env_instance
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = body.get("task_id", "basic_failures")
    scenario_id = body.get("scenario_id")
    seed = body.get("seed")
    episode_id = body.get("episode_id")

    kwargs: Dict[str, Any] = {"task_id": task_id}
    if scenario_id is not None:
        kwargs["scenario_id"] = scenario_id
    if seed is not None:
        kwargs["seed"] = seed
    if episode_id is not None:
        kwargs["episode_id"] = episode_id

    with _env_lock:
        _env_instance = TorchDebugEnvironment()
        obs = _env_instance.reset(**kwargs)

    return _serialize_obs(obs)


@app.post("/step")
async def stateful_step(request: Request):
    """Step: run action on the shared env from the last /reset."""
    global _env_instance
    try:
        body = await request.json()
    except Exception:
        body = {}
    action_data = body.get("action", body)

    # Unwrap legacy CallToolAction wrapper if present
    if "arguments" in action_data and "tool_name" in action_data:
        action_data = action_data["arguments"]

    with _env_lock:
        if _env_instance is None:
            _env_instance = TorchDebugEnvironment()

        try:
            action = TorchDebugAction.model_validate(action_data)
        except Exception:
            action = TorchDebugAction(
                action_type=action_data.get("action_type", "analyze_logs"),
                diagnosis=action_data.get("diagnosis"),
                fix_description=action_data.get("fix_description"),
                fix_code=action_data.get("fix_code"),
                parameters=action_data.get("parameters", {}),
            )

        obs = _env_instance.step(action)

    return _serialize_obs(obs)


@app.get("/")
def root():
    """Human-friendly root endpoint for HF Space landing checks."""
    return {
        "name": "torchdebug_env",
        "status": "ok",
        "message": "TorchDebug OpenEnv — PyTorch debugging environment. "
                   "Use /health, /reset, /step, /state, /schema, /ws.",
    }


def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the TorchDebug environment server with uvicorn."""
    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
