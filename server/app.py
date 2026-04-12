"""FastAPI application for the TorchDebug environment server."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict

from fastapi import FastAPI, Request

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core import create_app

try:
    from ..models import TorchDebugAction, TorchDebugObservation
except ImportError:
    from models import TorchDebugAction, TorchDebugObservation

# Support both in-repo and standalone imports (OpenEnv reference pattern)
try:
    from .torchdebug_environment import TorchDebugEnvironment
except ImportError:
    from server.torchdebug_environment import TorchDebugEnvironment

logger = logging.getLogger(__name__)

# create_app expects a callable factory (not a class instance)
app = create_app(
    env=TorchDebugEnvironment,
    action_cls=TorchDebugAction,
    observation_cls=TorchDebugObservation,
    env_name="torchdebug_env",
)

# ---------------------------------------------------------------------------
# Stateful HTTP overrides for /reset and /step
# ---------------------------------------------------------------------------
# The OpenEnv framework's REST endpoints are stateless: each HTTP request
# creates (and destroys) a fresh environment instance.  This means an
# HTTP POST /step after POST /reset runs against a *new* env that was
# never reset — producing "Error: Call reset() first." with reward 0.01.
#
# For multi-step episodes over plain HTTP (used by Phase-2 validators and
# by inference.py's HTTP fallback), we need a shared environment that
# persists between /reset and subsequent /step calls.
#
# WebSocket (/ws) sessions managed by the framework are unaffected.
# ---------------------------------------------------------------------------

_env_lock = threading.Lock()
_env_instance: TorchDebugEnvironment | None = None


def _serialize_obs(obs: TorchDebugObservation) -> Dict[str, Any]:
    """Serialize observation to match openenv-core's response format."""
    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    return {"observation": obs_dict, "reward": obs.reward, "done": obs.done}


# Remove the framework's stateless /reset and /step routes so ours take effect.
app.router.routes = [
    r
    for r in app.router.routes
    if not (hasattr(r, "path") and r.path in ("/reset", "/step"))
]


@app.post("/reset")
async def stateful_reset(request: Request):
    """Reset: create a fresh env, run reset(), keep instance for later /step."""
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
        "message": "TorchDebug OpenEnv is running. Use /health, /reset, /step, /state, /schema.",
    }


def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the TorchDebug environment server with uvicorn."""
    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
