"""FastAPI application for the TorchDebug environment server."""

from __future__ import annotations

import os

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

# create_app expects a callable factory (not a class instance)
app = create_app(
    env=TorchDebugEnvironment,
    action_cls=TorchDebugAction,
    observation_cls=TorchDebugObservation,
    env_name="torchdebug_env",
)


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
