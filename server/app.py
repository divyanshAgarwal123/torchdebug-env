"""
FastAPI application for the TorchDebug Environment.

Exposes TorchDebugEnvironment over HTTP and WebSocket endpoints
using the OpenEnv create_app factory.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core import create_app, CallToolAction, CallToolObservation

try:
    from .torchdebug_environment import TorchDebugEnvironment
except ImportError:
    from server.torchdebug_environment import TorchDebugEnvironment

# create_app expects a callable factory (not a class instance)
app = create_app(
    env=TorchDebugEnvironment,  # Class is callable — creates new instance per session
    action_cls=CallToolAction,
    observation_cls=CallToolObservation,
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


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
