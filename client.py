"""
TorchDebug Environment Client.

Client for connecting to a running TorchDebug Environment server.
Extends MCPToolClient for tool-calling interactions.

Example:
    >>> with TorchDebugEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...
    ...     # Get current observation
    ...     obs = env.call_tool("get_observation")
    ...
    ...     # Investigate
    ...     result = env.call_tool("take_action", action_type="analyze_logs")
    ...     result = env.call_tool("take_action", action_type="inspect_gradients")
    ...
    ...     # Diagnose
    ...     result = env.call_tool("take_action",
    ...         action_type="diagnose",
    ...         diagnosis="Learning rate is too high causing loss explosion"
    ...     )
    ...
    ...     # Prescribe fix
    ...     result = env.call_tool("take_action",
    ...         action_type="prescribe_fix",
    ...         fix_description="Reduce learning rate to 0.01",
    ...         fix_code="optimizer = SGD(lr=0.01)"
    ...     )

Example with Docker:
    >>> env = TorchDebugEnv.from_docker_image("torchdebug-env:latest")
    >>> try:
    ...     env.reset()
    ...     tools = env.list_tools()
    ... finally:
    ...     env.close()

Example with HuggingFace Space:
    >>> env = TorchDebugEnv.from_env("your-user/torchdebug-env")
    >>> try:
    ...     env.reset()
    ...     result = env.call_tool("get_observation")
    ... finally:
    ...     env.close()
"""

from openenv.core.mcp_client import MCPToolClient


class TorchDebugEnv(MCPToolClient):
    """
    Client for the TorchDebug Environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    """
    pass
