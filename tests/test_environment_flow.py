import math

try:
    from torchdebug_env.models import TorchDebugAction
    from torchdebug_env.server.torchdebug_environment import TorchDebugEnvironment
except ImportError:
    from models import TorchDebugAction
    from server.torchdebug_environment import TorchDebugEnvironment


def test_reset_and_step_flow():
    env = TorchDebugEnvironment()
    obs = env.reset(task_id="basic_failures", scenario_id="easy_lr_too_high")

    assert obs.done is False
    assert 0.0 <= obs.reward < 1.0
    assert obs.task_id == "easy_lr_too_high"

    o1 = env._process_action(TorchDebugAction(action_type="analyze_logs"))
    assert o1.step_number == 1
    assert o1.done is False

    o2 = env._process_action(
        TorchDebugAction(action_type="diagnose", diagnosis="learning rate too high")
    )
    assert o2.step_number == 2
    assert o2.done is False

    o3 = env._process_action(
        TorchDebugAction(
            action_type="prescribe_fix",
            fix_description="reduce learning rate to 0.01",
            fix_code="optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
        )
    )
    assert o3.done is True
    assert 0.0 < o3.reward < 1.0


def test_all_emitted_rewards_are_strict_open_interval():
    """Intermediate steps emit 0.0; only the terminal step carries the real
    score.  The task score (sum of all rewards) must satisfy 0 < score < 1."""
    env = TorchDebugEnvironment()
    obs = env.reset(task_id="basic_failures", scenario_id="easy_lr_too_high")
    assert 0.0 <= obs.reward < 1.0  # reset reward is 0.0

    # Intermediate steps: reward should be 0.0
    o1 = env._process_action(TorchDebugAction(action_type="analyze_logs"))
    assert 0.0 <= o1.reward < 1.0

    o2 = env._process_action(TorchDebugAction(action_type="diagnose", diagnosis=""))
    assert 0.0 <= o2.reward < 1.0

    o3 = env._process_action(TorchDebugAction(action_type="request_hint"))
    assert 0.0 <= o3.reward < 1.0

    # Terminal step: must be strictly in (0, 1)
    o4 = env._process_action(TorchDebugAction(
        action_type="prescribe_fix",
        fix_description="reduce lr", fix_code="lr=0.01",
    ))
    assert o4.done is True
    assert 0.0 < o4.reward < 1.0

    # Task score = sum of all rewards must be in (0, 1)
    total = obs.reward + o1.reward + o2.reward + o3.reward + o4.reward
    assert 0.0 < total < 1.0, f"Task score (sum) out of range: {total}"


def test_fixed_trajectory_is_deterministic():
    actions = [
        TorchDebugAction(action_type="analyze_logs"),
        TorchDebugAction(action_type="inspect_gradients"),
        TorchDebugAction(action_type="diagnose", diagnosis="learning rate too high causing divergence and nan"),
        TorchDebugAction(
            action_type="prescribe_fix",
            fix_description="lower learning rate to 0.01",
            fix_code="optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)",
        ),
    ]

    env1 = TorchDebugEnvironment()
    env1.reset(task_id="basic_failures", scenario_id="easy_lr_too_high")
    r1 = None
    for a in actions:
        o = env1._process_action(a)
        r1 = o.reward

    env2 = TorchDebugEnvironment()
    env2.reset(task_id="basic_failures", scenario_id="easy_lr_too_high")
    r2 = None
    for a in actions:
        o = env2._process_action(a)
        r2 = o.reward

    assert r1 is not None and r2 is not None
    assert math.isclose(r1, r2, rel_tol=0.0, abs_tol=1e-12)
