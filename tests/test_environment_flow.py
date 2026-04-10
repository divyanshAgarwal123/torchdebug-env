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
    assert obs.reward == 0.0
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
    assert 0.0 <= o3.reward <= 1.0


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
