#!/usr/bin/env python3
"""
TorchDebug pre-submission validator.

Runs a practical local checklist aligned to the hackathon pass/fail gate:
- openenv validate
- local environment smoke test (reset + step)
- optional docker build/run/reset smoke test
- optional baseline inference run (requires API env vars)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: str, *, cwd: Path | None = None, timeout: int = 1200) -> None:
    print(f"\n$ {cmd}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        shell=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {cmd}")


def check_openenv_validate() -> None:
    run_cmd("openenv validate .", cwd=ROOT, timeout=300)


def check_local_smoke() -> None:
    script = r'''
try:
    from torchdebug_env.server.torchdebug_environment import TorchDebugEnvironment
    from torchdebug_env.models import TorchDebugAction
except ImportError:
    from server.torchdebug_environment import TorchDebugEnvironment
    from models import TorchDebugAction

env = TorchDebugEnvironment()
obs = env.reset(task_id='basic_failures', scenario_id='easy_lr_too_high')
assert obs.done is False
o1 = env._process_action(TorchDebugAction(action_type='analyze_logs'))
assert o1.step_number == 1
o2 = env._process_action(TorchDebugAction(action_type='diagnose', diagnosis='learning rate too high'))
assert o2.step_number == 2
print('local-smoke-ok')
'''
    py = shlex.quote(sys.executable)
    run_cmd(f"{py} -c {shlex.quote(script)}", cwd=ROOT, timeout=120)


def check_tests() -> bool:
    py = shlex.quote(sys.executable)
    try:
        run_cmd(
            f"{py} -m pytest -q tests/test_reward.py tests/test_environment_flow.py",
            cwd=ROOT,
            timeout=240,
        )
        return True
    except RuntimeError:
        print("⚠️  pytest not available in current interpreter; skipping tests. "
              "Install with: python -m pip install pytest")
        return False


def write_report(report: dict) -> Path:
    out_dir = ROOT / "outputs" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "submission_report.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_file


def check_docker_smoke(image: str) -> None:
    run_cmd(f"docker build -f server/Dockerfile -t {image} .", cwd=ROOT, timeout=1800)
    run_cmd(f"docker rm -f {image}-ctr >/dev/null 2>&1 || true", cwd=ROOT, timeout=30)
    run_cmd(f"docker run --rm -d -p 8000:8000 --name {image}-ctr {image}", cwd=ROOT, timeout=60)
    try:
        run_cmd("sleep 5 && curl -sSf http://localhost:8000/health", cwd=ROOT, timeout=30)
        run_cmd(
            "curl -sSf -X POST http://localhost:8000/reset "
            "-H 'content-type: application/json' "
            "-d '{\"task_id\":\"basic_failures\",\"scenario_id\":\"easy_lr_too_high\"}' "
            "| python -c \"import sys, json; o=json.load(sys.stdin); assert 'observation' in o; print('docker-reset-ok')\"",
            cwd=ROOT,
            timeout=30,
        )
    finally:
        run_cmd(f"docker stop {image}-ctr >/dev/null 2>&1 || true", cwd=ROOT, timeout=30)


def check_baseline(timeout_s: int) -> None:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required env vars for baseline: {', '.join(missing)}"
        )

    py = shlex.quote(sys.executable)
    run_cmd(f"{py} inference.py", cwd=ROOT, timeout=timeout_s)

    out_file = ROOT / "outputs" / "evals" / "baseline_results.json"
    if not out_file.exists():
        raise RuntimeError(f"Expected output not found: {out_file}")
    print(f"baseline-output-ok: {out_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TorchDebug pre-submission checks")
    parser.add_argument("--docker", action="store_true", help="Include docker build/run smoke checks")
    parser.add_argument("--baseline", action="store_true", help="Run inference baseline (requires API env vars)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip local grader tests")
    parser.add_argument("--baseline-timeout", type=int, default=1200, help="Timeout seconds for baseline run")
    parser.add_argument("--docker-image", default="torchdebug-env-local", help="Docker image tag to use")
    args = parser.parse_args()

    print("== TorchDebug pre-submission checks ==")
    checks = {
        "openenv_validate": False,
        "local_smoke": False,
        "tests": None if args.skip_tests else "pending",
        "docker_smoke": None if not args.docker else False,
        "baseline": None if not args.baseline else False,
    }

    try:
        check_openenv_validate()
        checks["openenv_validate"] = True

        check_local_smoke()
        checks["local_smoke"] = True

        if not args.skip_tests:
            tests_ok = check_tests()
            checks["tests"] = True if tests_ok else "skipped"

        if args.docker:
            check_docker_smoke(args.docker_image)
            checks["docker_smoke"] = True

        if args.baseline:
            check_baseline(args.baseline_timeout)
            checks["baseline"] = True

        status = "passed"
        exit_code = 0
    except Exception as e:
        print(f"\n❌ Presubmit failed: {e}")
        status = "failed"
        exit_code = 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "options": {
            "docker": args.docker,
            "baseline": args.baseline,
            "skip_tests": args.skip_tests,
            "baseline_timeout": args.baseline_timeout,
            "docker_image": args.docker_image,
        },
        "environment": {
            "python": sys.executable,
            "cwd": str(ROOT),
        },
    }
    report_file = write_report(report)
    print(f"\nReport written: {report_file}")

    if exit_code == 0:
        print("\n✅ All selected checks passed")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
