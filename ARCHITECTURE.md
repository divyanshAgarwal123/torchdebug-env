# TorchDebug Environment Architecture

This project now follows the same structure used by strong OpenEnv environments (for example calendar, reasoning_gym, tbench2, carla, repl):

## 1) Package Contract Layer

- [models.py](models.py)
	- Defines typed contracts: `TorchDebugAction`, `TorchDebugObservation`, `TorchDebugState`.
- [__init__.py](__init__.py)
	- Exposes a clean public API.
	- Uses lazy export for `TorchDebugEnv` (import cost only when needed).

## 2) Client Layer

- [client.py](client.py)
	- Remote controller for the environment.
	- Keeps inference/evaluation code independent from server internals.

## 3) Server Layer

- [server/torchdebug_environment.py](server/torchdebug_environment.py)
	- Core environment behavior and grading logic.
	- Owns episode lifecycle and action processing.
- [server/app.py](server/app.py)
	- FastAPI entrypoint created via OpenEnv `create_app(...)`.
	- Dual import pattern (`.` + fallback) for both in-repo and standalone execution.
	- `main(host, port)` with `API_PORT` support for deployment flexibility.
- [server/__init__.py](server/__init__.py)
	- Explicit server exports (`app`, `TorchDebugEnvironment`).

## 4) Scenario + Reward Separation

- [scenarios/](scenarios)
	- Task definitions and deterministic scenario registry.
- [utils/reward.py](utils/reward.py)
	- Scoring/grading policy kept separate from environment orchestration.

## 5) Validation + Submission Layer

- [presubmit.py](presubmit.py)
	- Local + docker + baseline checks.
- [openenv.yaml](openenv.yaml)
	- Manifest for packaging/deployment.
- [inference.py](inference.py)
	- Baseline agent + strict structured stdout contract.

## Why this structure is strong

1. **Clear boundaries:** contracts, environment logic, server wiring, and evaluation are separated.
2. **Portable imports:** same code works in local repo and docker/Space runtime.
3. **Stable API surface:** package exports stay clean for downstream users.
4. **OpenEnv-native lifecycle:** `reset/step/state` path is explicit and testable.

