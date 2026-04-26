# Agents Guide

This file is for coding agents working in this repository. It is a fast map of what lives where and how to make safe, useful changes.

## Project snapshot

- Package name: `ai-traineree`
- Runtime package path: `aitraineree/`
- Build system: Hatch (`pyproject.toml`)
- Python: `>=3.12`
- Main domain: reinforcement learning agents, buffers, runners, tasks, and training examples

## Where things are

- `aitraineree/agents/` single-agent implementations (`dqn.py`, `ppo.py`, `sac.py`, `td3.py`, `ddpg.py`, `d4pg.py`, `rainbow.py`, `d3pg.py`)
- `aitraineree/multi_agents/` multi-agent implementations (`iql.py`, `maddpg.py`)
- `aitraineree/runners/` orchestration (`env_runner.py`, `multiagent_env_runner.py`, `multi_sync_env_runner.py`)
- `aitraineree/buffers/` replay and related buffers
- `aitraineree/networks/` model/network building blocks
- `aitraineree/tasks.py` environment/task adapters (Gymnasium, PettingZoo, Unity)
- `aitraineree/loggers/` experiment logging (`tensorboard_logger.py`, `file_logger.py`)
- `examples/` runnable training scripts grouped by use case
- `tests/` unit/integration tests mirroring package areas
- `docs/` Sphinx docs sources (`index.rst` and topic pages)
- `.github/workflows/` CI for lint, tests, and publish

## Common commands

- Install dev env: `uv sync --dev`
- Run tests: `uv run pytest`
- Run lint: `uvx ruff@0.3.0 check`
- Build package: `uv build`
- Run a basic example: `uv run examples/cart_dqn.py`

## Release pipeline

- Release workflow: `.github/workflows/pip-publish.yml`
- Trigger: pushing a tag matching `v*.*.*` (for example `v0.7.1`)
- The workflow validates that tag version equals `project.version` in `pyproject.toml`
- On success, it publishes to PyPI (`uv publish`) and creates a GitHub release with generated notes and `dist/*` assets
- If GitHub releases and tags differ, inspect with `gh release list` and `git ls-remote --tags origin`

## Change workflow for agents

1. Read `pyproject.toml` first for dependency/tooling truth.
2. Make focused edits in `aitraineree/` and matching tests in `tests/`.
3. Run targeted tests first, then broader `uv run pytest` if feasible.
4. Run lint (`uvx ruff@0.3.0 check`) before finishing.
5. Update docs (`README.md` and/or `docs/`) when behavior or APIs change.

## Conventions and guardrails

- Keep imports and style Ruff-clean; avoid introducing new formatting tools.
- Prefer small, local changes over broad refactors unless explicitly requested.
- Do not delete or rewrite unrelated user changes in a dirty worktree.
- Treat heavy artifacts as outputs, not source: `dist/`, `videos/`, `run_states/`, model `.net` files.
- Examples are informative and can drift; code + tests + `pyproject.toml` are stronger sources of truth.

## Known gotchas

- Package version source of truth is `pyproject.toml` (`project.version`); `aitraineree.__version__` resolves from package metadata and falls back to reading `pyproject.toml` in source checkouts.
- Some pytest modules are intentionally ignored via `tool.pytest.ini_options.addopts` in `pyproject.toml`.
- `ai-traineree/meta.yaml` is a Conda recipe area and separate from the runtime Python package in `aitraineree/`.
