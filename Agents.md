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
- `aitraineree/tasks.py` core environment/task adapters (Gymnasium, PettingZoo, Unity)
- `aitraineree/browser/` generic browser runtime support that can be reused by external environment integrations
- `aitraineree/loggers/` experiment logging (`tensorboard_logger.py`, `file_logger.py`)
- `extensions/browser_envs/common/` shared browser-env extension helpers (bridges, env lookup, local HTTP serving)
- `extensions/browser_envs/<env>/` environment-specific browser integrations, tasks, and runnable scripts that are intentionally outside the core AI Traineree package
- `examples/` runnable training scripts grouped by use case
- `tests/` unit/integration tests mirroring package areas
- `docs/` Sphinx docs sources (`index.rst` and topic pages)
- `.github/workflows/` CI for lint, tests, and publish

## Common commands

- Install dev env: `uv sync --dev`
- Run tests: `.venv/bin/python -m pytest` (uses installed ROCm torch) or `uv run python -m pytest` (uses project default)
- Run lint: `uvx ruff check`
- Build package: `uv build`
- Run a basic example: `uv run examples/cart_dqn.py`
- Run Python scripts requiring project deps: `uv run python ...` (uses project environment)
- Run Python scripts requiring custom packages (e.g., ROCm torch): `.venv/bin/python ...`
- Run Taskfile tasks from the repo root without `-t` when using the default Taskfile path in that directory.
- Prefer env-oriented task naming for env-specific actions, for example `browser-verify:<env-name>` instead of `<env-name>:browser-verify`.

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
4. Run lint (`uvx ruff check`) before finishing.
5. Update docs (`README.md` and/or `docs/`) when behavior or APIs change.

## Conventions and guardrails

- Keep imports and style Ruff-clean; avoid introducing new formatting tools.
- Prefer small, local changes over broad refactors unless explicitly requested.
- Do not delete or rewrite unrelated user changes in a dirty worktree.
- Treat heavy artifacts as outputs, not source: `dist/`, `videos/`, `run_states/`, model `.net` files.
- Examples are informative and can drift; code + tests + `pyproject.toml` are stronger sources of truth.
- When documenting or running Python commands for this repo, prefer using `.venv/bin/python` for scripts that require custom-installed packages (like ROCm torch), or `uv run python` for standard scripts that rely on the project's declared dependencies.
- When the user gives workflow notes or recurring corrections, update the relevant repo documentation such as `Agents.md` so the same mistake is less likely to repeat.
- For external environments under `envs/`, prefer runtime injection or monkey patching from AI Traineree-owned code over editing files in the checked-out environment itself.

## Known gotchas

- Package version source of truth is `pyproject.toml` (`project.version`); `aitraineree.__version__` resolves from package metadata and falls back to reading `pyproject.toml` in source checkouts.
- Some pytest modules are intentionally ignored via `tool.pytest.ini_options.addopts` in `pyproject.toml`.
- `ai-traineree/meta.yaml` is a Conda recipe area and separate from the runtime Python package in `aitraineree/`.
- External repositories for experiments live under `envs/`, not `gyms/`. An environment may be implemented with Gymnasium, but not every environment is a gym.
- `scripts/rope_man_smoke.py` exits after the smoke check by default. Use `uv run python scripts/rope_man_smoke.py --serve` if you need the local server to stay up for browser testing.
- Env-specific task commands should resolve environments by searching under `envs/` rather than hardcoding owner/repo paths when the env directory name is sufficient.
- Browser env integrations should keep the env repo pristine by default. Keep generic browser runtime support in `aitraineree/browser/`, but put environment-specific bridges/controllers/tasks/training helpers in extension-style areas such as `extensions/browser_envs/`.
- When running scripts that require a specific torch version (e.g., ROCm-enabled torch installed in `.venv/`), use `.venv/bin/python path/to/script.py` or `source .venv/bin/activate && python path/to/script.py` instead of `uv run python ...`. This is because `uv run` syncs the project environment and may reinstall a different torch version from what's specified in `pyproject.toml`, overriding your custom-installed packages.
