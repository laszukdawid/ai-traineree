# Browser Environments

This document captures the integration pattern learned from the first browser environment experiment (`rope-man-game`). The goal is to preserve the architectural decisions so future environment integrations start from the same baseline.

## Core rules

- External environments live under `envs/<owner>/<repo>`.
- AI Traineree owns the integration code; checked-out environments should remain unmodified by default.
- Prefer runtime injection or monkey patching over editing files inside `envs/`.
- Environment-specific commands should be env-oriented, for example `browser-verify:<env-name>`.
- Python commands for repo tooling should use `uv run python ...`.
- Keep generic browser runtime support in `aitraineree/browser/`, but keep environment-specific bridges, tasks, and training helpers in extension-style areas such as `extensions/browser_envs/`.

Current structure in this repo:

- `aitraineree/browser/` contains reusable browser runtime support
- `extensions/browser_envs/common/` contains shared extension helpers such as bridge snippets and local HTTP serving
- `extensions/browser_envs/<env>/` contains environment-specific tasks and scripts

## Why environments should stay unmodified

Keeping checked-out repositories pristine makes it easier to:

- update or re-clone an upstream environment
- compare local behavior with upstream behavior
- avoid mixing third-party code with AI Traineree integration code
- repeat the same integration strategy across multiple repositories

Direct modification of files under `envs/` should be treated as a temporary debugging technique, not the default architecture.

## Recommended browser integration pattern

For browser-based environments, use this order of preference:

1. Serve the checked-out repo locally over HTTP.
2. Launch a browser through Playwright.
3. Wait for the page to load.
4. Inject an AI Traineree-owned bridge script into the page at runtime.
5. Use that bridge to expose a stable control and observation interface.

The bridge should live in AI Traineree-owned code, not inside the checked-out environment.

## Bridge responsibilities

The injected bridge should provide a small stable API, for example:

- `snapshot()`
- `start(seed)`
- `reset(seed?)`
- `action(name, value?)`
- `releaseAll()`

The bridge should convert environment-specific globals and functions into a shape that Python can drive predictably.

## Rope Man example

For `rope-man-game`, the working pattern is:

- serve `envs/mitsuhiko/rope-man-game`
- open the page in Playwright
- inject the Rope Man bridge from `scripts/browser_env_bridges.py`
- drive the page via `window.__ropeManRl`

This proves that a browser environment can be adapted without modifying the checked-out source repository.

The current Rope Man numeric observation vector should be treated only as a control smoke test. It is useful for proving that Python actions can change environment state, but it is not sufficient for a generally solvable learning problem because the important failure signals are driven by rendered geometry such as obstacles and terrain collisions.

For browser environments like this, a realistic training setup will likely need one of these:

- browser screenshots as observations
- a rendered frame stream exposed to the task runtime
- a richer geometry/collision interface than the toy control bridge currently exposes

## Tower Game example

For `tower_game`, the same external integration pattern also works:

- serve `envs/iamkun/tower_game`
- open the page in Playwright
- inject the Tower Game bridge from `scripts/browser_env_bridges.py`
- drive the page via `window.__towerGameRl`

This second example is useful because it differs from Rope Man in important ways:

- the action space is much smaller (`noop` / `drop`)
- the game has explicit score and failure hooks on the page already
- reset is easiest through a page reload rather than a custom in-game reset path

This confirms that the browser-runtime pattern is reusable across at least two different browser games while keeping checked-out env repos unmodified.

Tower Game is also the cleaner early training example so far:

- tiny action space (`noop`, `drop`)
- explicit score and failure hooks
- clear short episodes
- frame-based DQN training now runs end-to-end through AI Traineree

## What to generalize next

After one or two more browser environment integrations, extract the reusable pieces into proper runtime/task modules, likely including:

- env lookup and checkout helpers
- static server lifecycle
- browser runtime lifecycle
- runtime bridge injection
- task wrapper for reset/step/observation/reward mapping

Do not generalize too early. First confirm the pattern across multiple repositories.
