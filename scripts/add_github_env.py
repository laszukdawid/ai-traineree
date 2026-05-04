import argparse
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS_ROOT = REPO_ROOT / "envs"
GITHUB_REPO_RE = re.compile(r"^(?:https://github\.com/)?(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def parse_repo(repo: str) -> tuple[str, str, str]:
    match = GITHUB_REPO_RE.match(repo.strip())
    if not match:
        raise ValueError(f"Unsupported GitHub repo format: {repo}")
    owner = match.group("owner")
    name = match.group("repo")
    url = f"https://github.com/{owner}/{name}.git"
    return owner, name, url


def run_git(args: list[str], workdir: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(workdir) if workdir else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Clone a GitHub repository into envs/<owner>/<repo>.")
    parser.add_argument("repo", help="GitHub repo URL or owner/repo string")
    args = parser.parse_args()

    owner, name, url = parse_repo(args.repo)
    target = ENVS_ROOT / owner / name
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        sha = run_git(["rev-parse", "HEAD"], target)
        print(f"Environment already exists: {target}")
        print(f"Current commit: {sha}")
        print("Next steps:")
        print(f"  1. Inspect the env at {target}")
        print(f"  2. If it is a browser game, serve it locally and add an adapter/smoke test")
        return

    run_git(["clone", url, str(target)])
    sha = run_git(["rev-parse", "HEAD"], target)
    print(f"Added environment: {target}")
    print(f"Cloned from: {url}")
    print(f"Pinned commit: {sha}")
    print("Next steps:")
    print(f"  1. Inspect the env at {target}")
    print("  2. Decide runtime type (browser, Python package, native app, etc.)")
    print("  3. Add the minimal adapter and smoke test for that env")


if __name__ == "__main__":
    main()
