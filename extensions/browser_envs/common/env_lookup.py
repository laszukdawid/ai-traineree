from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ENVS_ROOT = REPO_ROOT / "envs"


def resolve_env_path(env_name: str) -> Path:
    matches = sorted(ENVS_ROOT.glob(f"**/{env_name}"))
    dirs = [path for path in matches if path.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"Could not find env '{env_name}' under {ENVS_ROOT}")
    if len(dirs) > 1:
        options = ", ".join(str(path.relative_to(REPO_ROOT)) for path in dirs)
        raise ValueError(f"Env name '{env_name}' is ambiguous. Matches: {options}")
    return dirs[0]
