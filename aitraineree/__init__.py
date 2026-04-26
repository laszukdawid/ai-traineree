from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib
import torch


def _resolve_version() -> str:
    try:
        return version("ai-traineree")
    except PackageNotFoundError:
        try:
            pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
            with pyproject_path.open("rb") as file:
                return tomllib.load(file)["project"]["version"]
        except (FileNotFoundError, KeyError):
            return "0.0.0"


__version__ = _resolve_version()


try:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")
