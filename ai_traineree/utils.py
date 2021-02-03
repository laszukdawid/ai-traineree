import os
import torch

from numpy import ndarray
from pathlib import Path
from typing import List


def to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    else:
        return torch.tensor(x)


def save_gif(path, images: List[ndarray]) -> None:
    print(f"Saving as a gif to {path}")
    from PIL import Image
    imgs = [Image.fromarray(img[::2, ::2]) for img in images]  # Reduce /4 size; pick w/2 h/2 pix

    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], optimize=True, quality=85)
