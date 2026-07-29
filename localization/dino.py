"""Shared DINOv2 embedding helpers."""

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 14  # DINOv2 needs both dimensions divisible by its patch size

_MODELS = {}


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model(name, device):
    """Cached so repeated calls in one process don't reload the weights."""
    if name not in _MODELS:
        model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        _MODELS[name] = model.eval().to(device)
    return _MODELS[name]


def load_frame(path, short_side, crop_bottom=0.0, rotate=0):
    img = Image.open(path).convert("RGB")

    # crop before rotating -- the operator's head is at the bottom of the
    # frame as recorded, whatever we do to the image afterwards
    if crop_bottom:
        w, h = img.size
        img = img.crop((0, 0, w, int(h * (1 - crop_bottom))))

    if rotate:
        img = img.rotate(rotate, expand=True)

    w, h = img.size
    scale = short_side / min(w, h)
    size = tuple(max(PATCH, round(d * scale / PATCH) * PATCH) for d in (w, h))
    img = img.resize(size, Image.BICUBIC)

    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1)


def embed(paths, model_name="dinov2_vits14", short_side=518, crop_bottom=0.0,
          rotate=0, batch_size=16, quiet=False):
    """Return L2-normalized embeddings, one row per path."""
    device = pick_device()
    model = get_model(model_name, device)

    tensors = torch.stack([
        load_frame(p, short_side, crop_bottom, rotate) for p in paths
    ])
    if not quiet:
        print(f"embedding {len(paths)} frames at "
              f"{tensors.shape[3]}x{tensors.shape[2]}"
              + (f" rotated {rotate}deg" if rotate else ""))

    out = []
    with torch.inference_mode():
        for i in range(0, len(tensors), batch_size):
            out.append(model(tensors[i:i + batch_size].to(device)).float().cpu())

    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()
