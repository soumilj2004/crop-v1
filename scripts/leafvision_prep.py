"""
Prepare LeafVision SSL backbones for use as training init.

Converts LeafVision DINO checkpoints (git-lfs files, e.g. from
github.com/LABA-SNU/LeafVision) into backbone-only state dicts that
train_efficientnet.py can load via `--init leafvision_dino_<arch>`.

Expected source layout: <src>/LeafVision_DINO_<arch>.pth
Output: <out>/leafvision_dino_<arch>_backbone.pt

No hardcoded paths: repo root = parent of scripts/, override with CROPGUARD_BASE.
"""
import argparse, os
from pathlib import Path

import torch
from torchvision import models

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))

BACKBONES = {
    "efficientnet_b0": models.efficientnet_b0,
    "resnet50": models.resnet50,
    "resnet18": models.resnet18,
}


def extract_backbone(arch: str, src_dir: Path, out_dir: Path):
    src = src_dir / f"LeafVision_DINO_{arch}.pth"
    if not src.exists():
        print(f"SKIP {arch}: {src} not found")
        return False
    raw = torch.load(src, map_location="cpu")
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        state = raw
    elif isinstance(raw, dict):
        if "state_dict" in raw:
            state = raw["state_dict"]
        elif "model_state_dict" in raw:
            state = raw["model_state_dict"]
        else:
            cand = {k: v for k, v in raw.items() if isinstance(v, dict) and v
                    and all(isinstance(x, torch.Tensor) for x in v.values())}
            state = next(iter(cand.values())) if cand else None
    else:
        state = None
    if state is None:
        print(f"FAIL {arch}: could not locate state_dict in {src.name}")
        return False

    model = BACKBONES[arch](weights=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("classifier") and not k.startswith("fc")]
    if bad or unexpected:
        print(f"FAIL {arch}: missing={len(bad)} unexpected={len(unexpected)}; bad={bad[:5]} unexpected={unexpected[:5]}")
        return False

    out = out_dir / f"leafvision_dino_{arch}_backbone.pt"
    keep = {k: v for k, v in model.state_dict().items()
            if not k.startswith("classifier.") and not k.startswith("fc.")}
    torch.save(keep, out)
    print(f"OK {arch}: {len(state)} keys -> {out} ({out.stat().st_size/1e6:.1f} MB, {len(keep)} backbone keys)")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare LeafVision DINO backbones for training init")
    ap.add_argument("--src", type=str, default=str(REPO / "data" / "leafvision"),
                    help="folder containing LeafVision_DINO_*.pth (default: <repo>/data/leafvision)")
    ap.add_argument("--out", type=str, default=str(REPO / "models"),
                    help="output folder for backbone .pt files (default: <repo>/models)")
    ap.add_argument("--archs", type=str, nargs="+", default=list(BACKBONES),
                    help="backbones to extract (default: all supported)")
    args = ap.parse_args()
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = all(extract_backbone(a, src_dir, out_dir) for a in args.archs)
    raise SystemExit(0 if ok else 1)
