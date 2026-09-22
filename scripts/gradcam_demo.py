"""
Grad-CAM heatmap demo for app explainability (no extra deps, manual hooks).

For each of the first N images per class in a split, saves the original image
and a Grad-CAM overlay (jet colormap) into <repo>/data/gradcam/<class>/.

Supports CNN backbones (efficientnet_*, mobilenet_*, resnet*). Swin checkpoints
are skipped with a warning.

Usage:
  python scripts/gradcam_demo.py --model wheat [--ckpt models/wheat_efficientnet_best.pt] [--n 3] [--split val]
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

from inference import load_model, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

CLASSES = {
    "wheat": ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"],
    "rice": ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"],
    "wheat_stage": ["early", "mid", "late"],
}


def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def gradcam(model, conv, x, target_cls):
    act = {}

    def fh(module, inp, out):
        act["a"] = out.detach()

    def bh(module, grad_in, grad_out):
        act["g"] = grad_out[0].detach()

    h1 = conv.register_forward_hook(fh)
    h2 = conv.register_full_backward_hook(bh)
    out = model(x)
    onehot = torch.zeros_like(out)
    onehot[0, target_cls] = 1.0
    model.zero_grad()
    out.backward(gradient=onehot)
    h1.remove()
    h2.remove()
    weights = act["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam[0, 0].cpu().numpy()


def overlay(img_rgb, cam, alpha=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(img_rgb)
    axes[1].imshow(cam, cmap="jet", alpha=alpha)
    axes[1].set_title("grad-cam")
    axes[1].axis("off")
    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CLASSES))
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    ap.add_argument("--n", type=int, default=3, help="images per class")
    args = ap.parse_args()

    classes = CLASSES[args.model]
    ckpt = Path(args.ckpt) if args.ckpt else REPO / "models" / f"{args.model}_efficientnet_best.pt"
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 1
    model, _ = load_model(ckpt, len(classes), classes)
    model.eval()
    conv = find_last_conv(model)
    if conv is None:
        print(f"SKIP: no Conv2d found (checkpoint may be a transformer arch): {ckpt.name}")
        return 0
    size = getattr(model, "input_size", 224)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    out_root = REPO / "data" / "gradcam"
    done = 0
    for idx, cls in enumerate(classes):
            cdir = REPO / "data" / "split" / args.model / args.split / cls
            if not cdir.is_dir():
                continue
            files = sorted(f for f in cdir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            odir = out_root / cls
            odir.mkdir(parents=True, exist_ok=True)
            for f in files[: args.n]:
                img = Image.open(f).convert("RGB")
                x = tf(img).unsqueeze(0)
                logits = model(x)
                target = int(logits.argmax(1).item())
                cam = gradcam(model, conv, x, target)
                fig = overlay(np.array(img), cam)
                dest = odir / f"{f.stem}_pred-{classes[target]}.png"
                fig.savefig(dest, dpi=90, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(f"  {cls}: {f.name} -> predicted {classes[target]} -> {dest}")
                done += 1
    print(f"Saved {done} Grad-CAM overlays to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

