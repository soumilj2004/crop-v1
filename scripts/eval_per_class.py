"""
Per-class evaluation (the shipping gate): precision / recall / F1 per class,
macro F1, accuracy, and confusion matrix on a test split.

Usage:
  python scripts/eval_per_class.py --model wheat [--ckpt models/wheat_efficientnet_best.pt] [--split test]

Output: JSON report next to the checkpoint + printed table.
Paths resolve relative to the repo root; override with CROPGUARD_BASE.
"""
import argparse, json, os, sys
from pathlib import Path

import torch
from torchvision import transforms

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

from inference import load_model, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

CLASSES = {
    "wheat": ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"],
    "rice": ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"],
    "wheat_stage": ["early", "mid", "late"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CLASSES))
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    classes = CLASSES[args.model]
    ckpt = Path(args.ckpt) if args.ckpt else REPO / "models" / f"{args.model}_efficientnet_best.pt"
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 1

    model, _ = load_model(ckpt, len(classes), classes)
    model.eval()
    model.to("cpu")
    size = getattr(model, "input_size", 224)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    split_dir = REPO / "data" / "split" / args.model / args.split
    if not split_dir.is_dir():
        print(f"ERROR: split not found: {split_dir}")
        return 1

    images = []
    labels = []
    for idx, cls in enumerate(classes):
        cdir = split_dir / cls
        if not cdir.is_dir():
            print(f"  [warn] missing class dir: {cdir}")
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                images.append(f)
                labels.append(idx)
    if not images:
        print("ERROR: no images found")
        return 1
    print(f"Evaluating {len(images)} images ({args.split}, {args.model}) with {ckpt.name}")

    from PIL import Image
    preds = []
    with torch.no_grad():
        for f, lab in zip(images, labels):
            img = Image.open(f).convert("RGB")
            x = tf(img).unsqueeze(0)
            out = model(x)
            preds.append(int(out.argmax(1).item()))

    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for lab, p in zip(labels, preds):
        cm[lab][p] += 1

    rows = []
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n)) - tp
        fn = sum(cm[c][r] for r in range(n)) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": classes[c], "count": tp + fn, "tp": tp, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)})

    acc = sum(cm[c][c] for c in range(n)) / len(images)
    macro_f1 = sum(r["f1"] for r in rows) / n
    print(f"\n{'class':<22}{'count':>6}{'P':>9}{'R':>9}{'F1':>9}")
    for r in rows:
        flag = "  <-- FAIL" if min(r["precision"], r["recall"], r["f1"]) < 0.9 else ""
        print(f"{r['class']:<22}{r['count']:>6}{r['precision']:>9.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}{flag}")
    print(f"\nAccuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}  (gate: every P/R/F1 >= 0.90)")

    report = {
        "model": args.model, "ckpt": str(ckpt), "split": args.split,
        "images": len(images), "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "classes": rows, "confusion_matrix": cm,
        "gate_passed": all(min(r["precision"], r["recall"], r["f1"]) >= 0.9 for r in rows),
    }
    out = ckpt.with_name(f"{ckpt.stem}_perclass.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
