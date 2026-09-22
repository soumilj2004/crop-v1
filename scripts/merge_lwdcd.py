"""
Merge normalized LWDCD2020 images into the wheat TRAIN split only.

Design decision (documented in PLAN.md): the existing val/test splits stay
untouched so every model keeps the same honest benchmark. LWDCD2020 field
images are added ONLY to train, with an 'lwdcd_' filename prefix so provenance
is visible. Idempotent: files already merged are skipped.

No hardcoded paths: repo root = parent of scripts/, override with CROPGUARD_BASE.
Run on any machine with the same data layout (e.g. friend's GPU box).
"""
import argparse, json, os, shutil, sys
from pathlib import Path

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))

# LWDCD2020 folder name -> our train class folder name
CLASS_MAP = {
    "Crown and Root Rot": "Crown & Root Rot",
    "Healthy Wheat": "Healthy",
    "Leaf Rust": "Leaf Rust",
    "Wheat Loose Smut": "Loose Smut",
}


def merge(norm_dir: Path, split_dir: Path, crop: str = "wheat"):
    norm_dir = Path(norm_dir)
    split_dir = Path(split_dir)
    train_dir = split_dir / crop / "train"
    if not train_dir.is_dir():
        print(f"ERROR: {train_dir} not found. Point --split at the repo's data/split.")
        return None

    before = {}
    for cls in sorted(train_dir.iterdir()):
        if cls.is_dir():
            before[cls.name] = len(list(cls.glob("*")))

    copied, skipped, missing = 0, 0, []
    for src_cls, dst_cls in CLASS_MAP.items():
        src = norm_dir / src_cls
        dst = train_dir / dst_cls
        if not src.is_dir():
            missing.append(src_cls)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            dest = dst / f"lwdcd_{f.name}"
            if dest.exists():
                skipped += 1
            else:
                shutil.copy2(f, dest)
                copied += 1
        print(f"  {src_cls} -> {dst_cls}: {len(list(src.iterdir()))} images")

    after = {}
    for cls in sorted(train_dir.iterdir()):
        if cls.is_dir():
            after[cls.name] = len(list(cls.glob("*")))

    report = {
        "crop": crop,
        "copied": copied,
        "skipped_existing": skipped,
        "missing_source_classes": missing,
        "train_before": before,
        "train_after": after,
    }
    rep_path = REPO / "data" / "lwdcd_merge_report.json"
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print("BEFORE:", json.dumps(before))
    print("AFTER :", json.dumps(after))
    print(f"copied={copied} skipped={skipped} missing={missing}")
    print(f"Report: {rep_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge LWDCD2020 into wheat train split")
    ap.add_argument("--norm", type=str, default=str(REPO / "data" / "lwdcd2020_norm"),
                    help="normalized LWDCD folder (default: <repo>/data/lwdcd2020_norm)")
    ap.add_argument("--split", type=str, default=str(REPO / "data" / "split"),
                    help="data/split root (default: <repo>/data/split)")
    ap.add_argument("--crop", type=str, default="wheat", choices=["wheat"])
    args = ap.parse_args()
    report = merge(Path(args.norm), Path(args.split), args.crop)
    sys.exit(0 if report and not report["missing_source_classes"] else 1)
