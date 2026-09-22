"""
Normalize LWDCD2020 raw images into clean RGB JPEGs for training.

Converts every image (jpg/jfif/jpeg/png/gif) to RGB JPEG (max side capped),
skips corrupt files, reports stats. Idempotent: existing outputs are skipped.
No hardcoded paths: repo root = parent of scripts/, override with --src/--out.
"""
import argparse, json, os, sys
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))

CLASS_ALIAS = {}  # keep original folder names; mapping happens at merge time


def normalize(src_dir: Path, out_dir: Path, max_side: int = 1024, quality: int = 90):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    stats = {"inputs": 0, "outputs": 0, "skipped_existing": 0, "errors": {}}
    classes = sorted(d for d in src_dir.iterdir() if d.is_dir())
    for cls in classes:
        src_cls = src_dir / cls.name
        out_cls = out_dir / cls.name
        out_cls.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in src_cls.iterdir() if f.is_file())
        errs = []
        for i, f in enumerate(files):
            stats["inputs"] += 1
            dest = out_cls / f"{cls.name}__{i:05d}.jpg"
            if dest.exists():
                stats["skipped_existing"] += 1
                continue
            try:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    im.thumbnail((max_side, max_side), Image.LANCZOS)
                    im.save(dest, "JPEG", quality=quality)
                stats["outputs"] += 1
            except Exception as e:
                errs.append(f"{f.name}: {e}")
        if errs:
            stats["errors"][cls.name] = errs
        print(f"  {cls.name}: {len(files)} -> {len(files) - len(errs) - stats['skipped_existing']} new, {len(errs)} errors")
    report = out_dir / "normalize_report.json"
    with open(report, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"STATS: {json.dumps(stats)}")
    print(f"Report: {report}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Normalize LWDCD2020 images to RGB JPEG")
    ap.add_argument("--src", type=str, default=str(REPO / "data" / "lwdcd2020"),
                    help="raw LWDCD2020 folder (default: <repo>/data/lwdcd2020)")
    ap.add_argument("--out", type=str, default=str(REPO / "data" / "lwdcd2020_norm"),
                    help="output folder (default: <repo>/data/lwdcd2020_norm)")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()
    stats = normalize(Path(args.src), Path(args.out), args.max_side, args.quality)
    sys.exit(0 if stats["errors"] == {} else 1)
