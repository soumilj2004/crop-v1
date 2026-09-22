"""70/15/15 stratified split from deduped raw data using hard links."""

import os, shutil, random
from pathlib import Path

BASE = Path("C:/CropGuardAI/cropguard_ai")
SPLIT = BASE / "data" / "split"
RAW = BASE / "data" / "raw"
SEED = 42
random.seed(SEED)

def hardlink_split(src_dir, split_dir):
    split_dir.mkdir(parents=True, exist_ok=True)
    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        files = sorted([f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext])
        random.shuffle(files)
        n = len(files)
        n1 = int(n * 0.7)
        n2 = int(n * 0.15)
        slices = [("train", slice(0, n1)), ("val", slice(n1, n1+n2)), ("test", slice(n1+n2, None))]
        for name, sl in slices:
            out_dir = split_dir / name / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files[sl]:
                dst = out_dir / f.name
                if not dst.exists():
                    try:
                        os.link(str(f), str(dst))
                    except OSError:
                        shutil.copy2(str(f), str(dst))
        print(f"  {cls_dir.name}: {n} -> train:{n1} val:{n2} test:{n-n1-n2}")

# Wheat
print("=== Wheat ===")
hardlink_split(RAW / "wheat", SPLIT / "wheat")
for s in ["train","val","test"]:
    n = sum(1 for _ in (SPLIT/"wheat"/s).rglob("*") if _.is_file())
    print(f"  wheat/{s}: {n}")

# Rice
print("\n=== Rice ===")
hardlink_split(RAW / "rice", SPLIT / "rice")
for s in ["train","val","test"]:
    n = sum(1 for _ in (SPLIT/"rice"/s).rglob("*") if _.is_file())
    print(f"  rice/{s}: {n}")

print("\nDONE")
