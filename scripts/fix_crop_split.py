"""Recreate crop splits from existing rice + wheat splits using hard links."""
import os, shutil
from pathlib import Path

BASE = Path("C:/CropGuardAI/cropguard_ai")
SPLIT = BASE / "data" / "split"
CROP = SPLIT / "crop"

if CROP.exists():
    shutil.rmtree(CROP)
CROP.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    out = CROP / split
    out.mkdir(parents=True, exist_ok=True)
    
    for crop in ["rice", "wheat"]:
        src = SPLIT / crop / split
        if not src.exists():
            print(f"  Missing: {src}")
            continue
        crop_out = out / crop
        crop_out.mkdir(parents=True, exist_ok=True)
        for cls_dir in src.iterdir():
            if not cls_dir.is_dir():
                continue
            dst = crop_out / cls_dir.name
            if not dst.exists():
                # Use hard link instead of copy for speed
                try:
                    os.link(str(cls_dir), str(dst))
                except OSError:
                    shutil.copytree(cls_dir, dst)
        n = sum(1 for _ in src.rglob('*') if _.is_file())
        print(f"  {crop}/{split}: {n} files")

for s in ["train", "val", "test"]:
    n = sum(1 for _ in (CROP/s).rglob("*") if _.is_file())
    print(f"crop/{s}: {n}")