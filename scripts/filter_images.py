"""
Apply photographic + close-up filters to raw images.
Move rejected images to data/raw_excluded_non_photo/.
"""

import os, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path("C:/CropGuardAI/cropguard_ai")
RAW = BASE / "data" / "raw"
EXCLUDED = BASE / "data" / "raw_excluded_non_photo"
EXCLUDED.mkdir(parents=True, exist_ok=True)

def is_likely_photograph(img_path, min_mean_saturation=18):
    """Reject line drawings, diagrams, screenshots (low avg saturation)."""
    img = Image.open(img_path).convert("HSV")
    arr = np.array(img)
    mean_saturation = arr[:, :, 1].mean()
    return mean_saturation >= min_mean_saturation

def is_closeup_leaf_shot(img_path, min_vegetation_fraction=0.35):
    """Reject wide field shots where vegetation doesn't dominate frame."""
    img = Image.open(img_path).convert("HSV")
    arr = np.array(img)
    h, s, v = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    is_vegetation_like = (h >= 20) & (h <= 150) & (s > 25) & (v > 20) & (v < 240)
    return is_vegetation_like.mean() >= min_vegetation_fraction

def run_filters(crop):
    raw_dir = RAW / crop
    if not raw_dir.exists():
        print(f"  SKIP {crop}: no raw dir")
        return {}

    print(f"\n=== {crop} ===")
    results = {}
    for cls_dir in sorted(raw_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        valid_ext = ('.jpg','.jpeg','.png','.bmp')
        files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext]
        n_before = len(files)

        kept = []
        excluded_non_photo = []
        excluded_not_closeup = []
        excluded_both = []

        for f in files:
            try:
                is_photo = is_likely_photograph(str(f))
                if not is_photo:
                    excluded_non_photo.append(f)
                    continue
                is_closeup = is_closeup_leaf_shot(str(f))
                if not is_closeup:
                    excluded_not_closeup.append(f)
                    continue
                kept.append(f)
            except Exception as e:
                print(f"    ERROR {f.name}: {e}")
                kept.append(f)  # Keep on error

        # Move excluded files
        for f in excluded_non_photo + excluded_not_closeup:
            rel = f.relative_to(RAW)
            dst = EXCLUDED / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.move(str(f), str(dst))

        n_after = len(kept)
        n_non_photo = len(excluded_non_photo)
        n_not_closeup = len(excluded_not_closeup)
        results[cls_dir.name] = {
            "before": n_before,
            "after": n_after,
            "non_photo": n_non_photo,
            "not_closeup": n_not_closeup,
        }

        parts = []
        if n_non_photo:
            parts.append(f"non-photo={n_non_photo}")
        if n_not_closeup:
            parts.append(f"not-closeup={n_not_closeup}")
        suffix = ", ".join(parts) if parts else "all kept"
        print(f"  {cls_dir.name}: {n_before} -> {n_after} ({suffix})")

    return results

if __name__ == "__main__":
    for crop in ["wheat", "rice"]:
        results = run_filters(crop)
        if results:
            total_before = sum(r["before"] for r in results.values())
            total_after = sum(r["after"] for r in results.values())
            total_excluded = total_before - total_after
            print(f"  TOTAL {crop}: {total_before} -> {total_after} (excluded {total_excluded})")
    print("\nFiltering complete. Now re-run dedup.")
