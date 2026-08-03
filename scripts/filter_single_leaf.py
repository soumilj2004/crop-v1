"""
Option A: Single-leaf-dominant filter for rice.
Detects images where ONE large connected vegetation component dominates the frame
(rather than many overlapping clumps).
"""
import os, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

BASE = Path("C:/CropGuardAI/cropguard_ai")
RAW = BASE / "data" / "raw"
EXCLUDED = BASE / "data" / "raw_excluded_non_photo"
EXCLUDED.mkdir(parents=True, exist_ok=True)

def is_single_leaf_dominant(img_path, min_dominant_fraction=0.55):
    """
    Returns True if a single connected vegetation component covers >= min_dominant_fraction
    of the frame's vegetation pixels — i.e., one leaf blade dominates.
    """
    img = Image.open(img_path).convert("HSV")
    arr = np.array(img)
    h, s, v = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    
    # Broad vegetation mask (greens through yellows/browns)
    is_veg = (h >= 20) & (h <= 150) & (s > 25) & (v > 20) & (v < 240)
    veg_pixels = is_veg.sum()
    if veg_pixels == 0:
        return False
    
    # Connected components on vegetation mask
    labeled, n_components = ndimage.label(is_veg)
    if n_components == 0:
        return False
    
    # Find largest component size
    component_sizes = np.bincount(labeled.ravel())[1:]  # skip background (0)
    largest = component_sizes.max()
    dominant_fraction = largest / veg_pixels
    
    # Also check that the dominant component spans most of the frame height/width
    # (a single leaf blade should be long and run most of the frame)
    largest_label = np.argmax(component_sizes) + 1
    largest_mask = (labeled == largest_label)
    rows = np.where(largest_mask.any(axis=1))[0]
    cols = np.where(largest_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return False
    height_span = (rows.max() - rows.min() + 1) / is_veg.shape[0]
    width_span = (cols.max() - cols.min() + 1) / is_veg.shape[1]
    
    # A single leaf blade should span most of one dimension
    span_score = max(height_span, width_span)
    
    return dominant_fraction >= min_dominant_fraction and span_score >= 0.6

def apply_single_leaf_filter(crop="rice"):
    raw_dir = RAW / crop
    excluded_dir = EXCLUDED / crop
    excluded_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== {crop}: single-leaf filter ===")
    results = {}
    for cls_dir in sorted(raw_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        if cls_dir.name == "Healthy":
            print(f"  {cls_dir.name}: skipped (healthy excluded from stage labels)")
            continue
        valid_ext = ('.jpg','.jpeg','.png','.bmp')
        files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext]
        n_before = len(files)
        
        kept = []
        moved = 0
        for f in files:
            try:
                if is_single_leaf_dominant(str(f)):
                    kept.append(f)
                else:
                    dst = EXCLUDED / crop / f"{cls_dir.name}_not_single_leaf" / f.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        shutil.move(str(f), str(dst))
                        moved += 1
            except Exception as e:
                print(f"    ERROR {f.name}: {e}")
                kept.append(f)
        
        n_after = len(kept)
        results[cls_dir.name] = {"before": n_before, "after": n_after, "moved": moved}
        print(f"  {cls_dir.name}: {n_before} -> {n_after} (moved {moved})")
    return results

if __name__ == "__main__":
    apply_single_leaf_filter("rice")
    print("\nDone. Re-run dedup, split, and stage labels.")