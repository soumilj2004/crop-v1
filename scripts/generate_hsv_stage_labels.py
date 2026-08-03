"""Optimized HSV stage label generation with resume support."""
import json, os, time
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path("C:/CropGuardAI/cropguard_ai")

def lesion_ratio(img_path):
    img = Image.open(img_path).convert("HSV")
    arr = np.array(img, dtype=np.int32)
    h, s, v = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    is_green = (h >= 45) & (h <= 130) & (s > 40) & (v > 30)
    is_leaf_pixel = (v > 20) & ~((s < 20) & (v > 200))
    leaf_px = is_leaf_pixel.sum()
    if leaf_px == 0:
        return 0.0
    return float((is_leaf_pixel & ~is_green).sum() / leaf_px)

DISEASES = {
    "wheat": ["Crown & Root Rot", "Leaf Rust", "Loose Smut"],
    "rice": ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"],
}

def generate_labels(crop):
    split_dir = BASE / "data" / "split" / crop
    manifest_path = BASE / "data" / "stage_labels" / f"{crop}_hsv_stage_labels.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = BASE / "data" / "stage_labels" / f"{crop}_hsv_cache.json"

    # Load cached ratios if resuming
    cached = {}
    if cache_path.exists():
        cached = json.load(open(cache_path))

    per_disease = {d: {"items": [], "need_compute": []} for d in DISEASES[crop]}

    for split_name in ["train", "val", "test"]:
        split_path = split_dir / split_name
        if not split_path.exists():
            continue
        for cls in DISEASES[crop]:
            cls_dir = split_path / cls
            if not cls_dir.exists():
                continue
            for f in cls_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                    rel = str(f.relative_to(split_dir))
                    if rel in cached:
                        per_disease[cls]["items"].append((cached[rel], rel))
                    else:
                        per_disease[cls]["need_compute"].append((rel, f))

    # Compute missing lesion ratios
    for cls in DISEASES[crop]:
        items = per_disease[cls]
        if not items["need_compute"]:
            continue
        n = len(items["need_compute"])
        print(f"  Computing {cls}: {n} images...")
        t0 = time.time()
        for i, (rel, f) in enumerate(items["need_compute"]):
            try:
                lr = lesion_ratio(str(f))
                cached[rel] = lr
                items["items"].append((lr, rel))
            except Exception as e:
                print(f"    ERROR {f.name}: {e}")
            if (i+1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                remain = (n - i - 1) / rate
                print(f"    {i+1}/{n} ({rate:.1f}/s, {remain:.0f}s remain)")
                # Save cache every 500
                json.dump(cached, open(cache_path, "w"))
        print(f"    Done {cls}: {len(items['items'])} items in {time.time()-t0:.0f}s")
        json.dump(cached, open(cache_path, "w"))

    # Sort and tercile per disease
    manifest = {}
    for cls in DISEASES[crop]:
        items = per_disease[cls]["items"]
        if len(items) < 3:
            print(f"  {cls}: only {len(items)} images, all -> mid")
            for _, rel in items:
                manifest[rel] = "mid"
            continue

        items.sort(key=lambda x: x[0])
        n = len(items)
        n3 = n // 3
        r = n % 3
        boundaries = [0, n3, 2*n3 + (r > 0), n]

        for i in range(3):
            for idx in range(boundaries[i], boundaries[i+1]):
                manifest[items[idx][1]] = ["early", "mid", "late"][i]

        early_vals = [items[i][0] for i in range(boundaries[0], boundaries[1])]
        mid_vals = [items[i][0] for i in range(boundaries[1], boundaries[2])]
        late_vals = [items[i][0] for i in range(boundaries[2], boundaries[3])]
        print(f"  {cls} ({n}): early [{np.mean(early_vals):.3f}] mid [{np.mean(mid_vals):.3f}] late [{np.mean(late_vals):.3f}]")

    # Add Healthy -> early
    for split_name in ["train", "val", "test"]:
        split_path = split_dir / split_name
        healthy_dir = split_path / "Healthy"
        if healthy_dir.exists():
            for f in healthy_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                    rel = str(f.relative_to(split_dir))
                    manifest[rel] = "early"

    json.dump(manifest, open(manifest_path, "w"))
    print(f"\n  Manifest: {len(manifest)} labels -> {manifest_path}")

    stages = {"early": 0, "mid": 0, "late": 0}
    for v in manifest.values():
        stages[v] += 1
    for s, c in stages.items():
        print(f"    {s}: {c}")
    return manifest

def generate_contact_sheet(crop, manifest, output_path, samples_per_cell=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    split_dir = BASE / "data" / "split" / crop
    by_disease_stage = {}
    for d in DISEASES[crop]:
        by_disease_stage[d] = {"early": [], "mid": [], "late": []}

    for rel, stage in manifest.items():
        p = Path(rel)
        if len(p.parts) >= 3:
            disease = p.parts[1]
            if disease in by_disease_stage and stage in by_disease_stage[disease]:
                by_disease_stage[disease][stage].append(rel)

    n_diseases = len(DISEASES[crop])
    stage_colors = {"early": "green", "mid": "orange", "late": "red"}
    fig, axes = plt.subplots(n_diseases, 3, figsize=(12, n_diseases * 4))
    if n_diseases == 1:
        axes = axes.reshape(1, -1)

    for i, disease in enumerate(DISEASES[crop]):
        for j, stage in enumerate(["early", "mid", "late"]):
            ax = axes[i, j]
            samples = by_disease_stage[disease][stage][:samples_per_cell]
            ax.set_title(f"{disease}\n{stage} ({len(samples)} shown)", fontsize=10, color=stage_colors[stage])
            if not samples:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center")
                ax.set_facecolor('#f0f0f0')
            else:
                try:
                    img = Image.open(str(split_dir / samples[0]))
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Contact sheet: {output_path}")

if __name__ == "__main__":
    for crop in ["wheat", "rice"]:
        print(f"\n{'='*50}\n{crop}\n{'='*50}")
        manifest = generate_labels(crop)
        if manifest:
            generate_contact_sheet(crop, manifest, BASE / "models" / f"{crop}_stage_contact_sheet.png")
    print("\nALL DONE")
