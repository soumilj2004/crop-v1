#!/usr/bin/env python3
"""
Generate heuristic pseudo-labels for disease severity stage (early/mid/late).
Method: HSV-based lesion area ratio per disease class, split into terciles.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# HSV thresholds (PIL HSV is 0-255)
HUE_MIN, HUE_MAX = 45, 130
SAT_MIN = 40
VAL_MIN = 30
LEAF_VAL_MIN = 20

SPLIT_DIR = Path("data/split")
LABELS_DIR = Path("data/stage_labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)

RICE_DISEASES = ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"]
WHEAT_DISEASES = ["Leaf Rust", "Loose Smut", "Crown & Root Rot"]

def compute_lesion_ratio(image_path):
    """Compute fraction of leaf pixels that are non-green (lesion proxy)."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((160, 160))
        hsv = img.convert("HSV")
        hsv_arr = np.array(hsv)
        
        h = hsv_arr[:, :, 0]
        s = hsv_arr[:, :, 1]
        v = hsv_arr[:, :, 2]
        
        # Leaf mask
        leaf_mask = (v > LEAF_VAL_MIN) & (v < 250) & (s > 10)
        
        if leaf_mask.sum() == 0:
            return 0.0
        
        # Healthy green mask
        green_mask = (h >= HUE_MIN) & (h <= HUE_MAX) & (s >= SAT_MIN) & (v >= VAL_MIN)
        
        # Lesion = leaf but not healthy green
        lesion_mask = leaf_mask & ~green_mask
        
        ratio = lesion_mask.sum() / leaf_mask.sum()
        return float(ratio)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0.0

def generate_stage_labels(crop_name, disease_classes, split_name="train"):
    """Generate tercile-based stage labels for one crop's diseases."""
    print(f"\nGenerating stage labels for {crop_name} ({split_name})...")
    
    split_dir = SPLIT_DIR / crop_name / split_name
    if not split_dir.exists():
        print(f"  ERROR: {split_dir} does not exist")
        return {}
    
    # Compute ratios for all images
    ratios = {}
    for disease in disease_classes:
        disease_dir = split_dir / disease
        if not disease_dir.exists():
            print(f"  WARNING: {disease_dir} not found")
            continue
        
        print(f"  {disease}: computing ratios...")
        for img_path in tqdm(list(disease_dir.glob("*")), desc=f"    {disease}"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            ratio = compute_lesion_ratio(img_path)
            ratios[f"{disease}/{img_path.name}"] = (disease, ratio)
    
    # Split into terciles PER DISEASE CLASS
    stage_labels = defaultdict(dict)
    
    for disease in disease_classes:
        disease_ratios = {k: v for k, v in ratios.items() if v[0] == disease}
        if not disease_ratios:
            continue
        
        # Sort by ratio
        sorted_items = sorted(disease_ratios.items(), key=lambda x: x[1][1])
        n = len(sorted_items)
        
        # Tercile boundaries
        n_early = n // 3
        n_mid = n // 3
        # Remainder goes to late
        
        for i, (key, (d, ratio)) in enumerate(sorted_items):
            img_name = key.split("/")[-1]
            if i < n_early:
                stage = "early"
            elif i < n_early + n_mid:
                stage = "mid"
            else:
                stage = "late"
            stage_labels[disease][img_name] = stage
        
        # Print distribution
        dist = defaultdict(int)
        for s in stage_labels[disease].values():
            dist[s] += 1
        print(f"    {disease}: early={dist['early']}, mid={dist['mid']}, late={dist['late']}")
    
    return stage_labels

def create_validation_sheet(crop_name, disease_classes, split_name="train"):
    """Create a visual validation grid: sample images per disease per stage."""
    labels_file = LABELS_DIR / f"{crop_name}_{split_name}_stage_labels.json"
    if not labels_file.exists():
        return
    
    with open(labels_file) as f:
        stage_labels = json.load(f)
    
    split_dir = SPLIT_DIR / crop_name / split_name
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(disease_classes), 3, figsize=(9, 3*len(disease_classes)))
    if len(disease_classes) == 1:
        axes = axes.reshape(1, -1)
    
    for row, disease in enumerate(disease_classes):
        if disease not in stage_labels:
            continue
        
        for col, stage in enumerate(["early", "mid", "late"]):
            ax = axes[row, col]
            
            # Find sample images for this disease+stage
            samples = [img for img, s in stage_labels[disease].items() if s == stage]
            
            if samples:
                sample = samples[0]
                img_path = split_dir / disease / sample
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except:
                    ax.text(0.5, 0.5, "Error loading", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No samples", ha='center', va='center')
            
            ax.set_title(f"{disease}\n{stage}")
            ax.axis('off')
    
    plt.tight_layout()
    out_path = LABELS_DIR / f"{crop_name}_{split_name}_stage_validation_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Validation grid saved to {out_path}")

def main():
    print("=" * 60)
    print("GENERATING HEURISTIC STAGE PSEUDO-LABELS")
    print("=" * 60)
    print("Method: HSV lesion ratio -> terciles per disease class")
    print("NOTE: These are HEURISTIC labels, NOT ground truth!")
    print("      Validate the grid before training!")
    
    # Generate for rice train split
    rice_labels = generate_stage_labels("rice", RICE_DISEASES, "train")
    with open(LABELS_DIR / "rice_train_stage_labels.json", "w") as f:
        json.dump(rice_labels, f, indent=2)
    create_validation_sheet("rice", RICE_DISEASES, "train")
    
    # Generate for wheat train split
    wheat_labels = generate_stage_labels("wheat", WHEAT_DISEASES, "train")
    with open(LABELS_DIR / "wheat_train_stage_labels.json", "w") as f:
        json.dump(wheat_labels, f, indent=2)
    create_validation_sheet("wheat", WHEAT_DISEASES, "train")
    
    # Also generate for val/test if they exist
    for split in ["val", "test"]:
        for crop, diseases in [("rice", RICE_DISEASES), ("wheat", WHEAT_DISEASES)]:
            labels = generate_stage_labels(crop, diseases, split)
            if labels:
                with open(LABELS_DIR / f"{crop}_{split}_stage_labels.json", "w") as f:
                    json.dump(labels, f, indent=2)
    
    print("\n" + "=" * 60)
    print("STAGE LABELS GENERATED")
    print("=" * 60)
    print(f"Files saved to {LABELS_DIR}/")
    print("CRITICAL: Check the validation grid images before training!")
    print("  - rice_train_stage_validation_grid.png")
    print("  - wheat_train_stage_validation_grid.png")

if __name__ == "__main__":
    main()