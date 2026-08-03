#!/usr/bin/env python3
"""
Data splitting: deduplicate (MD5) -> stratified split (70/15/15) -> save.
Must run AFTER downloading and organizing raw images into data/raw/<crop>/<Class>/
"""

import os
import json
import hashlib
import shutil
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test

RAW_DIR = Path("data/raw")
SPLIT_DIR = Path("data/split")

RICE_CLASSES = ["Bacterial Blight", "Blast", "Brown Spot", "Tungro", "Healthy"]
WHEAT_CLASSES = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]

def md5_file(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def deduplicate_folder(folder_path):
    """Remove exact duplicate images by MD5 hash."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return 0, 0
    
    seen = {}
    removed = 0
    total = 0
    
    for img_path in folder_path.glob("*.jpg"):
        total += 1
        file_hash = md5_file(img_path)
        if file_hash in seen:
            img_path.unlink()
            removed += 1
        else:
            seen[file_hash] = img_path
    
    for img_path in folder_path.glob("*.jpeg"):
        total += 1
        file_hash = md5_file(img_path)
        if file_hash in seen:
            img_path.unlink()
            removed += 1
        else:
            seen[file_hash] = img_path
    
    for img_path in folder_path.glob("*.png"):
        total += 1
        file_hash = md5_file(img_path)
        if file_hash in seen:
            img_path.unlink()
            removed += 1
        else:
            seen[file_hash] = img_path
    
    return total, removed

def split_dataset(crop_name, classes):
    """Split deduplicated images into train/val/test."""
    print(f"\n{'='*50}")
    print(f"SPLITTING {crop_name.upper()}")
    print(f"{'='*50}")
    
    raw_crop_dir = RAW_DIR / crop_name
    split_crop_dir = SPLIT_DIR / crop_name
    
    # Collect all images per class
    class_images = {}
    total_before = 0
    total_after = 0
    total_removed = 0
    
    for cls in classes:
        cls_dir = raw_crop_dir / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} does not exist, skipping")
            class_images[cls] = []
            continue
        
        # Deduplicate
        before, removed = deduplicate_folder(cls_dir)
        total_before += before
        total_removed += removed
        after = before - removed
        total_after += after
        
        # Collect remaining images
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
        class_images[cls] = images
        print(f"  {cls}: {before} -> {after} images ({removed} duplicates removed)")
    
    print(f"  TOTAL: {total_before} -> {total_after} ({total_removed} duplicates removed, {100*total_removed/total_before:.1f}%)")
    
    # Stratified split
    for split_name, split_dir in [("train", split_crop_dir / "train"),
                                   ("val", split_crop_dir / "val"),
                                   ("test", split_crop_dir / "test")]:
        for cls in classes:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)
    
    random.seed(SEED)
    
    for cls, images in class_images.items():
        if not images:
            continue
        
        # Shuffle
        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for img in train_imgs:
            shutil.copy2(img, split_crop_dir / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy2(img, split_crop_dir / "val" / cls / img.name)
        for img in test_imgs:
            shutil.copy2(img, split_crop_dir / "test" / cls / img.name)
        
        print(f"  {cls}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    # Save split info
    split_info = {
        "crop": crop_name,
        "seed": SEED,
        "ratios": SPLIT_RATIOS,
        "classes": {cls: {"train": len(list((split_crop_dir/"train"/cls).glob("*"))),
                          "val": len(list((split_crop_dir/"val"/cls).glob("*"))),
                          "test": len(list((split_crop_dir/"test"/cls).glob("*")))}
                    for cls in classes}
    }
    with open(split_crop_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

def create_crop_split():
    """Create rice vs wheat binary classification dataset from existing splits."""
    print(f"\n{'='*50}")
    print(f"CREATING CROP CLASSIFIER DATASET (rice vs wheat)")
    print(f"{'='*50}")
    
    crop_dir = SPLIT_DIR / "crop"
    for split in ["train", "val", "test"]:
        (crop_dir / split / "rice").mkdir(parents=True, exist_ok=True)
        (crop_dir / split / "wheat").mkdir(parents=True, exist_ok=True)
    
    # Copy rice images labeled as "rice"
    for split in ["train", "val", "test"]:
        rice_src = SPLIT_DIR / "rice" / split
        wheat_src = SPLIT_DIR / "wheat" / split
        crop_dst = crop_dir / split
        
        rice_count = 0
        for cls_dir in rice_src.iterdir():
            if cls_dir.is_dir():
                for img in cls_dir.glob("*"):
                    dst = crop_dst / "rice" / f"rice_{cls_dir.name}_{img.name}"
                    shutil.copy2(img, dst)
                    rice_count += 1
        
        wheat_count = 0
        for cls_dir in wheat_src.iterdir():
            if cls_dir.is_dir():
                for img in cls_dir.glob("*"):
                    dst = crop_dst / "wheat" / f"wheat_{cls_dir.name}_{img.name}"
                    shutil.copy2(img, dst)
                    wheat_count += 1
        
        print(f"  {split}: rice={rice_count}, wheat={wheat_count}")

def create_stage_splits(crop_name, disease_classes):
    """Create stage classification splits from disease images only (exclude Healthy)."""
    print(f"\n{'='*50}")
    print(f"CREATING STAGE SPLITS FOR {crop_name.upper()}")
    print(f"{'='*50}")
    
    stage_dir = SPLIT_DIR / f"{crop_name}_stage"
    for split in ["train", "val", "test"]:
        for stage in ["early", "mid", "late"]:
            (stage_dir / split / stage).mkdir(parents=True, exist_ok=True)
    
    # Load stage labels
    labels_file = Path("data/stage_labels") / f"{crop_name}_train_stage_labels.json"
    if not labels_file.exists():
        print(f"  ERROR: Stage labels not found at {labels_file}")
        print("  Run generate_stage_labels.py first!")
        return
    
    with open(labels_file) as f:
        stage_labels = json.load(f)
    
    # For each disease class, copy images to stage folders based on labels
    for disease in disease_classes:
        if disease not in stage_labels:
            continue
        
        for split in ["train", "val", "test"]:
            src_dir = SPLIT_DIR / crop_name / split / disease
            if not src_dir.exists():
                continue
            
            for img_path in src_dir.glob("*"):
                label_info = stage_labels[disease].get(img_path.name)
                if label_info:
                    dst_dir = stage_dir / split / label_info
                    dst_path = dst_dir / f"{disease}_{img_path.name}"
                    shutil.copy2(img_path, dst_path)
    
    # Count
    for split in ["train", "val", "test"]:
        counts = {}
        for stage in ["early", "mid", "late"]:
            n = len(list((stage_dir / split / stage).glob("*")))
            counts[stage] = n
        print(f"  {split}: {counts}")

def main():
    print("=" * 60)
    print("DATA SPLITTING PIPELINE")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Ratios: train={SPLIT_RATIOS[0]}, val={SPLIT_RATIOS[1]}, test={SPLIT_RATIOS[2]}")
    
    # Split rice
    split_dataset("rice", RICE_CLASSES)
    
    # Split wheat
    split_dataset("wheat", WHEAT_CLASSES)
    
    # DEPRECATED: Crop classifier dataset is redundant — crop is determined by user input in the app.
    # Keeping only stage splits.
    # create_crop_split()
    
    # Create stage splits (requires stage labels to exist)
    create_stage_splits("rice", ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"])
    create_stage_splits("wheat", ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"])
    
    print("\n" + "=" * 60)
    print("ALL SPLITS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()