#!/usr/bin/env python3
"""Recreate all data splits - flat structure."""
import random
import shutil
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
random.seed(42)

rice_classes = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
wheat_classes = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]

# Disease splits
for dataset, classes in [("rice", rice_classes), ("wheat", wheat_classes)]:
    src_dir = RAW_DIR / dataset
    split_base = DATA_DIR / "split" / dataset

    for class_name in classes:
        class_dir = src_dir / class_name
        if not class_dir.exists():
            print(f"  SKIP {dataset}/{class_name} - not found")
            continue

        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            images.extend(class_dir.glob(ext))

        seen = set()
        unique_images = []
        for img in images:
            if img.name not in seen:
                seen.add(img.name)
                unique_images.append(img)

        random.shuffle(unique_images)
        n = len(unique_images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_imgs = unique_images[:n_train]
        val_imgs = unique_images[n_train:n_train + n_val]
        test_imgs = unique_images[n_train + n_val:]

        for split_name, split_images in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest = split_base / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for old_file in dest.glob("*.*"):
                old_file.unlink()
            for img in split_images:
                shutil.copy2(str(img), str(dest / img.name))

        print(f"  {dataset}/{class_name}: {n} -> train:{len(train_imgs)} val:{len(val_imgs)} test:{len(test_imgs)}")

# Crop type splits
print("\nCrop type splits:")
crop_split_base = DATA_DIR / "split" / "crop"
for crop_type in ["rice", "wheat"]:
    all_images = []
    for class_dir in (RAW_DIR / crop_type).iterdir():
        if class_dir.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                all_images.extend(class_dir.glob(ext))

    random.shuffle(all_images)
    n = len(all_images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train + n_val]
    test_imgs = all_images[n_train + n_val:]

    for split_name, split_images in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        dest = crop_split_base / split_name / crop_type
        dest.mkdir(parents=True, exist_ok=True)
        for old_file in dest.glob("*.*"):
            old_file.unlink()
        for img in split_images:
            shutil.copy2(str(img), str(dest / img.name))

    print(f"  crop/{crop_type}: {n} -> train:{len(train_imgs)} val:{len(val_imgs)} test:{len(test_imgs)}")

# Stage splits
print("\nStage splits:")
rice_stage_map = {
    "Bacterial Blight": "early", "Blast": "mid", "Brown Spot": "mid",
    "Tungro": "late", "Healthy": "early",
}
wheat_stage_map = {
    "Leaf Rust": "mid", "Loose Smut": "late",
    "Crown & Root Rot": "late", "Healthy": "early",
}

for dataset, stage_map in [("rice", rice_stage_map), ("wheat", wheat_stage_map)]:
    stage_split_base = DATA_DIR / "split" / f"{dataset}_stage"

    for stage in ["early", "mid", "late"]:
        all_images = []
        for cls, stg in stage_map.items():
            if stg == stage:
                cls_dir = RAW_DIR / dataset / cls
                if cls_dir.exists():
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                        all_images.extend(cls_dir.glob(ext))

        random.shuffle(all_images)
        n = len(all_images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_imgs = all_images[:n_train]
        val_imgs = all_images[n_train:n_train + n_val]
        test_imgs = all_images[n_train + n_val:]

        for split_name, split_images in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest = stage_split_base / split_name / stage
            dest.mkdir(parents=True, exist_ok=True)
            for old_file in dest.glob("*.*"):
                old_file.unlink()
            for img in split_images:
                shutil.copy2(str(img), str(dest / img.name))

        print(f"  {dataset}_stage/{stage}: {n} -> train:{len(train_imgs)} val:{len(val_imgs)} test:{len(test_imgs)}")

print("\nAll splits created!")
