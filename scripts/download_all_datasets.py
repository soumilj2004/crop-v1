#!/usr/bin/env python3
"""
Download and prepare all datasets for CropGuard AI using HuggingFace datasets.
Rice: minhhungg/rice-disease-dataset (37,978 images, 21 classes -> map to 5)
Wheat: Multiple sources combined
"""

import os
import sys
import hashlib
import shutil
import json
import random
from pathlib import Path
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
TEMP_DIR = DATA_DIR / "temp" / "hf_downloads"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def md5_file(path):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_image(image, save_path):
    """Save PIL image to path, handling format conversion."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        return False
    try:
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'P':
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')
        image.save(str(save_path), 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"  Error saving {save_path}: {e}")
        return False


def deduplicate_folder(folder_path):
    """Remove exact duplicate images by MD5 hash."""
    folder_path = Path(folder_path)
    seen_hashes = {}
    removed = 0
    for img_file in list(folder_path.glob("*.*")):
        if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        try:
            file_hash = md5_file(img_file)
            if file_hash in seen_hashes:
                img_file.unlink()
                removed += 1
            else:
                seen_hashes[file_hash] = img_file.name
        except Exception:
            pass
    return len(seen_hashes), removed


# ============================================================
# RICE DATASET - minhhungg/rice-disease-dataset (37,978 images)
# ============================================================
def download_rice_dataset():
    """
    Download from minhhungg/rice-disease-dataset.
    Maps 21 Vietnamese classes to our 5 classes.
    """
    print("=" * 70)
    print("DOWNLOADING RICE DATASET (minhhungg/rice-disease-dataset)")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        return False

    dest_dir = RAW_DIR / "rice"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have enough data
    total_existing = sum(
        len(list((dest_dir / c).glob("*.*")))
        for c in ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
        if (dest_dir / c).exists()
    )
    if total_existing > 5000:
        print(f"Rice dataset already has {total_existing} images. Skipping download.")
        return True

    # Class mapping from dataset labels to our labels
    # Dataset has 21 classes, we need 5
    CLASS_MAP = {
        # Bacterial Blight
        "Bacterial Leaf Blight": "Bacterial Blight",
        "bacterial_leaf_blight": "Bacterial Blight",
        # Blast
        "Blast": "Blast",
        "blast": "Blast",
        # Brown Spot
        "Brown Spot": "Brown Spot",
        "brown_spot": "Brown Spot",
        # Tungro
        "Tungro Virus": "Tungro",
        "tungro": "Tungro",
        "Tungro": "Tungro",
        # Healthy
        "Healthy": "Healthy",
        "healthy": "Healthy",
        "normal": "Healthy",
    }

    print("Loading dataset from HuggingFace...")
    print("This may take several minutes on first run...")

    try:
        # Try loading the full dataset
        ds = load_dataset("minhhungg/rice-disease-dataset", trust_remote_code=True)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")

        # Get the train split (has all data)
        if "train" in ds:
            split_data = ds["train"]
        else:
            split_data = ds[list(ds.keys())[0]]

        print(f"Total samples: {len(split_data)}")
        print(f"Features: {split_data.features}")

    except Exception as e:
        print(f"Error loading full dataset: {e}")
        print("Trying to load individual splits...")
        try:
            ds = load_dataset("minhhungg/rice-disease-dataset", split="train", trust_remote_code=True)
            split_data = ds
            print(f"Loaded train split: {len(split_data)} samples")
        except Exception as e2:
            print(f"Error: {e2}")
            return False

    # Map labels and save images
    counts = defaultdict(int)
    skipped = 0
    saved = 0

    # Check what field name the label uses
    label_field = None
    for field in ["label", "disease", "class", "category", "class_label"]:
        if field in split_data.column_names:
            label_field = field
            break

    if label_field is None:
        print(f"Available columns: {split_data.column_names}")
        print("Could not identify label field. Trying 'label'...")
        label_field = "label"

    print(f"\nUsing label field: '{label_field}'")
    print(f"Label distribution:")
    label_counts = defaultdict(int)
    for item in split_data:
        label_counts[item[label_field]] += 1
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        mapped = CLASS_MAP.get(label, "???")
        print(f"  {label} -> {mapped} ({count})")

    print(f"\nSaving images to {dest_dir}...")

    for i, item in enumerate(split_data):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(split_data)}...")

        raw_label = item[label_field]

        # Map to our class names
        if raw_label in CLASS_MAP:
            our_class = CLASS_MAP[raw_label]
        else:
            # Try case-insensitive match
            matched = False
            for src, dst in CLASS_MAP.items():
                if raw_label.lower() == src.lower():
                    our_class = dst
                    matched = True
                    break
            if not matched:
                skipped += 1
                continue

        # Get the image
        image = item.get("image")
        if image is None:
            skipped += 1
            continue

        # Generate filename
        ext = ".jpg"
        fname = f"rice_{i:06d}{ext}"
        save_path = dest_dir / our_class / fname

        if save_image(image, save_path):
            counts[our_class] += 1
            saved += 1
        else:
            skipped += 1

    print(f"\nRice dataset saved:")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count}")
    print(f"  Total saved: {saved}, Skipped: {skipped}")

    # Deduplicate
    print("\nDeduplicating rice classes...")
    for cls in ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]:
        cls_dir = dest_dir / cls
        if cls_dir.exists():
            unique, dupes = deduplicate_folder(cls_dir)
            print(f"  {cls}: {unique} unique, {dupes} duplicates removed")

    return True


# ============================================================
# WHEAT DATASET - Multiple sources
# ============================================================
def download_wheat_dataset():
    """
    Download wheat disease data from multiple HuggingFace sources.
    Sources:
    1. geraldmc/plantvillage-full (has Wheat___ classes)
    2. prithivMLmods/Rice-Leaf-Disease (for cross-crop augmentation)
    """
    print("\n" + "=" * 70)
    print("DOWNLOADING WHEAT DATASET")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed")
        return False

    dest_dir = RAW_DIR / "wheat"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    total_existing = sum(
        len(list((dest_dir / c).glob("*.*")))
        for c in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]
        if (dest_dir / c).exists()
    )
    if total_existing > 2000:
        print(f"Wheat dataset already has {total_existing} images. Skipping.")
        return True

    counts = defaultdict(int)

    # ---- Source 1: PlantVillage for wheat diseases ----
    print("\n--- Source 1: PlantVillage (wheat classes) ---")
    try:
        ds = load_dataset("geraldmc/plantvillage-full", trust_remote_code=True)
        print(f"PlantVillage loaded")

        # Check columns
        train_split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        print(f"Columns: {train_split.column_names}")
        print(f"Total: {len(train_split)}")

        # Find wheat classes
        wheat_pv_map = {
            "Wheat___Brown_rust": "Leaf Rust",
            "Wheat___Yellow_rust": "Leaf Rust",
            "Wheat___Healthy": "Healthy",
            "Wheat___Powdery_mildew": "Powdery Mildew",
            "Wheat___Septoria": "Septoria",
            "Wheat___Black_rust": "Leaf Rust",
        }

        # Find label column
        label_col = None
        for col in ["class_label", "label", "class", "category"]:
            if col in train_split.column_names:
                label_col = col
                break

        if label_col:
            # Count wheat images
            wheat_labels = [l for l in wheat_pv_map.keys()]
            wheat_count = sum(1 for item in train_split if item[label_col] in wheat_labels)
            print(f"Wheat images found in PlantVillage: {wheat_count}")

            saved = 0
            for i, item in enumerate(train_split):
                label = item[label_col]
                if label not in wheat_pv_map:
                    continue

                our_class = wheat_pv_map[label]
                image = item.get("image")
                if image is None:
                    continue

                fname = f"pv_wheat_{i:06d}.jpg"
                save_path = dest_dir / our_class / fname

                if save_image(image, save_path):
                    counts[our_class] += 1
                    saved += 1

            print(f"  Saved {saved} wheat images from PlantVillage")
        else:
            print(f"  Could not find label column. Available: {train_split.column_names}")

    except Exception as e:
        print(f"  Error with PlantVillage: {e}")

    # ---- Source 2: musfiqurtuhin/BCDD (Bangladeshi Crops Disease Dataset) ----
    print("\n--- Source 2: BCDD (Bangladeshi Crops Disease Dataset) ---")
    try:
        ds = load_dataset("musfiqurtuhin/BCDD", trust_remote_code=True)
        print(f"BCDD loaded")
        print(f"Splits: {list(ds.keys())}")

        # Get train split
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        bcdd_data = ds[split_name]
        print(f"Columns: {bcdd_data.column_names}")
        print(f"Total: {len(bcdd_data)}")

        # BCDD has wheat diseases
        bcdd_wheat_map = {
            "Wheat_Healthy": "Healthy",
            "Wheat_Rust": "Leaf Rust",
            "Wheat_Leaf_Rust": "Leaf Rust",
            "Wheat_Brown_Rust": "Leaf Rust",
            "Wheat_Yellow_Rust": "Leaf Rust",
            "Wheat_Powdery_Mildew": "Powdery Mildew",
            "Wheat_Smut": "Loose Smut",
            "Wheat_Loose_Smut": "Loose Smut",
            "Wheat_Black_Smut": "Loose Smut",
        }

        # Find label column
        label_col = None
        for col in ["label", "class", "category", "disease", "crop_disease"]:
            if col in bcdd_data.column_names:
                label_col = col
                break

        if label_col:
            print(f"Label distribution:")
            dist = defaultdict(int)
            for item in bcdd_data:
                dist[item[label_col]] += 1
            for k, v in sorted(dist.items(), key=lambda x: -x[1])[:20]:
                print(f"  {k}: {v}")

            saved = 0
            for i, item in enumerate(bcdd_data):
                label = item[label_col]

                # Try exact match first
                our_class = None
                if label in bcdd_wheat_map:
                    our_class = bcdd_wheat_map[label]
                else:
                    # Try fuzzy match
                    label_lower = label.lower()
                    if "wheat" in label_lower:
                        if "healthy" in label_lower:
                            our_class = "Healthy"
                        elif any(x in label_lower for x in ["rust", "leaf"]):
                            our_class = "Leaf Rust"
                        elif "mildew" in label_lower:
                            our_class = "Powdery Mildew"
                        elif "smut" in label_lower:
                            our_class = "Loose Smut"
                        elif "rot" in label_lower:
                            our_class = "Crown & Root Rot"

                if our_class is None:
                    continue

                image = item.get("image")
                if image is None:
                    continue

                fname = f"bcdd_wheat_{i:06d}.jpg"
                save_path = dest_dir / our_class / fname

                if save_image(image, save_path):
                    counts[our_class] += 1
                    saved += 1

            print(f"  Saved {saved} wheat images from BCDD")
        else:
            print(f"  Could not find label column. Available: {bcdd_data.column_names}")

    except Exception as e:
        print(f"  Error with BCDD: {e}")

    # ---- Source 3: Saon110/bd-crop-vegetable-plant-disease-dataset ----
    print("\n--- Source 3: BD Crop Vegetable Disease Dataset ---")
    try:
        ds = load_dataset("Saon110/bd-crop-vegetable-plant-disease-dataset", trust_remote_code=True)
        print(f"BD Crop loaded")
        print(f"Splits: {list(ds.keys())}")

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        bd_data = ds[split_name]
        print(f"Columns: {bd_data.column_names}")

        # Find wheat-related entries
        label_col = None
        for col in ["label", "class", "category", "disease", "class_label", "crop_disease"]:
            if col in bd_data.column_names:
                label_col = col
                break

        if label_col:
            # Find wheat entries
            wheat_entries = []
            for item in bd_data:
                label = str(item[label_col]).lower()
                if "wheat" in label or "gandom" in label:
                    wheat_entries.append(item)

            print(f"  Wheat entries found: {len(wheat_entries)}")

            saved = 0
            for i, item in enumerate(wheat_entries):
                label = str(item[label_col]).lower()
                if "healthy" in label:
                    our_class = "Healthy"
                elif "rust" in label:
                    our_class = "Leaf Rust"
                elif "smut" in label:
                    our_class = "Loose Smut"
                elif "rot" in label or "blight" in label:
                    our_class = "Crown & Root Rot"
                else:
                    continue

                image = item.get("image")
                if image is None:
                    continue

                fname = f"bd_wheat_{i:06d}.jpg"
                save_path = dest_dir / our_class / fname
                if save_image(image, save_path):
                    counts[our_class] += 1
                    saved += 1

            print(f"  Saved {saved} wheat images from BD Crop")
        else:
            print(f"  Could not find label column. Available: {bd_data.column_names}")

    except Exception as e:
        print(f"  Error with BD Crop: {e}")

    # ---- Source 4: Voxel51/PlantWild (in-the-wild plant disease) ----
    print("\n--- Source 4: PlantWild (in-the-wild wheat diseases) ---")
    try:
        ds = load_dataset("uqtwei2/PlantWild", trust_remote_code=True)
        print(f"PlantWild loaded")
        print(f"Splits: {list(ds.keys())}")

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        pw_data = ds[split_name]
        print(f"Columns: {pw_data.column_names}")

        label_col = None
        for col in ["label", "class", "ground_truth", "disease", "class_label"]:
            if col in pw_data.column_names:
                label_col = col
                break

        if label_col:
            # Find wheat entries
            wheat_entries = []
            for item in pw_data:
                label = str(item[label_col]).lower()
                if "wheat" in label:
                    wheat_entries.append(item)

            print(f"  Wheat entries found: {len(wheat_entries)}")

            saved = 0
            for i, item in enumerate(wheat_entries):
                label = str(item[label_col]).lower()
                if "healthy" in label:
                    our_class = "Healthy"
                elif "rust" in label:
                    our_class = "Leaf Rust"
                elif "smut" in label:
                    our_class = "Loose Smut"
                elif "rot" in label or "blight" in label:
                    our_class = "Crown & Root Rot"
                else:
                    continue

                image = item.get("image")
                if image is None:
                    continue

                fname = f"pw_wheat_{i:06d}.jpg"
                save_path = dest_dir / our_class / fname
                if save_image(image, save_path):
                    counts[our_class] += 1
                    saved += 1

            print(f"  Saved {saved} wheat images from PlantWild")

    except Exception as e:
        print(f"  Error with PlantWild: {e}")

    # ---- Source 5: Additional rice datasets for wheat augmentation ----
    print("\n--- Source 5: Project-AgML Rice Disease (Bangladesh) ---")
    try:
        ds = load_dataset("Project-AgML/rice_disease_classification_bangladesh", trust_remote_code=True)
        print(f"Project-AgML loaded")

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        agml_data = ds[split_name]
        print(f"Columns: {agml_data.column_names}")

        # This dataset has rice diseases that look similar to wheat diseases
        # Use for cross-crop augmentation of disease patterns
        agml_map = {
            "Rice Blast": "Blast",
            "Leaf Scald": "Bacterial Blight",
            "Brown Spot": "Brown Spot",
            "Rice Tungro": "Tungro",
            "Healthy": "Healthy",
        }

        label_col = None
        for col in ["label", "class", "category", "disease"]:
            if col in agml_data.column_names:
                label_col = col
                break

        if label_col:
            # Save to a separate rice augmentation directory
            aug_dir = RAW_DIR / "rice_augmented"
            aug_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for i, item in enumerate(agml_data):
                label = item[label_col]
                if label not in agml_map:
                    continue

                our_class = agml_map[label]
                image = item.get("image")
                if image is None:
                    continue

                fname = f"agml_{i:06d}.jpg"
                save_path = aug_dir / our_class / fname
                if save_image(image, save_path):
                    saved += 1

            print(f"  Saved {saved} augmented rice images")

    except Exception as e:
        print(f"  Error with Project-AgML: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("WHEAT DATASET SUMMARY")
    print("=" * 70)
    for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        cls_dir = dest_dir / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.*")))
            print(f"  {cls}: {count}")
        else:
            print(f"  {cls}: 0")

    # Deduplicate
    print("\nDeduplicating wheat classes...")
    for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        cls_dir = dest_dir / cls
        if cls_dir.exists():
            unique, dupes = deduplicate_folder(cls_dir)
            print(f"  {cls}: {unique} unique, {dupes} duplicates removed")

    return True


# ============================================================
# MERGE AUGMENTED DATA
# ============================================================
def merge_augmented_rice():
    """Merge augmented rice data into main rice directory."""
    aug_dir = RAW_DIR / "rice_augmented"
    main_dir = RAW_DIR / "rice"

    if not aug_dir.exists():
        return

    print("\nMerging augmented rice data...")
    for cls in ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]:
        src = aug_dir / cls
        dst = main_dir / cls
        if not src.exists():
            continue

        dst.mkdir(parents=True, exist_ok=True)
        existing = len(list(dst.glob("*.*")))
        count = 0
        for img in src.glob("*.*"):
            new_name = f"aug_{cls.lower().replace(' ', '_')}_{count:06d}{img.suffix}"
            dst_path = dst / new_name
            if not dst_path.exists():
                shutil.copy2(str(img), str(dst_path))
                count += 1

        print(f"  {cls}: added {count} augmented images (total now: {existing + count})")

    # Cleanup
    shutil.rmtree(str(aug_dir), ignore_errors=True)


# ============================================================
# CREATE SPLITS
# ============================================================
def create_splits(dataset_name, class_names, train_pct=0.7, val_pct=0.15, test_pct=0.15, seed=42):
    """Create train/val/test splits for a dataset."""
    print(f"\nCreating splits for {dataset_name}...")

    src_dir = RAW_DIR / dataset_name
    split_base = DATA_DIR / "split" / dataset_name

    random.seed(seed)

    for class_name in class_names:
        class_dir = src_dir / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found")
            continue

        # Get all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            images.extend(class_dir.glob(ext))

        # Remove duplicates by filename
        seen = set()
        unique_images = []
        for img in images:
            if img.name not in seen:
                seen.add(img.name)
                unique_images.append(img)

        random.shuffle(unique_images)

        n = len(unique_images)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        splits = {
            'train': unique_images[:n_train],
            'val': unique_images[n_train:n_train + n_val],
            'test': unique_images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest = split_base / split_name / dataset_name / class_name
            dest.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                dest_path = dest / img.name
                if not dest_path.exists():
                    shutil.copy2(str(img), str(dest_path))

        print(f"  {class_name}: {n} total -> train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")


def create_crop_splits():
    """Create crop type classifier splits (rice vs wheat)."""
    print("\nCreating crop type splits...")

    split_base = DATA_DIR / "split" / "crop"
    rice_dir = RAW_DIR / "rice"
    wheat_dir = RAW_DIR / "wheat"

    random.seed(42)

    for crop_type, src_dir in [("rice", rice_dir), ("wheat", wheat_dir)]:
        all_images = []
        for class_dir in src_dir.iterdir():
            if class_dir.is_dir():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                    all_images.extend(class_dir.glob(ext))

        random.shuffle(all_images)
        n = len(all_images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        splits = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train + n_val],
            'test': all_images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest = split_base / split_name / "crop" / crop_type
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                dest_path = dest / img.name
                if not dest_path.exists():
                    shutil.copy2(str(img), str(dest_path))

        print(f"  {crop_type}: {n} total -> train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")


def create_stage_splits(dataset_name, class_names):
    """Create growth stage splits from disease labels."""
    print(f"\nCreating stage splits for {dataset_name}...")

    split_base = DATA_DIR / "split" / f"{dataset_name}_stage"
    src_dir = RAW_DIR / dataset_name

    # Map disease classes to growth stages
    if dataset_name == "rice":
        stage_map = {
            "Bacterial Blight": "early",
            "Blast": "mid",
            "Brown Spot": "mid",
            "Tungro": "late",
            "Healthy": "early",
        }
    else:  # wheat
        stage_map = {
            "Leaf Rust": "mid",
            "Loose Smut": "late",
            "Crown & Root Rot": "late",
            "Healthy": "early",
        }

    random.seed(42)

    for stage in ["early", "mid", "late"]:
        all_images = []
        for class_name, stage_label in stage_map.items():
            if stage_label == stage:
                class_dir = src_dir / class_name
                if class_dir.exists():
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                        all_images.extend(class_dir.glob(ext))

        random.shuffle(all_images)
        n = len(all_images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        splits = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train + n_val],
            'test': all_images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest = split_base / split_name / f"{dataset_name}_stage" / stage
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                dest_path = dest / img.name
                if not dest_path.exists():
                    shutil.copy2(str(img), str(dest_path))

        print(f"  {stage}: {n} total -> train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("CropGuard AI - Dataset Download & Preparation")
    print("=" * 70)

    # Step 1: Download rice dataset
    download_rice_dataset()

    # Step 2: Download wheat dataset
    download_wheat_dataset()

    # Step 3: Merge augmented data
    merge_augmented_rice()

    # Step 4: Create splits
    rice_classes = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
    wheat_classes = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]

    create_splits("rice", rice_classes)
    create_splits("wheat", wheat_classes)
    # DEPRECATED: create_crop_splits() — crop is determined by user input, not ML.
    create_stage_splits("rice", rice_classes)
    create_stage_splits("wheat", wheat_classes)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    for dataset, classes in [("rice", rice_classes), ("wheat", wheat_classes)]:
        print(f"\n{dataset.upper()}:")
        for cls in classes:
            split_dir = DATA_DIR / "split" / dataset
            train_count = len(list((split_dir / "train" / dataset / cls).glob("*.*"))) if (split_dir / "train" / dataset / cls).exists() else 0
            val_count = len(list((split_dir / "val" / dataset / cls).glob("*.*"))) if (split_dir / "val" / dataset / cls).exists() else 0
            test_count = len(list((split_dir / "test" / dataset / cls).glob("*.*"))) if (split_dir / "test" / dataset / cls).exists() else 0
            print(f"  {cls}: train={train_count} val={val_count} test={test_count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
