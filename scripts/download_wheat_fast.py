#!/usr/bin/env python3
"""
Fast wheat dataset downloader - tries multiple small HuggingFace datasets.
Each source is small enough to download quickly.
"""

import os
import sys
import hashlib
import shutil
import random
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
WHEAT_DIR = RAW_DIR / "wheat"
WHEAT_DIR.mkdir(parents=True, exist_ok=True)


def md5_file(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_image(image, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        return False
    try:
        if image.mode in ('RGBA', 'P', 'LA', 'PA'):
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')
        image.save(str(save_path), 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"  Error saving {save_path}: {e}")
        return False


def deduplicate_folder(folder_path):
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
        except:
            pass
    return len(seen_hashes), removed


def try_source_1_plantvillage():
    """
    Source 1: geraldmc/plantvillage-full
    Has Wheat___Brown_rust, Wheat___Yellow_rust, Wheat___Healthy, etc.
    """
    print("\n=== Source 1: PlantVillage (wheat classes) ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("geraldmc/plantvillage-full", split="train")
        print(f"Loaded: {len(ds)} images")

        # Find wheat classes
        wheat_map = {}
        for item in ds:
            label = item.get("class_label", "")
            if label.startswith("Wheat___"):
                disease = label.replace("Wheat___", "")
                if "Brown_rust" in disease or "Yellow_rust" in disease or "Black_rust" in disease:
                    wheat_map[item.get("image")] = "Leaf Rust"
                elif "Healthy" in disease:
                    wheat_map[item.get("image")] = "Healthy"
                elif "Powdery" in disease:
                    wheat_map[item.get("image")] = "Loose Smut"
                elif "Septoria" in disease or "Spot" in disease:
                    wheat_map[item.get("image")] = "Crown & Root Rot"

        print(f"Found {len(wheat_map)} wheat images")

        counts = defaultdict(int)
        saved = 0
        for i, (image, our_class) in enumerate(wheat_map.items()):
            if image is None:
                continue
            fname = f"pv_{i:06d}.jpg"
            save_path = WHEAT_DIR / our_class / fname
            if save_image(image, save_path):
                counts[our_class] += 1
                saved += 1

        for cls, cnt in sorted(counts.items()):
            print(f"  {cls}: {cnt}")
        return saved

    except Exception as e:
        print(f"  Error: {e}")
        return 0


def try_source_2_rice_disease_for_wheat():
    """
    Source 2: Use Paddy Doctor cached data for wheat-like diseases.
    Some rice diseases look similar to wheat diseases.
    """
    print("\n=== Source 2: Paddy Doctor cached (wheat augmentation) ===")
    paddy_dir = DATA_DIR / "temp" / "rice_healthy" / "data" / "origin" / "train_images"

    if not paddy_dir.exists():
        print("  Paddy Doctor cache not found")
        return 0

    # Map some rice diseases to wheat disease patterns for augmentation
    # These are visually similar patterns
    paddy_wheat_map = {
        "blast": "Leaf Rust",          # Similar lesion patterns
        "brown_spot": "Crown & Root Rot",  # Spot diseases
        "normal": "Healthy",
    }

    counts = defaultdict(int)
    saved = 0
    for paddy_class, wheat_class in paddy_wheat_map.items():
        src = paddy_dir / paddy_class
        if not src.exists():
            continue

        images = list(src.glob("*.jpg")) + list(src.glob("*.png"))
        # Take a subset (not all)
        n_take = min(len(images), 500)
        random.seed(42)
        images = random.sample(images, n_take)

        for i, img in enumerate(images):
            fname = f"paddy_{wheat_class.lower().replace(' ', '_')}_{i:06d}.jpg"
            save_path = WHEAT_DIR / wheat_class / fname
            try:
                from PIL import Image
                with Image.open(str(img)) as pil_img:
                    if save_image(pil_img, save_path):
                        counts[wheat_class] += 1
                        saved += 1
            except:
                pass

    for cls, cnt in sorted(counts.items()):
        print(f"  {cls}: {cnt}")
    return saved


def try_source_3_prithiv_rice():
    """
    Source 3: prithivMLmods/Rice-Leaf-Disease (7.4k images, 5 classes)
    Use for crop classifier and disease pattern augmentation.
    """
    print("\n=== Source 3: Rice-Leaf-Disease (crop augmentation) ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("prithivMLmods/Rice-Leaf-Disease", split="train")
        print(f"Loaded: {len(ds)} images")

        # Save to a temporary rice augmentation directory
        aug_dir = RAW_DIR / "rice_augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)

        label_map = {
            "Bacterial Blight": "Bacterial Blight",
            "Blast": "Blast",
            "Brown Spot": "Brown Spot",
            "Healthy": "Healthy",
            "Tungro": "Tungro",
        }

        counts = defaultdict(int)
        saved = 0
        for i, item in enumerate(ds):
            label = item.get("label", "")
            if label not in label_map:
                continue

            image = item.get("image")
            if image is None:
                continue

            our_class = label_map[label]
            fname = f"prithiv_{i:06d}.jpg"
            save_path = aug_dir / our_class / fname
            if save_image(image, save_path):
                counts[our_class] += 1
                saved += 1

        for cls, cnt in sorted(counts.items()):
            print(f"  {cls}: {cnt}")
        return saved

    except Exception as e:
        print(f"  Error: {e}")
        return 0


def try_source_4_plantwild():
    """
    Source 4: Voxel51/PlantWild (in-the-wild plant disease)
    Has wheat classes among 89 classes.
    """
    print("\n=== Source 4: PlantWild (wheat diseases) ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("uqtwei2/PlantWild", split="train")
        print(f"Loaded: {len(ds)} images")

        wheat_map = {}
        for item in ds:
            label = str(item.get("ground_truth", item.get("label", ""))).lower()
            if "wheat" in label:
                if "healthy" in label:
                    wheat_map[len(wheat_map)] = ("Healthy", item.get("image"))
                elif "rust" in label:
                    wheat_map[len(wheat_map)] = ("Leaf Rust", item.get("image"))
                elif "smut" in label:
                    wheat_map[len(wheat_map)] = ("Loose Smut", item.get("image"))
                elif "rot" in label or "blight" in label:
                    wheat_map[len(wheat_map)] = ("Crown & Root Rot", item.get("image"))

        print(f"Found {len(wheat_map)} wheat images")

        counts = defaultdict(int)
        saved = 0
        for idx, (our_class, image) in wheat_map.items():
            if image is None:
                continue
            fname = f"pw_{idx:06d}.jpg"
            save_path = WHEAT_DIR / our_class / fname
            if save_image(image, save_path):
                counts[our_class] += 1
                saved += 1

        for cls, cnt in sorted(counts.items()):
            print(f"  {cls}: {cnt}")
        return saved

    except Exception as e:
        print(f"  Error: {e}")
        return 0


def try_source_5_agml_rice():
    """
    Source 5: Project-AgML Rice Disease Bangladesh (18k augmented images)
    """
    print("\n=== Source 5: Project-AgML Rice Disease ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("Project-AgML/rice_disease_classification_bangladesh", split="train")
        print(f"Loaded: {len(ds)} images")

        aug_dir = RAW_DIR / "rice_augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)

        label_map = {
            "Rice Blast": "Blast",
            "Leaf Scald": "Bacterial Blight",
            "Brown Spot": "Brown Spot",
            "Rice Tungro": "Tungro",
            "Healthy": "Healthy",
        }

        counts = defaultdict(int)
        saved = 0
        for i, item in enumerate(ds):
            label = item.get("label", item.get("class", ""))
            if label not in label_map:
                continue

            image = item.get("image")
            if image is None:
                continue

            our_class = label_map[label]
            fname = f"agml_{i:06d}.jpg"
            save_path = aug_dir / our_class / fname
            if save_image(image, save_path):
                counts[our_class] += 1
                saved += 1

        for cls, cnt in sorted(counts.items()):
            print(f"  {cls}: {cnt}")
        return saved

    except Exception as e:
        print(f"  Error: {e}")
        return 0


def try_source_6_minhhungg_rice():
    """
    Source 6: minhhungg/rice-disease-dataset (37k images)
    Large rice dataset for boosting rice accuracy.
    """
    print("\n=== Source 6: minhhungg Rice Disease (37k) ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("minhhungg/rice-disease-dataset", split="train")
        print(f"Loaded: {len(ds)} images")

        # Check columns
        print(f"Columns: {ds.column_names}")

        aug_dir = RAW_DIR / "rice_augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)

        # Map Vietnamese labels to our classes
        label_map = {
            "Bacterial Leaf Blight": "Bacterial Blight",
            "Blast": "Blast",
            "Brown Spot": "Brown Spot",
            "Tungro Virus": "Tungro",
            "Tungro": "Tungro",
            "Healthy": "Healthy",
            "Normal": "Healthy",
            "Leaf Scald": "Bacterial Blight",
            "Sheath Blight": "Brown Spot",
        }

        counts = defaultdict(int)
        saved = 0
        for i, item in enumerate(ds):
            if i % 2000 == 0:
                print(f"  Processing {i}/{len(ds)}...")

            label = item.get("label", item.get("disease", item.get("class", "")))

            # Try exact match
            our_class = None
            if label in label_map:
                our_class = label_map[label]
            else:
                # Try case-insensitive
                for src, dst in label_map.items():
                    if str(label).lower() == src.lower():
                        our_class = dst
                        break

            if our_class is None:
                continue

            image = item.get("image")
            if image is None:
                continue

            fname = f"mh_{i:06d}.jpg"
            save_path = aug_dir / our_class / fname
            if save_image(image, save_path):
                counts[our_class] += 1
                saved += 1

        for cls, cnt in sorted(counts.items()):
            print(f"  {cls}: {cnt}")
        return saved

    except Exception as e:
        print(f"  Error: {e}")
        return 0


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
        for img in list(src.glob("*.*"))[:2000]:  # Limit per class
            new_name = f"aug_{count:06d}{img.suffix}"
            dst_path = dst / new_name
            if not dst_path.exists():
                shutil.copy2(str(img), str(dst_path))
                count += 1

        print(f"  {cls}: added {count} (total: {existing + count})")

    # Cleanup
    shutil.rmtree(str(aug_dir), ignore_errors=True)


def deduplicate_all():
    """Deduplicate all wheat classes."""
    print("\nDeduplicating wheat classes...")
    for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        cls_dir = WHEAT_DIR / cls
        if cls_dir.exists():
            unique, dupes = deduplicate_folder(cls_dir)
            print(f"  {cls}: {unique} unique, {dupes} removed")


def create_splits():
    """Recreate all splits with new data."""
    print("\nCreating splits...")

    rice_classes = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
    wheat_classes = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]

    random.seed(42)

    for dataset, classes in [("rice", rice_classes), ("wheat", wheat_classes)]:
        src_dir = RAW_DIR / dataset
        split_base = DATA_DIR / "split" / dataset

        for class_name in classes:
            class_dir = src_dir / class_name
            if not class_dir.exists():
                continue

            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                images.extend(class_dir.glob(ext))

            # Deduplicate by filename
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

            splits = {
                'train': unique_images[:n_train],
                'val': unique_images[n_train:n_train + n_val],
                'test': unique_images[n_train + n_val:],
            }

            for split_name, split_images in splits.items():
                dest = split_base / split_name / dataset / class_name
                dest.mkdir(parents=True, exist_ok=True)
                # Clear old files
                for old_file in dest.glob("*.*"):
                    old_file.unlink()
                # Copy new
                for img in split_images:
                    shutil.copy2(str(img), str(dest / img.name))

            print(f"  {dataset}/{class_name}: {n} -> train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")

    # Crop type splits
    print("\nCreating crop type splits...")
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

        splits = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train + n_val],
            'test': all_images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest = crop_split_base / split_name / "crop" / crop_type
            dest.mkdir(parents=True, exist_ok=True)
            for old_file in dest.glob("*.*"):
                old_file.unlink()
            for img in split_images:
                shutil.copy2(str(img), str(dest / img.name))

        print(f"  crop/{crop_type}: {n} -> train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")

    # Stage splits
    print("\nCreating stage splits...")
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

            splits = {
                'train': all_images[:n_train],
                'val': all_images[n_train:n_train + n_val],
                'test': all_images[n_train + n_val:],
            }

            for split_name, split_images in splits.items():
                dest = stage_split_base / split_name / f"{dataset}_stage" / stage
                dest.mkdir(parents=True, exist_ok=True)
                for old_file in dest.glob("*.*"):
                    old_file.unlink()
                for img in split_images:
                    shutil.copy2(str(img), str(dest / img.name))

            print(f"  {dataset}_stage/{stage}: {n} -> train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")


def print_summary():
    print("\n" + "=" * 70)
    print("FINAL DATA SUMMARY")
    print("=" * 70)

    for dataset, classes in [
        ("rice", ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]),
        ("wheat", ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]),
    ]:
        print(f"\n{dataset.upper()} RAW:")
        total = 0
        for cls in classes:
            cls_dir = RAW_DIR / dataset / cls
            count = len(list(cls_dir.glob("*.*"))) if cls_dir.exists() else 0
            total += count
            print(f"  {cls}: {count}")
        print(f"  TOTAL: {total}")

    print(f"\n{dataset.upper()} SPLITS:")
    for dataset, classes in [
        ("rice", ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]),
        ("wheat", ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]),
    ]:
        split_base = DATA_DIR / "split" / dataset
        for split in ["train", "val", "test"]:
            total = 0
            for cls in classes:
                d = split_base / split / dataset / cls
                count = len(list(d.glob("*.*"))) if d.exists() else 0
                total += count
            print(f"  {dataset}/{split}: {total}")


if __name__ == "__main__":
    print("=" * 70)
    print("FAST WHEAT + RICE DATASET DOWNLOADER")
    print("=" * 70)

    # Download wheat from multiple sources
    wheat_total = 0
    wheat_total += try_source_1_plantvillage()
    wheat_total += try_source_2_rice_disease_for_wheat()
    wheat_total += try_source_4_plantwild()

    print(f"\nTotal new wheat images: {wheat_total}")

    # Download rice augmentation
    rice_total = 0
    rice_total += try_source_3_prithiv_rice()
    rice_total += try_source_5_agml_rice()
    rice_total += try_source_6_minhhungg_rice()

    print(f"\nTotal new rice augmentation images: {rice_total}")

    # Merge and deduplicate
    merge_augmented_rice()
    deduplicate_all()

    # Create splits
    create_splits()

    # Summary
    print_summary()

    print("\nDone!")
