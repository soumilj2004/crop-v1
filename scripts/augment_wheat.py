#!/usr/bin/env python3
"""
Generate augmented wheat data from existing 50 images per class.
Uses aggressive augmentation to create 500+ images per class.
Also downloads from HuggingFace with streaming (faster).
"""

import os
import sys
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance, ImageTransform
import io

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
WHEAT_DIR = RAW_DIR / "wheat"


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
        return False


def augment_image(img, augmentation_type, seed):
    """Apply a specific augmentation to an image."""
    random.seed(seed)

    if augmentation_type == 'rotate_15':
        return img.rotate(15, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    elif augmentation_type == 'rotate_neg15':
        return img.rotate(-15, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    elif augmentation_type == 'rotate_30':
        return img.rotate(30, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    elif augmentation_type == 'rotate_neg30':
        return img.rotate(-30, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    elif augmentation_type == 'flip_h':
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif augmentation_type == 'flip_v':
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif augmentation_type == 'brightness_up':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.3)
    elif augmentation_type == 'brightness_down':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.7)
    elif augmentation_type == 'contrast_up':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.4)
    elif augmentation_type == 'contrast_down':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.6)
    elif augmentation_type == 'saturation_up':
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(1.5)
    elif augmentation_type == 'saturation_down':
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(0.5)
    elif augmentation_type == 'sharpness_up':
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(2.0)
    elif augmentation_type == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))
    elif augmentation_type == 'crop_center':
        w, h = img.size
        new_w, new_h = int(w * 0.8), int(h * 0.8)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BICUBIC)
    elif augmentation_type == 'crop_top_left':
        w, h = img.size
        new_w, new_h = int(w * 0.8), int(h * 0.8)
        cropped = img.crop((0, 0, new_w, new_h))
        return cropped.resize((w, h), Image.BICUBIC)
    elif augmentation_type == 'crop_bottom_right':
        w, h = img.size
        new_w, new_h = int(w * 0.8), int(h * 0.8)
        cropped = img.crop((w - new_w, h - new_h, w, h))
        return cropped.resize((w, h), Image.BICUBIC)
    elif augmentation_type == 'shear_h':
        w, h = img.size
        shear_factor = 0.2
        return img.transform((w, h), Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), fillcolor=(128, 128, 128))
    elif augmentation_type == 'shear_v':
        w, h = img.size
        shear_factor = 0.2
        return img.transform((w, h), Image.AFFINE, (1, 0, 0, shear_factor, 1, 0), fillcolor=(128, 128, 128))
    elif augmentation_type == 'posterize':
        from PIL import ImageOps
        return ImageOps.posterize(img, 3)
    elif augmentation_type == 'solarize':
        from PIL import ImageOps
        return ImageOps.solarize(img, 128)
    elif augmentation_type == 'invert':
        from PIL import ImageOps
        return ImageOps.invert(img)
    elif augmentation_type == 'scale_up':
        w, h = img.size
        new_w, new_h = int(w * 1.2), int(h * 1.2)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    elif augmentation_type == 'noise':
        import numpy as np
        arr = np.array(img)
        noise = np.random.normal(0, 25, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    elif augmentation_type == 'color_shift':
        import numpy as np
        arr = np.array(img).astype(np.float32)
        # Shift color channels
        shift = random.uniform(-30, 30)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + shift, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] - shift / 2, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] + shift / 3, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    elif augmentation_type == 'gamma':
        import numpy as np
        gamma = random.uniform(0.7, 1.5)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        return Image.fromarray((arr * 255).astype(np.uint8))
    else:
        return img


AUGMENTATION_TYPES = [
    'rotate_15', 'rotate_neg15', 'rotate_30', 'rotate_neg30',
    'flip_h', 'flip_v',
    'brightness_up', 'brightness_down',
    'contrast_up', 'contrast_down',
    'saturation_up', 'saturation_down',
    'sharpness_up',
    'blur',
    'crop_center', 'crop_top_left', 'crop_bottom_right',
    'shear_h', 'shear_v',
    'posterize', 'solarize',
    'scale_up', 'noise', 'color_shift', 'gamma',
]


def augment_class(class_dir, target_count=600):
    """Augment a single class to reach target_count."""
    class_dir = Path(class_dir)
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        images.extend(class_dir.glob(ext))

    current_count = len(images)
    if current_count >= target_count:
        return current_count

    needed = target_count - current_count
    print(f"  Augmenting {class_dir.name}: {current_count} -> {target_count} (need {needed} more)")

    augmented = 0
    for i in range(needed):
        # Pick a random source image
        src_img_path = random.choice(images)
        # Pick a random augmentation
        aug_type = random.choice(AUGMENTATION_TYPES)
        seed = random.randint(0, 100000)

        try:
            with Image.open(str(src_img_path)) as src_img:
                aug_img = augment_image(src_img, aug_type, seed)
                fname = f"aug_{augmented:06d}_{aug_type}.jpg"
                save_path = class_dir / fname
                if save_image(aug_img, save_path):
                    augmented += 1
        except Exception as e:
            pass

    final_count = len(list(class_dir.glob("*.*")))
    print(f"    Added {augmented} augmented images. Total: {final_count}")
    return final_count


def augment_all_wheat():
    """Augment all wheat classes to 600+ images each."""
    print("=" * 70)
    print("AUGMENTING WHEAT DATASET")
    print("=" * 70)

    wheat_classes = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]
    target_per_class = 600

    for cls in wheat_classes:
        cls_dir = WHEAT_DIR / cls
        if cls_dir.exists():
            augment_class(cls_dir, target_per_class)
        else:
            print(f"  Warning: {cls_dir} not found")


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


def deduplicate_all():
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
                for old_file in dest.glob("*.*"):
                    old_file.unlink()
                for img in split_images:
                    import shutil
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
                import shutil
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
                    import shutil
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
    augment_all_wheat()
    deduplicate_all()
    create_splits()
    print_summary()
    print("\nDone!")
