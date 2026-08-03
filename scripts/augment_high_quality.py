#!/usr/bin/env python3
"""
High-quality augmentation to boost wheat data from 50 to 1500+ per class.
Also downloads from multiple web sources.
"""

import os
import sys
import hashlib
import shutil
import random
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
WHEAT_DIR = RAW_DIR / "wheat"

random.seed(42)
np.random.seed(42)

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_image(image, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        return False
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(str(save_path), 'JPEG', quality=95)
        return True
    except:
        return False


# ============================================================
# HIGH-QUALITY AUGMENTATION
# ============================================================
def augment_image(img, aug_type, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    w, h = img.size
    
    if aug_type == 'rotate_10':
        return img.rotate(10, resample=Image.BICUBIC, fillcolor=(128,128,128))
    elif aug_type == 'rotate_neg10':
        return img.rotate(-10, resample=Image.BICUBIC, fillcolor=(128,128,128))
    elif aug_type == 'rotate_20':
        return img.rotate(20, resample=Image.BICUBIC, fillcolor=(128,128,128))
    elif aug_type == 'rotate_neg20':
        return img.rotate(-20, resample=Image.BICUBIC, fillcolor=(128,128,128))
    elif aug_type == 'flip_h':
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == 'flip_v':
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif aug_type == 'flip_both':
        return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    elif aug_type == 'brightness_up':
        return ImageEnhance.Brightness(img).enhance(1.2)
    elif aug_type == 'brightness_down':
        return ImageEnhance.Brightness(img).enhance(0.8)
    elif aug_type == 'contrast_up':
        return ImageEnhance.Contrast(img).enhance(1.3)
    elif aug_type == 'contrast_down':
        return ImageEnhance.Contrast(img).enhance(0.7)
    elif aug_type == 'saturation_up':
        return ImageEnhance.Color(img).enhance(1.3)
    elif aug_type == 'saturation_down':
        return ImageEnhance.Color(img).enhance(0.7)
    elif aug_type == 'sharpness_up':
        return ImageEnhance.Sharpness(img).enhance(1.5)
    elif aug_type == 'sharpness_down':
        return ImageEnhance.Sharpness(img).enhance(0.5)
    elif aug_type == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif aug_type == 'crop_center':
        new_w, new_h = int(w*0.85), int(h*0.85)
        left = (w-new_w)//2
        top = (h-new_h)//2
        return img.crop((left, top, left+new_w, top+new_h)).resize((w,h), Image.BICUBIC)
    elif aug_type == 'crop_top_left':
        new_w, new_h = int(w*0.85), int(h*0.85)
        return img.crop((0, 0, new_w, new_h)).resize((w,h), Image.BICUBIC)
    elif aug_type == 'crop_bottom_right':
        new_w, new_h = int(w*0.85), int(h*0.85)
        return img.crop((w-new_w, h-new_h, w, h)).resize((w,h), Image.BICUBIC)
    elif aug_type == 'shear_h':
        return img.transform((w,h), Image.AFFINE, (1, 0.15, 0, 0, 1, 0), fillcolor=(128,128,128))
    elif aug_type == 'shear_v':
        return img.transform((w,h), Image.AFFINE, (1, 0, 0, 0.15, 1, 0), fillcolor=(128,128,128))
    elif aug_type == 'posterize':
        return ImageOps.posterize(img, 4)
    elif aug_type == 'scale_up':
        new_w, new_h = int(w*1.15), int(h*1.15)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w-w)//2
        top = (new_h-h)//2
        return resized.crop((left, top, left+w, top+h))
    elif aug_type == 'scale_down':
        new_w, new_h = int(w*0.85), int(h*0.85)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        bg = Image.new('RGB', (w,h), (128,128,128))
        left = (w-new_w)//2
        top = (h-new_h)//2
        bg.paste(resized, (left, top))
        return bg
    elif aug_type == 'noise':
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    elif aug_type == 'color_shift':
        arr = np.array(img).astype(np.float32)
        shift = random.uniform(-20, 20)
        arr[:,:,0] = np.clip(arr[:,:,0] + shift, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] - shift/2, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] + shift/3, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    elif aug_type == 'gamma':
        gamma = random.uniform(0.8, 1.3)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        return Image.fromarray((arr * 255).astype(np.uint8))
    elif aug_type == 'elastic':
        # Simple elastic-like transform using affine
        angle = random.uniform(-5, 5)
        return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(128,128,128))
    elif aug_type == 'compose_flip_rotate':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(15, resample=Image.BICUBIC, fillcolor=(128,128,128))
        return img
    elif aug_type == 'compose_flip_brightness':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = ImageEnhance.Brightness(img).enhance(1.15)
        return img
    elif aug_type == 'compose_rotate_contrast':
        img = img.rotate(-15, resample=Image.BICUBIC, fillcolor=(128,128,128))
        img = ImageEnhance.Contrast(img).enhance(1.2)
        return img
    elif aug_type == 'compose_crop_flip':
        new_w, new_h = int(w*0.9), int(h*0.9)
        left = (w-new_w)//2
        top = (h-new_h)//2
        img = img.crop((left, top, left+new_w, top+new_h)).resize((w,h), Image.BICUBIC)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    elif aug_type == 'compose_color_noise':
        arr = np.array(img).astype(np.float32)
        arr = arr * random.uniform(0.9, 1.1)
        arr = arr + np.random.normal(0, 10, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    else:
        return img


AUGMENTATION_TYPES = [
    'rotate_10', 'rotate_neg10', 'rotate_20', 'rotate_neg20',
    'flip_h', 'flip_v', 'flip_both',
    'brightness_up', 'brightness_down',
    'contrast_up', 'contrast_down',
    'saturation_up', 'saturation_down',
    'sharpness_up', 'sharpness_down',
    'blur',
    'crop_center', 'crop_top_left', 'crop_bottom_right',
    'shear_h', 'shear_v',
    'posterize',
    'scale_up', 'scale_down',
    'noise', 'color_shift', 'gamma', 'elastic',
    'compose_flip_rotate', 'compose_flip_brightness',
    'compose_rotate_contrast', 'compose_crop_flip', 'compose_color_noise',
]


def augment_class(class_dir, target_count=1500):
    """Augment a class to target_count images."""
    class_dir = Path(class_dir)
    images = [f for f in class_dir.glob("*.*") 
              if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')]
    
    current = len(images)
    if current >= target_count:
        print(f"  {class_dir.name}: already {current} images")
        return current
    
    needed = target_count - current
    print(f"  {class_dir.name}: {current} -> {target_count} (need {needed} more)")
    
    augmented = 0
    for i in range(needed):
        src = random.choice(images)
        aug = random.choice(AUGMENTATION_TYPES)
        seed = random.randint(0, 1000000)
        
        try:
            with Image.open(str(src)) as img:
                aug_img = augment_image(img, aug, seed)
                fname = f"aug_{augmented:06d}_{aug}.jpg"
                if save_image(aug_img, class_dir / fname):
                    augmented += 1
        except:
            pass
    
    final = len(list(class_dir.glob("*.*")))
    print(f"    Added {augmented}. Total: {final}")
    return final


# ============================================================
# DEDUPLICATE
# ============================================================
def deduplicate_folder(folder_path):
    folder_path = Path(folder_path)
    seen = {}
    removed = 0
    for f in list(folder_path.glob("*.*")):
        if f.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.webp'):
            continue
        try:
            h = md5_file(f)
            if h in seen:
                f.unlink()
                removed += 1
            else:
                seen[h] = f
        except:
            pass
    return len(seen), removed


# ============================================================
# CREATE STAGE DATA
# ============================================================
def create_stage_data():
    print("\nCreating wheat_stage data...")
    stage_dir = RAW_DIR / "wheat_stage"
    for s in ["early", "mid", "late"]:
        (stage_dir / s).mkdir(parents=True, exist_ok=True)
    
    mapping = {
        "Healthy": "early",
        "Leaf Rust": "mid",
        "Loose Smut": "late",
        "Crown & Root Rot": "late",
    }
    
    counts = defaultdict(int)
    for disease, stage in mapping.items():
        src = RAW_DIR / "wheat" / disease
        if not src.exists():
            continue
        for img in src.glob("*.*"):
            if img.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'):
                fname = f"{stage}_{disease.lower().replace(' ','_')}_{counts[stage]:06d}{img.suffix}"
                dst = stage_dir / stage / fname
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))
                    counts[stage] += 1
    
    for s, n in sorted(counts.items()):
        print(f"  {s}: {n}")
    return sum(counts.values())


# ============================================================
# CREATE SPLITS
# ============================================================
def create_splits():
    print("\nCreating splits...")
    random.seed(42)
    
    # Wheat disease
    wheat_classes = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]
    print("\n--- Wheat Disease ---")
    for cls in wheat_classes:
        d = WHEAT_DIR / cls
        if not d.exists():
            continue
        imgs = [f for f in d.glob("*.*") if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits = {'train': imgs[:n_train], 'val': imgs[n_train:n_train+n_val], 'test': imgs[n_train+n_val:]}
        for sn, si in splits.items():
            dest = DATA_DIR / "split" / "wheat" / sn / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in dest.glob("*.*"): f.unlink()
            for img in si:
                shutil.copy2(str(img), str(dest / img.name))
        print(f"  {cls}: {n} -> train:{n_train} val:{n_val} test:{n-n_train-n_val}")
    
    # Wheat stage
    print("\n--- Wheat Stage ---")
    for stage in ["early", "mid", "late"]:
        d = RAW_DIR / "wheat_stage" / stage
        if not d.exists():
            continue
        imgs = [f for f in d.glob("*.*") if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits = {'train': imgs[:n_train], 'val': imgs[n_train:n_train+n_val], 'test': imgs[n_train+n_val:]}
        for sn, si in splits.items():
            dest = DATA_DIR / "split" / "wheat_stage" / sn / stage
            dest.mkdir(parents=True, exist_ok=True)
            for f in dest.glob("*.*"): f.unlink()
            for img in si:
                shutil.copy2(str(img), str(dest / img.name))
        print(f"  {stage}: {n} -> train:{n_train} val:{n_val} test:{n-n_train-n_val}")
    
    # Rice (keep existing)
    print("\n--- Rice ---")
    rice_classes = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
    for cls in rice_classes:
        d = RAW_DIR / "rice" / cls
        if not d.exists():
            continue
        imgs = [f for f in d.glob("*.*") if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits = {'train': imgs[:n_train], 'val': imgs[n_train:n_train+n_val], 'test': imgs[n_train+n_val:]}
        for sn, si in splits.items():
            dest = DATA_DIR / "split" / "rice" / sn / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in dest.glob("*.*"): f.unlink()
            for img in si:
                shutil.copy2(str(img), str(dest / img.name))
        print(f"  {cls}: {n} -> train:{n_train} val:{n_val} test:{n-n_train-n_val}")
    
    # Rice stage
    print("\n--- Rice Stage ---")
    rice_stage_map = {"Bacterial Blight":"early","Blast":"mid","Brown Spot":"mid","Tungro":"late","Healthy":"early"}
    for stage in ["early", "mid", "late"]:
        all_imgs = []
        for cls, stg in rice_stage_map.items():
            if stg == stage:
                d = RAW_DIR / "rice" / cls
                if d.exists():
                    all_imgs.extend([f for f in d.glob("*.*") if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')])
        random.shuffle(all_imgs)
        n = len(all_imgs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits = {'train': all_imgs[:n_train], 'val': all_imgs[n_train:n_train+n_val], 'test': all_imgs[n_train+n_val:]}
        for sn, si in splits.items():
            dest = DATA_DIR / "split" / "rice_stage" / sn / stage
            dest.mkdir(parents=True, exist_ok=True)
            for f in dest.glob("*.*"): f.unlink()
            for img in si:
                shutil.copy2(str(img), str(dest / img.name))
        print(f"  {stage}: {n} -> train:{n_train} val:{n_val} test:{n-n_train-n_val}")
    
    # Crop type
    print("\n--- Crop Type ---")
    for crop_type in ["rice", "wheat"]:
        all_imgs = []
        for cls_dir in (RAW_DIR / crop_type).iterdir():
            if cls_dir.is_dir():
                all_imgs.extend([f for f in cls_dir.glob("*.*") if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')])
        random.shuffle(all_imgs)
        n = len(all_imgs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits = {'train': all_imgs[:n_train], 'val': all_imgs[n_train:n_train+n_val], 'test': all_imgs[n_train+n_val:]}
        for sn, si in splits.items():
            dest = DATA_DIR / "split" / "crop" / sn / crop_type
            dest.mkdir(parents=True, exist_ok=True)
            for f in dest.glob("*.*"): f.unlink()
            for img in si:
                shutil.copy2(str(img), str(dest / img.name))
        print(f"  {crop_type}: {n} -> train:{n_train} val:{n_val} test:{n-n_train-n_val}")


def print_summary():
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for split in ["train", "val", "test"]:
        print(f"\n  {split}:")
        for dataset, classes in [
            ("wheat", ["Leaf Rust","Loose Smut","Crown & Root Rot","Healthy"]),
            ("rice", ["Bacterial Blight","Blast","Brown Spot","Healthy","Tungro"]),
            ("wheat_stage", ["early","mid","late"]),
            ("rice_stage", ["early","mid","late"]),
        ]:
            total = 0
            for cls in classes:
                d = DATA_DIR / "split" / dataset / split / cls
                total += len(list(d.glob("*.*"))) if d.exists() else 0
            print(f"    {dataset}: {total}")


if __name__ == "__main__":
    print("="*60)
    print("HIGH-QUALITY AUGMENTATION PIPELINE")
    print("="*60)
    
    # Augment each wheat class to 1500
    print("\n--- Augmenting Wheat Classes ---")
    for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        cls_dir = WHEAT_DIR / cls
        if cls_dir.exists():
            augment_class(cls_dir, target_count=1500)
    
    # Deduplicate
    print("\n--- Deduplicating ---")
    for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        d = WHEAT_DIR / cls
        if d.exists():
            u, r = deduplicate_folder(d)
            print(f"  {cls}: {u} unique, {r} removed")
    
    # Create stage data
    create_stage_data()
    
    # Create splits
    create_splits()
    
    # Summary
    print_summary()
    
    print("\nDone!")
