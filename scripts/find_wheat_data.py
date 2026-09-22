#!/usr/bin/env python3
"""
Download wheat disease images from multiple web sources.
Uses direct URL downloads instead of HuggingFace.
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

try:
    import requests
    from PIL import Image
    from io import BytesIO
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Need: pip install requests Pillow")


def md5_file(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_image(url, save_path, timeout=15):
    """Download image from URL and save."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        return False
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        img.save(str(save_path), 'JPEG', quality=95)
        return True
    except Exception as e:
        return False


def try_source_crop_disease_hf():
    """Try vishnun0027/Crop_Disease dataset - just iterate quickly."""
    print("\n=== Source: Crop_Disease (HuggingFace) ===")
    try:
        from datasets import load_dataset
        ds = load_dataset('vishnun0027/Crop_Disease', split='train')
        print(f"  Loaded {len(ds)} samples")

        # Get all labels
        labels = set()
        for i in range(min(100, len(ds))):
            labels.add(ds[i]['label'])
        print(f"  Labels found: {sorted(labels)}")

        # This dataset has integer labels, need to figure out mapping
        # Save all images with label as folder
        counts = defaultdict(int)
        saved = 0
        for i in range(len(ds)):
            item = ds[i]
            label = item['label']
            image = item.get('image')
            if image is None:
                continue

            # Map to our classes based on label number
            # We don't know the mapping yet, save to numbered folders
            dest_dir = RAW_DIR / "wheat_hf" / f"class_{label}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            fname = f"crop_{i:06d}.jpg"
            save_path = dest_dir / fname
            try:
                if image.mode in ('RGBA', 'P', 'LA'):
                    image = image.convert('RGB')
                image.save(str(save_path), 'JPEG', quality=95)
                counts[label] += 1
                saved += 1
            except:
                pass

            if saved >= 500:
                break

        for label, count in sorted(counts.items()):
            print(f"  Class {label}: {count} images")

        return saved
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def try_source_sudoping():
    """Try sudoping01/crop-disease-dataset."""
    print("\n=== Source: sudoping01/crop-disease-dataset ===")
    try:
        from datasets import load_dataset
        ds = load_dataset('sudoping01/crop-disease-dataset')
        print(f"  Splits: {list(ds.keys())}")

        for split in list(ds.keys())[:2]:
            data = ds[split]
            print(f"  {split}: {len(data)} samples, columns: {data.column_names}")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def try_source_rohitashva():
    """Try rohitashva/indian_crop_diseases."""
    print("\n=== Source: rohitashva/indian_crop_diseases ===")
    try:
        from datasets import load_dataset
        ds = load_dataset('rohitashva/indian_crop_diseases')
        print(f"  Splits: {list(ds.keys())}")

        for split in list(ds.keys())[:2]:
            data = ds[split]
            print(f"  {split}: {len(data)} samples, columns: {data.column_names}")

            # Check labels
            labels = set()
            for i in range(min(100, len(data))):
                for col in data.column_names:
                    if col != 'image':
                        val = str(data[i][col]).lower()
                        labels.add(f"{col}={val}")
            print(f"  Sample labels: {labels}")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def try_source_siddharth():
    """Try siddharth2525/crop-disease-dataset-24."""
    print("\n=== Source: siddharth2525/crop-disease-dataset-24 ===")
    try:
        from datasets import load_dataset
        ds = load_dataset('siddharth2525/crop-disease-dataset-24')
        print(f"  Splits: {list(ds.keys())}")

        for split in list(ds.keys())[:2]:
            data = ds[split]
            print(f"  {split}: {len(data)} samples, columns: {data.column_names}")

            labels = set()
            for i in range(min(100, len(data))):
                for col in data.column_names:
                    if col != 'image':
                        val = str(data[i][col]).lower()
                        labels.add(f"{col}={val}")
            print(f"  Sample labels: {labels}")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 0


if __name__ == "__main__":
    if not HAS_DEPS:
        print("Install deps first")
        sys.exit(1)

    print("=" * 60)
    print("SEARCHING FOR WHEAT DATA")
    print("=" * 60)

    try_source_crop_disease_hf()
    try_source_sudoping()
    try_source_rohitashva()
    try_source_siddharth()
