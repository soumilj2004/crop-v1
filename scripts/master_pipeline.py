#!/usr/bin/env python3
"""
Master download + split + train script.
Downloads real images from multiple sources, rebuilds splits, trains all models.
"""

import os
import sys
import hashlib
import shutil
import random
import json
import time
import math
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

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
# SOURCE 1: PlantVillage (correct config name)
# ============================================================
def download_plantvillage():
    print("\n" + "="*60)
    print("SOURCE 1: PlantVillage (mohanty/PlantVillage)")
    print("="*60)
    try:
        from datasets import load_dataset
        ds = load_dataset("mohanty/PlantVillage", split="train")
        print(f"Loaded: {len(ds)} images, columns: {ds.column_names}")

        wheat_dir = RAW_DIR / "wheat"
        for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
            (wheat_dir / cls).mkdir(parents=True, exist_ok=True)

        counts = defaultdict(int)
        saved = 0
        for i, item in enumerate(ds):
            label = str(item.get("label", "")).lower()
            image = item.get("image")
            if image is None:
                continue

            cls = None
            if "wheat" in label:
                if "rust" in label or "brown" in label or "yellow" in label or "black" in label:
                    cls = "Leaf Rust"
                elif "healthy" in label:
                    cls = "Healthy"
                elif "powdery" in label or "mildew" in label:
                    cls = "Powdery Mildew"
                elif "septoria" in label:
                    cls = "Septoria"
                elif "spot" in label or "blight" in label:
                    cls = "Crown & Root Rot"

            if cls:
                fname = f"pv_{i:06d}.jpg"
                if save_image(image, wheat_dir / cls / fname):
                    counts[cls] += 1
                    saved += 1

        for c, n in sorted(counts.items()):
            print(f"  {c}: {n}")
        print(f"Total wheat saved: {saved}")
        return saved
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
        return 0


# ============================================================
# SOURCE 2: musfiqurtuhin/BCDD (wheat leaf disease)
# ============================================================
def download_bcdd():
    print("\n" + "="*60)
    print("SOURCE 2: musfiqurtuhin/BCDD")
    print("="*60)
    try:
        from datasets import load_dataset
        ds = load_dataset("musfiqurtuhin/BCDD", split="train")
        print(f"Loaded: {len(ds)} images, columns: {ds.column_names}")

        wheat_dir = RAW_DIR / "wheat"
        counts = defaultdict(int)
        saved = 0
        for i, item in enumerate(ds):
            label = str(item.get("label", item.get("disease", ""))).lower()
            image = item.get("image")
            if image is None:
                continue

            cls = None
            if "wheat" in label or "leaf" in label:
                if "rust" in label:
                    cls = "Leaf Rust"
                elif "smut" in label:
                    cls = "Loose Smut"
                elif "blight" in label or "rot" in label or "spot" in label:
                    cls = "Crown & Root Rot"
                elif "healthy" in label or "normal" in label:
                    cls = "Healthy"

            if cls:
                fname = f"bcdd_{i:06d}.jpg"
                if save_image(image, wheat_dir / cls / fname):
                    counts[cls] += 1
                    saved += 1

        for c, n in sorted(counts.items()):
            print(f"  {c}: {n}")
        print(f"Total saved: {saved}")
        return saved
    except Exception as e:
        print(f"  Error: {e}")
        return 0


# ============================================================
# SOURCE 3: geraldmc/plantvillage-full
# ============================================================
def download_plantvillage_full():
    print("\n" + "="*60)
    print("SOURCE 3: geraldmc/plantvillage-full")
    print("="*60)
    try:
        from datasets import load_dataset
        ds = load_dataset("geraldmc/plantvillage-full", split="train")
        print(f"Loaded: {len(ds)} images, columns: {ds.column_names}")

        wheat_dir = RAW_DIR / "wheat"
        counts = defaultdict(int)
        saved = 0
        for i, item in enumerate(ds):
            label = str(item.get("class_label", "")).lower()
            image = item.get("image")
            if image is None:
                continue

            cls = None
            if "wheat" in label:
                if "rust" in label or "brown" in label or "yellow" in label:
                    cls = "Leaf Rust"
                elif "healthy" in label:
                    cls = "Healthy"
                elif "powdery" in label or "mildew" in label:
                    cls = "Loose Smut"
                elif "septoria" in label or "spot" in label:
                    cls = "Crown & Root Rot"

            if cls:
                fname = f"pvf_{i:06d}.jpg"
                if save_image(image, wheat_dir / cls / fname):
                    counts[cls] += 1
                    saved += 1

        for c, n in sorted(counts.items()):
            print(f"  {c}: {n}")
        print(f"Total saved: {saved}")
        return saved
    except Exception as e:
        print(f"  Error: {e}")
        return 0


# ============================================================
# SOURCE 4: ashu010/agridrone-data (wheat subset)
# ============================================================
def download_agridrone():
    print("\n" + "="*60)
    print("SOURCE 4: ashu010/agridrone-data (wheat)")
    print("="*60)
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id="ashu010/agridrone-data",
            repo_type="dataset",
            allow_patterns=["train/wheat_*/**", "val/wheat_*/**", "test/wheat_*/**"],
        )
        print(f"Downloaded to: {path}")

        wheat_dir = RAW_DIR / "wheat"
        class_map = {
            "wheat_leaf_rust": "Leaf Rust",
            "wheat_smut": "Loose Smut",
            "wheat_blast": "Crown & Root Rot",
            "wheat_healthy": "Healthy",
            "wheat_yellow_rust": "Leaf Rust",
            "wheat_aphid": "Crown & Root Rot",
            "fusarium_head_blight": "Crown & Root Rot",
            "leaf_blight": "Crown & Root Rot",
            "powdery_mildew": "Powdery Mildew",
            "septoria": "Septoria",
            "tan_spot": "Crown & Root Rot",
        }

        counts = defaultdict(int)
        saved = 0
        for split in ["train", "val", "test"]:
            split_dir = Path(path) / split
            if not split_dir.exists():
                continue
            for folder in split_dir.iterdir():
                if not folder.is_dir():
                    continue
                folder_name = folder.name.lower()
                cls = None
                for pattern, target in class_map.items():
                    if pattern in folder_name:
                        cls = target
                        break
                if cls is None:
                    continue

                for img in folder.glob("*.*"):
                    if img.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                        fname = f"adrone_{split}_{saved:06d}{img.suffix}"
                        if save_image(Image.open(str(img)) if not hasattr(img, 'save') else img, wheat_dir / cls / fname):
                            counts[cls] += 1
                            saved += 1

        for c, n in sorted(counts.items()):
            print(f"  {c}: {n}")
        print(f"Total saved: {saved}")
        return saved
    except Exception as e:
        print(f"  Error: {e}")
        return 0


# ============================================================
# SOURCE 5: Wheat growth stages from literature
# (Crown Root/Tillering=early, Mid Vegetative/Booting=mid, Heading/Anthesis/Milking=late)
# ============================================================
def create_wheat_stage_data():
    print("\n" + "="*60)
    print("Creating wheat_stage data from disease classes")
    print("="*60)

    stage_dir = RAW_DIR / "wheat_stage"
    for s in ["early", "mid", "late"]:
        (stage_dir / s).mkdir(parents=True, exist_ok=True)

    # Map based on when diseases appear in growth cycle:
    # Healthy = early (plant is young/healthy)
    # Leaf Rust = mid (appears mid-season)
    # Loose Smut = late (appears at heading)
    # Crown & Root Rot = late (appears late)
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
# DEDUPLICATE ALL
# ============================================================
def deduplicate_all():
    print("\n" + "="*60)
    print("Deduplicating all classes")
    print("="*60)
    for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        d = RAW_DIR / "wheat" / cls
        if d.exists():
            u, r = deduplicate_folder(d)
            print(f"  wheat/{cls}: {u} unique, {r} removed")
    for s in ["early", "mid", "late"]:
        d = RAW_DIR / "wheat_stage" / s
        if d.exists():
            u, r = deduplicate_folder(d)
            print(f"  wheat_stage/{s}: {u} unique, {r} removed")


# ============================================================
# CREATE SPLITS
# ============================================================
def create_splits():
    print("\n" + "="*60)
    print("Creating train/val/test splits")
    print("="*60)
    random.seed(42)

    # Wheat disease
    wheat_classes = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]
    print("\n--- Wheat Disease ---")
    for cls in wheat_classes:
        d = RAW_DIR / "wheat" / cls
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
    print("\n--- Rice (keeping existing) ---")
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
    print("MASTER DATA PIPELINE")
    print("="*60)

    # Clean old augmented data, keep only originals
    print("\nCleaning old augmented wheat data...")
    wheat_dir = RAW_DIR / "wheat"
    for cls in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        cls_dir = wheat_dir / cls
        if cls_dir.exists():
            for f in cls_dir.glob("aug_*.*"):
                f.unlink()
            for f in cls_dir.glob("bcdd_*.*"):
                f.unlink()
            count = len(list(cls_dir.glob("*.*")))
            print(f"  {cls}: {count} remaining after cleanup")

    # Download from all sources
    total = 0
    total += download_plantvillage()
    total += download_plantvillage_full()
    total += download_bcdd()

    print(f"\nTotal new wheat images: {total}")

    # Create stage data
    create_wheat_stage_data()

    # Deduplicate
    deduplicate_all()

    # Create splits
    create_splits()

    # Summary
    print_summary()

    print("\nData pipeline complete!")
