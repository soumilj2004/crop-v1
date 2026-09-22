#!/usr/bin/env python3
"""
Download wheat leaf disease dataset from GitHub.
Source: aadium/wheat-disease-detection
Classes: Wheat Leaf Rust, Wheat Loose Smut, Wheat Crown & Root Rot, Wheat Healthy
"""

import os
import subprocess
import hashlib
from pathlib import Path
from tqdm import tqdm

WHEAT_REPO_URL = "https://github.com/aadium/wheat-disease-detection.git"
WHEAT_CLASSES = ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]
WHEAT_SOURCE_FOLDERS = ["Wheat Leaf Rust", "Wheat Loose Smut", "Wheat Crown & Root Rot", "Wheat Healthy"]

DATA_DIR = Path("data/raw/wheat")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def md5_file(path):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clone_repo_sparse(repo_url, target_dir, sparse_paths):
    """Clone a repo with sparse checkout to only get specific folders."""
    target_dir = Path(target_dir)
    if target_dir.exists():
        print(f"Directory {target_dir} already exists, skipping clone")
        return
    
    print(f"Cloning {repo_url} (sparse) to {target_dir}...")
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(target_dir)], check=True)
    repo_path = target_dir
    subprocess.run(["git", "-C", str(repo_path), "sparse-checkout", "init", "--cone"], check=True)
    subprocess.run(["git", "-C", str(repo_path), "sparse-checkout", "set"] + sparse_paths, check=True)
    subprocess.run(["git", "-C", str(repo_path), "checkout"], check=True)

def copy_images_to_class_folders(src_root, class_mapping, dest_root):
    """Copy images from source structure to flat class folders."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    
    for src_class, dest_class in class_mapping.items():
        src_dir = Path(src_root) / src_class
        if not src_dir.exists():
            print(f"Warning: {src_dir} does not exist")
            continue
        dest_dir = dest_root / dest_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for img_file in src_dir.glob("*.jpg"):
            dest_file = dest_dir / img_file.name
            if not dest_file.exists():
                import shutil
                shutil.copy2(img_file, dest_file)
                count += 1
            else:
                base = img_file.stem
                ext = img_file.suffix
                i = 1
                while (dest_dir / f"{base}_{i}{ext}").exists():
                    i += 1
                import shutil
                shutil.copy2(img_file, dest_dir / f"{base}_{i}{ext}")
                count += 1
        
        for img_file in src_dir.glob("*.jpeg"):
            dest_file = dest_dir / img_file.name
            if not dest_file.exists():
                import shutil
                shutil.copy2(img_file, dest_file)
                count += 1
            else:
                base = img_file.stem
                ext = img_file.suffix
                i = 1
                while (dest_dir / f"{base}_{i}{ext}").exists():
                    i += 1
                import shutil
                shutil.copy2(img_file, dest_dir / f"{base}_{i}{ext}")
                count += 1
        
        for img_file in src_dir.glob("*.png"):
            dest_file = dest_dir / img_file.name
            if not dest_file.exists():
                import shutil
                shutil.copy2(img_file, dest_file)
                count += 1
            else:
                base = img_file.stem
                ext = img_file.suffix
                i = 1
                while (dest_dir / f"{base}_{i}{ext}").exists():
                    i += 1
                import shutil
                shutil.copy2(img_file, dest_dir / f"{base}_{i}{ext}")
                count += 1
        
        print(f"  {dest_class}: copied {count} images")

def deduplicate_folder(folder_path):
    """Remove exact duplicate images (by MD5) from a folder."""
    folder_path = Path(folder_path)
    seen_hashes = {}
    duplicates = 0
    
    for img_file in folder_path.glob("*"):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            file_hash = md5_file(img_file)
            if file_hash in seen_hashes:
                img_file.unlink()
                duplicates += 1
            else:
                seen_hashes[file_hash] = img_file.name
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    return len(seen_hashes), duplicates

def download_wheat_dataset():
    """Download wheat disease images from GitHub."""
    print("=" * 60)
    print("Downloading Wheat Disease Dataset")
    print("=" * 60)
    
    temp_dir = Path("data/temp/wheat")
    clone_repo_sparse(WHEAT_REPO_URL, temp_dir, ["cropDiseaseDataset"])
    
    dataset_dir = temp_dir / "cropDiseaseDataset"
    if not dataset_dir.exists():
        # Try alternative paths
        for sub in temp_dir.glob("**/cropDiseaseDataset"):
            dataset_dir = sub
            break
    
    if not dataset_dir.exists():
        print(f"Could not find cropDiseaseDataset folder in {temp_dir}")
        return
    
    # Map source folders to our class names
    class_mapping = {}
    for src, dst in zip(WHEAT_SOURCE_FOLDERS, WHEAT_CLASSES):
        src_path = dataset_dir / src
        if src_path.exists():
            class_mapping[src] = dst
        else:
            print(f"Warning: {src_path} not found")
    
    copy_images_to_class_folders(dataset_dir, class_mapping, DATA_DIR)
    
    # Deduplicate each class
    print("\nDeduplicating wheat disease images...")
    total_unique = 0
    total_dupes = 0
    for class_name in WHEAT_CLASSES:
        class_dir = DATA_DIR / class_name
        if class_dir.exists():
            unique, dupes = deduplicate_folder(class_dir)
            total_unique += unique
            total_dupes += dupes
            print(f"  {class_name}: {unique} unique, {dupes} duplicates removed")
    
    print(f"\nWheat total: {total_unique} unique images, {total_dupes} duplicates removed")

def main():
    download_wheat_dataset()
    
    print("\n" + "=" * 60)
    print("WHEAT DATASET SUMMARY")
    print("=" * 60)
    for class_name in WHEAT_CLASSES:
        class_dir = DATA_DIR / class_name
        if class_dir.exists():
            count = len([f for f in class_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            print(f"  {class_name}: {count} images")
        else:
            print(f"  {class_name}: 0 images (MISSING)")

if __name__ == "__main__":
    main()