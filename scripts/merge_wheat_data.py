"""Merge kushagra3204 wheat diseases into LWDCD2020 for richer wheat dataset."""
import os, shutil, random
from pathlib import Path
from collections import Counter

BASE = Path("C:/CropGuardAI/cropguard_ai")
RAW_WHEAT = BASE / "data" / "raw" / "wheat"
KAGGLE = BASE / "data" / "external" / "data" / "train"

# Map kushagra3204 classes to our 4 disease classes
MAPPING = {
    "Brown Rust": "Leaf Rust",
    "Yellow Rust": "Leaf Rust",
    "Smut": "Loose Smut",
    "Common Root Rot": "Crown & Root Rot",
    "Healthy": "Healthy",
    "Black Rust": "Leaf Rust",
    "Leaf Blight": "Crown & Root Rot",
    "Tan spot": "Crown & Root Rot",
    "Septoria": "Crown & Root Rot",
    "Fusarium Head Blight": "Loose Smut",
    "Mildew": "Leaf Rust",
    # Skip: Aphid, Mite, Stem fly, Blast (not wheat diseases or pests)
}

# Skip classes that don't map well
SKIP = {"Aphid", "Mite", "Stem fly", "Blast"}

def merge():
    print("=== Merging kushagra3204 into LWDCD2020 ===")
    
    # Count existing
    for d in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        src = RAW_WHEAT / d
        count = len(list(src.glob("*"))) if src.exists() else 0
        print(f"  Existing {d}: {count}")
    
    added = Counter()
    for cls, files in [(c, list((KAGGLE / c).glob("*"))) for c in MAPPING if (KAGGLE / c).exists()]:
        target = MAPPING[cls]
        target_dir = RAW_WHEAT / target
        target_dir.mkdir(parents=True, exist_ok=True)
        
        existing = set(f.name for f in target_dir.glob("*"))
        count = 0
        for f in files:
            if f.name not in existing:
                shutil.copy2(f, target_dir / f.name)
                count += 1
        added[target] += count
        print(f"  Added {count} from {cls} -> {target}")
    
    print("\n=== After merge ===")
    for d in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"]:
        src = RAW_WHEAT / d
        count = len(list(src.glob("*"))) if src.exists() else 0
        print(f"  {d}: {count}")
    
    total = sum(len(list((RAW_WHEAT / d).glob("*"))) for d in ["Leaf Rust", "Loose Smut", "Crown & Root Rot", "Healthy"])
    print(f"  TOTAL: {total}")

if __name__ == "__main__":
    merge()
