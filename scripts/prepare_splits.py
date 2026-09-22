"""Create train/val/test splits for all models from raw data."""
import os, shutil, random, json
from pathlib import Path
from collections import defaultdict

BASE = Path("C:/CropGuardAI/cropguard_ai")
SPLIT_DIR = BASE / "data" / "split"
RAW_DIR = BASE / "data" / "raw"
STAGE_DIR = BASE / "data" / "stage_labels"
STAGE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

# Disease -> Stage mapping
WHEAT_STAGE = {
    "Leaf Rust": "mid",
    "Loose Smut": "late",
    "Crown & Root Rot": "late",
    "Healthy": "early",
}

RICE_STAGE = {
    "Bacterial Blight": "early",
    "Healthy": "early",
    "Blast": "mid",
    "Brown Spot": "mid",
    "Tungro": "late",
}

def create_split(src_dir, split_dir, ratios=(0.7, 0.15, 0.15)):
    """Split data into train/val/test."""
    split_dir.mkdir(parents=True, exist_ok=True)
    for cls_dir in src_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        files = [f for f in cls_dir.iterdir() if f.is_file()]
        random.shuffle(files)
        
        n = len(files)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:],
        }
        
        for split_name, split_files in splits.items():
            out = split_dir / split_name / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out / f.name)

def create_stage_labels(split_dir, stage_map, stage_name):
    """Create stage labels JSON from disease splits."""
    for split in ["train", "val", "test"]:
        split_path = split_dir / split
        labels = {}
        for cls_dir in split_path.iterdir():
            if not cls_dir.is_dir():
                continue
            disease = cls_dir.name
            if disease not in stage_map:
                continue
            stage = stage_map[disease]
            for f in cls_dir.iterdir():
                if f.is_file():
                    # Use relative path from split dir
                    rel = str(f.relative_to(split_dir))
                    labels[rel] = stage
        
        out_file = STAGE_DIR / f"{stage_name}_{split}_stage_labels.json"
        with open(out_file, "w") as fh:
            json.dump(labels, fh, indent=2)
        print(f"  {out_file.name}: {len(labels)} labels")

def merge_rice_sources():
    """Merge rice data from both raw sources."""
    rice_dir = RAW_DIR / "rice"
    rice_dir.mkdir(parents=True, exist_ok=True)
    
    # Source 1: our existing rice diseases
    src1 = BASE / "data" / "external" / "rice_diseases"
    if src1.exists():
        for cls_dir in src1.iterdir():
            if cls_dir.is_dir():
                target = rice_dir / cls_dir.name
                target.mkdir(parents=True, exist_ok=True)
                existing = set(f.name for f in target.glob("*"))
                count = 0
                for f in cls_dir.iterdir():
                    if f.is_file() and f.name not in existing:
                        shutil.copy2(f, target / f.name)
                        count += 1
                if count:
                    print(f"  Merged {count} from rice_diseases/{cls_dir.name}")
    
    # Source 2: RiceLeafDiseaseBD (Mendeley)
    src2 = BASE / "data" / "external" / "RiceLeafDiseaseBD"
    if src2.exists():
        for cls_dir in src2.iterdir():
            if cls_dir.is_dir():
                # Map to our class names
                name_map = {
                    "Healthy": "Healthy",
                    "Blast": "Blast",
                    "Brown Spot": "Brown Spot",
                    "Leaf Smut": "Tungro",  # closest match
                    "Rice Tungro": "Tungro",
                    "Sheath Blight": "Brown Spot",  # closest match
                }
                target_name = name_map.get(cls_dir.name, cls_dir.name)
                target = rice_dir / target_name
                target.mkdir(parents=True, exist_ok=True)
                existing = set(f.name for f in target.glob("*"))
                count = 0
                for f in cls_dir.iterdir():
                    if f.is_file() and f.name not in existing:
                        shutil.copy2(f, target / f.name)
                        count += 1
                if count:
                    print(f"  Merged {count} from RiceLeafDiseaseBD/{cls_dir.name} -> {target_name}")

def make_crop_split():
    """Create rice vs wheat binary split."""
    crop_dir = SPLIT_DIR / "crop"
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        out = crop_dir / split
        out.mkdir(parents=True, exist_ok=True)
        
        rice_src = SPLIT_DIR / "rice" / split
        wheat_src = SPLIT_DIR / "wheat" / split
        
        if rice_src.exists():
            rice_out = out / "rice"
            rice_out.mkdir(parents=True, exist_ok=True)
            for cls_dir in rice_src.iterdir():
                if cls_dir.is_dir():
                    dst = rice_out / cls_dir.name
                    if not dst.exists():
                        shutil.copytree(cls_dir, dst)
        
        if wheat_src.exists():
            wheat_out = out / "wheat"
            wheat_out.mkdir(parents=True, exist_ok=True)
            for cls_dir in wheat_src.iterdir():
                if cls_dir.is_dir():
                    dst = wheat_out / cls_dir.name
                    if not dst.exists():
                        shutil.copytree(cls_dir, dst)

def main():
    print("=== Step 1: Merge rice sources ===")
    merge_rice_sources()
    
    print("\n=== Step 2: Create wheat splits ===")
    wheat_raw = RAW_DIR / "wheat"
    wheat_split = SPLIT_DIR / "wheat"
    if wheat_raw.exists():
        create_split(wheat_raw, wheat_split)
        for s in ["train", "val", "test"]:
            n = sum(1 for _ in (wheat_split / s).rglob("*") if _.is_file())
            print(f"  wheat/{s}: {n}")
    
    print("\n=== Step 3: Create wheat_stage splits ===")
    ws_split = SPLIT_DIR / "wheat_stage"
    # Create stage directories from wheat disease data
    for stage in ["early", "mid", "late"]:
        (RAW_DIR / "wheat_stage" / stage).mkdir(parents=True, exist_ok=True)
    
    for cls, stage in WHEAT_STAGE.items():
        src = wheat_split / "train" / cls
        if src.exists():
            dst = RAW_DIR / "wheat_stage" / stage
            for f in src.iterdir():
                if f.is_file() and not (dst / f.name).exists():
                    shutil.copy2(f, dst / f.name)
    
    # Split wheat_stage
    ws_raw = RAW_DIR / "wheat_stage"
    create_split(ws_raw, ws_split)
    for s in ["train", "val", "test"]:
        n = sum(1 for _ in (ws_split / s).rglob("*") if _.is_file())
        print(f"  wheat_stage/{s}: {n}")
    
    print("\n=== Step 4: Create rice splits ===")
    rice_raw = RAW_DIR / "rice"
    rice_split = SPLIT_DIR / "rice"
    if rice_raw.exists():
        create_split(rice_raw, rice_split)
        for s in ["train", "val", "test"]:
            n = sum(1 for _ in (rice_split / s).rglob("*") if _.is_file())
            print(f"  rice/{s}: {n}")
    
    print("\n=== Step 5: Create rice_stage splits ===")
    rs_split = SPLIT_DIR / "rice_stage"
    for stage in ["early", "mid", "late"]:
        (RAW_DIR / "rice_stage" / stage).mkdir(parents=True, exist_ok=True)
    
    for cls, stage in RICE_STAGE.items():
        src = rice_split / "train" / cls
        if src.exists():
            dst = RAW_DIR / "rice_stage" / stage
            for f in src.iterdir():
                if f.is_file() and not (dst / f.name).exists():
                    shutil.copy2(f, dst / f.name)
    
    create_split(RAW_DIR / "rice_stage", rs_split)
    for s in ["train", "val", "test"]:
        n = sum(1 for _ in (rs_split / s).rglob("*") if _.is_file())
        print(f"  rice_stage/{s}: {n}")
    
    print("\n=== Step 6: Create crop split ===")
    make_crop_split()
    crop_dir = SPLIT_DIR / "crop"
    for s in ["train", "val", "test"]:
        n = sum(1 for _ in (crop_dir / s).rglob("*") if _.is_file()) if (crop_dir / s).exists() else 0
        print(f"  crop/{s}: {n}")
    
    print("\n=== Step 7: Create stage labels ===")
    create_stage_labels(wheat_split, WHEAT_STAGE, "wheat")
    create_stage_labels(rice_split, RICE_STAGE, "rice")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
