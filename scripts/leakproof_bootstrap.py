"""
LEAKPROOF DATA PIPELINE - Guarantees zero train/val/test leakage
- Global deduplication BEFORE splitting
- Hash manifests for full audit trail
- Septoria/Powdery Mildew label cleaning
- Strict 70/15/15 split from CLEAN pool
- Hash manifests saved for full audit trail
"""
import hashlib, json, os, random, shutil, sys, subprocess
from pathlib import Path

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))

# ======= CONFIG =======
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train/val/test
SEED = 42
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.jfif'}

# Label cleaning: map known bad labels to correct ones or discard
LABEL_CLEAN_MAP = {
    # Known mislabels from earlier data pulls
    "Septoria": None,           # DISCARD - not in our 4-class scheme
    "Powdery Mildew": None,     # DISCARD - not in our 4-class scheme
    "Brown Rust": "Leaf Rust",  # Map to existing
    "Yellow Rust": "Leaf Rust", # Map to existing
    "Stem Rust": "Leaf Rust",   # Map to existing
    "Black Rust": "Leaf Rust",  # Map to existing
    "Rust": "Leaf Rust",        # Generic -> specific
    "Crown Rot": "Crown & Root Rot",
    "Root Rot": "Crown & Root Rot",
    "Loose Smut": "Loose Smut",
    "Wheat Loose Smut": "Loose Smut",
    "Healthy Wheat": "Healthy",
    "Healthy": "Healthy",
    "Crown & Root Rot": "Crown & Root Rot",
    "Leaf Rust": "Leaf Rust",
    "Loose Smut": "Loose Smut",
}

# Our 4 target classes (exact folder names)
TARGET_CLASSES = ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"]

# ======= UTILITIES =======
def log(msg): print(f"[LEAKPROOF] {msg}", flush=True)

def file_hash(path):
    """MD5 hash of file contents"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def scan_images(root):
    """Return list of (path, hash, original_label) for all images under root"""
    results = []
    for cls_dir in Path(root).iterdir():
        if not cls_dir.is_dir():
            continue
        original_label = cls_dir.name
        for img in cls_dir.iterdir():
            if img.is_file() and img.suffix.lower() in VALID_EXTS:
                results.append((img, original_label))
    return results

def clean_label(original):
    """Apply label cleaning map"""
    return LABEL_CLEAN_MAP.get(original, original)

def verify_zero_overlap(splits_dict):
    """Verify zero hash overlap between any splits"""
    hashes = {split: set() for split in splits_dict}
    for split, items in splits_dict.items():
        for item in items:
            p = item[0] if isinstance(item, tuple) else item
            hashes[split].add(file_hash(p))
    
    splits_list = list(splits_dict.keys())
    for i in range(len(splits_list)):
        for j in range(i+1, len(splits_list)):
            a, b = splits_list[i], splits_list[j]
            overlap = hashes[a] & hashes[b]
            if overlap:
                log(f"[FAIL] LEAK DETECTED: {a} ∩ {b} = {len(overlap)} images")
                return False
    log("Zero overlap verified across " + str(len(splits_dict)) + " splits")
    return True

def save_manifest(split_name, items, out_dir):
    """Save hash manifest for audit"""
    manifest = []
    for item in items:
        p, label = item if isinstance(item, tuple) else (item, None)
        entry = {"path": str(p), "hash": file_hash(p)}
        if label is not None:
            entry["label"] = label
        manifest.append(entry)
    out_path = Path(out_dir) / f"manifest_{split_name}.json"
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log(f"Manifest saved: {out_path} ({len(manifest)} images)")

# ======= MAIN PIPELINE =======
def main():
    random.seed(SEED)
    
    log("=" * 60)
    log("LEAKPROOF DATA PIPELINE STARTING")
    log("=" * 60)
    
    # Paths
    raw_dir = REPO / "data" / "raw" / "wheat"
    lwdcd_dir = REPO / "data" / "lwdcd2020_norm"
    # NOTE: writing to *_fixed dirs, not overwriting the live data/split/wheat or
    # data/manifests in place -- review counts, then swap in deliberately.
    split_dir = REPO / "data" / "split" / "wheat_fixed"
    manifest_dir = REPO / "data" / "manifests_fixed"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- PHASE 1: COLLECT ALL RAW IMAGES ----
    log("Phase 1: Collecting all raw images...")
    all_images = []  # (src_path, cleaned_label)
    
    # Source A: Main raw wheat data
    if raw_dir.exists():
        for img_path, orig_label in scan_images(raw_dir):
            clean = clean_label(orig_label)
            if clean in TARGET_CLASSES:
                all_images.append((img_path, clean))
            else:
                log(f"  Discarded (bad label): {orig_label} -> {clean}")
        log(f"  From raw/wheat: {len([x for x in all_images if x[1] in TARGET_CLASSES])} valid images")
    else:
        log(f"  WARNING: {raw_dir} not found")
    
    # Source B: LWDCD2020 normalized (already clean class names)
    if lwdcd_dir.exists():
        lwdcd_map = {
            "Crown and Root Rot": "Crown & Root Rot",
            "Healthy Wheat": "Healthy",
            "Leaf Rust": "Leaf Rust",
            "Wheat Loose Smut": "Loose Smut",
        }
        for img_path, orig_label in scan_images(lwdcd_dir):
            clean = lwdcd_map.get(orig_label)
            if clean in TARGET_CLASSES:
                all_images.append((img_path, clean))
        log(f"  From lwdcd2020_norm: {len([x for x in all_images if 'lwdcd' in str(x[0])])} images")
    else:
        log(f"  WARNING: {lwdcd_dir} not found")
    
    log(f"Total collected: {len(all_images)} images")
    
    # ---- PHASE 2: GLOBAL DEDUPLICATION ----
    log("Phase 2: Global deduplication (hash-based)...")
    hash_to_best = {}  # hash -> (path, label)
    duplicates = 0
    for path, label in all_images:
        h = file_hash(path)
        if h in hash_to_best:
            duplicates += 1
            # Keep the first occurrence (arbitrary but deterministic)
        else:
            hash_to_best[h] = (path, label)
    
    unique_images = list(hash_to_best.values())
    log(f"  Before dedup: {len(all_images)}")
    log(f"  Duplicates removed: {duplicates}")
    log(f"  Unique images: {len(unique_images)}")
    
    # ---- PHASE 3: CLASS BALANCE CHECK ----
    log("Phase 3: Class distribution...")
    by_class = {c: [] for c in TARGET_CLASSES}
    for path, label in unique_images:
        by_class[label].append(path)
    
    for c in TARGET_CLASSES:
        log(f"  {c}: {len(by_class[c])} images")
    
    min_class = min(len(v) for v in by_class.values())
    if min_class < 50:
        log(f"[WARN] WARNING: Smallest class has only {min_class} images")
    
    # ---- PHASE 4: STRATIFIED SPLIT (LEAKPROOF) ----
    log("Phase 4: Stratified split (70/15/15)...")
    splits = {"train": [], "val": [], "test": []}
    
    for label, paths in by_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        splits["train"].extend((p, label) for p in paths[:n_train])
        splits["val"].extend((p, label) for p in paths[n_train:n_train + n_val])
        splits["test"].extend((p, label) for p in paths[n_train + n_val:])
    
    # Shuffle each split (deterministic)
    for split in splits:
        random.shuffle(splits[split])
    
    # ---- PHASE 5: ZERO LEAKAGE VERIFICATION ----
    log("Phase 5: Zero leakage verification...")
    if not verify_zero_overlap(splits):
        log("[FAIL] PIPELINE FAILED: Leakage detected!")
        sys.exit(1)
    
    # ---- PHASE 6: WRITE CLEAN SPLITS ----
    log("Phase 6: Writing clean splits...")
    # Writing to a fresh *_fixed directory -- never deletes the existing live split.
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in splits.items():
        by_label = {}
        for p, label in items:
            by_label.setdefault(label, []).append(p)
        
        for label, paths in by_label.items():
            dest = split_dir / split_name / label
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(paths):
                dst = dest / f"{label.replace(' ', '_').replace('&','and')}_{i:05d}{src.suffix}"
                shutil.copy2(src, dst)
    
    # ---- PHASE 7: SAVE HASH MANIFESTS ----
    log("Phase 7: Saving hash manifests...")
    for split_name, paths in splits.items():
        save_manifest(split_name, paths, manifest_dir)
    
    # Final summary
    log("=" * 60)
    log("LEAKPROOF PIPELINE COMPLETE")
    log("=" * 60)
    for split_name, items in splits.items():
        by_label = {}
        for p, label in items:
            by_label[label] = by_label.get(label, 0) + 1
        log(f"  {split_name}: {len(items)} total - " + ", ".join(f"{k}:{v}" for k,v in by_label.items()))
    log(f"Manifests saved to: {manifest_dir}")
    log("[OK] READY FOR TRAINING - Zero leakage guaranteed")
    return 0

if __name__ == "__main__":
    sys.exit(main())