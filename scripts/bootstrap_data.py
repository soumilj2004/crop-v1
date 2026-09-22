"""
Bootstrap the full CropGuardAI dataset pipeline.

Does the right thing automatically:
- If base splits exist -> just merge LWDCD
- If raw data exists -> create splits + merge
- If nothing -> download base wheat from Kaggle + create splits + merge

Usage:
  python scripts/bootstrap_data.py [--force] [--skip-download]
"""
import argparse, json, os, shutil, sys, subprocess
from pathlib import Path

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))

def log(msg): print(f"[BOOTSTRAP] {msg}", flush=True)

def check_base_splits():
    """Return True if base wheat splits exist with expected classes."""
    split_dir = REPO / "data" / "split" / "wheat"
    required = {"train", "val", "test"}
    if not split_dir.exists():
        return False
    for split in required:
        sd = split_dir / split
        if not sd.exists():
            return False
        classes = {d.name for d in sd.iterdir() if d.is_dir()}
        expected = {"Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"}
        if not expected.issubset(classes):
            return False
    return True

def check_raw_wheat():
    """Return True if raw wheat data exists with expected classes."""
    raw_dir = REPO / "data" / "raw" / "wheat"
    if not raw_dir.exists():
        return False
    classes = {d.name for d in raw_dir.iterdir() if d.is_dir()}
    expected = {"Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"}
    return expected.issubset(classes)

def download_base_wheat():
    """Download base wheat dataset from Kaggle (kushagra3204/wheat-diseases)."""
    log("Downloading base wheat dataset from Kaggle...")
    raw_dir = REPO / "data" / "raw" / "wheat"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Try kaggle CLI first
    try:
        result = subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "kushagra3204/wheat-diseases",
            "-p", str(raw_dir),
            "--unzip"
        ], capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            log("Kaggle download succeeded")
            return True
        else:
            log(f"Kaggle CLI failed: {result.stderr}")
    except FileNotFoundError:
        log("Kaggle CLI not found")
    except subprocess.TimeoutExpired:
        log("Kaggle download timed out")
    
    # Fallback: try direct download via Python requests
    log("Trying direct download...")
    try:
        import requests, zipfile, io
        url = "https://www.kaggle.com/api/v1/datasets/download/kushagra3204/wheat-diseases"
        # Note: Kaggle requires auth for API. Without credentials, this will fail.
        # We'll just inform the user.
        log("Kaggle API requires authentication. Please either:")
        log("  1. Install kaggle CLI and run: kaggle datasets download -d kushagra3204/wheat-diseases -p data/raw/wheat --unzip")
        log("  2. Or manually download from https://www.kaggle.com/datasets/kushagra3204/wheat-diseases")
        log("  3. Extract to data/raw/wheat/ preserving class folders")
        return False
    except Exception as e:
        log(f"Direct download failed: {e}")
        return False

def run_prepare_splits():
    """Run the prepare_splits.py script."""
    log("Running prepare_splits.py to create all splits...")
    script = REPO / "scripts" / "prepare_splits.py"
    if not script.exists():
        log(f"ERROR: {script} not found")
        return False
    result = subprocess.run([sys.executable, str(script)], cwd=REPO, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        log("prepare_splits.py completed")
        print(result.stdout)
        return True
    else:
        log(f"prepare_splits.py failed: {result.stderr}")
        return False

def run_merge_lwdcd():
    """Run merge_lwdcd.py to add LWDCD images to train."""
    log("Running merge_lwdcd.py to add LWDCD images...")
    script = REPO / "scripts" / "merge_lwdcd.py"
    if not script.exists():
        log(f"ERROR: {script} not found")
        return False
    result = subprocess.run([sys.executable, str(script)], cwd=REPO, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        log("merge_lwdcd.py completed")
        print(result.stdout)
        return True
    else:
        log(f"merge_lwdcd.py failed: {result.stderr}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Bootstrap CropGuardAI dataset")
    ap.add_argument("--force", action="store_true", help="Force re-create splits even if they exist")
    ap.add_argument("--skip-download", action="store_true", help="Skip download attempt, only use local data")
    args = ap.parse_args()

    log("=== CropGuardAI Dataset Bootstrap ===")

    # Step 1: Check if base splits already exist
    if not args.force and check_base_splits():
        log("Base wheat splits already exist. Skipping split creation.")
    else:
        log("Base splits missing or --force specified.")
        
        # Check for raw data
        if check_raw_wheat():
            log("Raw wheat data found. Creating splits...")
            if not run_prepare_splits():
                return 1
        else:
            log("Raw wheat data not found.")
            if not args.skip_download:
                if download_base_wheat():
                    if not run_prepare_splits():
                        return 1
                else:
                    log("ERROR: Could not obtain base wheat dataset.")
                    log("Please manually download from https://www.kaggle.com/datasets/kushagra3204/wheat-diseases")
                    log("Extract to data/raw/wheat/ preserving class folders:")
                    log("  data/raw/wheat/Leaf Rust/")
                    log("  data/raw/wheat/Crown & Root Rot/")
                    log("  data/raw/wheat/Healthy/")
                    log("  data/raw/wheat/Loose Smut/")
                    log("Then re-run this script.")
                    return 1
            else:
                log("ERROR: No base wheat data found and --skip-download specified.")
                return 1

    # Step 2: Merge LWDCD into train
    log("Merging LWDCD2020 into train split...")
    if not run_merge_lwdcd():
        return 1

    # Verify
    if check_base_splits():
        log("✅ Bootstrap complete! Dataset ready for training.")
        return 0
    else:
        log("❌ Bootstrap failed - splits still missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())