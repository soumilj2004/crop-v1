#!/usr/bin/env python3
"""Launch all training jobs in parallel using multiprocessing."""
import subprocess
import sys
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def train_model(args):
    """Train a single model."""
    crop, epochs = args
    print(f"[START] {crop} ({epochs} epochs)")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "scripts/train_fast.py", "--crop", crop, "--epochs", str(epochs)],
            timeout=5400,
            capture_output=True,
            text=True,
        )
        dt = time.time() - t0
        # Find accuracy from output
        acc = "N/A"
        for line in result.stdout.split("\n"):
            if "Test Accuracy" in line:
                acc = line.strip()
            elif "Best val" in line:
                acc = line.strip()
        return crop, "OK", dt, acc, result.stdout[-500:] if result.stdout else ""
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        return crop, "TIMEOUT", dt, "N/A", "Timed out"
    except Exception as e:
        dt = time.time() - t0
        return crop, "ERROR", dt, str(e), ""

if __name__ == "__main__":
    os.chdir(r"C:\CropGuardAI\cropguard_ai")

    jobs = [
        ("crop", 12),
        ("rice_stage", 12),
        ("wheat_stage", 12),
        ("rice", 10),
        ("wheat", 10),
    ]

    print("=" * 60)
    print(f"LAUNCHING {len(jobs)} PARALLEL TRAINING JOBS")
    print("=" * 60)

    start = time.time()

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(train_model, job): job for job in jobs}

        for future in as_completed(futures):
            crop, status, dt, acc, last_lines = future.result()
            print(f"\n[DONE] {crop}: {status} in {dt/60:.1f} min")
            print(f"  Result: {acc}")
            if last_lines:
                # Show last meaningful lines
                lines = [l for l in last_lines.split("\n") if l.strip()][-5:]
                for l in lines:
                    print(f"  {l}")

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL JOBS COMPLETED in {total/60:.1f} min")
    print(f"{'='*60}")
