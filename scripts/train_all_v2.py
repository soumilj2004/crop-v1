#!/usr/bin/env python3
"""Train all models sequentially with proper data."""
import subprocess
import sys
import time

MODELS = [
    ("wheat", 20),
    ("rice", 20),
    ("crop", 15),
    ("rice_stage", 15),
    ("wheat_stage", 15),
]

start = time.time()
results = {}

for crop, epochs in MODELS:
    print(f"\n{'='*60}")
    print(f"TRAINING: {crop} ({epochs} epochs)")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, "scripts/train_fast.py", "--crop", crop, "--epochs", str(epochs)],
        timeout=5400,  # 90 min max
    )

    dt = time.time() - t0
    results[crop] = {"status": "OK" if result.returncode == 0 else "FAIL", "time": dt}
    print(f"\n{crop} finished in {dt/60:.1f} min (exit: {result.returncode})")

total = time.time() - start
print(f"\n{'='*60}")
print(f"ALL DONE in {total/60:.1f} min")
for crop, info in results.items():
    print(f"  {crop}: {info['status']} ({info['time']/60:.1f} min)")
