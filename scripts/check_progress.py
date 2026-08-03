import json
from pathlib import Path

BASE = Path("C:/CropGuardAI/cropguard_ai/models")
for f in BASE.glob("*_history.json"):
    h = json.load(open(f))
    epochs = len(h["val_acc"])
    best = max(h["val_acc"])
    name = f.name.replace("_history.json", "")
    print(f"{name}: {epochs} epochs, best={best*100:.1f}%")