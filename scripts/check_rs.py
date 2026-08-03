import json
from pathlib import Path
p = Path('C:/CropGuardAI/cropguard_ai/models/rice_stage_history.json')
if p.exists():
    h = json.load(open(p))
    epochs = len(h['val_acc'])
    print(f'epochs: {epochs}')
    for i in range(epochs):
        print(f'Epoch {i+1}: train={h["train_acc"][i]:.4f} val={h["val_acc"][i]:.4f}')