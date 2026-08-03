import os
import shutil
import random
from pathlib import Path

random.seed(42)

def create_splits(raw_dir, split_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/val/test splits from raw directory."""
    raw_path = Path(raw_dir)
    split_path = Path(split_dir)
    
    # Remove old splits
    if split_path.exists():
        shutil.rmtree(split_path)
    
    for class_name in sorted(os.listdir(raw_path)):
        class_dir = raw_path / class_name
        if not class_dir.is_dir():
            continue
        
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.jfif')]
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            dest = split_path / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dest / img.name)
        
        print(f"  {class_name}: {n} total -> train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

# Create wheat splits
print("Creating wheat splits...")
create_splits(
    "C:/CropGuardAI/cropguard_ai/data/raw/wheat",
    "C:/CropGuardAI/cropguard_ai/data/split/wheat"
)

# Create wheat_stage splits
print("\nCreating wheat_stage splits...")
create_splits(
    "C:/CropGuardAI/cropguard_ai/data/raw/wheat_stage",
    "C:/CropGuardAI/cropguard_ai/data/split/wheat_stage"
)

# Print totals
for split_type in ['train', 'val', 'test']:
    wheat_dir = Path(f"C:/CropGuardAI/cropguard_ai/data/split/wheat/{split_type}")
    if wheat_dir.exists():
        total = sum(len(list((wheat_dir / d).iterdir())) for d in os.listdir(wheat_dir) if (wheat_dir / d).is_dir())
        print(f"wheat {split_type}: {total}")
    
    stage_dir = Path(f"C:/CropGuardAI/cropguard_ai/data/split/wheat_stage/{split_type}")
    if stage_dir.exists():
        total = sum(len(list((stage_dir / d).iterdir())) for d in os.listdir(stage_dir) if (stage_dir / d).is_dir())
        print(f"wheat_stage {split_type}: {total}")
