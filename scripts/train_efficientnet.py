"""
Train wheat/rice/crop classifiers using EfficientNet-B0.
3 diseases + Healthy per crop (4 classes each).
Two-phase: freeze backbone (6 epochs), then unfreeze (6 epochs).
Cosine annealing LR, label smoothing, class weighting.
"""
import os, sys, json, time, argparse, random
from pathlib import Path
from collections import Counter

import time, os
_log_path = "C:/CropGuardAI/training_script.log"
# Try to open; if locked, use a PID-suffixed path
try:
    _log_file = open(_log_path, "a", buffering=1)
except PermissionError:
    _log_path = f"C:/CropGuardAI/training_{os.getpid()}.log"
    _log_file = open(_log_path, "a", buffering=1)

# Redirect stdout/stderr to log file only when run directly (not at import time)
if __name__ == "__main__":
    sys.stdout = _log_file
    sys.stderr = _log_file

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

# Prevent system sleep during training
try:
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
except Exception:
    pass

# Suppress verbose PIL warnings
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency")

BASE = Path(os.environ.get("CROPGUARD_BASE", "C:/CropGuardAI/cropguard_ai"))
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 32
NUM_WORKERS = 2
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    BATCH_SIZE = 64
    NUM_WORKERS = 4
else:
    torch.set_num_threads(4)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Wheat: 3 diseases + Healthy | Rice: 4 diseases + Healthy
DISEASE_CLASSES = {
    "wheat": ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"],
    "rice": ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"],
    "wheat_stage": ["early", "mid", "late"],
    "rice_stage": ["early", "mid", "late"],
}
ALL_CLASSES = {
    "wheat": DISEASE_CLASSES["wheat"],
    "rice": DISEASE_CLASSES["rice"],
    "crop": ["rice", "wheat"],
    "wheat_stage": DISEASE_CLASSES["wheat_stage"],
    "rice_stage": DISEASE_CLASSES["rice_stage"],
}

def get_transforms(phase="train"):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def _safe_load_image(path):
    """Load image with simple error handling (corrupted images pre-scanned)."""
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception:
        return None

class DiseaseDataset(Dataset):
    def __init__(self, crop, split, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        classes = DISEASE_CLASSES[crop]
        split_dir = BASE / "data" / "split" / crop / split
        for idx, cls in enumerate(classes):
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')]
            for f in files:
                self.samples.append(f)
                self.labels.append(idx)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = _safe_load_image(self.samples[idx])
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class CropDataset(Dataset):
    def __init__(self, split, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        for crop_idx, crop in enumerate(["rice", "wheat"]):
            for cls_dir in (BASE / "data" / "split" / crop / split).iterdir():
                if not cls_dir.is_dir():
                    continue
                valid_ext = ('.jpg','.jpeg','.png','.bmp')
                files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext]
                for f in files:
                    self.samples.append(f)
                    self.labels.append(crop_idx)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = _safe_load_image(self.samples[idx])
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataset(model_name, split, transform):
    if model_name in ("wheat", "rice", "wheat_stage", "rice_stage"):
        return DiseaseDataset(model_name, split, transform)
    elif model_name == "crop":
        return CropDataset(split, transform)

def build_model(num_classes):
    """EfficientNet-B0 with custom classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    for p in model.features.parameters():
        p.requires_grad = False
    return model.to(DEVICE)

def unfreeze_last_n(model, n=5):
    """Unfreeze last n MBConv blocks of EfficientNet-B0."""
    blocks = list(model.features)
    for i, block in enumerate(blocks):
        if i >= len(blocks) - n:
            for p in block.parameters():
                p.requires_grad = True

def get_trainable(model):
    return [p for p in model.parameters() if p.requires_grad]

def compute_class_weight(dataset):
    counts = Counter(dataset.labels)
    total = len(dataset.labels)
    n = len(counts)
    weights = [0.0] * n
    for cls, cnt in counts.items():
        weights[cls] = total / (n * cnt)
    return torch.FloatTensor(weights).to(DEVICE)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.classes = classes
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def cutmix_batch(imgs, labels, alpha=1.0):
    """CutMix: replace a rectangular patch of each image with another image's patch."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0))
    shuffled_imgs, shuffled_labels = imgs[idx], labels[idx]
    _, _, h, w = imgs.shape
    cut_h = int(h * np.sqrt(1.0 - lam))
    cut_w = int(w * np.sqrt(1.0 - lam))
    cx = np.random.randint(h)
    cy = np.random.randint(w)
    x1 = max(0, cx - cut_h // 2); x2 = min(h, cx + cut_h // 2)
    y1 = max(0, cy - cut_w // 2); y2 = min(w, cy + cut_w // 2)
    imgs_mix = imgs.clone()
    imgs_mix[:, :, x1:x2, y1:y2] = shuffled_imgs[:, :, x1:x2, y1:y2]
    return imgs_mix, labels, shuffled_labels, lam

def train_one_epoch(model, loader, criterion, optimizer, ep_str="", use_cutmix=False, cutmix_prob=0.5):
    model.train()
    total_loss, correct, total = 0, 0, 0
    n = len(loader)
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if use_cutmix and np.random.rand() < cutmix_prob:
            imgs, labels, shuffled, lam = cutmix_batch(imgs, labels)
            outputs = model(imgs)
            loss = lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, shuffled)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        if (i+1) % 30 == 0 or i == n-1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  {ep_str} batch {i+1}/{n} | loss:{total_loss/total:.4f} acc:{correct/total:.4f} lr:{lr:.1e}")
    return total_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

def train(model_name, max_epochs=20, unfreeze_n=7, phase2_lr=2e-5, phase1_epochs=8, resume=False):
    num_classes = len(ALL_CLASSES[model_name])
    best_path = MODELS_DIR / f"{model_name}_efficientnet_best.pt"
    ckpt_path = MODELS_DIR / f"{model_name}_efficientnet_checkpoint.pt"

    print(f"\n{'='*60}")
    print(f"Training: {model_name} ({num_classes} classes) on {DEVICE}")
    print(f"{'='*60}")

    train_ds = get_dataset(model_name, "train", get_transforms("train"))
    val_ds = get_dataset(model_name, "val", get_transforms("val"))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    if len(train_ds) == 0:
        print("  ERROR: No training data!")
        return 0.0

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    class_weights = compute_class_weight(train_ds)
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)

    history_path = MODELS_DIR / f"{model_name}_efficientnet_history.json"

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        start_epoch = ckpt["epoch"] + 1
        phase = ckpt["phase"]
        best_acc = ckpt["best_acc"]
        model = build_model(num_classes)
        if phase == 2:
            unfreeze_last_n(model, n=unfreeze_n)
        model.load_state_dict(ckpt["model_state_dict"])
        if phase == 1:
            optimizer = optim.AdamW(get_trainable(model), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        else:
            optimizer = optim.AdamW(get_trainable(model), lr=phase2_lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - phase1_epochs, eta_min=1e-7)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for _ in range(start_epoch):
            scheduler.step()
        if history_path.exists():
            history = json.load(open(history_path))
        else:
            history = {"phase": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        print(f"  Resumed from epoch {start_epoch+1}/{max_epochs}, phase {phase}, best {best_acc:.4f}")
    else:
        model = build_model(num_classes)
        phase = 1
        best_acc = 0.0
        start_epoch = 0
        history = {"phase": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        if history_path.exists():
            with open(history_path) as f:
                prior = json.load(f)
            prior_best = max(prior.get("val_acc", [0.0]))
            if prior_best > best_acc:
                best_acc = prior_best
                print(f"  Loaded existing best from disk: {best_acc:.4f}")
        optimizer = optim.AdamW(get_trainable(model), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    for epoch in range(start_epoch, max_epochs):
        if epoch == phase1_epochs and phase == 1:
            print(f"  >>> Phase 2: unfreezing last {unfreeze_n} blocks, LR -> {phase2_lr:.0e}")
            unfreeze_last_n(model, n=unfreeze_n)
            phase = 2
            optimizer = optim.AdamW(get_trainable(model), lr=phase2_lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - phase1_epochs, eta_min=1e-7)
            if history_path.exists():
                with open(history_path) as f:
                    prior = json.load(f)
                prior_best = max(prior.get("val_acc", [0.0]))
                if prior_best > best_acc:
                    best_acc = prior_best
            print(f"  Best entering phase 2: {best_acc:.4f}")

        t0 = time.time()
        ep_str = f"Ep{epoch+1}P{phase}"
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, ep_str, use_cutmix=True)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        history["phase"].append(phase)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            tag = " *BEST*"
        elif val_acc >= best_acc * 0.97:
            tag = f" (best {best_acc:.4f})"

        torch.save({
            "epoch": epoch, "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }, ckpt_path)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{max_epochs} P{phase} | Train:{train_acc:.4f} Val:{val_acc:.4f} Best:{best_acc:.4f} LR:{lr:.1e} {elapsed:.0f}s{tag}")

        with open(MODELS_DIR / f"{model_name}_efficientnet_history.json", "w") as f:
            json.dump(history, f)

        target = 0.99 if model_name == "crop" else 0.95
        if best_acc >= target and phase == 2:
            print(f"  Target {target:.0%} reached!")
            break

    print(f"  DONE: {model_name} best={best_acc:.4f}")
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["wheat", "rice", "crop", "wheat_stage", "rice_stage"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--unfreeze", type=int, default=7, help="Number of blocks to unfreeze")
    parser.add_argument("--phase2_lr", type=float, default=2e-5, help="Phase 2 learning rate")
    parser.add_argument("--phase1_epochs", type=int, default=8, help="Epochs in phase 1 (frozen)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    train(args.model, args.epochs, args.unfreeze, args.phase2_lr, args.phase1_epochs, args.resume)
