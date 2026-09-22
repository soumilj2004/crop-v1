"""
Train wheat/rice/crop classifiers (arch-aware).
3 diseases + Healthy per crop (4 classes each).
Two-phase: freeze backbone (phase1_epochs), then unfreeze.
Cosine annealing LR, label smoothing, class weighting, mixup/cutmix, EMA.
Paths are relative to the repo root (parent of scripts/) unless CROPGUARD_BASE is set.
"""
import os, sys, json, time, argparse, random
from pathlib import Path
from collections import Counter

import time, os
REPO_ROOT = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))
_log_dir = REPO_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)
_log_path = str(_log_dir / "training_script.log")
# Try to open; if locked, use a PID-suffixed path
try:
    _log_file = open(_log_path, "a", buffering=1)
except PermissionError:
    _log_path = str(_log_dir / f"training_{os.getpid()}.log")
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
    # ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x1) | ES_DISPLAY_REQUIRED(0x2).
    # The previous value (0x80000002) set DISPLAY_REQUIRED only, which does NOT
    # keep the system awake -- a long CPU run would die on sleep. Fixed 2026-08-31.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
except Exception:
    pass

# Suppress verbose PIL warnings
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency")

BASE = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 32
NUM_WORKERS = 2
INPUT_SIZE = 224
ARCH = "efficientnet_b0"
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

def get_backbone_blocks(model):
    """Iterable of backbone blocks (arch-agnostic)."""
    if hasattr(model, "features"):
        return list(model.features)
    if hasattr(model, "layer4"):
        return [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
    return []



class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention).

    Lets the network learn to weight *which* regions of a busy field photo
    (lots of overlapping leaves, background water/soil) actually carry the
    disease signal, instead of treating the whole image uniformly. Cheap:
    a few thousand extra params on top of a frozen/fine-tuned backbone.
    Reference: Woo et al. 2018, "CBAM: Convolutional Block Attention Module".
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        ch_att = self.sigmoid_channel(avg_out + max_out)
        x = x * ch_att
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_sp, max_sp], dim=1)))
        x = x * sp_att
        return x


class EfficientNetCBAM(nn.Module):
    """EfficientNet-B0 backbone + CBAM attention before pooling/classifier.

    Keeps `.features` as the ORIGINAL torchvision nn.Sequential (unchanged)
    so get_backbone_blocks()/unfreeze_last_n() keep working exactly as they
    do for plain efficientnet_b0 -- only the forward pass changes (features
    -> CBAM -> avgpool -> classifier). `.classifier` is also kept as a
    direct attribute for the same reason (existing code paths touch it).
    """
    def __init__(self, base_model, channels):
        super().__init__()
        self.features = base_model.features
        self.cbam = CBAM(channels)
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(num_classes, arch="efficientnet_b0", init_path=None):
    """Build backbone with custom classifier head.

    init_path: optional backbone state dict (e.g. LeafVision SSL) to load
    instead of ImageNet weights. Ignored for archs without a provided init.
    """
    use_inet = init_path is None
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_inet else None)
        in_features = model.classifier[0].in_features
    elif arch == "efficientnet_b0_cbam":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_inet else None)
        in_features = base.classifier[1].in_features
        model = EfficientNetCBAM(base, channels=in_features)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.cbam.parameters():
            p.requires_grad = True
        if init_path:
            _load_init(model, init_path)
        return model.to(DEVICE)
    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if use_inet else None)
        in_features = model.classifier[1].in_features
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_inet else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        for block in get_backbone_blocks(model):
            for p in block.parameters():
                p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        if init_path:
            _load_init(model, init_path)
        return model.to(DEVICE)
    elif arch in ("swin_t", "swin_s"):
        if arch == "swin_t":
            model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        else:
            model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        if init_path:
            print(f"  [warn] --init ignored for {arch} (no SSL weights available); using ImageNet")
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        for p in model.features.parameters():
            p.requires_grad = False
        return model.to(DEVICE)
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_inet else None)
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
    if init_path:
        _load_init(model, init_path)
    return model.to(DEVICE)


def _load_init(model, init_path):
    import torch
    sd = torch.load(init_path, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    cur = model.state_dict()
    sd = {k: v for k, v in sd.items()
          if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("classifier") and not k.startswith("fc")]
    print(f"  Init from {init_path}: loaded={len(sd)} missing={len(missing)} (classifier only: {len(bad)==0}) unexpected={len(unexpected)}")
    if bad or unexpected:
        print(f"  [warn] unexpected init keys: {unexpected[:5]}, missing: {bad[:5]}")

def unfreeze_last_n(model, n=5):
    """Unfreeze last n backbone blocks."""
    blocks = get_backbone_blocks(model)
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
    def __init__(self, classes, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.classes = classes
        self.weight = weight
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * pred, dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[target]
        return torch.mean(loss)

def mixup_batch(imgs, labels, alpha=0.2):
    """MixUp: convex combination of two images and their labels."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0))
    mixed = lam * imgs + (1.0 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam

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

class EMA:
    """Exponential moving average of model weights (float params + buffers)."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = self._snapshot(model)

    def _snapshot(self, model):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        self.shadow = {k: v.detach().clone() for k, v in sd.items()}


def train_one_epoch(model, loader, criterion, optimizer, ep_str="", aug="both", aug_prob=0.5, ema=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    n = len(loader)
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if aug in ("cutmix", "both") and np.random.rand() < aug_prob:
            imgs, labels, shuffled, lam = cutmix_batch(imgs, labels)
            outputs = model(imgs)
            loss = lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, shuffled)
        elif aug in ("mixup", "both") and np.random.rand() < aug_prob:
            imgs, labels, shuffled, lam = mixup_batch(imgs, labels)
            outputs = model(imgs)
            loss = lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, shuffled)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # EMA must track every optimizer step. Updating once per epoch (the old
        # behaviour) left the shadow weights ~0.999**epochs of the UNTRAINED
        # initialisation -- 98% untrained after 20 epochs -- so every reported
        # val_acc measured an untrained model and the saved "best" checkpoint
        # was EMA garbage. Cost ~48 accuracy points. Fixed 2026-08-31.
        if ema is not None:
            ema.update(model)
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

def resolve_init(init, models_dir):
    """Resolve --init value to a backbone file path (or None)."""
    if not init:
        return None
    p = Path(init)
    if p.exists():
        return str(p)
    if init.startswith("leafvision_dino_"):
        cand = models_dir / f"{init}_backbone.pt"
        if cand.exists():
            return str(cand)
    print(f"  [warn] init '{init}' not found; falling back to ImageNet weights")
    return None


def train(model_name, max_epochs=20, unfreeze_n=7, phase2_lr=2e-5, phase1_epochs=8, resume=False,
          arch="efficientnet_b0", img_size=224, aug="both", ema_decay=0.999, balanced=True,
          init_path=None):
    num_classes = len(ALL_CLASSES[model_name])
    best_path = MODELS_DIR / f"{model_name}_efficientnet_best.pt"
    ckpt_path = MODELS_DIR / f"{model_name}_efficientnet_checkpoint.pt"

    print(f"\n{'='*60}")
    print(f"Training: {model_name} ({num_classes} classes) on {DEVICE} arch={arch} img={img_size}"
          + (f" init={init_path}" if init_path else ""))
    print(f"{'='*60}")

    train_ds = get_dataset(model_name, "train", get_transforms("train"))
    val_ds = get_dataset(model_name, "val", get_transforms("val"))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    if len(train_ds) == 0:
        print("  ERROR: No training data!")
        return 0.0

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    class_weights = compute_class_weight(train_ds) if balanced else None
    if class_weights is not None:
        print(f"  Class weights: {[round(w, 3) for w in class_weights.tolist()]}")
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1, weight=class_weights)

    history_path = MODELS_DIR / f"{model_name}_efficientnet_history.json"

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        start_epoch = ckpt["epoch"] + 1
        phase = ckpt["phase"]
        best_acc = ckpt["best_acc"]
        model = build_model(num_classes, arch, init_path)
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
        ema = EMA(model, ema_decay) if ema_decay > 0 else None
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if history_path.exists():
            history = json.load(open(history_path))
        else:
            history = {"phase": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        print(f"  Resumed from epoch {start_epoch+1}/{max_epochs}, phase {phase}, best {best_acc:.4f}")
    else:
        model = build_model(num_classes, arch, init_path)
        phase = 1
        best_acc = 0.0
        start_epoch = 0
        history = {"phase": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        ema = EMA(model, ema_decay) if ema_decay > 0 else None
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
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, ep_str, aug=aug, ema=ema)
        # Score the raw model AND the EMA, then keep whichever is genuinely
        # better. Never trust one blindly -- that is what hid the bug above.
        val_loss, val_acc = validate(model, val_loader, criterion)
        best_is_ema = False
        if ema is not None:
            raw_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict())
            ema_loss, ema_acc = validate(model, val_loader, criterion)
            model.load_state_dict(raw_weights)
            print(f"  val raw:{val_acc:.4f}  ema:{ema_acc:.4f}")
            if ema_acc > val_acc:
                val_loss, val_acc, best_is_ema = ema_loss, ema_acc, True
        scheduler.step()

        history["phase"].append(phase)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            _sd = ema.state_dict() if (best_is_ema and ema is not None) else model.state_dict()
            torch.save({"arch": arch, "state_dict": _sd, "val_acc": val_acc,
                        "source": "ema" if best_is_ema else "raw"}, best_path)
            tag = " *BEST*"
        elif val_acc >= best_acc * 0.97:
            tag = f" (best {best_acc:.4f})"

        ckpt = {
            "arch": arch,
            "epoch": epoch, "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }
        if ema is not None:
            ckpt["ema_state_dict"] = ema.state_dict()
        torch.save(ckpt, ckpt_path)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{max_epochs} P{phase} | Train:{train_acc:.4f} Val:{val_acc:.4f} Best:{best_acc:.4f} LR:{lr:.1e} {elapsed:.0f}s{tag}")

        # Sanity guard: if validation is still at chance after phase 1, something
        # is broken -- stop instead of burning hours producing a useless model.
        _chance = 1.0 / num_classes
        if epoch + 1 >= phase1_epochs + 2 and best_acc < _chance * 1.25:
            print(f"\n  ABORT: val accuracy {best_acc:.4f} is still near chance "
                  f"({_chance:.4f}) after {epoch+1} epochs. Training is not working -- "
                  f"stopping rather than wasting GPU time. Send the log back.")
            break

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
    parser.add_argument("--arch", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "efficientnet_b0_cbam", "mobilenet_v3_small", "efficientnet_v2_s", "resnet50", "swin_t", "swin_s"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--init", type=str, default=None,
                        help="Backbone init: leafvision_dino_efficientnet_b0 | leafvision_dino_resnet50 | path-to-.pt")
    parser.add_argument("--aug", type=str, default="both", choices=["none", "cutmix", "mixup", "both"],
                        help="Batch augmentation: cutmix, mixup, both, none")
    parser.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="Use inverse-frequency class weights in loss")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    ARCH = args.arch
    INPUT_SIZE = args.img_size
    init_path = resolve_init(args.init, MODELS_DIR)
    train(args.model, args.epochs, args.unfreeze, args.phase2_lr, args.phase1_epochs, args.resume,
          args.arch, args.img_size, args.aug, args.ema, args.balanced, init_path)

