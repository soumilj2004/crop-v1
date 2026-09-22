#!/usr/bin/env python3
"""Phase-2 fine-tuning restart with full LR control (bypasses resume LR clobber).

Loads the best checkpoint, unfreezes the last N blocks, trains with a fresh
optimizer + cosine schedule. Appends results to the main history JSON so
/model-info stays consistent.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from scripts.train_efficientnet import (
        DEVICE, BATCH_SIZE, NUM_WORKERS, MODELS_DIR,
        ALL_CLASSES, DiseaseDataset, get_transforms,
        build_model, unfreeze_last_n, get_trainable,
        compute_class_weight, LabelSmoothingLoss,
        train_one_epoch, validate,
    )
    import scripts.train_efficientnet as _te
except ImportError:
    from train_efficientnet import (
        DEVICE, BATCH_SIZE, NUM_WORKERS, MODELS_DIR,
        ALL_CLASSES, DiseaseDataset, get_transforms,
        build_model, unfreeze_last_n, get_trainable,
        compute_class_weight, LabelSmoothingLoss,
        train_one_epoch, validate,
    )
    import train_efficientnet as _te

MODEL = "wheat"
ARCH = "efficientnet_b0"
RCKPT = MODELS_DIR / f"{MODEL}_restart_checkpoint.pt"


def restart_phase2(lr=2e-5, epochs=20, unfreeze_n=7, resume=False, model=MODEL, arch=ARCH):
    num_classes = len(ALL_CLASSES[model])
    best_path = MODELS_DIR / f"{model}_efficientnet_best.pt"
    ckpt_path = MODELS_DIR / f"{model}_efficientnet_checkpoint.pt"
    history_path = MODELS_DIR / f"{model}_efficientnet_history.json"
    rckpt_path = MODELS_DIR / f"{model}_restart_checkpoint.pt"

    train_ds = DiseaseDataset(model, "train", get_transforms("train"))
    val_ds = DiseaseDataset(model, "val", get_transforms("val"))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)

    model_ = build_model(num_classes, arch)
    unfreeze_last_n(model_, n=unfreeze_n)
    start_epoch = 0
    if resume and rckpt_path.exists():
        rck = torch.load(rckpt_path, map_location=DEVICE)
        model_.load_state_dict(rck["model_state_dict"])
        start_epoch = rck["epoch"] + 1
        print(f"  Resumed restart at epoch {start_epoch+1}/{epochs}")
    else:
        raw = torch.load(best_path, map_location=DEVICE)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        model_.load_state_dict(raw)
        print(f"  Loaded best model, unfroze last {unfreeze_n} blocks")

    optimizer = optim.AdamW(get_trainable(model_), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if resume and rckpt_path.exists():
        optimizer.load_state_dict(torch.load(rckpt_path, map_location=DEVICE)["optimizer_state_dict"])
        for _ in range(start_epoch):
            scheduler.step()

    history = {"phase": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    if history_path.exists():
        history = json.load(open(history_path))
    best_acc = max(history.get("val_acc", [0.0]))

    for epoch in range(start_epoch, epochs):
        ep_str = f"R{epoch+1}/{epochs}"
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model_, train_loader, criterion, optimizer, ep_str, use_cutmix=True)
        val_loss, val_acc = validate(model_, val_loader, criterion)
        scheduler.step()

        history["phase"].append(2)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            if arch == "efficientnet_b0":
                torch.save(model_.state_dict(), best_path)
            else:
                torch.save({"arch": arch, "state_dict": model_.state_dict()}, best_path)
            tag = " *BEST*"
        elif val_acc >= best_acc * 0.97:
            tag = f" (best {best_acc:.4f})"

        torch.save({
            "arch": arch, "epoch": epoch, "phase": 2,
            "model_state_dict": model_.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }, ckpt_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }, rckpt_path)
        json.dump(history, open(history_path, "w"))

        print(f"  Epoch {epoch+1}/{epochs} P2 | Train:{train_acc:.4f} Val:{val_acc:.4f} Best:{best_acc:.4f} LR:{scheduler.get_last_lr()[0]:.1e} {time.time()-t0:.0f}s{tag}")

    print(f"DONE: {model} best={best_acc:.4f}")


if __name__ == "__main__":
    args_ = sys.argv[1:]
    lr = float(args_[0]) if args_ and not args_[0].startswith("--") else 2e-5
    epochs = int(args_[1]) if len(args_) > 1 and not args_[1].startswith("--") else 20
    unfreeze_n = int(args_[2]) if len(args_) > 2 and not args_[2].startswith("--") else 7
    resume = "--resume" in args_
    model = "wheat_stage" if "--model" in args_ else MODEL
    arch = "mobilenet_v3_small" if "--arch" in args_ else ARCH
    _te.INPUT_SIZE = 128 if arch == "mobilenet_v3_small" else 224
    restart_phase2(lr=lr, epochs=epochs, unfreeze_n=unfreeze_n, resume=resume, model=model, arch=arch)
