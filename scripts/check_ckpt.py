import torch, os
ckpt_path = "C:/CropGuardAI/cropguard_ai/models/wheat_checkpoint.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Epoch: {ckpt['epoch']}, Best: {ckpt['best_acc']:.4f}")
else:
    print("No checkpoint")
