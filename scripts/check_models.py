import torch, os

models_dir = 'C:/CropGuardAI/cropguard_ai/models'
files = ['crop.pt','rice.pt','rice_stage.pt','wheat_best.pt','wheat_checkpoint.pt']
for f in files:
    p = os.path.join(models_dir, f)
    if os.path.exists(p):
        ckpt = torch.load(p, map_location='cpu')
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            print(f"{f}: keys={keys}")
            for k in ['best_acc','accuracy','val_acc','epoch']:
                if k in ckpt:
                    print(f"  {k}={ckpt[k]}")
            if 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
                last_key = [k for k in sd.keys() if 'classifier' in k or 'fc' in k]
                print(f"  classifier keys: {last_key}")
                for lk in last_key:
                    print(f"    {lk}: shape={sd[lk].shape}")
        else:
            print(f"{f}: raw state_dict (not checkpoint dict)")
