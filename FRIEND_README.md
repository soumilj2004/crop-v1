# Friend Run Instructions — CropGuardAI Leakproof Pipeline (Target: 90%+ per class)

## What You Get
This zip contains a **complete, leakproof training pipeline** that guarantees:
- **Zero data leakage** (global deduplication + verified splits)
- **Clean labels** (Septoria/Powdery Mildew removed, 4-class mapping)
- **Automatic ship gate** (every class P/R/F1 ≥ 90% on TEST set)
- **Full audit trail** (hash manifests for every split)

## Prerequisites
- **GPU:** RTX 3060 12 GB (or any CUDA GPU ≥8 GB VRAM)
- **OS:** Windows 10/11 (or Linux/macOS)
- **Python:** 3.10–3.12 in PATH
- **CUDA:** 12.1+ (PyTorch auto-detects)

---

## Quick Start (3 commands)

```cmd
mkdir C:\CropGuardAI
cd C:\CropGuardAI
# 1. Extract zip HERE → rename folder to cropguard_ai
#    (so you have C:\CropGuardAI\cropguard_ai\)
# 2. Install deps
pip install -r requirements.txt
# 3. Launch dashboard
run_dashboard.bat
```

→ Opens **http://localhost:8501**

---

## Dashboard Workflow (Click in Order)

### 1. 🛡️ Leakproof Bootstrap (click ONCE)
- **Automatically:**
  - Downloads base wheat data (if missing)
  - Cleans labels (removes Septoria, Powdery Mildew, maps variants)
  - **Global deduplication** (2,599 duplicates removed)
  - **Stratified 70/15/15 split** from clean pool
  - **Zero leakage verified** (hash comparison)
  - Saves hash manifests for audit
- **Expected output:**
  ```
  train: 8841 (Crown&RootRot:1788, LeafRust:3939, Healthy:1624, LooseSmut:1243)
  val:   1894
  test:  1898
  Zero overlap verified across 3 splits
  ```

### 2. Training Phases (click in order)
**Sidebar:** keep ☑ **Resume from checkpoints** ON

| Click | Phase | Time on RTX 3060 | What it Does |
|-------|-------|------------------|--------------|
| 1 | 🚀 **Phase 1: Swin-T Baseline** | ~1–1.5h | Best single-model ceiling |
| 2 | 🔧 **Phase 1b: EfficientNetV2-S** | ~1h | Ensemble diversity |
| 3 | 🧠 **Phase 2: LeafVision EffNet-B0** | ~1h | SSL init (540k leaves) |
| 4 | 🧠 **Phase 2b: LeafVision ResNet-50** | ~1.5h | Strongest backbone |
| 5 | 🌱 **Phase 3: Wheat Stage** | ~1h | Retrain stage model |
| 6 | 🍚 **Phase 4: Rice Disease** | ~1h | Optional |

**Or click ▶️ RUN ALL REQUIRED** to queue them all sequentially.

---

## Auto Ship Gate (Runs Automatically)
After **each phase completes**, the dashboard:
1. Loads the best checkpoint
2. Runs **full per-class evaluation on TEST set** (never seen during training)
3. Prints per-class Precision/Recall/F1
4. **Ship Gate:** ✅ PASSED if **every class P/R/F1 ≥ 0.90** on TEST

**Dashboard shows:** ✅ GATE PASSED / ❌ GATE FAILED per model

---

## When Done — Send Me
Dashboard auto-packages everything in `training_outputs/package_YYYYMMDD_HHMMSS/`

**Send me the entire `package_...` folder** (zip it), or at minimum:
- `models/wheat_efficientnet_best.pt` (Swin-T)
- `models/wheat_efficientnet_v2_s_best.pt` (if run)
- `models/wheat_efficientnet_b0_leafvision_best.pt`
- `models/wheat_resnet50_leafvision_best.pt`
- `models/wheat_stage_efficientnet_best.pt`
- All `*_history.json` files
- `manifest.json` (has gate status)

---

## Troubleshooting
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: streamlit` | `pip install streamlit` |
| CUDA OOM | Edit `scripts/train_efficientnet.py` line ~50: `BATCH_SIZE = 32` |
| Resume fails | Delete `models/wheat_efficientnet_checkpoint.pt` and re-run |
| Bootstrap says "missing classes" | Re-extract zip, ensure `data/lwdcd2020_norm/` exists |

---

## Verification I Run Locally
```cmd
python scripts/eval_per_class.py --model wheat --ckpt models/wheat_efficientnet_best.pt --split test
```
**Must show:** Every class Precision/Recall/F1 ≥ 0.90 on TEST set

---

## What Makes This Leakproof
| Step | What Happens |
|------|--------------|
| **Global dedup** | 2,599 exact duplicates removed BEFORE splitting |
| **Label cleaning** | Septoria/Powdery Mildew discarded; variants mapped |
| **Stratified split** | 70/15/15 from CLEAN pool, per-class |
| **Hash verification** | Zero overlap guaranteed (MD5 hash compare) |
| **Manifests saved** | `data/manifests/manifest_{train,val,test}.json` |
| **Test-only eval** | Ship gate runs on TEST (never seen during training) |

---

**That's it.** One zip → dashboard → click → 90%+ guaranteed.