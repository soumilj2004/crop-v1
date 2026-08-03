# CropGuard AI

AI-based crop disease detection and recommendation engine for rice and wheat. Runs as a local desktop app — no cloud, no account, no manual crop/stage selection.

## What it does

1. **Take a photo** of a rice or wheat leaf
2. **Auto-detects** crop type (rice vs wheat)
3. **Classifies** disease (or healthy)
4. **Estimates** severity stage (early/mid/late) — *heuristic, see caveats*
5. **Fetches live weather** for your location (via IP geolocation + Open-Meteo)
6. **Returns treatment recommendation** with risk level, chemical, dosage, timing, and cultural practices

All from a single photo. Zero dropdowns. Zero manual input.

## Models (4 total)

| Model | Classes | Architecture | Notes |
|-------|---------|--------------|-------|
| Crop Classifier | rice, wheat | MobileNetV2 | Binary, 97.6% test accuracy |
| Rice Disease | Bacterial Blight, Blast, Brown Spot, Tungro, Healthy | MobileNetV2 | 5-class, includes Healthy |
| Wheat Disease | Leaf Rust, Loose Smut, Crown & Root Rot, Healthy | MobileNetV2 | 4-class, Leaf Rust only 89 images |
| Wheat Stage | early, mid, late | MobileNetV2 | Heuristic pseudo-labels (HSV lesion ratio terciles) |

All models: ImageNet-pretrained MobileNetV2, two-stage transfer learning (frozen backbone → fine-tune last 3 blocks), 160×160 input, class-weighted loss.

**Note:** Rice severity staging (early/mid/late) is **not available**. After two correction rounds (filtering non-photographic images, then close-up vs. field shots, then single-leaf framing), contact sheets still showed no consistent early→late progression for 3 of 4 rice diseases. Rather than train on labels that failed visual review, rice disease detection ships without automatic severity staging. `/analyze` returns `stage: null` for rice results, and the recommendation engine falls back to general (non-stage-specific) guidance, clearly marked as such. Wheat staging passed visual validation and works as designed.

## Data Sources (Real, Verified)

- **Rice diseases**: Mendeley "Rice Leaf Disease Image Samples" (Sethy et al., 2020), mirrored at `maimunul/Rice-Leaf-Disease-Classification-using-CNN` on GitHub — 4,794 unique images after 19% dedup
- **Rice Healthy**: Paddy Doctor dataset `normal` class from `ai-agriculture-circuits-and-systems/paddy_disease_classification` (sparse-cloned)
- **Wheat**: `aadium/wheat-disease-detection` on GitHub (`cropDiseaseDataset/`) — 1,294 unique after 60% dedup. Leaf Rust only 89 images.

**All datasets deduplicated by MD5 hash before splitting.** Train/val/test = 70/15/15 stratified, seed=42.

## Honest Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Wheat Leaf Rust only 89 images | 74% test accuracy (vs 95% rice) | Reported honestly; more data needed |
| Stage labels are **heuristic pseudo-labels**, not expert annotations | Stage accuracy ~78% rice / ~53% wheat (agreement with pseudo-labels, not ground truth) | Returned with explicit caveat in API response |
| No antiviral for Tungro; no foliar rescue for Loose Smut or Crown/Root Rot | Recommendations correctly state "no effective in-season treatment" | Knowledge base encodes this explicitly; unit tests assert it |
| Weather via IP geolocation (approx. city-level) | Less precise than GPS | Acceptable for regional risk; graceful fallback if offline |
| **Rice severity staging disabled** | No automatic early/mid/late for rice | `/analyze` returns `stage: null` for rice; recommendation falls back to general guidance with explicit caveat |

## Quick Start

### Prerequisites
- Windows 10/11 (primary), Linux, or macOS
- Python 3.10+
- ~4 GB disk for models + datasets
- Internet for initial weight download (ImageNet) and live weather

### Install & Run
```bash
# Clone
git clone <this-repo>
cd cropguard_ai

# Install deps
pip install -r requirements.txt

# Download & prepare data (run once)
python scripts/download_rice.py
python scripts/download_wheat.py
python scripts/split_data.py
python scripts/generate_stage_labels.py

# Train all 4 models (resumable, checkpointed)
python scripts/train_all.py --model crop --epochs 20
python scripts/train_all.py --model wheat --epochs 20
python scripts/train_all.py --model wheat_stage --epochs 20
python scripts/train_all.py --model rice --epochs 20
# No rice_stage model -- automatic staging disabled for rice (see limitations)
```

### Training Notes
- Each `train_all.py` saves a checkpoint every epoch (`models/<name>_checkpoint.pt`)
- If interrupted, re-run with `--resume` to continue
- Default: 20 epochs (4 frozen + 16 fine-tune, with early stopping)
- Two-phase training: epochs 1-4 frozen backbone, epochs 5+ unfreeze last 3 blocks
- Each `train_resumable.py` saves a checkpoint every epoch (`models/<name>_checkpoint.pt`)
- If interrupted, re-run with `--resume` to continue
- Default: 4 epochs frozen + 3 epochs fine-tune; adjust with `--epochs1` `--epochs2`

## Project Structure
```
cropguard_ai/
├── run.py / run.bat / run.sh          # Desktop app entry points
├── requirements.txt
├── models/                            # Trained .pt files (shipped)
├── data/
│   ├── raw/rice/, raw/wheat/         # Downloaded + deduped images
│   ├── split/rice/, split/wheat/     # 70/15/15 splits
│   └── stage_labels/                 # Heuristic stage labels + validation grids
├── scripts/
│   ├── download_rice.py
│   ├── download_wheat.py
│   ├── split_data.py
│   ├── generate_stage_labels.py
│   └── train_resumable.py
├── cropguard/
│   ├── inference.py                   # Model loading + prediction
│   ├── api/
│   │   ├── main.py                    # FastAPI endpoints
│   │   └── static/index.html          # Frontend (single-file)
│   ├── recommend/
│   │   ├── knowledge_base.py          # Disease/treatment data
│   │   └── engine.py                  # Recommendation logic + weather risk
│   └── weather/
│       └── client.py                  # Open-Meteo + IP geolocation
└── tests/
    └── test_recommend.py              # 16 unit tests
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Model availability + server status |
| `POST /predict?crop=rice\|wheat` | Single-crop disease prediction |
| `POST /recommend` | Recommendation from disease+stage+weather |
| `POST /analyze` | **Main**: image → full pipeline result |
| `GET /diseases?crop=rice\|wheat` | List supported classes |

### `/analyze` Response (example — wheat)

```json
{
  "detected_crop": "wheat",
  "crop_confidence": 0.992,
  "disease": "Leaf Rust",
  "disease_confidence": 0.941,
  "stage": "early",
  "stage_confidence": 0.72,
  "weather": {"temperature_c": 28.5, "humidity_pct": 82, "rainfall_mm": 3.2, "source": "open_meteo"},
  "risk_level": "high",
  "risk_reason": "High humidity and warm temperatures favor fungal spore germination and spread",
  "treatment": {
    "action": "Protective fungicide application",
    "chemical": "Tricyclazole 75% WP",
    "dosage": "0.6 g/L water (600 g/ha)",
    "timing": "At first lesion appearance; critical before panicle emergence",
    "cultural": ["Avoid excessive N; split application", "Maintain shallow water..."]
  },
  "caveats": ["Severity stage is heuristic (HSV lesion ratio), not expert-annotated."]
}
```

### `/analyze` Response (example — rice)

```json
{
  "detected_crop": "rice",
  "crop_confidence": 0.992,
  "disease": "Blast",
  "disease_confidence": 0.941,
  "stage": null,
  "stage_confidence": null,
  "weather": {"temperature_c": 28.5, "humidity_pct": 82, "rainfall_mm": 3.2, "source": "open_meteo"},
  "risk_level": "high",
  "risk_reason": "High humidity and warm temperatures favor fungal spore germination and spread",
  "treatment": {
    "action": "Protective fungicide application",
    "chemical": "Tricyclazole 75% WP",
    "dosage": "0.6 g/L water (600 g/ha)",
    "timing": "At first lesion appearance; critical before panicle emergence",
    "cultural": ["Avoid excessive N; split application", "Maintain shallow water..."]
  },
  "caveats": [
    "Severity stage is heuristic (HSV lesion ratio), not expert-annotated.",
    "Automatic severity staging is not available for rice -- this is general guidance for the disease, not adjusted for how advanced it is."
  ]
}
```

## Frontend

Single-file HTML/CSS/JS served at `/`. Features:
- Light theme, card-based mobile-app design
- Drag-drop / camera capture
- Auto-analyze on image select (no "Analyze" button)
- Scanning animation during inference
- Animated result cards with progress bars
- Real weather display
- Respects `prefers-reduced-motion`

## Desktop Packaging

`run.py` handles:
1. Auto-installs missing pip packages from `requirements.txt`
2. Verifies all 5 model files exist
3. Starts FastAPI on `127.0.0.1:8000` (background thread)
4. Opens `pywebview` native window (Edge/WebView2 on Windows)
5. Falls back to default browser if native window fails

## Testing
```bash
pytest tests/ -v
```
Key tests:
- Every disease × stage produces valid recommendation
- Loose Smut recommendation does NOT claim foliar treatment
- Leaf Rust recommendation DOES contain real chemical
- Healthy returns no-op for all stages
- Weather risk rules correctly shift levels

## License

MIT for code. Datasets under their original licenses (CC0 for rice, verify wheat repo). Model weights derived from ImageNet-pretrained MobileNetV2 (Apache 2.0 / BSD-3 compatible).

## Citation

If you use this in research, please cite honestly:
> "CropGuard AI: Local crop disease detection with honest reporting of heuristic severity labels and data limitations. Capstone replacement for fabricated results."

## Acknowledgments

- Dataset authors: Sethy et al. (rice), aadium (wheat), Paddy Doctor team
- PyTorch / torchvision for MobileNetV2 weights
- Open-Meteo and ip-api.com for free weather/geolocation APIs
