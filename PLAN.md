# CropGuardAI — Plan to get every wheat metric >= 90%

Status: active (2026-08-07)
GPU target: friend's RTX 3060 12GB. Laptop CPU = data/eval/inference only, no training.

## Current baseline (to beat)
- Local model (EffNet-B0, 224px, 3 seeds): val 87.31%, test 86.74% (1,621 imgs)
- Friend's run (EffNet-B0): val 86.25%, test 84.58% -> ours stays the deployed model
- Stage model: MobileNetV3-Small @128px best 64.27% (old B0 stage: 67.1%)

## Q1 answered: color filters
- RGB level: ExG/ExR/ExGR + Otsu = 97.4% plant/background segmentation, lighting-robust.
  Optional 4th channel / on-device preprocess. Not the main lever.
- SWIR/hyperspectral (1420/1920nm water bands) detects pre-symptom disease (stem rust
  F1 0.94-0.96) but cameras $20-50k -> roadmap (drone/UAV) only, not MVP.
- Augmentations: avoid blur/grayscale/solarization (hurt plant SSL); use affine/posterize/color jitter.

## Q2 answered: transfer approach
- LeafVision DINO SSL pretraining on 540k leaf images (arXiv 2606.10676, public weights):
  at 30 img/class ResNet-50 94.6->98.5%, EffNet-B0 89.0->98.1%. Domain init >> ImageNet init.
- GPU allows: EfficientNetV2-S (20.4M / 2.9 GMACs, torchvision) or Swin-T (timm).
  Swin's window attention handles field backgrounds (SWIN 88% vs CNN 53% real-world).

## Q3 answered: HAT-Net / 99.75% ensemble (LWDCD2020 = our exact 4 classes)
- HAT-Net (Swin+DeiT fusion, cosine annealing): 99.2% acc / 97.79% cross-val (Springer s40537-025-01353-w)
- W-STNet 96.72% (fastest), W-DENet 97.3%
- 99.75% = simple voting ensemble (Xception + InceptionV3 + ResNet50)
- Grad-CAM/LIME used for explainability -> ship heatmaps in app
- Data: LWDCD2020 on Kaggle `lyxbash/lcdcd-2020-dataset` (~12k, field backgrounds)

## Q4 answered: VLM approach
- SCOLD (HF `enalis/scold`): Swin-T visual + RoBERTa text, 186k image-caption pairs, 97 concepts,
  trained on RTX 3060 Ti 12GB (friend's card). Zero-shot wheat 42%, 16-shot fine-tune 98%.
- Use: LoRA fine-tune on our 4 classes with symptom-text prompts (or linear-probe frozen embeddings).

## Plan (all training on friend's GPU)

### Phase 0 - Data (CPU) - DONE 2026-08-07
- [x] Downloaded LWDCD2020 (Kaggle `lyxbash/lcdcd-2020-dataset`, 3,861 imgs: C&RR 1,033 / Healthy 820 / Leaf Rust 1,267 / Loose Smut 741) -> validated (0 corrupt)
- [x] Normalized to RGB JPEG max 1024px (`data/lwdcd2020_norm/` via `scripts/normalize_lwdcd.py`)
- [x] Merged into TRAIN only via `scripts/merge_lwdcd.py` (val 1,615 / test 1,621 UNTOUCHED - honest benchmark)
      Train: 7,544 -> 11,405 (C&RR 2,825 / Healthy 2,092 / Leaf Rust 4,694 / Loose Smut 1,794)
      DECISION: no physical oversampling - `--balanced` class weights in loss instead (inverse frequency)
- [ ] Optional later: WFD (wfd.sysbio.ru, 2,414), FWDI (2,643), AgriPath-LF16 30k (HF `hamzamooraj99/AgriPath-LF16-30k`)
- [x] Diff zip built (`C:\CropGuardAI\transfer\wheat_v2_diff.zip`) with scripts + LWDCD data + FRIEND_README.md

### Phase 1 - GPU baseline (friend, ~1-2h)
- [ ] Swin-T @224px, ImageNet init, 60 epochs, phase1 10 frozen + phase2 2e-5
- [ ] CutMix+MixUp, label smoothing 0.1, AdamW, cosine, EMA 0.999, balanced class weights
- [ ] Expect val 92-95% (data +5% vs old split, arch +2-4%)

### Phase 2 - Domain init (friend, ~1-2h per run)
- [x] LeafVision weights obtained + verified (git-lfs clone, 810MB; flat torchvision-style dicts, load clean)
- [x] `scripts/leafvision_prep.py` extracts backbone-only inits -> `models/leafvision_dino_{effnet_b0,resnet18,resnet50}_backbone.pt` (verified: loaded=358, classifier-only missing)
- [x] `--init` flag in train_efficientnet.py (smoke-tested)
- [ ] Friend runs: `--arch efficientnet_b0 --init leafvision_dino_efficientnet_b0` and `--arch resnet50 --init leafvision_dino_resnet50`
- [ ] Expect val 92-96%

### Phase 3 - Ensemble (friend)
- [ ] 3 diverse backbones (Swin-T + EfficientNetV2-S + ConvNeXt) + weighted voting
- [ ] Grad-CAM/LIME overlay for app evidence - PROTOTYPED locally (`scripts/gradcam_demo.py`, 8 overlays saved to `data/gradcam/`)
- [ ] Per-class P/R/F1 gate tool built (`scripts/eval_per_class.py`); CURRENT BASELINE on val: acc 87.31, macro-F1 87.12 — ALL classes FAIL the 90% gate (worst: C&RR 0.805 F1)
- [ ] Expect 95-97%; per-class P/R/F1 >= 90% gate

### Phase 4 - Escalation only if a class < 90%
- [ ] SCOLD 16-shot LoRA fine-tune (98% wheat ceiling)

### Phase 5 - Stage model (friend)
- [ ] Retrain wheat_stage with same stack (EffNetV2-S, expect >75%); MobileNetV3-Small obsolete
- [ ] Rice stays as-is unless below threshold

### Phase 6 - Ship
- [ ] Rebuild friend's package: wheat + wheat_stage + rice + ensemble voting
- [ ] Per-class P/R/F1 report on test, gate >= 90% every class
- [ ] Sync history JSON to /model-info; keep current model until replaced

## Timeline
- CPU today: data prep (Phase 0) + gallery + eval tooling
- Friend's GPU: Phases 1-5 (~2-4 days total training wall-clock)
- Gate: all classes >= 90% before packaging
