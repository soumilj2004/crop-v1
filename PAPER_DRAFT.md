# CropGuardAI: A Leakage-Aware, Reproducible Crop Disease Classification Pipeline with Weather-Aware Decision Support for Wheat and Rice

**Status: WORKING DRAFT — Wheat results pending (training in progress on external GPU as of 2026-08-30). Rice, crop-classification, and methodology sections reflect verified, current repository state.**

---

## Abstract

*(To be finalized once Wheat results are available — do not write this section until Results is complete; an abstract written against pending numbers risks becoming inconsistent with the final findings.)*

---

## 1. Introduction

Automated crop disease diagnosis from leaf imagery has been studied extensively using convolutional neural networks and, more recently, vision transformers, typically reporting validation accuracies in the 90-99% range on public datasets such as PlantVillage and its derivatives. A recurring, under-reported problem in this literature is data leakage: near-identical or exactly duplicated images distributed across training, validation, and test splits inflate reported performance without reflecting a model's ability to generalize to unseen leaves. This project encountered that problem directly during the development of a Wheat and Rice disease classification pipeline, and the response to it — rather than the initial (misleading) accuracy figures — is the primary methodological contribution reported here.

CropGuardAI is not framed as "an AI plant disease detection app." It is framed as a leakage-aware, reproducible crop disease classification pipeline for Wheat and Rice, integrated with weather-aware agricultural decision support. The system takes a single leaf photograph and produces: (1) an automatic crop-type classification (Wheat vs. Rice), (2) a disease classification specific to that crop, (3) where supported, a severity/stage estimate, (4) live local weather context, and (5) a rule-based treatment recommendation. The end-to-end pipeline is described in Section 5; the components most relevant to the paper's central claim — clean, honestly evaluated classification performance — are described in Sections 4 and 6-8.

## 2. Related Work

*(To be expanded with a literature review pass; the following reflects sources already identified and used during pipeline development, not a complete survey.)*

Prior work on wheat and rice leaf disease classification using the LWDCD2020 dataset (the same source used for part of this project's Wheat data) has reported strong results with transformer-hybrid architectures: Springer's *Journal of Big Data* (2026, DOI prefix s40537-025-01353-w) reports a Swin+DeiT fusion architecture (HAT-Net) reaching 99.2% accuracy / 97.79% cross-validation accuracy on LWDCD2020, with a plain Swin-T fine-tune (referred to in that work as W-STNet) reaching 96.72% cross-validation accuracy — a useful point of comparison for this project's own Swin-T runs, though direct comparison requires matching train/test protocol, which the cited work does not fully disclose relative to leakage controls.

Domain-specific self-supervised pretraining for plant imagery has been explored in LeafVision (arXiv:2606.10676), which trains DINO-style self-supervised representations on 540,000 leaf images and reports substantial low-shot gains over ImageNet initialization (e.g., EfficientNet-B0 improving from 89.0% to 98.1% at 30 images/class). This project incorporates LeafVision backbone weights as an optional initialization path (`--init` flag in `train_efficientnet.py`) but has not yet run a controlled comparison against ImageNet initialization on the leakage-corrected data; this remains an open experiment (Section 10).

Vision-language approaches have also been explored for agricultural imagery; SCOLD (Swin-T + RoBERTa, trained on 186k image-caption pairs) reports strong few-shot classification when fine-tuned or linear-probed, which motivated considering it as an ensemble or comparison candidate, though it has not been integrated into this pipeline.

## 3. Research Objectives and Questions

The project is organized around four research questions, none of which should be read as pre-established conclusions:

**RQ1.** How does dataset deduplication and leakage auditing affect the reliability of plant disease classification evaluation?

**RQ2.** How effectively can deep-learning models classify Wheat and Rice leaf diseases under a controlled train/validation/test protocol?

**RQ3.** Do domain-specific pretrained representations (LeafVision/DINO) provide a measurable advantage over ImageNet initialization?

**RQ4.** Can disease predictions be combined with live weather information and a rule-based agricultural knowledge base to provide context-aware recommendations?

RQ1 is the most directly answered by the current state of the project (Section 8.3). RQ2 is partially answered for Rice and unanswered for Wheat pending the in-progress training run. RQ3 is not yet answered — the infrastructure exists but the controlled comparison has not been run. RQ4 is answered at the implementation level (Section 7) but not validated against expert agronomic ground truth.

## 4. Materials and Datasets

### 4.1 Wheat

Wheat disease data was assembled from two sources: a base dataset (11,394 images) and LWDCD2020 (Kaggle `lyxbash/lcdcd-2020-dataset`, 3,861-3,862 images), covering four target classes: Crown & Root Rot, Healthy, Leaf Rust, and Loose Smut. The combined candidate pool, after label normalization (folder-name variants such as "Crown and Root Rot," "Healthy Wheat," and rust variants such as "Brown Rust"/"Yellow Rust"/"Stem Rust" mapped to the canonical four classes; out-of-scheme labels such as Septoria and Powdery Mildew discarded), totals 15,255 candidate images.

Global MD5 hashing of this pool identifies 2,616 exact-duplicate images (a duplication rate of approximately 17%), leaving 12,639 unique images. A stratified 70/15/15 split (seed 42) over this deduplicated pool yields 8,845 training, 1,894 validation, and 1,900 test images. Hash-based verification confirms zero exact-duplicate overlap between any pair of splits.

*Methodological note (leakage-audit provenance):* an earlier version of this pipeline reported 12,286 images after deduplication and splitting rather than the 12,639 reported above. The discrepancy (347 images) was traced to a class-label matching defect in the split-writing step of `leakproof_bootstrap.py`: images correctly relabeled from the LWDCD2020 source folder name "Crown and Root Rot" to the canonical class "Crown & Root Rot" during collection had that label discarded during the split step and re-derived via a substring match against the file's source path, which fails for LWDCD-sourced files because their path retains the original folder name without the ampersand. This silently excluded 247 training, 45 validation, and 55 test images, all from the Crown & Root Rot class, from every prior Wheat training run. The defect has been fixed (the corrected label is now carried through the pipeline rather than re-derived from the path string), and the corrected split (12,639 images) has been regenerated and hash-verified, but **no model has yet been trained on the corrected data** as of this draft.

### 4.2 Rice

Rice disease data totals 6,395 images across five classes: Bacterial Blight (884), Blast (1,735), Brown Spot (947), Healthy (1,751), and Tungro (1,078). Sources include the Mendeley "Rice Leaf Disease Image Samples" dataset (Sethy et al., 2020; mirrored at `maimunul/Rice-Leaf-Disease-Classification-using-CNN`) with reported deduplication to 4,794 unique images from that source alone, supplemented with Healthy-class images from the Paddy Doctor dataset. A stratified 70/15/15 split (seed 42) yields 4,473 training, 957 validation, and 965 test images, with hash-verified zero overlap across splits.

Global MD5 hashing of the full raw Rice pool (6,395 images) was performed as part of this draft's preparation and finds **zero exact-duplicate images**. This is a narrower guarantee than a full leakage audit: it rules out byte-identical duplicates but does not rule out near-duplicates (cropped, rotated, or lightly re-encoded versions of the same photograph), which were not checked and remain an open item (Section 10).

### 4.3 Crop-type (Wheat vs. Rice) classification data

The crop-type classifier's train/validation/test splits are constructed by directly reusing the corresponding Wheat and Rice disease-classification splits (`scripts/fix_crop_split.py` hard-links each disease class's split directory into a combined crop-level directory). Consequently, the crop classifier's "test" images are the identical photographs used to evaluate the Wheat and Rice disease classifiers, not an independently sampled set. This is disclosed explicitly here because it affects how the crop-classifier's reported accuracy (Section 8.2) should be interpreted.

## 5. Methodology

### 5.1 Leakage-aware split construction

The Wheat leakage-aware pipeline (`leakproof_bootstrap.py`) follows a fixed sequence: (1) collect candidate images from all raw sources, (2) normalize class labels via an explicit mapping table, (3) compute an MD5 hash of every candidate image, (4) discard all but the first-encountered occurrence of each hash (global deduplication *before* any split boundary is drawn), (5) perform a stratified 70/15/15 split per class with a fixed random seed (42), (6) verify by hash comparison that no image hash appears in more than one split, and (7) write per-split hash manifests for audit. Rice follows the same split ratio and seed but has not undergone the equivalent multi-source deduplication step described above (Section 4.2).

### 5.2 Discovery of the original leakage problem

The leakage-aware pipeline was adopted in direct response to an earlier failure mode, not as a default design choice. An initial Wheat training run using a naively constructed split showed validation accuracy (98.10% for a Swin-T model at epoch 60) far exceeding training accuracy (72-74%, stagnant across epochs) — a training/validation gap inconsistent with normal learning dynamics and consistent with the specific signature of train/test leakage (the model partially memorizing images it had already seen, reappearing across the split boundary under a different assigned role). Investigation identified 2,599 exact MD5 duplicates distributed across the original train/validation/test assignment for the Wheat pool then in use. This finding motivated the redesign described in Section 5.1 and is treated as this paper's primary methodological narrative (Section 9).

### 5.3 Training configuration

Unless otherwise noted, models are trained with: AdamW optimizer, weight decay 1e-4, two-phase transfer learning (Phase 1: frozen backbone, 8 epochs, learning rate 1e-3; Phase 2: last 7 blocks unfrozen, learning rate 2e-5), maximum 20 epochs total, cosine annealing learning-rate schedule, dropout 0.3, label smoothing 0.1, class-balanced (inverse-frequency-weighted) loss, MixUp (alpha 0.2) and CutMix (alpha 1.0) batch augmentation each applied with probability 0.5, and an exponential moving average of model weights (decay 0.999) used for validation and checkpoint selection. Image augmentation includes resize-then-crop (256 to 224), horizontal flip, vertical flip (p=0.3), random rotation (±25°), color jitter, and ImageNet normalization. Not every reported model in Section 8 necessarily used every element of this configuration; per-run configuration should be verified against the corresponding training log before being used in a final results table.

### 5.4 Model architectures

The training codebase (`train_efficientnet.py`) supports EfficientNet-B0, EfficientNetV2-S, Swin Transformer Tiny/Small, ResNet-50, and MobileNetV3-Small as interchangeable backbones behind a common classifier head (dropout-Linear-ReLU-dropout-Linear), selected via a command-line `--arch` flag, with an optional `--init` flag to load a LeafVision DINO-pretrained backbone in place of ImageNet weights. All four production models currently reported in Section 8 (crop classifier, Wheat disease, Rice disease, Wheat severity stage) use EfficientNet-B0 specifically; the architecture-comparison experiment implied by RQ2/RQ3 has not yet been run on the leakage-corrected data.

## 6. Deep Learning Models (Current Inventory)

| Model | Task | Classes | Architecture | Status |
|---|---|---|---|---|
| Crop classifier | Wheat vs. Rice (binary) | 2 | EfficientNet-B0 | Trained; test evaluation not independent (Section 8.2) |
| Rice disease | 5-class disease classification | 5 | EfficientNet-B0 | Trained; TEST evaluation complete (Section 8.1) |
| Wheat disease | 4-class disease classification | 4 | EfficientNet-B0 | Existing checkpoint trained on leakage-affected data only; no clean-data checkpoint yet (Section 8.3) |
| Wheat severity stage | early/mid/late | 3 | EfficientNet-B0 | Trained on heuristic pseudo-labels (Section 6.1) |
| Rice severity stage | early/mid/late | 3 | — | Not shipped (Section 6.2) |

### 6.1 Wheat severity/stage

Stage labels are generated by a heuristic HSV-based lesion-area ratio, binned into terciles computed globally across the dataset (not per-split), and are therefore pseudo-labels rather than expert-annotated ground truth. This should be described in any results table as "agreement with heuristic pseudo-labels," not "accuracy," since there is no independent ground truth to validate against.

### 6.2 Rice severity/stage — deliberately not shipped

An automatic Rice severity/stage model was attempted but not shipped. After several correction passes (filtering non-photographic images, then close-up-vs-field-shot filtering, then single-leaf framing filtering, then switching from per-split to global tercile boundaries), manual visual review of contact sheets still found no consistent early-to-late progression for three of the four Rice diseases under the heuristic labeling scheme. Rather than train and ship a model on labels that had failed visual review, Rice severity staging was disabled: the inference API returns `stage: null` for Rice, and the recommendation engine falls back to general (non-stage-specific) guidance. This is a deliberate scope decision, not an unresolved defect, and is documented here to preempt it being mistaken for one.

## 7. Weather Integration and Recommendation Engine

Live weather (current conditions and a 5-day/3-hour forecast: temperature, humidity, rainfall, wind) is retrieved from Open-Meteo, with approximate user location obtained via IP geolocation (ip-api.com). Weather-driven disease risk is assessed against five rule-based threshold sets keyed to broad pathogen categories (fungal: humidity ≥80%, 22-32°C; bacterial: humidity ≥75%, rainfall >5mm; viral/vector-driven: 25-32°C; seedborne fungal: 18-24°C, humidity ≥60%; soilborne fungal: rainfall >15mm or humidity ≥85%).

The recommendation engine combines the predicted disease, predicted stage (where available), model confidence, and weather-derived risk level against a rule-based knowledge base covering all four Wheat and five Rice classes, returning an action, chemical/dosage guidance, timing, and cultural-practice recommendations. The engine correctly withholds chemical treatment recommendations where none exists agronomically — for example, no antiviral treatment is offered for Tungro, and no in-season foliar rescue is offered for Loose Smut or Crown & Root Rot once systemic, with guidance instead directed at next-season seed treatment and resistant-variety selection. This behavior is covered by unit tests.

A sourcing audit of this knowledge base (conducted as part of this draft's preparation) found 53 distinct chemical/dosage entries across the two crops' treatment plans, all internally consistent with commonly published extension-service formulation and rate ranges, but **none individually cited** to a specific extension bulletin, regulatory guidance document, or peer-reviewed field trial — the only sourcing statement in the code is a generic module-level comment. The five weather-risk thresholds are in the same position. This is treated in Section 10 as a limitation requiring resolution before any of this guidance is presented as validated rather than as implemented rule-based logic.

## 8. Results

### 8.1 Rice

Evaluated on the held-out TEST split (965 images, never used in training or model selection) using the existing EfficientNet-B0 checkpoint (94.65% best validation accuracy during training):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterial Blight | 0.928 | 0.963 | 0.945 | 134 |
| Blast | 0.976 | 0.927 | 0.951 | 261 |
| Brown Spot | 0.978 | 0.937 | 0.957 | 143 |
| Healthy | 0.970 | 0.977 | 0.974 | 264 |
| Tungro | 0.914 | 0.982 | 0.947 | 163 |

**Overall accuracy: 95.65%. Macro-F1: 95.47%.** Every class exceeds the project's own per-class acceptance threshold (precision, recall, and F1 all ≥0.90), meeting the project's internal ship gate. This is a project-specific acceptance criterion adopted for this pipeline, not a claim of universal scientific adequacy.

### 8.2 Crop-type (Wheat vs. Rice) classification

Evaluated on a stratified 534-image subsample (60 images per class-crop combination, seed 42) of the crop-level test directory using the existing EfficientNet-B0 checkpoint (98.86% best validation accuracy during training): **98.13% accuracy** (Rice: precision 0.971 / recall 0.997 / F1 0.984; Wheat: precision 0.996 / recall 0.962 / F1 0.978). As established in Section 4.3, this test set is constructed from the same photographs used to evaluate the Wheat and Rice disease classifiers rather than an independently sampled set, and this figure should not be presented as an independent generalization estimate without that caveat.

### 8.3 Wheat

**Pending.** No model has yet been trained on the leakage-corrected 12,639-image split described in Section 4.1. All historical Wheat results predate the leakage fix and are reported in Section 9 as motivating evidence for the methodology, not as findings:

| Run | Validation | Test | Note |
|---|---|---|---|
| EfficientNet-B0 (local) | 87.31% | 86.74% | Leakage-affected split |
| EfficientNet-B0 (external GPU) | 86.25% | 84.58% | Leakage-affected split |
| Swin-T, 60 epochs | ~98.10% | not evaluated | Leakage-affected split; train/val gap (72-74% train vs. 98.10% val) is the signature that triggered the leakage audit |

Training on the corrected split is in progress on external GPU hardware as of this draft. This section will be completed with per-class precision/recall/F1 against the project's 90% ship gate once that run concludes.

## 9. Discussion

The central finding to date is methodological rather than a performance number: a 17% duplication rate in the raw Wheat candidate pool was sufficient to produce a validation accuracy figure (98.10%) that looked like a strong result but was an artifact of the same or near-identical images appearing in both the training and validation partitions. The diagnostic signal was not the high validation number by itself but its relationship to training accuracy — a 24-point gap that widened rather than closed as training progressed is inconsistent with genuine generalization and should be treated as a leakage red flag in future work on this or comparable datasets, particularly when combining multiple third-party sources (as this project did with its base Wheat dataset and LWDCD2020) that may share upstream provenance.

The Rice results (Section 8.1), representing the only complete, leakage-controlled classification result available at the time of this draft, meet the project's per-class ship gate on a genuinely held-out test set and are the strongest evidence to date that the underlying architecture and training recipe (EfficientNet-B0, two-phase transfer learning, class-balanced loss, MixUp/CutMix, EMA) are adequate for this task class, conditional on clean data. Whether the same holds for Wheat — a four-class problem drawn from a more heterogeneous, multi-source pool with a higher measured duplication rate than Rice — is the open question this paper cannot yet answer.

## 10. Limitations

- **Wheat has no clean-data result as of this draft** (Section 8.3); every claim about Wheat model adequacy is deferred pending that run.
- **Rice has been checked only for exact (MD5) duplication**, not near-duplication; a perceptual-hash or embedding-similarity audit has not been performed and could reveal additional leakage risk not captured by exact hashing.
- **The crop-type classifier's test set is not independent** of the Wheat/Rice disease-classifier test sets (Section 4.3, 8.2); a genuinely separate test set has not yet been constructed.
- **Severity/stage labels for Wheat are heuristic pseudo-labels** (Section 6.1), not expert-annotated ground truth, and are reported accordingly.
- **Recommendation-engine dosages and weather-risk thresholds are uncited** (Section 7); they should be understood as implemented rule-based logic, not as agronomically validated prescriptions, until sourced.
- **No independent field or out-of-distribution evaluation has been performed.** All reported numbers reflect in-distribution, controlled-split performance; nothing in this paper should be read as a claim of field robustness or deployment readiness.
- **The LeafVision domain-specific pretraining comparison (RQ3) has not been run** on the leakage-corrected data; the infrastructure to do so exists but the controlled experiment does not yet exist.
- **Location handling uses coarse IP geolocation**, and no explicit deployment region has been established for the agricultural recommendations; regional applicability of the treatment guidance should not be assumed beyond what is disclosed in Section 7.

## 11. Future Work

In priority order given the current project state: (1) complete Wheat training and ship-gate evaluation on the corrected split; (2) if time and resources allow, run the architecture comparison (EfficientNet-B0 vs. EfficientNetV2-S vs. Swin-T vs. LeafVision-initialized variants) implied by RQ2/RQ3 on both crops; (3) construct a genuinely independent crop-classifier test set; (4) perform near-duplicate detection on Rice; (5) source and cite the recommendation-engine knowledge base; (6) pursue an independent field/out-of-distribution evaluation set (e.g., PlantDoc-style field imagery) to separate controlled-split performance from real-world generalization, structured explicitly as two parallel metrics (clean-test F1 vs. external-field F1) rather than a single blended number; (7) consider confidence calibration and an abstention mechanism for low-confidence predictions before any deployment scenario; (8) consider expert-annotated severity labels as a replacement for the current heuristic pseudo-labels.

## 12. Conclusion

*(To be completed once Section 8.3 is filled in. A conclusion drawn before the Wheat result exists would necessarily overstate what has been established.)*

## References

*(Partial list reflecting sources explicitly used or cited during pipeline development; a full literature review pass is still needed before submission. Entries are given with the identifiers available in the project's own records; full bibliographic details — venue, page numbers, complete author lists — should be verified against the original source before this paper is finalized, as they have not been independently re-verified here.)*

1. Sethy, P. K., et al. (2020). Rice Leaf Disease Image Samples. Mendeley Data.
2. LWDCD2020 (Large Wheat Disease Classification Dataset). Kaggle dataset `lyxbash/lcdcd-2020-dataset`.
3. Paddy Doctor dataset (Healthy-class Rice images). `ai-agriculture-circuits-and-systems/paddy_disease_classification`.
4. LeafVision: domain-specific self-supervised (DINO) representation learning for leaf imagery. arXiv:2606.10676.
5. HAT-Net / W-STNet / W-DENet transformer architectures evaluated on LWDCD2020. *Journal of Big Data*, Springer, DOI prefix s40537-025-01353-w.
6. SCOLD vision-language model for agricultural imagery. Hugging Face `enalis/scold`.
7. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
8. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*.
9. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
10. Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
11. Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). *ICCV*.

---

*Draft prepared 2026-08-30. Sections 1, 3, 4, 5, 6.2, 7 (implementation), 8.1, 8.2 reflect verified current repository state as of this date. Sections 8.3, 9 (Wheat-dependent portions), Abstract, and Conclusion are explicitly incomplete pending the Wheat training run described in Section 8.3. Section 2 (Related Work) requires a fuller literature search before this can be considered submission-ready.*
