"""
CropGuard AI - Model Inference Module
Loads trained models and runs predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

ARCH_INPUT_SIZE = {
    "efficientnet_b0": 224,
    "efficientnet_v2_s": 224,
    "mobilenet_v3_small": 128,
    "resnet50": 224,
    "swin_t": 224,
    "swin_s": 224,
}


def _build_arch(arch: str, num_classes: int):
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        model.input_size = ARCH_INPUT_SIZE.get(arch, INPUT_SIZE)
        return model
    elif arch in ("swin_t", "swin_s"):
        model = models.swin_t(weights=None) if arch == "swin_t" else models.swin_s(weights=None)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        model.input_size = ARCH_INPUT_SIZE.get(arch, INPUT_SIZE)
        return model
    else:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    model.input_size = ARCH_INPUT_SIZE.get(arch, INPUT_SIZE)
    return model


def _transform_for(model: nn.Module):
    size = getattr(model, "input_size", INPUT_SIZE)
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(model_path: Path, num_classes: int, class_names: List[str]) -> Tuple[nn.Module, Dict]:
    """Load a trained model from checkpoint (arch-aware)."""
    raw = torch.load(model_path, map_location=DEVICE)
    if isinstance(raw, dict) and "arch" in raw and "state_dict" in raw:
        arch, state_dict = raw["arch"], raw["state_dict"]
    else:
        arch, state_dict = "efficientnet_b0", raw
    model = _build_arch(arch, num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    idx_to_class = {i: name for i, name in enumerate(class_names)}
    return model, idx_to_class


def load_mobilenet_model(model_path: Path, num_classes: int, class_names: List[str]) -> Tuple[nn.Module, Dict]:
    """Load a legacy MobileNetV2 model (used for wheat_stage)."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    return model, idx_to_class


def predict_image(model: nn.Module, image: Image.Image, idx_to_class: Dict) -> Tuple[str, float, Dict[str, float]]:
    """Run inference on a single image."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = _transform_for(model)(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    
    pred_class = idx_to_class[pred_idx.item()]
    confidence = conf.item()
    
    all_probs = {idx_to_class[i]: probs[i].item() for i in range(len(probs))}
    
    return pred_class, confidence, all_probs



def estimate_rice_severity(image: Image.Image) -> dict:
    """
    Heuristic severity (stage) estimate for rice, based on HSV lesion-area
    analysis -- NOT a trained classifier.

    Why a heuristic instead of a trained model: no labeled severity
    (early/mid/late) dataset exists for rice in this project -- only
    disease-class labels (Bacterial Blight / Blast / Brown Spot / Healthy /
    Tungro). Training a supervised stage classifier would require either
    fabricating labels or sourcing/annotating a new dataset, neither of
    which is honest to do "ASAP". This heuristic gives a real, working
    stage signal today; see recommend/engine.py's caveat text, which
    already anticipated exactly this method.

    Method:
      1. Resize to a fixed size for speed/consistency.
      2. Isolate leaf-tissue pixels (moderate-to-high saturation) from
         background, falling back to the whole frame if isolation fails
         (e.g. a tight macro crop with little background).
      3. Within leaf tissue, flag pixels outside the healthy-green hue
         band, or very dark (necrotic / blast centers / bacterial ooze
         shadow), as lesion pixels.
      4. affected_ratio = lesion_pixels / leaf_pixels.
      5. Bucket into early / mid / late via fixed thresholds.

    Returns the same shape as the wheat stage model's output so callers
    (ModelManager.predict_stage, api/main.py) don't need to branch on
    "is this a real model or a heuristic".
    """
    img = image.convert("RGB").resize((256, 256))
    hsv = np.array(img.convert("HSV")).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Leaf-tissue mask: reasonably saturated & not near-black/near-white background
    leaf_mask = (s > 40) & (v > 25) & (v < 250)
    leaf_px = int(leaf_mask.sum())
    if leaf_px < 200:
        leaf_mask = np.ones_like(leaf_mask, dtype=bool)
        leaf_px = int(leaf_mask.size)

    # PIL HSV hue is 0-255 (mapped from 0-360deg). Healthy green leaf hue
    # band (~70-170deg) maps to roughly 50-125 on the PIL 0-255 scale.
    healthy_green = (h >= 45) & (h <= 125) & (s > 60)
    very_dark = v < 60
    lesion_mask = leaf_mask & (~healthy_green | very_dark)
    lesion_px = int(lesion_mask.sum())

    affected_ratio = lesion_px / max(leaf_px, 1)

    if affected_ratio < 0.14:
        stage = "early"
    elif affected_ratio < 0.32:
        stage = "mid"
    else:
        stage = "late"

    thresholds = {"early": (0.0, 0.14), "mid": (0.14, 0.32), "late": (0.32, 1.0)}
    lo, hi = thresholds[stage]
    span = (hi - lo) or 1.0
    centered = 1 - abs(((min(max(affected_ratio, lo), hi) - lo) / span) - 0.5) * 2
    confidence = round(0.5 + 0.4 * max(0.0, min(1.0, centered)), 4)

    return {
        "stage": stage,
        "confidence": confidence,
        "affected_ratio": round(affected_ratio, 4),
        "caveat": (
            "Estimated from heuristic HSV lesion-area analysis, not a trained "
            "classifier or expert-annotated ground truth -- no labeled rice "
            "severity dataset exists for this project. Treat as a rough "
            "visual indicator only, not a diagnosis."
        ),
    }


class ModelManager:
    """Manages up to 5 models with lazy loading (rice_stage optional -- see REQUIRED_MODELS)."""
    
    LOADERS = {
        "crop": load_model,
        "rice": load_model,
        "wheat": load_model,
        "wheat_stage": load_model,
        "rice_stage": load_model,
    }

    REQUIRED_MODELS = {
        "crop": "crop_efficientnet_best.pt",
        "rice": "rice_efficientnet_best.pt",
        "wheat": "wheat_efficientnet_best.pt",
        "wheat_stage": "wheat_stage_efficientnet_best.pt",
        # rice_stage is OPTIONAL: trained via
        # `python scripts/train_efficientnet.py --model rice_stage --epochs 20`
        # against data/split/rice_stage (early/mid/late, HSV-heuristic-derived
        # labels -- see scripts/generate_hsv_stage_labels.py). Until that
        # checkpoint exists on disk, predict_stage() falls back to
        # estimate_rice_severity() (the live HSV heuristic) automatically --
        # this key is checked for existence, not required at startup.
        "rice_stage": "rice_stage_efficientnet_best.pt",
    }
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._models = {}
        self._class_maps = {}
        self._loaded = set()
    
    def _load_model(self, model_key: str, filename: str, num_classes: int, class_names: List[str]):
        """Load a specific model if not already loaded."""
        if model_key in self._loaded:
            return
        
        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        loader = self.LOADERS.get(model_key, load_model)
        model, idx_to_class = loader(model_path, num_classes, class_names)
        self._models[model_key] = model
        self._class_maps[model_key] = idx_to_class
        self._loaded.add(model_key)
        print(f"Loaded model: {model_key}")
    
    def get_crop_classifier(self):
        """Rice vs Wheat binary classifier."""
        self._load_model("crop", self.REQUIRED_MODELS["crop"], 2, ["rice", "wheat"])
        return self._models["crop"], self._class_maps["crop"]
    
    def get_rice_classifier(self):
        """Rice disease classifier (5 classes including Brown Spot)."""
        classes = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
        self._load_model("rice", self.REQUIRED_MODELS["rice"], 5, classes)
        return self._models["rice"], self._class_maps["rice"]
    
    def get_wheat_classifier(self):
        """Wheat disease classifier (4 classes including Healthy)."""
        classes = ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"]
        self._load_model("wheat", self.REQUIRED_MODELS["wheat"], 4, classes)
        return self._models["wheat"], self._class_maps["wheat"]
    
    def get_wheat_stage_classifier(self):
        """Wheat severity stage (early, mid, late)."""
        classes = ["early", "mid", "late"]
        self._load_model("wheat_stage", self.REQUIRED_MODELS["wheat_stage"], 3, classes)
        return self._models["wheat_stage"], self._class_maps["wheat_stage"]

    def has_rice_stage_model(self) -> bool:
        """True once rice_stage_efficientnet_best.pt has been trained and placed in models/."""
        return (self.models_dir / self.REQUIRED_MODELS["rice_stage"]).exists()

    def get_rice_stage_classifier(self):
        """Rice severity stage (early, mid, late) -- trained classifier, once available."""
        classes = ["early", "mid", "late"]
        self._load_model("rice_stage", self.REQUIRED_MODELS["rice_stage"], 3, classes)
        return self._models["rice_stage"], self._class_maps["rice_stage"]
    
    def predict_crop(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        model, idx_to_class = self.get_crop_classifier()
        return predict_image(model, image, idx_to_class)
    
    def predict_rice_disease(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        model, idx_to_class = self.get_rice_classifier()
        return predict_image(model, image, idx_to_class)
    
    def predict_wheat_disease(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        model, idx_to_class = self.get_wheat_classifier()
        return predict_image(model, image, idx_to_class)
    
    def predict_wheat_stage(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        model, idx_to_class = self.get_wheat_stage_classifier()
        return predict_image(model, image, idx_to_class)

    def predict_rice_stage(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        model, idx_to_class = self.get_rice_stage_classifier()
        return predict_image(model, image, idx_to_class)
    
    def predict_stage(self, crop: str, image: Image.Image) -> Optional[dict]:
        """
        Returns a stage prediction dict for both wheat and rice.

        Wheat: a trained EfficientNet classifier (wheat_stage_efficientnet_best.pt)
        over real early/mid/late labels.

        Rice: no labeled severity dataset exists for this project, so this
        falls back to estimate_rice_severity() -- a heuristic HSV lesion-area
        estimate, not a trained classifier. It is a real, working signal, just
        a different (and less rigorous) kind of one. Both paths return the
        same dict shape so callers don't need to branch on the method.
        """
        if crop == "wheat":
            logits = self._predict_stage_logits(image)
            probs = torch.softmax(logits, dim=1)[0]
            idx = probs.argmax().item()
            return {
                "stage": ["early", "mid", "late"][idx],
                "confidence": round(probs[idx].item(), 4),
                "caveat": "Derived from a trained EfficientNet stage classifier "
                          "over labeled early/mid/late wheat images.",
            }
        elif crop == "rice":
            if self.has_rice_stage_model():
                stage, confidence, _ = self.predict_rice_stage(image)
                return {
                    "stage": stage,
                    "confidence": round(confidence, 4),
                    "caveat": "Derived from a trained EfficientNet stage classifier over "
                              "HSV-heuristic-derived labels (data/split/rice_stage) -- not "
                              "expert-annotated ground truth, but a real trained model, "
                              "same as wheat's stage classifier.",
                }
            # Fallback while rice_stage hasn't been trained/placed yet.
            return estimate_rice_severity(image)
        else:
            raise ValueError(f"Unknown crop: {crop}")
    
    def _predict_stage_logits(self, image: Image.Image):
        """Internal helper to get logits from wheat stage model."""
        model, _ = self.get_wheat_stage_classifier()
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = _transform_for(model)(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return model(tensor)

    def health_check(self) -> Dict[str, bool]:
        """Check which models are available on disk."""
        return {key: (self.models_dir / fname).exists() for key, fname in self.REQUIRED_MODELS.items()}


# Global instance
_model_manager: Optional[ModelManager] = None

def get_model_manager(models_dir: str = "models") -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(Path(models_dir))
    return _model_manager


if __name__ == "__main__":
    # Quick test if models exist
    mm = get_model_manager("models")
    print("Model availability:", mm.health_check())