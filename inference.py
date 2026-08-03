"""
CropGuard AI - Model Inference Module
Loads trained models and runs predictions.
"""

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


def load_model(model_path: Path, num_classes: int, class_names: List[str]) -> Tuple[nn.Module, Dict]:
    """Load a trained EfficientNet-B0 model from checkpoint."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    
    pred_class = idx_to_class[pred_idx.item()]
    confidence = conf.item()
    
    all_probs = {idx_to_class[i]: probs[i].item() for i in range(len(probs))}
    
    return pred_class, confidence, all_probs


class ModelManager:
    """Manages all 4 models with lazy loading (rice_stage removed)."""
    
    LOADERS = {
        "crop": load_model,
        "rice": load_model,
        "wheat": load_model,
        "wheat_stage": load_model,
    }

    REQUIRED_MODELS = {
        "crop": "crop_efficientnet_best.pt",
        "rice": "rice_efficientnet_best.pt",
        "wheat": "wheat_efficientnet_best.pt",
        "wheat_stage": "wheat_stage_efficientnet_best.pt",
        # rice_stage intentionally omitted -- see README "Known Limitations"
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
    
    def predict_stage(self, crop: str, image: Image.Image) -> Optional[dict]:
        """
        Returns a stage prediction dict for wheat, or None for rice.
        None is a deliberate, permanent signal -- not a missing-model bug.
        Callers must handle it, not treat it as an error.
        """
        if crop == "wheat":
            logits = self._predict_stage_logits(image)
            probs = torch.softmax(logits, dim=1)[0]
            idx = probs.argmax().item()
            return {
                "stage": ["early", "mid", "late"][idx],
                "confidence": round(probs[idx].item(), 4),
                "caveat": "Derived from a heuristic lesion-severity proxy, not "
                          "expert-annotated ground truth -- treat as a second "
                          "opinion, not a verified diagnosis.",
            }
        elif crop == "rice":
            return None
        else:
            raise ValueError(f"Unknown crop: {crop}")
    
    def _predict_stage_logits(self, image: Image.Image):
        """Internal helper to get logits from wheat stage model."""
        model, _ = self.get_wheat_stage_classifier()
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
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