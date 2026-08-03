import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
sys.path.insert(0, 'C:\\CropGuardAI')

from cropguard_ai.inference import get_model_manager
from cropguard_ai.recommend.engine import get_engine
from cropguard_ai.weather.client import fetch_live_weather

BASE = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model manager
mm = get_model_manager(str(BASE / "models"))
print("Model availability:", mm.health_check())

# Test full pipeline on a rice image
img = Image.open(BASE / 'data/split/rice/train/Blast/100004.jpg').convert('RGB')

# 1. Crop detection
crop, crop_conf, crop_probs = mm.predict_crop(img)
print(f"\n1. Crop detection: {crop} (conf: {crop_conf:.3f})")

# 2. Disease detection
if crop == 'rice':
    disease, disease_conf, disease_probs = mm.predict_rice_disease(img)
else:
    disease, disease_conf, disease_probs = mm.predict_wheat_disease(img)
print(f"2. Disease: {disease} (conf: {disease_conf:.3f})")

# 3. Stage detection (if not healthy)
if disease != 'Healthy':
    stage_result = mm.predict_stage(crop, img)
    if stage_result:
        stage = stage_result['stage']
        stage_conf = stage_result['confidence']
        print(f"3. Stage: {stage} (conf: {stage_conf:.3f})")
    else:
        stage = None
        print("3. Stage: N/A (rice stage disabled)")
else:
    stage = None
    print("3. Stage: N/A (healthy)")

# 4. Weather
weather = fetch_live_weather()
if weather:
    print(f"4. Weather: {weather.temperature_c:.1f}°C, {weather.humidity_pct:.0f}% humidity, {weather.rainfall_mm:.1f}mm rain")
else:
    print("4. Weather: unavailable")
    from cropguard_ai.weather.client import WeatherData
    weather = WeatherData(temperature_c=25, humidity_pct=70, rainfall_mm=5, wind_kph=5)

# 5. Recommendation
engine = get_engine()
result = engine.get_recommendation(
    crop=crop,
    disease_class=disease,
    stage=stage,
    confidence=disease_conf,
    weather=weather
)

print(f"\n5. Recommendation:")
print(f"   Disease: {result.disease_name} ({result.pathogen})")
print(f"   Stage: {result.stage.value if result.stage else 'N/A'}")
print(f"   Risk: {result.risk_level.value} - {result.risk_reason}")
print(f"   Action: {result.treatment.action}")
print(f"   Chemical: {result.treatment.chemical}")
print(f"   Dosage: {result.treatment.dosage}")
print(f"   Timing: {result.treatment.timing}")
print(f"   Cultural: {result.treatment.cultural}")
print(f"   Caveats: {result.caveats}".encode('ascii', 'ignore').decode())