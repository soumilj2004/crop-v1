"""
CropGuard AI - FastAPI Backend
Endpoints: /health, /predict, /recommend, /analyze
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
import io
from PIL import Image

from inference import get_model_manager, ModelManager
from recommend.engine import get_engine, RecommendationResult, WeatherData
from weather.client import fetch_live_weather, WeatherClient

app = FastAPI(title="CropGuard AI", version="1.0.0")

# Mount static files (frontend)
frontend_dir = Path(__file__).parent / "static"
frontend_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Global model manager
model_manager: Optional[ModelManager] = None


@app.on_event("startup")
async def startup_event():
    global model_manager
    model_manager = get_model_manager("models")
    print("Models loaded:", model_manager.health_check())


@app.get("/health")
async def health_check():
    """Liveness + model availability check."""
    if model_manager is None:
        return {"status": "initializing", "models": {}}
    return {
        "status": "ok",
        "models": model_manager.health_check(),
        "device": str(model_manager._models.get("crop_classifier", {}).next().device) if model_manager._models else "unknown"
    }


class PredictRequest(BaseModel):
    crop: str  # "rice" or "wheat"


class PredictResponse(BaseModel):
    crop: str
    disease: str
    confidence: float
    all_probabilities: Dict[str, float]


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), crop: str = Form(...)):
    """Predict disease for a known crop."""
    if crop not in ["rice", "wheat"]:
        raise HTTPException(status_code=400, detail="Crop must be 'rice' or 'wheat'")
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    if crop == "rice":
        disease, confidence, probs = model_manager.predict_rice_disease(image)
    else:
        disease, confidence, probs = model_manager.predict_wheat_disease(image)
    
    return PredictResponse(
        crop=crop,
        disease=disease,
        confidence=confidence,
        all_probabilities=probs
    )


class RecommendRequest(BaseModel):
    crop: str
    disease: str
    stage: Optional[str] = None
    confidence: float
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_kph: Optional[float] = None


class RecommendResponse(BaseModel):
    crop: str
    disease: str
    pathogen: str
    stage: Optional[str]
    confidence: float
    risk_level: str
    risk_reason: str
    treatment: Dict[str, Any]
    weather: Optional[Dict[str, Any]]
    caveats: List[str]


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Generate recommendation from disease+stage+weather (no image)."""
    weather = None
    if req.temperature_c is not None:
        weather = WeatherData(
            temperature_c=req.temperature_c,
            humidity_pct=req.humidity_pct or 50,
            rainfall_mm=req.rainfall_mm or 0,
            wind_kph=req.wind_kph or 0
        )
    
    result = get_engine().get_recommendation(
        crop=req.crop,
        disease_class=req.disease,
        stage=req.stage,
        confidence=req.confidence,
        weather=weather
    )
    
    return RecommendResponse(
        crop=result.crop,
        disease=result.disease_name,
        pathogen=result.pathogen,
        stage=result.stage.value if result.stage else None,
        confidence=result.confidence,
        risk_level=result.risk_level.value,
        risk_reason=result.risk_reason,
        treatment={
            "action": result.treatment.action,
            "chemical": result.treatment.chemical,
            "dosage": result.treatment.dosage,
            "timing": result.treatment.timing,
            "cultural": result.treatment.cultural
        },
        weather={
            "temperature_c": weather.temperature_c if weather else None,
            "humidity_pct": weather.humidity_pct if weather else None,
            "rainfall_mm": weather.rainfall_mm if weather else None,
            "wind_kph": weather.wind_kph if weather else None,
            "source": weather.source if weather else "none"
        } if weather else None,
        caveats=result.caveats
    )


class AnalyzeResponse(BaseModel):
    # Crop detection
    detected_crop: str
    crop_confidence: float
    crop_probs: Dict[str, float]
    # Disease detection
    disease: str
    disease_confidence: float
    disease_probs: Dict[str, float]
    # Stage (if diseased)
    stage: Optional[str]
    stage_confidence: Optional[float]
    stage_probs: Optional[Dict[str, float]]
    # Weather
    weather: Optional[Dict[str, Any]]
    # Recommendation
    risk_level: str
    risk_reason: str
    treatment: Dict[str, Any]
    caveats: List[str]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Full automated pipeline: image -> crop -> disease -> stage -> weather -> recommendation.
    Zero user input required beyond the image.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # 1. Detect crop
    crop, crop_conf, crop_probs = model_manager.predict_crop(image)
    
    # 2. Detect disease (crop-specific)
    if crop == "rice":
        disease, disease_conf, disease_probs = model_manager.predict_rice_disease(image)
    else:
        disease, disease_conf, disease_probs = model_manager.predict_wheat_disease(image)
    
    # 3. Check if healthy
    is_healthy = disease == "Healthy"
    
    # 4. Predict stage (if diseased)
    stage = None
    stage_conf = None
    stage_probs = None
    
    if not is_healthy:
        stage_result = model_manager.predict_stage(crop, image)
        if stage_result:
            stage = stage_result["stage"]
            stage_conf = stage_result["confidence"]
            # stage_probs not computed here to keep it simple; could add if needed
    
    # 5. Fetch live weather
    weather_data = fetch_live_weather()
    
    # 6. Generate recommendation
    result = get_engine().get_recommendation(
        crop=crop,
        disease_class=disease,
        stage=stage,
        confidence=disease_conf,
        weather=weather_data
    )
    
    return AnalyzeResponse(
        detected_crop=crop,
        crop_confidence=crop_conf,
        crop_probs=crop_probs,
        disease=disease,
        disease_confidence=disease_conf,
        disease_probs=disease_probs,
        stage=stage,
        stage_confidence=stage_conf,
        stage_probs=stage_probs,
        weather={
            "temperature_c": weather_data.temperature_c,
            "humidity_pct": weather_data.humidity_pct,
            "rainfall_mm": weather_data.rainfall_mm,
            "wind_kph": weather_data.wind_kph,
            "source": weather_data.source
        } if weather_data else None,
        risk_level=result.risk_level.value,
        risk_reason=result.risk_reason,
        treatment={
            "action": result.treatment.action,
            "chemical": result.treatment.chemical,
            "dosage": result.treatment.dosage,
            "timing": result.treatment.timing,
            "cultural": result.treatment.cultural
        },
        caveats=result.caveats
    )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML."""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>CropGuard AI</h1><p>Frontend not built yet</p>", status_code=200)


@app.get("/diseases")
async def list_diseases(crop: str):
    """List supported diseases for a crop."""
    if crop == "rice":
        return {
            "crop": "rice",
            "diseases": ["Bacterial Blight", "Blast", "Healthy", "Tungro"]
        }
    elif crop == "wheat":
        return {
            "crop": "wheat",
            "diseases": ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"]
        }
    else:
        raise HTTPException(status_code=400, detail="Crop must be 'rice' or 'wheat'")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)