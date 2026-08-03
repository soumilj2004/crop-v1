"""
DEPRECATED — Use recommend.engine instead.

This file is a refactoring prototype that was never wired into the API.
All treatment data lives in knowledge.knowledge_base (the canonical data layer).
Recommendation logic lives in recommend.engine (the canonical engine, which
imports data from knowledge.knowledge_base).

This file is kept for reference only and should not be imported by any
production code. All paths converge on recommend.engine.get_engine().
"""

from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

from knowledge.knowledge_base import (
    get_knowledge_base, DISEASE_KEY_MAP, DiseaseInfo, TreatmentPlan, Stage, DiseaseType
)


class RiskLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class WeatherContext:
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_kph: Optional[float] = None
    source: str = "live_api"  # "live_api", "mock", "none"

    def is_complete(self) -> bool:
        return all(v is not None for v in [self.temperature_c, self.humidity_pct, self.rainfall_mm])


@dataclass
class RecommendationResult:
    crop: str
    disease: str
    disease_display: str
    pathogen: str
    disease_type: str
    stage: Optional[str]
    confidence: float
    risk_level: RiskLevel
    risk_reason: str
    treatment: TreatmentPlan
    weather: WeatherContext
    typical_symptoms: list[str]
    spread_conditions: str
    caveat: Optional[str] = None


# Weather risk rules per disease type
WEATHER_RISK_RULES = {
    DiseaseType.FUNGAL: {
        "threshold": lambda t, h, r: (h is not None and h >= 80 and t is not None and 22 <= t <= 32),
        "message": "High humidity and favorable temperature for fungal spore germination and spread"
    },
    DiseaseType.BACTERIAL: {
        "threshold": lambda t, h, r: (h is not None and h >= 75 and r is not None and r > 5),
        "message": "High humidity with rainfall favors bacterial ooze and splash dispersal"
    },
    DiseaseType.VIRAL: {
        "threshold": lambda t, h, r: (t is not None and 25 <= t <= 32),
        "message": "Warm temperatures favor leafhopper vector population growth and virus transmission"
    },
    DiseaseType.SEEDBORNE_FUNGAL: {
        "threshold": lambda t, h, r: (t is not None and 18 <= t <= 24 and h is not None and h >= 60),
        "message": "Moderate temperatures and humidity at flowering favor spore release and seed infection"
    },
    DiseaseType.SOILBORNE_FUNGAL: {
        "threshold": lambda t, h, r: (r is not None and r > 15) or (h is not None and h >= 85),
        "message": "Excessive rainfall or very high humidity creates waterlogged conditions favoring root/crown pathogens"
    },
}


def assess_weather_risk(disease_type: DiseaseType, weather: WeatherContext) -> tuple[RiskLevel, str]:
    """Assess disease risk based on current weather conditions."""
    if not weather.is_complete():
        return RiskLevel.UNKNOWN, "Weather data incomplete; risk assessment unavailable"
    
    rule = WEATHER_RISK_RULES.get(disease_type)
    if not rule:
        return RiskLevel.UNKNOWN, "No weather risk rule for this disease type"
    
    t = weather.temperature_c
    h = weather.humidity_pct
    r = weather.rainfall_mm
    
    try:
        if rule["threshold"](t, h, r):
            return RiskLevel.HIGH, rule["message"]
        else:
            # Check moderate conditions
            if disease_type == DiseaseType.FUNGAL:
                if h is not None and h >= 70 and t is not None and 18 <= t <= 35:
                    return RiskLevel.MODERATE, "Moderate humidity and temperature; some fungal risk"
            elif disease_type == DiseaseType.BACTERIAL:
                if h is not None and h >= 65:
                    return RiskLevel.MODERATE, "Elevated humidity; bacterial spread possible"
            elif disease_type == DiseaseType.VIRAL:
                if t is not None and 20 <= t <= 35:
                    return RiskLevel.MODERATE, "Temperatures favor vector activity"
            return RiskLevel.LOW, "Current conditions not highly favorable for disease spread"
    except Exception:
        return RiskLevel.UNKNOWN, "Error evaluating weather risk"


def get_recommendation(
    crop: str,
    disease_class: str,
    stage: Optional[str],
    confidence: float,
    weather: Optional[WeatherContext] = None
) -> RecommendationResult:
    """
    Generate full treatment recommendation.
    
    Args:
        crop: "rice" or "wheat"
        disease_class: Model output class name (e.g., "Bacterial Blight", "Leaf Rust")
        stage: "early", "mid", "late", or None (for Healthy)
        confidence: Model confidence (0-1)
        weather: Optional WeatherContext
    
    Returns:
        RecommendationResult with all details
    """
    crop = crop.lower()
    if crop not in ["rice", "wheat"]:
        raise ValueError(f"Unsupported crop: {crop}")
    
    # Map model class to knowledge base key
    key_map = DISEASE_KEY_MAP.get(crop, {})
    kb_key = key_map.get(disease_class)
    if not kb_key:
        raise ValueError(f"Unknown disease class for {crop}: {disease_class}")
    
    kb = get_knowledge_base(crop)
    disease_info: DiseaseInfo = kb[kb_key]
    
    # Handle Healthy - no stage, no risk
    if kb_key == "healthy":
        stage_enum = None
        treatment = disease_info.treatments.get(Stage.EARLY, list(disease_info.treatments.values())[0])
        weather = weather or WeatherContext(source="none")
        return RecommendationResult(
            crop=crop,
            disease=kb_key,
            disease_display=disease_info.display_name,
            pathogen=disease_info.pathogen,
            disease_type=disease_info.disease_type.value,
            stage=None,
            confidence=confidence,
            risk_level=RiskLevel.LOW,
            risk_reason="No disease detected",
            treatment=treatment,
            weather=weather,
            typical_symptoms=disease_info.typical_symptoms,
            spread_conditions=disease_info.spread_conditions,
            caveat=None
        )
    
    # Parse stage
    if stage:
        stage = stage.lower()
        try:
            stage_enum = Stage(stage)
        except ValueError:
            stage_enum = Stage.EARLY  # default
    else:
        stage_enum = Stage.EARLY  # default if missing
    
    treatment = disease_info.treatments.get(stage_enum, disease_info.treatments[Stage.EARLY])
    
    # Weather risk
    weather = weather or WeatherContext(source="none")
    risk_level, risk_reason = assess_weather_risk(disease_info.disease_type, weather)
    
    # Caveat for stage classifiers (heuristic labels)
    caveat = ("Severity stage estimated from heuristic image analysis (HSV lesion ratio terciles), "
              "not expert-annotated ground truth. Treat as indicative only.")
    
    return RecommendationResult(
        crop=crop,
        disease=kb_key,
        disease_display=disease_info.display_name,
        pathogen=disease_info.pathogen,
        disease_type=disease_info.disease_type.value,
        stage=stage_enum.value if stage_enum else None,
        confidence=confidence,
        risk_level=risk_level,
        risk_reason=risk_reason,
        treatment=treatment,
        weather=weather,
        typical_symptoms=disease_info.typical_symptoms,
        spread_conditions=disease_info.spread_conditions,
        caveat=caveat
    )


def format_recommendation_text(result: RecommendationResult) -> str:
    """Format recommendation as human-readable text."""
    lines = []
    lines.append(f"=== CropGuard AI Recommendation ===")
    lines.append(f"Crop: {result.crop.capitalize()}")
    lines.append(f"Disease: {result.disease_display} ({result.pathogen})")
    lines.append(f"Type: {result.disease_type.replace('_', ' ').title()}")
    lines.append(f"Confidence: {result.confidence:.1%}")
    if result.stage:
        lines.append(f"Severity Stage: {result.stage.capitalize()}")
    lines.append(f"")
    lines.append(f"Risk Level: {result.risk_level.value.upper()}")
    lines.append(f"Risk Reason: {result.risk_reason}")
    lines.append(f"")
    lines.append(f"--- Treatment ---")
    lines.append(f"Action: {result.treatment.action}")
    lines.append(f"Chemical: {result.treatment.chemical}")
    lines.append(f"Dosage: {result.treatment.dosage}")
    lines.append(f"Timing: {result.treatment.timing}")
    lines.append(f"")
    lines.append(f"--- Cultural Controls ---")
    for c in result.treatment.cultural:
        lines.append(f"  • {c}")
    lines.append(f"")
    lines.append(f"--- Weather Context ---")
    if result.weather.is_complete():
        lines.append(f"Temperature: {result.weather.temperature_c}°C")
        lines.append(f"Humidity: {result.weather.humidity_pct}%")
        lines.append(f"Rainfall: {result.weather.rainfall_mm} mm")
        if result.weather.wind_kph:
            lines.append(f"Wind: {result.weather.wind_kph} km/h")
    else:
        lines.append(f"Weather data: {result.weather.source} (incomplete)")
    if result.caveat:
        lines.append(f"")
        lines.append(f"⚠ CAVEAT: {result.caveat}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    weather = WeatherContext(temperature_c=28, humidity_pct=85, rainfall_mm=10)
    result = get_recommendation("rice", "Blast", "early", 0.92, weather)
    print(format_recommendation_text(result))
    
    print("\n" + "="*50 + "\n")
    
    # Test loose smut (no foliar treatment)
    result2 = get_recommendation("wheat", "Loose Smut", "mid", 0.88, weather)
    print(format_recommendation_text(result2))