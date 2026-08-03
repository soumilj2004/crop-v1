"""
CropGuard AI - Recommendation Engine
Pure Python, no ML dependencies. Deterministic, fully testable.

Knowledge data lives in knowledge.knowledge_base (the canonical data layer).
This engine imports from there and adds risk assessment + recommendation logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
try:
    from cropguard_ai.weather.client import WeatherData
except ImportError:
    from weather.client import WeatherData

try:
    from cropguard_ai.knowledge.knowledge_base import (
        Stage, DiseaseType, TreatmentPlan, DiseaseInfo,
        RICE_KNOWLEDGE, WHEAT_KNOWLEDGE, DISEASE_KEY_MAP,
    )
except ImportError:
    from knowledge.knowledge_base import (
        Stage, DiseaseType, TreatmentPlan, DiseaseInfo,
        RICE_KNOWLEDGE, WHEAT_KNOWLEDGE, DISEASE_KEY_MAP,
    )


class RiskLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class RecommendationResult:
    crop: str
    disease_key: str
    disease_name: str
    pathogen: str
    disease_type: str
    typical_symptoms: List[str]
    stage: Stage
    confidence: float
    risk_level: RiskLevel
    risk_reason: str
    treatment: TreatmentPlan
    weather: Optional[WeatherData]
    caveats: List[str]


RICE_CLASS_MAP = DISEASE_KEY_MAP["rice"]
WHEAT_CLASS_MAP = DISEASE_KEY_MAP["wheat"]


# ============================================================
# WEATHER RISK MODULATION
# ============================================================
WEATHER_RISK_RULES = {
    DiseaseType.FUNGAL: {
        "threshold": lambda t, h, r: h >= 80 and 22 <= t <= 32,
        "high_msg": "High humidity and warm temperatures favor fungal spore germination and spread",
        "low_msg": "Conditions less favorable for fungal development"
    },
    DiseaseType.BACTERIAL: {
        "threshold": lambda t, h, r: h >= 75 and r > 5,
        "high_msg": "High humidity with rainfall favors bacterial ooze and splash dispersal",
        "low_msg": "Dry conditions reduce bacterial spread"
    },
    DiseaseType.VIRAL: {
        "threshold": lambda t, h, r: 25 <= t <= 32,
        "high_msg": "Warm temperatures favor leafhopper vector population growth",
        "low_msg": "Cooler temperatures reduce vector activity"
    },
    DiseaseType.SEEDBORNE_FUNGAL: {
        "threshold": lambda t, h, r: 18 <= t <= 24 and h >= 60,
        "high_msg": "Moderate temperatures with humidity favor spore release at flowering",
        "low_msg": "Conditions less favorable for spore dispersal"
    },
    DiseaseType.SOILBORNE_FUNGAL: {
        "threshold": lambda t, h, r: r > 15 or h >= 85,
        "high_msg": "Waterlogged soil and high humidity favor root/crown pathogen development",
        "low_msg": "Drier conditions reduce soil-borne pathogen pressure"
    },
}


def assess_weather_risk(disease_type: DiseaseType, weather: Optional[WeatherData]) -> tuple[RiskLevel, str]:
    """Assess disease risk based on current weather."""
    if weather is None:
        return RiskLevel.UNKNOWN, "Weather data unavailable; cannot assess environmental risk"
    
    rule = WEATHER_RISK_RULES.get(disease_type)
    if not rule:
        return RiskLevel.UNKNOWN, "No risk rule defined for this disease type"
    
    t = weather.temperature_c
    h = weather.humidity_pct
    r = weather.rainfall_mm
    
    # If any required value is None, return UNKNOWN
    if t is None or h is None or r is None:
        return RiskLevel.UNKNOWN, "Weather data incomplete; cannot assess environmental risk"
    
    is_high = rule["threshold"](t, h, r)
    
    if is_high:
        return RiskLevel.HIGH, rule["high_msg"]
    
    # Check moderate conditions
    if disease_type == DiseaseType.FUNGAL:
        if h >= 70 and 18 <= t <= 35:
            return RiskLevel.MODERATE, "Moderate humidity and temperature; some fungal risk"
    elif disease_type == DiseaseType.BACTERIAL:
        if h >= 65:
            return RiskLevel.MODERATE, "Elevated humidity; bacterial spread possible"
    elif disease_type == DiseaseType.VIRAL:
        if 20 <= t <= 35:
            return RiskLevel.MODERATE, "Temperatures favor vector activity"
    elif disease_type == DiseaseType.SEEDBORNE_FUNGAL:
        if 18 <= t <= 24 and h >= 60:
            return RiskLevel.MODERATE, "Moderate conditions for spore release at flowering"
    elif disease_type == DiseaseType.SOILBORNE_FUNGAL:
        if r > 15 or h >= 85:
            return RiskLevel.MODERATE, "Elevated soil moisture; root pathogen risk"
    
    return RiskLevel.LOW, rule["low_msg"]


# ============================================================
# MAIN RECOMMENDATION ENGINE
# ============================================================
class RecommendationEngine:
    """Generates treatment recommendations from diagnosis + weather."""
    
    def __init__(self):
        self.rice_kb = RICE_KNOWLEDGE
        self.wheat_kb = WHEAT_KNOWLEDGE
        self.rice_map = RICE_CLASS_MAP
        self.wheat_map = WHEAT_CLASS_MAP
    
    def get_recommendation(
        self,
        crop: str,
        disease_class: str,
        stage: Optional[str],
        confidence: float,
        weather: Optional[WeatherData] = None
    ) -> RecommendationResult:
        """
        Generate full recommendation.
        
        Args:
            crop: "rice" or "wheat"
            disease_class: Model output class name (e.g., "Blast", "Leaf Rust")
            stage: "early", "mid", "late", or None (for Healthy or rice without stage)
            confidence: Model confidence (0-1)
            weather: Optional WeatherData for risk assessment
        """
        crop = crop.lower()
        
        if crop == "rice":
            kb = self.rice_kb
            class_map = self.rice_map
        elif crop == "wheat":
            kb = self.wheat_kb
            class_map = self.wheat_map
        else:
            raise ValueError(f"Unsupported crop: {crop}")
        
        # Map model class to KB key
        kb_key = class_map.get(disease_class)
        if not kb_key:
            raise ValueError(f"Unknown disease class for {crop}: {disease_class}")
        
        disease_info = kb[kb_key]
        
        # Determine stage
        if disease_class == "Healthy" or kb_key == "healthy":
            stage_enum = Stage.NONE
        elif stage:
            try:
                stage_enum = Stage(stage.lower())
            except ValueError:
                stage_enum = Stage.MID  # fallback
        else:
            # No stage provided (e.g., rice without stage model)
            stage_enum = None
        
        # Get treatment
        if stage_enum is not None:
            fallback_stage = Stage.NONE if stage_enum == Stage.NONE else Stage.MID
            treatment = disease_info.treatments.get(stage_enum, disease_info.treatments[fallback_stage])
        else:
            # No stage available (rice without stage model) - use MID as general guidance
            treatment = disease_info.treatments.get(Stage.MID, disease_info.treatments[Stage.NONE])
        
        # Weather risk - Healthy always LOW risk
        if kb_key == "healthy":
            risk_level = RiskLevel.LOW
            risk_reason = "No disease detected"
        else:
            risk_level, risk_reason = assess_weather_risk(disease_info.disease_type, weather)
        
        # Caveats
        caveats = []
        if stage_enum is not None and stage_enum != Stage.NONE:
            caveats.append(
                "⚠ Severity stage is estimated from heuristic image analysis (HSV lesion ratio), "
                "not expert-annotated ground truth. Treat as indicative only."
            )
        elif stage_enum is None:
            # Rice without stage model
            caveats.append(
                "Automatic severity staging is not available for rice -- "
                "this is general guidance for the disease, not adjusted for "
                "how advanced it is. See README for why."
            )
        if disease_info.disease_type == DiseaseType.VIRAL:
            caveats.append("No antiviral treatment exists; management is 100% vector control.")
        if disease_info.disease_type in (DiseaseType.SEEDBORNE_FUNGAL, DiseaseType.SOILBORNE_FUNGAL):
            caveats.append(
                "No effective in-season foliar treatment exists. "
                "Control is via seed treatment (next season) and cultural practices."
            )
        
        return RecommendationResult(
            crop=crop.capitalize(),
            disease_key=kb_key,
            disease_name=disease_info.display_name,
            pathogen=disease_info.pathogen,
            disease_type=disease_info.disease_type.value,
            typical_symptoms=disease_info.typical_symptoms,
            stage=stage_enum,
            confidence=confidence,
            risk_level=risk_level,
            risk_reason=risk_reason,
            treatment=treatment,
            weather=weather,
            caveats=caveats
        )


# Singleton
_engine: Optional[RecommendationEngine] = None

def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


def recommend(
    crop: str,
    disease_class: str,
    stage: Optional[str],
    confidence: float,
    weather: Optional[WeatherData] = None
) -> RecommendationResult:
    """Convenience function."""
    return get_engine().get_recommendation(crop, disease_class, stage, confidence, weather)


def format_recommendation_text(result: RecommendationResult) -> str:
    """Format recommendation as human-readable text."""
    lines = []
    lines.append(f"=== CropGuard AI Recommendation ===")
    lines.append(f"Crop: {result.crop}")
    lines.append(f"Disease: {result.disease_name} ({result.pathogen})")
    lines.append(f"Type: {result.disease_type.replace('_', ' ').title()}")
    lines.append(f"Confidence: {result.confidence:.1%}")
    if result.stage and result.stage != Stage.NONE:
        lines.append(f"Severity Stage: {result.stage.value.capitalize()}")
    lines.append("")
    lines.append(f"Risk Level: {result.risk_level.value.upper()}")
    lines.append(f"Risk Reason: {result.risk_reason}")
    lines.append("")
    lines.append("--- Treatment ---")
    lines.append(f"Action: {result.treatment.action}")
    lines.append(f"Chemical: {result.treatment.chemical}")
    lines.append(f"Dosage: {result.treatment.dosage}")
    lines.append(f"Timing: {result.treatment.timing}")
    lines.append("")
    lines.append("--- Cultural Controls ---")
    for c in result.treatment.cultural:
        lines.append(f"  * {c}")
    lines.append("")
    lines.append("--- Weather Context ---")
    w = result.weather
    if w is not None and all(getattr(w, a, None) is not None for a in ('temperature_c', 'humidity_pct', 'rainfall_mm')):
        lines.append(f"Temperature: {w.temperature_c} C")
        lines.append(f"Humidity: {w.humidity_pct}%")
        lines.append(f"Rainfall: {w.rainfall_mm} mm")
        if w.wind_kph:
            lines.append(f"Wind: {w.wind_kph} km/h")
    else:
        lines.append("Weather data unavailable")
    if result.caveats:
        lines.append("")
        for caveat in result.caveats:
            lines.append(f"CAVEAT: {caveat}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    engine = RecommendationEngine()
    
    # Test rice blast early
    result = engine.get_recommendation("rice", "Blast", "early", 0.95)
    print(f"Rice Blast Early:")
    print(f"  Action: {result.treatment.action}")
    print(f"  Chemical: {result.treatment.chemical}")
    print(f"  Dosage: {result.treatment.dosage}")
    print(f"  Caveats: {result.caveats}")
    
    # Test wheat loose smut
    result = engine.get_recommendation("wheat", "Loose Smut", "mid", 0.90)
    print(f"\nWheat Loose Smut Mid:")
    print(f"  Action: {result.treatment.action}")
    print(f"  Chemical: {result.treatment.chemical}")
    print(f"  Caveats: {result.caveats}")
    
    # Test weather risk
    weather = WeatherData(temperature_c=28, humidity_pct=85, rainfall_mm=10, wind_kph=5)
    result = engine.get_recommendation("rice", "Blast", "early", 0.95, weather)
    print(f"\nWith weather (28°C, 85% humidity, 10mm rain):")
    print(f"  Risk: {result.risk_level.value} - {result.risk_reason}")