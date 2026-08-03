"""
CropGuard AI - Recommendation Engine
Pure Python, no ML dependencies. Deterministic, fully testable.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
try:
    from cropguard_ai.weather.client import WeatherData
except ImportError:
    from weather.client import WeatherData


class DiseaseType(str, Enum):
    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    SEEDBORNE_FUNGAL = "seedborne_fungal"
    SOILBORNE_FUNGAL = "soilborne_fungal"


class Stage(str, Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    NONE = "none"  # For Healthy class


class RiskLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class TreatmentPlan:
    action: str
    chemical: str
    dosage: str
    timing: str
    cultural: List[str]


@dataclass
class DiseaseInfo:
    display_name: str
    pathogen: str
    disease_type: DiseaseType
    typical_symptoms: List[str]
    spread_conditions: str
    treatments: Dict[Stage, TreatmentPlan]


@dataclass
class RecommendationResult:
    crop: str
    disease_key: str
    disease_name: str
    pathogen: str
    disease_type: str
    stage: Stage
    confidence: float
    risk_level: RiskLevel
    risk_reason: str
    treatment: TreatmentPlan
    weather: Optional[WeatherData]
    caveats: List[str]


# ============================================================
# RICE KNOWLEDGE BASE
# ============================================================
RICE_KNOWLEDGE: Dict[str, DiseaseInfo] = {
    "bacterial_blight": DiseaseInfo(
        display_name="Bacterial Blight",
        pathogen="Xanthomonas oryzae pv. oryzae",
        disease_type=DiseaseType.BACTERIAL,
        typical_symptoms=[
            "Water-soaked lesions on leaf margins",
            "Yellow to white stripes along leaf veins",
            "Lesions coalesce, leaves dry and die",
            "Bacterial ooze visible on lesions in morning"
        ],
        spread_conditions="High humidity (≥75%), rainfall >5mm, 25-34°C; spreads via wind-driven rain",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="Copper-based spray + field sanitation",
                chemical="Copper oxychloride 50% WP",
                dosage="2.5-3 g/L water",
                timing="Apply at first sign; repeat at 10-14 day intervals if conditions persist",
                cultural=[
                    "Remove and destroy infected plant debris",
                    "Avoid overhead irrigation when possible",
                    "Ensure balanced N fertilization (excess N increases susceptibility)"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="Copper spray + streptomycin if severe",
                chemical="Copper oxychloride 50% WP + Streptomycin sulfate",
                dosage="2.5-3 g/L + 0.5 g/L",
                timing="Immediate; protect flag leaf and panicle",
                cultural=[
                    "Rogue severely infected plants",
                    "Improve field drainage",
                    "Reduce plant density if excessive"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="Limited curative options; focus on next crop",
                chemical="Copper oxychloride (protect remaining foliage)",
                dosage="3 g/L",
                timing="If panicle not yet emerged; otherwise prepare for next season",
                cultural=[
                    "No effective rescue treatment once systemic",
                    "Plan resistant variety for next season",
                    "Seed treatment with streptomycin for next crop",
                    "Crop rotation with non-host"
                ]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Maintain good agronomic practices", "Balanced fertilization", "Proper water management"]
            ),
        }
    ),
    "blast": DiseaseInfo(
        display_name="Blast",
        pathogen="Magnaporthe oryzae",
        disease_type=DiseaseType.FUNGAL,
        typical_symptoms=[
            "Diamond-shaped lesions with gray center and brown margin",
            "Lesions on leaves, nodes, panicles (neck blast)",
            "White to gray sporulation in lesion centers under humidity",
            "Neck blast causes panicle breakage and whiteheads"
        ],
        spread_conditions="High humidity (≥90%), 22-28°C, leaf wetness >10h; wind disperses conidia",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="Protective fungicide application",
                chemical="Tricyclazole 75% WP",
                dosage="0.6 g/L water (600 g/ha)",
                timing="At first lesion appearance; critical before panicle emergence",
                cultural=[
                    "Avoid excessive N; split application",
                    "Maintain shallow water (not flooded) during susceptible stages",
                    "Use resistant varieties where available"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="Curative + protective fungicide",
                chemical="Tricyclazole 75% WP or Isoprothiolane 40% EC",
                dosage="Tricyclazole: 0.6 g/L; Isoprothiolane: 1.5 mL/L",
                timing="Immediate; prioritize flag leaf and panicle protection",
                cultural=[
                    "Drain field if waterlogged",
                    "Remove collateral weed hosts",
                    "Synchronize planting in area to reduce inoculum buildup"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="Neck blast emergency spray",
                chemical="Tricyclazole 75% WP or Azoxystrobin 23% SC",
                dosage="Tricyclazole: 0.6 g/L; Azoxystrobin: 1 mL/L",
                timing="At booting/heading if neck blast risk high",
                cultural=[
                    "Harvest early if severe",
                    "Burn straw after harvest",
                    "Seed treatment for next season (tricyclazole)"
                ]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Maintain good agronomic practices", "Monitor regularly"]
            ),
        }
    ),
    "brown_spot": DiseaseInfo(
        display_name="Brown Spot",
        pathogen="Bipolaris oryzae (Helminthosporium oryzae)",
        disease_type=DiseaseType.FUNGAL,
        typical_symptoms=[
            "Small circular to oval brown spots with yellow halo",
            "Spots coalesce into larger patches",
            "Glumes show dark brown to black discoloration",
            "Seedling blight causes patchy stand"
        ],
        spread_conditions="High humidity, 25-30°C; favored by K-deficient soils, drought stress",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="Mancozeb spray + correct soil nutrition",
                chemical="Mancozeb 75% WP",
                dosage="2.5 g/L water",
                timing="At first symptoms; repeat at 10-14 day intervals",
                cultural=[
                    "Apply potassium fertilizer (K deficiency is key predisposing factor)",
                    "Use disease-free seed",
                    "Seed treatment with carbendazim"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="Systemic fungicide + nutrition correction",
                chemical="Propiconazole 25% EC or Tebuconazole 25.9% EC",
                dosage="Propiconazole: 1 mL/L; Tebuconazole: 0.75 mL/L",
                timing="When spots cover >5% leaf area",
                cultural=[
                    "Foliar K spray (K2SO4 1%) if deficiency confirmed",
                    "Improve drainage",
                    "Avoid water stress"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="Protect panicles; plan next season",
                chemical="Propiconazole 25% EC",
                dosage="1 mL/L",
                timing="If grain filling not complete",
                cultural=[
                    "Harvest and destroy infected straw",
                    "Balanced NPK for next crop",
                    "Seed treatment with carbendazim + mancozeb"
                ]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Maintain balanced nutrition", "Monitor regularly"]
            ),
        }
    ),
    "tungro": DiseaseInfo(
        display_name="Tungro",
        pathogen="Rice tungro bacilliform virus (RTBV) + Rice tungro spherical virus (RTSV)",
        disease_type=DiseaseType.VIRAL,
        typical_symptoms=[
            "Yellow to orange discoloration of entire plant",
            "Stunted growth, reduced tillering",
            "Leaves twist and curl downward",
            "Small, partially filled grains"
        ],
        spread_conditions="Transmitted by green leafhopper (Nephotettix virescens); 25-32°C favors vector population",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="VECTOR CONTROL ONLY - no antiviral treatment exists",
                chemical="Systemic insecticide for leafhopper control",
                dosage="Imidacloprid 17.8% SL: 0.3 mL/L or Thiamethoxam 25% WG: 0.5 g/L",
                timing="At first sign of leafhopper infestation or tungro symptoms",
                cultural=[
                    "Rogue and destroy infected plants immediately",
                    "Synchronous planting across area",
                    "Remove grassy weeds (alternate hosts)",
                    "Use resistant varieties (e.g., IR64, Vikramarya)"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="Intensive vector control + rogueing",
                chemical="Fipronil 5% SC or Dinotefuran 20% SG",
                dosage="Fipronil: 2 mL/L; Dinotefuran: 0.4 g/L",
                timing="Repeated sprays as leafhopper populations dictate",
                cultural=[
                    "Continue rogueing",
                    "Light traps for adult leafhopper monitoring",
                    "Do NOT apply fungicides/bactericides - ineffective"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="No recovery; vector control to mitigate spread to neighboring fields",
                chemical="Insecticide for leafhopper vector suppression only",
                dosage="As per label for leafhopper",
                timing="If nearby fields at risk",
                cultural=[
                    "Harvest early if severely affected",
                    "Destroy crop residue",
                    "Plan resistant variety + seed treatment next season",
                    "Area-wide leafhopper management"
                ]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Monitor for leafhoppers", "Standard crop management"]
            ),
        }
    ),
    "healthy": DiseaseInfo(
        display_name="Healthy",
        pathogen="None",
        disease_type=DiseaseType.FUNGAL,  # dummy
        typical_symptoms=["Green, vigorous leaves", "No lesions or discoloration", "Normal growth and tillering"],
        spread_conditions="N/A",
        treatments={
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Maintain good agronomic practices", "Balanced fertilization", "Proper water management"]
            ),
        }
    ),
}

# Map model class names to knowledge base keys
RICE_CLASS_MAP = {
    "Bacterial Blight": "bacterial_blight",
    "Blast": "blast",
    "Healthy": "healthy",
    "Tungro": "tungro",
}

# ============================================================
# WHEAT KNOWLEDGE BASE
# ============================================================
WHEAT_KNOWLEDGE: Dict[str, DiseaseInfo] = {
    "leaf_rust": DiseaseInfo(
        display_name="Leaf Rust (Brown Rust)",
        pathogen="Puccinia triticina",
        disease_type=DiseaseType.FUNGAL,
        typical_symptoms=[
            "Small, circular to oval orange-brown pustules on leaf upper surface",
            "Pustules scattered, not in stripes",
            "Yellow halo around pustules on susceptible varieties",
            "Premature leaf senescence if severe"
        ],
        spread_conditions="High humidity, 15-25°C; wind spreads urediniospores long distances",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="Protective triazole spray at flag leaf emergence",
                chemical="Propiconazole 25% EC or Tebuconazole 25.9% EC",
                dosage="Propiconazole: 1 mL/L (200 mL/ha); Tebuconazole: 0.75 mL/L (150 mL/ha)",
                timing="GS 37-39 (flag leaf emerging); before pustules cover >5% flag leaf",
                cultural=[
                    "Plant resistant varieties",
                    "Avoid excessive N (delays maturity, increases susceptibility)",
                    "Early planting to escape peak rust"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="Curative triazole or SDHI+triazole mix",
                chemical="Prothioconazole 25% EC or Prothioconazole + Tebuconazole",
                dosage="Prothioconazole: 0.5 L/ha; Mix: per label",
                timing="When pustules on flag leaf or F1, before 50% leaf area affected",
                cultural=[
                    "Monitor nearby fields for early warning",
                    "Ensure good spray coverage (water volume 200 L/ha)"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="Late spray only if grain fill not complete",
                chemical="Tebuconazole 25.9% EC",
                dosage="0.75 mL/L (150 mL/ha)",
                timing="Up to GS 71 (watery ripe); observe PHI",
                cultural=[
                    "Harvest promptly to minimize yield loss",
                    "Destroy volunteer wheat (green bridge)",
                    "Crop rotation with non-host"
                ]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Standard crop management", "Monitor for rust"]
            ),
        }
    ),
    "loose_smut": DiseaseInfo(
        display_name="Loose Smut",
        pathogen="Ustilago tritici",
        disease_type=DiseaseType.SEEDBORNE_FUNGAL,
        typical_symptoms=[
            "Entire head replaced by black spore mass (olive-brown)",
            "Only bare rachis remains at maturity",
            "Spores disperse by wind, leaving rachis",
            "Infected plants slightly taller, earlier heading"
        ],
        spread_conditions="Seed-borne; spores infect embryo during flowering; 18-24°C, high humidity at flowering favors infection",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="NO FOLIAR TREATMENT EXISTS - seed-borne, systemic in seed",
                chemical="Seed treatment ONLY (for NEXT season)",
                dosage="Difenoconazole 3% WS: 2 mL/kg seed or Tebuconazole 2% DS: 1.25 g/kg or Triticonazole 2% WS: 1.25 mL/kg",
                timing="Treat seed BEFORE next planting",
                cultural=[
                    "Use certified smut-free seed",
                    "Hot water treatment (52°C, 10 min) as alternative",
                    "Remove smutted heads before spore release (limited effect)"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="NO FOLIAR TREATMENT EXISTS - seed-borne, systemic in seed",
                chemical="Seed treatment ONLY (for NEXT season)",
                dosage="Difenoconazole 3% WS: 2 mL/kg seed or Tebuconazole 2% DS: 1.25 g/kg or Triticonazole 2% WS: 1.25 mL/kg",
                timing="Treat seed BEFORE next planting",
                cultural=[
                    "Use certified smut-free seed",
                    "Hot water treatment (52°C, 10 min) as alternative",
                    "No in-season chemical rescue possible"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="NO FOLIAR TREATMENT EXISTS - seed-borne, systemic in seed",
                chemical="Seed treatment ONLY (for NEXT season)",
                dosage="Difenoconazole 3% WS: 2 mL/kg seed or Tebuconazole 2% DS: 1.25 g/kg or Triticonazole 2% WS: 1.25 mL/kg",
                timing="Treat seed BEFORE next planting",
                cultural=[
                    "Use certified smut-free seed",
                    "Hot water treatment (52°C, 10 min) as alternative",
                    "No in-season chemical rescue possible at any stage"
                ]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Use certified seed", "Standard management"]
            ),
        }
    ),
    "crown_root_rot": DiseaseInfo(
        display_name="Crown & Root Rot",
        pathogen="Fusarium spp. / Rhizoctonia spp.",
        disease_type=DiseaseType.SOILBORNE_FUNGAL,
        typical_symptoms=[
            "Plants stunted, yellowing, wilting",
            "Crown and root tissue brown to black, rotted",
            "Poor root development, easy to pull up",
            "Patchy distribution in field"
        ],
        spread_conditions="Soil-borne; favored by waterlogged soil, compaction, continuous wheat; rainfall >15mm or humidity ≥85%",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="NO EFFECTIVE IN-SEASON CHEMICAL RESCUE",
                chemical="Seed treatment for NEXT season (fluquinconazole/silthiofam)",
                dosage="Fluquinconazole 1.5% FS: 400 mL/100kg seed or Silthiofam 125 g/L FS: 200 mL/100kg seed",
                timing="Treat seed BEFORE next planting",
                cultural=[
                    "Improve field drainage",
                    "Crop rotation away from continuous wheat",
                    "Avoid deep sowing",
                    "Address soil compaction"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="NO EFFECTIVE IN-SEASON CHEMICAL RESCUE",
                chemical="Seed treatment for NEXT season (fluquinconazole/silthiofam)",
                dosage="Fluquinconazole 1.5% FS: 400 mL/100kg seed or Silthiofam 125 g/L FS: 200 mL/100kg seed",
                timing="Treat seed BEFORE next planting",
                cultural=[
                    "Improve field drainage",
                    "Crop rotation away from continuous wheat",
                    "No effective foliar fungicide for root/crown rot"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="NO EFFECTIVE IN-SEASON CHEMICAL RESCUE",
                chemical="Seed treatment for NEXT season (fluquinconazole/silthiofam)",
                dosage="Fluquinconazole 1.5% FS: 400 mL/100kg seed or Silthiofam 125 g/L FS: 200 mL/100kg seed",
                timing="Treat seed BEFORE next planting",
                cultural=[
                    "Improve field drainage",
                    "Crop rotation away from continuous wheat",
                    "Harvest affected areas separately if possible"
                ]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Maintain good drainage", "Crop rotation", "Standard management"]
            ),
        }
    ),
    "healthy": DiseaseInfo(
        display_name="Healthy",
        pathogen="None",
        disease_type=DiseaseType.FUNGAL,  # dummy
        typical_symptoms=["Green, vigorous plants", "No lesions or discoloration", "Normal root and crown development"],
        spread_conditions="N/A",
        treatments={
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Maintain good agronomic practices", "Balanced fertilization", "Proper water management"]
            ),
        }
    ),
}

WHEAT_CLASS_MAP = {
    "Leaf Rust": "leaf_rust",
    "Loose Smut": "loose_smut",
    "Crown & Root Rot": "crown_root_rot",
    "Healthy": "healthy",
}


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
    else:
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