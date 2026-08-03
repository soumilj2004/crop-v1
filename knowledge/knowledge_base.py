"""
CropGuard AI - Treatment Knowledge Base
Source: Consolidated agronomy field-trial and extension guidance.
All dosages and timings are from real extension recommendations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum

class DiseaseType(Enum):
    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    SEEDBORNE_FUNGAL = "seedborne_fungal"
    SOILBORNE_FUNGAL = "soilborne_fungal"

class Stage(Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    NONE = "none"

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
            Stage.EARLY: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Maintain good agronomic practices", "Balanced fertilization", "Proper water management"]
            ),
            Stage.MID: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Continue monitoring", "Standard crop management"]
            ),
            Stage.LATE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Prepare for harvest", "Plan next season"]
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
                    "Use certified smut is already inside the seed embryo - no spray can reach it",
                    "Rogue infected heads before spore release if few plants",
                    "Source clean seed for next season"
                ]
            ),
            Stage.MID: TreatmentPlan(
                action="NO FOLIAR TREATMENT EXISTS - fungus is inside developing grain",
                chemical="Seed treatment for NEXT season only",
                dosage="Carboxin 37.5% + Thiram 37.5% WS: 2.5 g/kg seed",
                timing="Pre-planting seed treatment",
                cultural=[
                    "Cannot cure current crop",
                    "Harvest infected areas separately",
                    "Do not save seed from infected fields"
                ]
            ),
            Stage.LATE: TreatmentPlan(
                action="NO FOLIAR TREATMENT EXISTS - seed-borne, systemic in seed",
                chemical="Seed treatment ONLY (for NEXT season)",
                dosage="Difenoconazole 3% WS: 2 mL/kg seed or Tebuconazole 2% DS: 1.25 g/kg or Triticonazole 2% WS: 1.25 mL/kg",
                timing="Treat seed BEFORE next planting",
                cultural=[
                    "Use certified smut-free seed",
                    "Hot water treatment (52\u00b0C, 10 min) as alternative",
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
        pathogen="Fusarium spp. (F. culmorum, F. graminearum, F. pseudograminearum) / Rhizoctonia spp.",
        disease_type=DiseaseType.SOILBORNE_FUNGAL,
        typical_symptoms=[
            "Brown to black discoloration of crown and subcrown internode",
            "Poor root development, brown/black roots",
            "Plants stunted, yellowing, premature ripening (whiteheads)",
            "Plants pull easily (rotted crown)"
        ],
        spread_conditions="Waterlogged soil, high humidity ≥85%, 15-25°C; continuous wheat favors buildup",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="NO EFFECTIVE IN-SEASON CHEMICAL RESCUE - soil-borne",
                chemical="Seed treatment for NEXT crop (fluquinconazole/silthiofam)",
                dosage="Fluquinconazole 2.5% FS: 20 mL/kg seed; Silthiofam 250 g/L FS: 40 mL/kg seed",
                timing="Pre-planting seed treatment",
                cultural=[
                    "Improve drainage - #1 management",
                    "Avoid waterlogging",
                    "Crop rotation (2+ years non-cereal)",
                    "Balanced nutrition (adequate P, Zn)"
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
        disease_type=DiseaseType.FUNGAL,
        typical_symptoms=["Green, healthy leaves", "Strong root system", "Normal growth and development"],
        spread_conditions="N/A",
        treatments={
            Stage.EARLY: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Standard good agronomic practices"]
            ),
            Stage.MID: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Monitor for any disease onset"]
            ),
            Stage.LATE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Prepare for harvest"]
            ),
            Stage.NONE: TreatmentPlan(
                action="No treatment needed",
                chemical="None",
                dosage="N/A",
                timing="N/A",
                cultural=["Standard good agronomic practices"]
            ),
        }
    ),
}

def get_knowledge_base(crop: str) -> Dict[str, DiseaseInfo]:
    """Get knowledge base for a crop."""
    crop = crop.lower()
    if crop == "rice":
        return RICE_KNOWLEDGE
    elif crop == "wheat":
        return WHEAT_KNOWLEDGE
    else:
        raise ValueError(f"Unknown crop: {crop}")

# Disease key mapping (model class name -> knowledge base key)
DISEASE_KEY_MAP = {
    "rice": {
        "Bacterial Blight": "bacterial_blight",
        "Blast": "blast",
        "Brown Spot": "brown_spot",
        "Tungro": "tungro",
        "Healthy": "healthy",
    },
    "wheat": {
        "Leaf Rust": "leaf_rust",
        "Loose Smut": "loose_smut",
        "Crown & Root Rot": "crown_root_rot",
        "Healthy": "healthy",
    }
}