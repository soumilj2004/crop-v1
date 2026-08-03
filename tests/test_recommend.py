"""
Unit tests for CropGuard AI recommendation engine.
Tests cover all diseases, stages, edge cases, and critical assertions.
"""

import pytest
from cropguard_ai.recommend.engine import (
    get_engine, RecommendationEngine,
    RecommendationResult, RiskLevel, Stage, DiseaseType
)
from cropguard_ai.weather.client import WeatherData


class TestRecommendationEngine:
    """Test the recommendation engine logic."""
    
    @pytest.fixture
    def engine(self):
        return get_engine()
    
    # ===== RICE TESTS =====
    
    def test_rice_bacterial_blight_all_stages(self, engine):
        """Test Bacterial Blight recommendations for all stages."""
        for stage in [Stage.EARLY, Stage.MID, Stage.LATE]:
            result = engine.get_recommendation("rice", "Bacterial Blight", stage.value, 0.9)
            assert result.crop == "Rice"
            assert result.disease_name == "Bacterial Blight"
            assert result.stage == stage
            assert result.treatment.action
            assert result.treatment.chemical
            assert result.treatment.dosage
            assert result.treatment.timing
            assert len(result.treatment.cultural) > 0
            assert result.caveats
    
    def test_rice_blast_all_stages(self, engine):
        """Test Blast recommendations for all stages."""
        for stage in [Stage.EARLY, Stage.MID, Stage.LATE]:
            result = engine.get_recommendation("rice", "Blast", stage.value, 0.95)
            assert result.disease_name == "Blast"
            assert "Tricyclazole" in result.treatment.chemical or "Isoprothiolane" in result.treatment.chemical
            assert result.treatment.action
    
    def test_rice_brown_spot(self, engine):
        """Brown Spot is supported (knowledge base includes it)."""
        for stage in [Stage.EARLY, Stage.MID, Stage.LATE]:
            result = engine.get_recommendation("rice", "Brown Spot", stage.value, 0.88)
            assert result.disease_name == "Brown Spot"
            assert result.disease_type == DiseaseType.FUNGAL.value
            assert result.treatment.action
    
    def test_rice_tungro_all_stages(self, engine):
        """Test Tungro (viral) - no antiviral treatment exists."""
        for stage in [Stage.EARLY, Stage.MID, Stage.LATE]:
            result = engine.get_recommendation("rice", "Tungro", stage.value, 0.9)
            assert result.disease_name == "Tungro"
            assert result.disease_type == DiseaseType.VIRAL.value
            # Should mention vector control only
            assert "leafhopper" in result.treatment.action.lower() or "vector" in result.treatment.action.lower()
            # Caveat should mention no antiviral
            assert any("antiviral" in c.lower() for c in result.caveats)
    
    def test_rice_healthy_no_stage(self, engine):
        """Test Healthy rice - no stage, no treatment needed."""
        result = engine.get_recommendation("rice", "Healthy", None, 0.99)
        assert result.disease_name == "Healthy"
        assert result.stage == Stage.NONE
        assert result.risk_level == RiskLevel.LOW
        assert result.treatment.action == "No treatment needed"
        assert result.treatment.chemical == "None"
    
    # ===== WHEAT TESTS =====
    
    def test_wheat_leaf_rust_all_stages(self, engine):
        """Test Leaf Rust - has real foliar treatment."""
        for stage in [Stage.EARLY, Stage.MID, Stage.LATE]:
            result = engine.get_recommendation("wheat", "Leaf Rust", stage.value, 0.85)
            assert result.disease_name == "Leaf Rust (Brown Rust)"
            assert "Propiconazole" in result.treatment.chemical or "Tebuconazole" in result.treatment.chemical or "Prothioconazole" in result.treatment.chemical
            assert result.treatment.action
    
    def test_wheat_loose_smut_no_foliar_treatment(self, engine):
        """CRITICAL: Loose Smut has NO foliar treatment at ANY stage."""
        for stage in [Stage.EARLY, Stage.MID, Stage.LATE]:
            result = engine.get_recommendation("wheat", "Loose Smut", stage.value, 0.9)
            assert result.disease_name == "Loose Smut"
            assert result.disease_type == DiseaseType.SEEDBORNE_FUNGAL.value
            # Must NOT claim foliar spray exists
            assert "NO FOLIAR TREATMENT EXISTS" in result.treatment.action.upper()
            # Should reference seed treatment for next season
            assert "seed treatment" in result.treatment.chemical.lower() or "next season" in result.treatment.timing.lower()
            assert "seed treatment" in result.treatment.chemical.lower()
            # Caveat should mention seed-borne
            assert any("seed" in c.lower() for c in result.caveats)
    
    def test_wheat_crown_root_rot_no_foliar_rescue(self, engine):
        """CRITICAL: Crown/Root Rot has NO in-season chemical rescue."""
        for stage in [Stage.EARLY, Stage.MID, Stage.LATE]:
            result = engine.get_recommendation("wheat", "Crown & Root Rot", stage.value, 0.88)
            assert result.disease_name == "Crown & Root Rot"
            assert result.disease_type == DiseaseType.SOILBORNE_FUNGAL.value
            assert "NO EFFECTIVE IN-SEASON CHEMICAL RESCUE" in result.treatment.action.upper()
            assert "seed treatment" in result.treatment.chemical.lower()
            assert any("drainage" in c.lower() for c in result.treatment.cultural)
    
    def test_wheat_healthy(self, engine):
        """Test Healthy wheat."""
        result = engine.get_recommendation("wheat", "Healthy", None, 0.99)
        assert result.disease_name == "Healthy"
        assert result.stage == Stage.NONE
        assert result.risk_level == RiskLevel.LOW
    
    # ===== EDGE CASES =====
    
    def test_unknown_crop_raises(self, engine):
        with pytest.raises(ValueError, match="Unsupported crop"):
            engine.get_recommendation("corn", "Some Disease", "early", 0.9)
    
    def test_unknown_disease_raises(self, engine):
        with pytest.raises(ValueError, match="Unknown disease"):
            engine.get_recommendation("rice", "Fake Disease", "early", 0.9)
    
    def test_unknown_stage_defaults_to_mid(self, engine):
        """Invalid stage should default to MID."""
        result = engine.get_recommendation("rice", "Blast", "invalid_stage", 0.9)
        assert result.stage == Stage.MID
    
    # ===== WEATHER RISK TESTS =====
    
    def test_fungal_high_risk_weather(self, engine):
        """Fungal disease + high humidity + warm temp = HIGH risk."""
        weather = WeatherData(temperature_c=28, humidity_pct=85, rainfall_mm=10, wind_kph=5)
        result = engine.get_recommendation("rice", "Blast", "early", 0.9, weather)
        assert result.risk_level == RiskLevel.HIGH
        assert "humidity" in result.risk_reason.lower() or "temperature" in result.risk_reason.lower()
    
    def test_fungal_low_risk_weather(self, engine):
        """Fungal disease + dry/cool = LOW risk."""
        weather = WeatherData(temperature_c=15, humidity_pct=40, rainfall_mm=0, wind_kph=10)
        result = engine.get_recommendation("rice", "Blast", "early", 0.9, weather)
        assert result.risk_level == RiskLevel.LOW
    
    def test_bacterial_high_risk_weather(self, engine):
        """Bacterial + high humidity + rain = HIGH risk."""
        weather = WeatherData(temperature_c=30, humidity_pct=80, rainfall_mm=10, wind_kph=5)
        result = engine.get_recommendation("rice", "Bacterial Blight", "early", 0.9, weather)
        assert result.risk_level == RiskLevel.HIGH
    
    def test_viral_risk_temp_driven(self, engine):
        """Viral risk driven by temperature (vector activity)."""
        weather = WeatherData(temperature_c=28, humidity_pct=50, rainfall_mm=0, wind_kph=5)
        result = engine.get_recommendation("rice", "Tungro", "early", 0.9, weather)
        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.MODERATE]
    
    def test_no_weather_returns_unknown(self, engine):
        """Missing weather returns UNKNOWN risk."""
        result = engine.get_recommendation("rice", "Blast", "early", 0.9, None)
        assert result.risk_level == RiskLevel.UNKNOWN
        assert "unavailable" in result.risk_reason.lower() or "incomplete" in result.risk_reason.lower()
    
    def test_incomplete_weather_returns_unknown(self, engine):
        """Partial weather data returns UNKNOWN."""
        weather = WeatherData(temperature_c=25, humidity_pct=None, rainfall_mm=5, wind_kph=0)
        result = engine.get_recommendation("rice", "Blast", "early", 0.9, weather)
        assert result.risk_level == RiskLevel.UNKNOWN
    
    # ===== SEEDBORNE/SOILBORNE SPECIFIC WEATHER =====
    
    def test_seedborne_fungal_weather(self, engine):
        """Seedborne fungal: moderate temp + humidity at flowering."""
        weather = WeatherData(temperature_c=20, humidity_pct=65, rainfall_mm=0, wind_kph=5)
        result = engine.get_recommendation("wheat", "Loose Smut", "mid", 0.9, weather)
        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.MODERATE]
    
    def test_soilborne_fungal_weather(self, engine):
        """Soilborne fungal: heavy rain or very high humidity."""
        weather = WeatherData(temperature_c=20, humidity_pct=90, rainfall_mm=20, wind_kph=5)
        result = engine.get_recommendation("wheat", "Crown & Root Rot", "mid", 0.9, weather)
        assert result.risk_level == RiskLevel.HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])