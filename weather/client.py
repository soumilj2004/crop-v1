"""
CropGuard AI - Real Weather Integration
Uses Open-Meteo (free, no API key) + IP-based geolocation.
"""

import time
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocationData:
    latitude: float
    longitude: float
    city: str
    country: str
    source: str = "ip_api"


@dataclass
class WeatherData:
    temperature_c: float
    humidity_pct: float
    rainfall_mm: float
    wind_kph: float
    source: str = "open_meteo"


class WeatherClient:
    """Client for fetching real weather data."""
    
    def __init__(self):
        self._location_cache: Optional[LocationData] = None
        self._location_cache_time: float = 0
        self._weather_cache: Optional[WeatherData] = None
        self._weather_cache_time: float = 0
        self.LOCATION_CACHE_TTL = 3600  # 1 hour
        self.WEATHER_CACHE_TTL = 300    # 5 minutes
    
    def get_location(self) -> Optional[LocationData]:
        """Get user's approximate location via IP geolocation."""
        # Check cache
        if self._location_cache and (time.time() - self._location_cache_time) < self.LOCATION_CACHE_TTL:
            return self._location_cache
        
        try:
            # Use ip-api.com (free, no key, generous limits)
            resp = requests.get("http://ip-api.com/json/", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") == "success":
                loc = LocationData(
                    latitude=data["lat"],
                    longitude=data["lon"],
                    city=data.get("city", "Unknown"),
                    country=data.get("country", "Unknown"),
                    source="ip_api"
                )
                self._location_cache = loc
                self._location_cache_time = time.time()
                return loc
            else:
                print(f"IP geolocation failed: {data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"IP geolocation error: {e}")
            return None
    
    def get_weather(self, latitude: float, longitude: float) -> Optional[WeatherData]:
        """Fetch current weather from Open-Meteo."""
        # Check cache
        if self._weather_cache and (time.time() - self._weather_cache_time) < self.WEATHER_CACHE_TTL:
            return self._weather_cache
        
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true",
                "hourly": "relative_humidity_2m,precipitation",
                "timezone": "auto",
            }
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            current = data.get("current_weather", {})
            hourly = data.get("hourly", {})
            
            temp = current.get("temperature")
            wind = current.get("windspeed")
            
            if temp is None:
                return None
            
            # Get current hour's humidity and rainfall
            humidity = 50.0  # fallback
            rainfall = 0.0
            
            if "relative_humidity_2m" in hourly and "time" in hourly:
                from datetime import datetime
                now_str = datetime.now().strftime("%Y-%m-%dT%H:00")
                times = hourly["time"]
                if now_str in times:
                    idx = times.index(now_str)
                    humidity = hourly["relative_humidity_2m"][idx]
                    rainfall = hourly.get("precipitation", [0])[idx]
            
            weather = WeatherData(
                temperature_c=temp,
                humidity_pct=humidity,
                rainfall_mm=rainfall,
                wind_kph=wind if wind is not None else 0.0,
                source="open_meteo"
            )
            self._weather_cache = weather
            self._weather_cache_time = time.time()
            return weather
            
        except Exception as e:
            print(f"Open-Meteo weather error: {e}")
            return None
    
    def get_weather_for_current_location(self) -> Optional[WeatherData]:
        """Convenience: get location then weather."""
        loc = self.get_location()
        if not loc:
            return None
        return self.get_weather(loc.latitude, loc.longitude)
    
    def clear_cache(self):
        self._location_cache = None
        self._location_cache_time = 0
        self._weather_cache = None
        self._weather_cache_time = 0


# Singleton
_weather_client: Optional[WeatherClient] = None

def get_weather_client() -> WeatherClient:
    global _weather_client
    if _weather_client is None:
        _weather_client = WeatherClient()
    return _weather_client


def fetch_live_weather() -> Optional[WeatherData]:
    """One-shot function to get current weather for the user's location."""
    return get_weather_client().get_weather_for_current_location()


if __name__ == "__main__":
    # Quick test
    weather = fetch_live_weather()
    if weather:
        print(f"Temperature: {weather.temperature_c}°C")
        print(f"Humidity: {weather.humidity_pct}%")
        print(f"Rainfall: {weather.rainfall_mm} mm")
        print(f"Wind: {weather.wind_kph} km/h")
        print(f"Source: {weather.source}")
    else:
        print("Failed to fetch weather - check internet connection")