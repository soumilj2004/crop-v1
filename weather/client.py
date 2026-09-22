"""
CropGuard AI - Real Weather Integration
Uses Open-Meteo (free, no API key) + IP-based geolocation.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests
from dataclasses import dataclass
from typing import Optional

# Disk-backed fallback so a dead/blocked network cannot take the demo down.
# Written on every successful live fetch; read only when the live fetch fails.
_CACHE_FILE = Path(__file__).resolve().parent.parent / "logs" / "weather_cache.json"
_NET_TIMEOUT = 4  # seconds - short, so an unreachable API fails fast instead of hanging the UI


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
            resp = requests.get("http://ip-api.com/json/", timeout=_NET_TIMEOUT)
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
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=_NET_TIMEOUT)
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
            self._save_disk_cache(weather, latitude, longitude)
            return weather
            
        except Exception as e:
            print(f"Open-Meteo weather error: {e}")
            return None
    
    # ---------- disk-backed fallback ----------

    def _save_disk_cache(self, weather: "WeatherData", latitude: float, longitude: float) -> None:
        """Persist the last good live reading so it can be reused if the network dies."""
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps({
                "fetched_at": time.time(),
                "fetched_at_human": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "latitude": latitude,
                "longitude": longitude,
                "temperature_c": weather.temperature_c,
                "humidity_pct": weather.humidity_pct,
                "rainfall_mm": weather.rainfall_mm,
                "wind_kph": weather.wind_kph,
            }, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Weather cache write failed (non-fatal): {e}")

    def _load_disk_cache(self) -> Optional["WeatherData"]:
        """Return the last good live reading, labelled as cached. None if unavailable."""
        try:
            if not _CACHE_FILE.exists():
                return None
            d = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            age_h = (time.time() - float(d.get("fetched_at", 0))) / 3600.0
            if age_h < 1:
                age = f"{int(age_h * 60)} min old"
            elif age_h < 48:
                age = f"{age_h:.0f} h old"
            else:
                age = f"{age_h / 24:.0f} d old"
            return WeatherData(
                temperature_c=d["temperature_c"],
                humidity_pct=d["humidity_pct"],
                rainfall_mm=d["rainfall_mm"],
                wind_kph=d["wind_kph"],
                # The UI shows this string verbatim - it must always say the reading is not live.
                source=f"open_meteo (cached {d.get('fetched_at_human', '?')} - {age}, network unavailable)",
            )
        except Exception as e:
            print(f"Weather cache read failed: {e}")
            return None

    def get_weather_for_current_location(self) -> Optional[WeatherData]:
        """Live weather for the current location; falls back to the last good reading.

        A cached reading is always labelled as cached in `source`, which the API passes
        through to the UI. Never silently presents stale data as live.
        """
        loc = self.get_location()
        if loc:
            live = self.get_weather(loc.latitude, loc.longitude)
            if live:
                return live
        cached = self._load_disk_cache()
        if cached:
            print("Weather: live fetch failed, using cached reading.")
        return cached

    def prime_cache(self) -> bool:
        """Fetch and store a live reading now. Run this before a demo, on a good network."""
        self.clear_cache()
        w = self.get_weather_for_current_location()
        ok = bool(w) and "cached" not in (w.source or "")
        print("Weather cache primed." if ok else "Could NOT prime weather cache - check the network.")
        if w:
            print(f"  {w.temperature_c} C / {w.humidity_pct}% RH / {w.rainfall_mm} mm / {w.wind_kph} kph  [{w.source}]")
        return ok
    
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
    # `python -m weather.client --prime`  -> fetch a live reading and store it on disk,
    #                                        so the demo survives a dead network later.
    # `python -m weather.client`          -> just show what the pipeline would use now.
    import sys
    client = get_weather_client()
    if "--prime" in sys.argv:
        sys.exit(0 if client.prime_cache() else 1)

    weather = client.get_weather_for_current_location()
    if weather:
        print(f"Temperature: {weather.temperature_c} C")
        print(f"Humidity:    {weather.humidity_pct} %")
        print(f"Rainfall:    {weather.rainfall_mm} mm")
        print(f"Wind:        {weather.wind_kph} kph")
        print(f"Source:      {weather.source}")
    else:
        print("No weather available (live fetch failed and no cached reading on disk).")
        print("Run `python -m weather.client --prime` on a working network first.")
