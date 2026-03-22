from __future__ import annotations

from typing import Dict, Optional

# Known place labels and coordinates for map links.
KNOWN_PLACES: Dict[str, Dict[str, float | str]] = {
    "plage": {"name": "Plage", "lat": 36.8065, "lon": 10.1815},
    "rue_urbaine": {"name": "Rue urbaine", "lat": 48.8566, "lon": 2.3522},
    "montagne": {"name": "Montagne", "lat": 46.2276, "lon": 2.2137},
    "parc": {"name": "Parc", "lat": 45.7640, "lon": 4.8357},
    "restaurant": {"name": "Restaurant", "lat": 43.2965, "lon": 5.3698},
}


def place_metadata(label: str) -> Optional[Dict[str, float | str]]:
    return KNOWN_PLACES.get(label)


def google_maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"
