import math


def update_gps(lat: float, lon: float, speed_kmph: float, heading_deg: float, dt_seconds: float) -> tuple[float, float]:
    speed_mps = speed_kmph * 1000.0 / 3600.0
    distance_m = speed_mps * dt_seconds

    earth_radius = 6371000.0

    dlat = distance_m * math.cos(math.radians(heading_deg)) / earth_radius
    dlon = distance_m * math.sin(math.radians(heading_deg)) / (
        earth_radius * max(math.cos(math.radians(lat)), 1e-6)
    )

    new_lat = lat + math.degrees(dlat)
    new_lon = lon + math.degrees(dlon)

    return new_lat, new_lon