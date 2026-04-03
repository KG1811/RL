import math


SERVICE_CENTERS = [
    {
        "name": "AutoFix Delhi",
        "lat": 28.620000,
        "lon": 77.210000,
        "slots_available": 4,
        "eta_minutes": 18,
    },
    {
        "name": "CarCare Noida",
        "lat": 28.570000,
        "lon": 77.320000,
        "slots_available": 6,
        "eta_minutes": 28,
    },
    {
        "name": "MechPro Gurgaon",
        "lat": 28.460000,
        "lon": 77.030000,
        "slots_available": 3,
        "eta_minutes": 34,
    },
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def find_nearest_service(lat: float, lon: float) -> dict:
    nearest = None
    best_distance = float("inf")

    for center in SERVICE_CENTERS:
        dist = haversine_km(lat, lon, center["lat"], center["lon"])
        if dist < best_distance:
            best_distance = dist
            nearest = {
                **center,
                "distance_km": round(dist, 2),
            }

    return nearest if nearest is not None else {}