# ==============================
# AutoMind OpenEnv - Vehicle Dynamics
# ==============================

from __future__ import annotations

from typing import Literal


RoadCondition = Literal["dry", "wet", "rain"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_friction_factor(road_condition: str) -> float:
    if road_condition == "dry":
        return 1.0
    if road_condition == "wet":
        return 0.8
    if road_condition == "rain":
        return 0.65
    return 1.0


def apply_speed_decay(speed: float) -> float:
    """
    Natural passive speed decay per step.
    """
    return clamp(speed - 1.5, 0.0, 200.0)


def apply_action_to_speed(
    speed: float,
    action_type: str,
    action_value: float,
    road_condition: str,
    brake_failure: bool,
) -> float:
    """
    Update speed based on agent action.

    action_value is expected in [0, 1].
    """
    friction = get_friction_factor(road_condition)
    updated_speed = speed

    if action_type == "accelerate":
        delta = 12.0 * action_value
        updated_speed += delta

    elif action_type == "brake":
        brake_effectiveness = 18.0 * action_value * friction
        if brake_failure:
            brake_effectiveness *= 0.55
        updated_speed -= brake_effectiveness

    elif action_type == "stop":
        stop_effectiveness = 28.0 * friction
        if brake_failure:
            stop_effectiveness *= 0.55
        updated_speed -= stop_effectiveness

    elif action_type in {"turn_left", "turn_right"}:
        updated_speed -= 4.0 * friction

    elif action_type == "continue":
        updated_speed -= 0.5

    elif action_type == "request_service":
        updated_speed -= 6.0

    return clamp(updated_speed, 0.0, 200.0)


def update_distance_to_obstacle(
    current_distance: float,
    speed: float,
    obstacle_relative_motion: float,
) -> float:
    """
    Lower distance as the vehicle moves forward.
    Positive obstacle_relative_motion means obstacle is moving away.
    """
    distance_delta = (speed / 12.0) - obstacle_relative_motion
    new_distance = current_distance - distance_delta
    return clamp(new_distance, 0.0, 200.0)


def update_engine_temperature(
    engine_temp: float,
    rpm: float,
    action_type: str,
    road_condition: str,
    overheating_active: bool,
    rng,
) -> float:
    """
    Physics-based temperature target seeking model derived from ECU logic.
    """
    coolantTarget = 82 + (rpm / 5200.0) * 25

    # Cooler weather/road effect
    if road_condition == "rain":
        coolantTarget -= 5.0
    elif road_condition == "wet":
        coolantTarget -= 2.0

    # Existing fault worsens temperature
    if overheating_active:
        coolantTarget += 30.0

    if action_type == "stop":
        # Cools down when stopped
        noise = rng.uniform(0.02, 0.1)
        temp = engine_temp - noise * 10.0 
        return clamp(temp, 25.0, 150.0)
    elif action_type == "brake":
        # Cools down when braking
        noise = rng.uniform(0.02, 0.08)
        temp = engine_temp - noise * 8.0 
        return clamp(temp, 40.0, 150.0)
    else:
        noise = rng.uniform(-0.2, 0.4)
        temp = engine_temp + (coolantTarget - engine_temp) * 0.08 + noise
        return clamp(temp, 60.0, 150.0)


def estimate_collision_risk(
    speed: float,
    distance_to_obstacle: float,
    road_condition: str,
    brake_failure: bool,
) -> float:
    """
    Risk in [0, 1].
    """
    friction = get_friction_factor(road_condition)

    if distance_to_obstacle <= 0:
        return 1.0

    base = (speed / max(distance_to_obstacle, 1.0)) * 0.12

    if road_condition == "wet":
        base += 0.10
    elif road_condition == "rain":
        base += 0.18

    if brake_failure:
        base += 0.22

    base += (1.0 - friction) * 0.1

    return clamp(base, 0.0, 1.0)
