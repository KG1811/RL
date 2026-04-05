# ==============================
# AutoMind OpenEnv - Failure Engine
# ==============================

from __future__ import annotations

from models import FailureState


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def update_oil_level(
    oil_level: float,
    speed: float,
    engine_temp: float,
    low_oil_active: bool,
    action_type: str,
) -> float:
    """
    Oil decreases slowly, faster under stress and if low-oil fault exists.
    """
    drain = 0.15 + (speed / 500.0)

    if engine_temp > 105:
        drain += 0.25

    if low_oil_active:
        drain += 0.4

    if action_type in {"brake", "stop", "request_service"}:
        drain -= 0.18

    if speed < 10:
        drain -= 0.08

    return clamp(oil_level - drain, 0.0, 100.0)


def update_battery_health(
    battery_health: float,
    action_type: str,
    battery_issue_active: bool,
) -> float:
    """
    Battery health degrades very slowly.
    """
    drain = 0.03

    if action_type == "stop":
        drain += 0.01

    if battery_issue_active:
        drain += 0.25

    return clamp(battery_health - drain, 0.0, 100.0)


def infer_failure_state(
    current_failures: FailureState,
    engine_temp: float,
    oil_level: float,
    battery_health: float,
) -> FailureState:
    """
    Keep explicit faults, but also infer evolving failures from state.
    """
    return FailureState(
        brake_failure=current_failures.brake_failure,
        sensor_failure=current_failures.sensor_failure,
        engine_overheating=current_failures.engine_overheating or engine_temp >= 110.0,
        low_oil=current_failures.low_oil or oil_level <= 25.0,
        battery_issue=current_failures.battery_issue or battery_health <= 20.0,
    )


def is_engine_failure(engine_temp: float, oil_level: float) -> bool:
    """
    Catastrophic engine failure trigger.
    """
    return engine_temp >= 125.0 or oil_level <= 5.0
