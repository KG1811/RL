# ==============================
# AutoMind OpenEnv - Tasks & Graders (Phase 7 FINAL)
# ==============================

from __future__ import annotations

from typing import Optional
from models import Observation, Action, Metrics


# =====================================
# TASK CONFIG
# =====================================

TASK_CONFIG = {
    "fault_diagnosis": {
        "allowed_actions": ["diagnose"],
        "goal": "Identify vehicle fault correctly",
    },
    "driving_decision": {
        "allowed_actions": [
            "brake",
            "accelerate",
            "turn_left",
            "turn_right",
            "continue",
            "stop",
        ],
        "goal": "Choose safest immediate driving action",
    },
    "autonomous_control": {
        "allowed_actions": [
            "brake",
            "accelerate",
            "turn_left",
            "turn_right",
            "continue",
            "stop",
            "request_service",
        ],
        "goal": "Full control with safety + diagnosis + efficiency",
    },
}


# =====================================
# TASK 1 — FAULT DIAGNOSIS
# =====================================

def detect_true_fault(observation: Observation) -> str:
    """
    Deterministic ground truth fault detection.
    """

    if observation.engine_temp >= 105:
        return "engine_overheating"

    if observation.oil_level <= 25:
        return "low_oil"

    if observation.battery_health <= 20:
        return "battery_issue"

    return "no_fault"


def grade_fault_diagnosis(action: Optional[Action], observation: Observation) -> float:
    """
    Strict deterministic scoring.
    """

    if action is None or action.action_type != "diagnose":
        return 0.0

    predicted_fault = action.reason.strip().lower()
    true_fault = detect_true_fault(observation)

    if predicted_fault == true_fault:
        return 1.0

    if true_fault != "no_fault" and predicted_fault != "no_fault":
        return 0.5

    return 0.0


# =====================================
# TASK 2 — DRIVING DECISION
# =====================================

def get_safe_action(observation: Observation) -> str:
    """
    Deterministic safe driving policy.
    """

    if (
        observation.failures.brake_failure
        or observation.failures.engine_overheating
        or observation.engine_temp > 108
        or observation.oil_level < 24
    ):
        if observation.distance_to_obstacle < 18:
            return "brake"
        return "stop"

    if observation.distance_to_obstacle < 15:
        return "brake"

    if observation.distance_to_obstacle < 28 and observation.speed > 35:
        return "brake"

    if observation.speed > 90 or (observation.speed > 70 and observation.acceleration > 2.0):
        return "brake"

    if observation.speed > 55:
        return "continue"

    if observation.speed < 40:
        return "accelerate"

    return "continue"


def grade_driving_decision(action: Action, observation: Observation) -> float:
    """
    Deterministic grading with partial credit.
    """

    action_type = action.action_type
    correct_action = get_safe_action(observation)

    if action_type == correct_action:
        return 1.0

    if correct_action == "brake" and action_type == "stop":
        return 0.7

    if correct_action == "stop" and action_type == "brake":
        return 0.7

    if correct_action == "continue" and action_type == "accelerate":
        return 0.6

    if correct_action == "accelerate" and action_type == "continue":
        return 0.6

    if correct_action == "continue" and action_type == "brake":
        return 0.4

    if observation.distance_to_obstacle < 20 and action_type == "continue":
        return 0.0

    return 0.0


# =====================================
# TASK 3 — FULL AUTONOMOUS CONTROL
# =====================================

def grade_autonomous_control(
    metrics: Metrics,
    info: Optional[dict] = None,
    action: Optional[Action] = None,
) -> float:
    """
    Final weighted deterministic score.
    """

    score = (
        0.35 * metrics.safety_score
        + 0.20 * metrics.diagnosis_score
        + 0.15 * metrics.efficiency_score
        + 0.10 * metrics.sequence_score
    )

    if info:
        outcome = info.get("outcome", "")
        alerts = info.get("alerts", [])
        service_booking = info.get("service_booking")
        service_recommended = info.get("service_recommended")
        health_score = float(info.get("health_score", 0))
        collision_risk = float(info.get("collision_risk", 1.0))
        severe_alert = any(
            alert in alerts
            for alert in ["ENGINE OVERHEATING", "BRAKE FAILURE", "BATTERY ISSUE", "LOW OIL"]
        )

        if outcome == "success_safe_stop":
            score += 0.12
        elif outcome == "episode_timeout" and collision_risk < 0.35 and health_score >= 45:
            score += 0.14
        elif outcome.startswith("failure"):
            score -= 0.20

        if severe_alert and service_booking:
            score += 0.12
        elif severe_alert and service_recommended and action and action.action_type == "request_service":
            score += 0.08
        elif severe_alert and action and action.action_type == "accelerate":
            score -= 0.10

    return round(max(0.0, min(1.0, score)), 3)


# =====================================
# MASTER EVALUATION FUNCTION
# =====================================

def evaluate_task(
    task_name: str,
    action: Optional[Action],
    observation: Observation,
    metrics: Optional[Metrics],
    info: Optional[dict] = None,
) -> float:

    if task_name == "fault_diagnosis":
        return grade_fault_diagnosis(action, observation)

    if task_name == "driving_decision":
        if action is None:
            return 0.0
        score = grade_driving_decision(action, observation)
        if info and info.get("outcome") == "failure_unsafe_decision":
            score = min(score, 0.5)
        return score

    if task_name == "autonomous_control":
        if metrics is None:
            return 0.0
        return grade_autonomous_control(metrics, info=info, action=action)

    raise ValueError(f"Unknown task: {task_name}")
