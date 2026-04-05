# ==============================
# AutoMind Agent (FINAL UPGRADED)
# ==============================

from models import Action


# ---------------------------------
# HELPER: TEMPERATURE TREND
# ---------------------------------
def get_temp_trend(history):
    if len(history) < 3:
        return "stable"

    temps = [h.state_summary.get("engine_temp", 0) for h in history[-3:]]

    if temps[2] > temps[1] > temps[0]:
        return "rising"

    if temps[2] < temps[1] < temps[0]:
        return "falling"

    return "stable"


# ---------------------------------
# FAULT DIAGNOSIS (UPGRADED)
# ---------------------------------
def diagnose_fault(obs):
    """
    Multi-factor realistic diagnosis
    """

    # 🔥 Combined logic (better than simple threshold)
    if obs.engine_temp > 105 and obs.oil_level < 30:
        return "engine_overheating", "high"

    if obs.engine_temp > 110:
        return "engine_overheating", "high"

    if obs.oil_level < 25:
        return "low_oil", "medium"

    if obs.battery_health < 40:
        return "battery_issue", "medium"

    return "no_fault", "low"


# ---------------------------------
# DECISION POLICY (UPGRADED)
# ---------------------------------
def decide_action(obs, fault, urgency):

    temp_trend = get_temp_trend(obs.history)

    if obs.distance_to_obstacle < 12:
        return "brake", 1.0, "critical obstacle distance"

    if obs.distance_to_obstacle < 22 and obs.speed > 25:
        return "brake", 0.9, "closing in on obstacle"

    # 🚨 1. ESCALATION LOGIC
    if temp_trend == "rising" and obs.engine_temp > 105:
        return "stop", 0.9, "temperature rising consistently → preventive stop"

    # ⚠️ 2. FAULT HANDLING
    if fault == "engine_overheating" and urgency == "high":
        return "stop", 1.0, f"critical overheating ({obs.engine_temp})"

    if fault == "low_oil":
        return "stop", 0.7, "low oil detected"
        
    if fault == "battery_issue":
        return "stop", 0.8, "critical battery issue"

    # 🚗 3. NORMAL DRIVING / SAFELY CONTINUING
    if obs.speed > 55:
        return "continue", 0.4, "cruising safely at higher speeds"
        
    if obs.speed < 40:
        return "accelerate", 0.6, "safe acceleration"

    return "continue", 0.5, "stable driving"


def decide_action_v2(obs, fault, urgency):
    temp_trend = get_temp_trend(obs.history)

    if obs.distance_to_obstacle < 12:
        return "brake", 1.0, "critical obstacle distance"

    if obs.distance_to_obstacle < 22 and obs.speed > 25:
        return "brake", 0.9, "closing in on obstacle"

    if temp_trend == "rising" and obs.engine_temp > 105:
        if obs.distance_to_obstacle < 35 or obs.speed > 30:
            return "brake", 0.9, "temperature rising with close obstacle"
        return "stop", 0.9, "temperature rising consistently"

    if fault == "engine_overheating" and urgency == "high":
        if obs.distance_to_obstacle < 35 or obs.speed > 25:
            return "brake", 1.0, f"critical overheating with close obstacle ({obs.engine_temp})"
        return "request_service", 1.0, f"critical overheating ({obs.engine_temp})"

    if fault == "low_oil":
        if obs.distance_to_obstacle < 30 or obs.speed > 25:
            return "brake", 0.9, "low oil with close obstacle"
        return "request_service", 0.8, "low oil detected"

    if fault == "battery_issue":
        return "request_service", 0.8, "critical battery issue"

    if obs.road_condition in {"wet", "rain"} and obs.speed > 50:
        return "brake", 0.7, "reduce speed on low-traction road"

    if obs.speed > 55:
        return "continue", 0.4, "cruising safely at higher speeds"

    if obs.speed < 40:
        return "accelerate", 0.6, "safe acceleration"

    return "continue", 0.5, "stable driving"


def choose_immediate_safe_action(obs):
    if (
        obs.distance_to_obstacle < 12
        or (obs.distance_to_obstacle < 22 and obs.speed > 25)
        or obs.failures.brake_failure
        or obs.failures.engine_overheating
        or obs.engine_temp > 108
        or obs.oil_level < 24
    ):
        return "brake", 1.0, "highest-priority safety maneuver"

    if obs.speed > 85 or (obs.road_condition in {"wet", "rain"} and obs.speed > 60):
        return "brake", 0.8, "reduce speed for current road risk"

    if obs.speed < 35 and obs.distance_to_obstacle > 35:
        return "accelerate", 0.5, "safe acceleration window"

    return "continue", 0.4, "maintain safe trajectory"


# ---------------------------------
# MAIN AGENT
# ---------------------------------
def agent_step(observation, task_name="autonomous_control"):

    fault, urgency = diagnose_fault(observation)

    if task_name == "fault_diagnosis":
        return Action(action_type="diagnose", value=1.0, reason=fault)

    if task_name == "driving_decision":
        action_type, value, reason = choose_immediate_safe_action(observation)
        return Action(action_type=action_type, value=value, reason=reason)

    action_type, value, reason = decide_action_v2(
        observation, fault, urgency
    )

    return Action(
        action_type=action_type,
        value=value,
        reason=reason,
    )
