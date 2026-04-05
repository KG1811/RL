from __future__ import annotations

import random

from models import Observation, HistoryItem, TelemetryState, FailureState
from vehicle_dynamics import (
    apply_speed_decay,
    update_distance_to_obstacle,
    update_engine_temperature,
    estimate_collision_risk,
)
from failure_engine import (
    update_oil_level,
    update_battery_health,
    infer_failure_state,
    is_engine_failure,
)
from traffic_engine import (
    get_obstacle_relative_motion,
    get_traffic_pressure,
)
from noise_engine import (
    add_sensor_noise,
    maybe_corrupt_distance,
)
from gps_engine import update_gps


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class AutoMindSimulator:
    def __init__(self, seed: int = 42, dt_seconds: float = 10.0) -> None:
        self.rng = random.Random(seed)
        self.dt_seconds = dt_seconds

    def _next_drive_mode(self, speed: float, throttle: float) -> str:
        if speed < 1:
            return "idle"
        if throttle > 65:
            return "sport"
        if speed < 35:
            return "city"
        return "cruise"

    def _compute_powertrain(
        self,
        state: TelemetryState,
        action_type: str,
        action_value: float,
    ) -> dict:
        
        # Determine targets based on action
        if action_type == "accelerate":
            targetSpeed = 45.0 + 75.0 * action_value
            targetThrottle = 40.0 + 45.0 * action_value
        elif action_type == "brake":
            targetSpeed = max(0.0, state.speed - 30.0 * action_value)
            targetThrottle = 0.0
        elif action_type == "stop":
            targetSpeed = 0.0
            targetThrottle = 0.0
        elif action_type == "request_service":
            targetSpeed = max(0.0, state.speed - 10.0)
            targetThrottle = max(0.0, state.throttle - 15.0)
        elif action_type in {"turn_left", "turn_right"}:
            targetSpeed = max(0.0, state.speed - 15.0)
            targetThrottle = max(0.0, state.throttle - 10.0)
        else: # continue
            # Simulate dynamic real-world driving behavior
            if state.distance_to_obstacle > 90:
                targetSpeed = clamp(state.speed + 12.0 + self.rng.uniform(-2, 10), 40.0, 130.0)
                targetThrottle = clamp(state.throttle + 10.0, 40.0, 85.0)
            elif state.distance_to_obstacle > 40:
                targetSpeed = clamp(state.speed + self.rng.uniform(-5, 10), 20.0, 75.0)
                targetThrottle = clamp(state.throttle + self.rng.uniform(-8, 8), 20.0, 55.0)
            elif state.distance_to_obstacle < 20:
                targetSpeed = max(0.0, state.speed - 25.0)
                targetThrottle = 0.0
            else:
                targetSpeed = state.speed + self.rng.uniform(-4.0, 4.0)
                targetThrottle = state.throttle + self.rng.uniform(-5.0, 5.0)

        # Apply ECU physics to reach targets
        speed_delta = (targetSpeed - state.speed) * 0.20 + self.rng.uniform(-2.0, 2.0)
        speed = clamp(state.speed + speed_delta, 0.0, 160.0)
        
        throttle_delta = (targetThrottle - state.throttle) * 0.25 + self.rng.uniform(-3.0, 3.0)
        throttle = clamp(state.throttle + throttle_delta, 0.0, 100.0)
        
        # Apply road condition friction and brake failures
        if action_type in {"brake", "stop"}:
            brake_effect = 1.0
            if state.failures.brake_failure:
                brake_effect = 0.35
            
            friction = 1.0
            if state.road_condition == "wet":
                friction = 0.8
            elif state.road_condition == "rain":
                friction = 0.65
                
            drop_factor = 15.0 * action_value if action_type == "brake" else 25.0
            speed = clamp(state.speed - drop_factor * friction * brake_effect, 0.0, 160.0)

        acceleration = speed - state.speed

        if speed < 1:
            gear = 0
        elif speed < 20:
            gear = 1
        elif speed < 40:
            gear = 2
        elif speed < 60:
            gear = 3
        elif speed < 85:
            gear = 4
        elif speed < 120:
            gear = 5
        else:
            gear = 6

        # ECU mapped RPM: 800 + speed*32 + throttle*18 + noise
        rpm = clamp(800.0 + speed * 32.0 + throttle * 18.0 + self.rng.uniform(-120.0, 120.0), 700.0, 5200.0)
        if gear == 0:
            rpm = clamp(750.0 + throttle * 4.0 + self.rng.uniform(-50.0, 50.0), 700.0, 5200.0)

        engine_load = clamp((throttle * 0.75) + (speed * 0.35), 0.0, 100.0)
        transmission_load = clamp((speed * 0.45) + (gear * 6.0), 0.0, 100.0)
        fuel_rate = clamp(0.6 + (rpm / 2800.0) + (throttle / 55.0), 0.0, 40.0)
        drive_mode = self._next_drive_mode(speed=speed, throttle=throttle)

        return {
            "speed": speed,
            "throttle": throttle,
            "acceleration": clamp(acceleration, -12.0, 12.0),
            "gear": gear,
            "rpm": rpm,
            "engine_load": engine_load,
            "transmission_load": transmission_load,
            "fuel_rate": fuel_rate,
            "drive_mode": drive_mode,
        }

    def transition(
        self,
        state: TelemetryState,
        previous_observation: Observation,
        action_type: str,
        action_value: float,
        difficulty: str,
    ) -> dict:
        powertrain = self._compute_powertrain(
            state=state,
            action_type=action_type,
            action_value=action_value,
        )

        obstacle_relative_motion = get_obstacle_relative_motion(
            rng=self.rng,
            difficulty=difficulty,
        )

        traffic_pressure = get_traffic_pressure(
            rng=self.rng,
            difficulty=difficulty,
        )

        distance_to_obstacle = update_distance_to_obstacle(
            current_distance=state.distance_to_obstacle,
            speed=powertrain["speed"],
            obstacle_relative_motion=obstacle_relative_motion,
        )

        distance_to_obstacle = max(0.0, distance_to_obstacle - (traffic_pressure * 2.2))

        engine_temp = update_engine_temperature(
            engine_temp=state.engine_temp,
            rpm=powertrain["rpm"],
            action_type=action_type,
            road_condition=state.road_condition,
            overheating_active=state.failures.engine_overheating,
            rng=self.rng,
        )

        oil_level = update_oil_level(
            oil_level=state.oil_level,
            speed=powertrain["speed"],
            engine_temp=engine_temp,
            low_oil_active=state.failures.low_oil,
            action_type=action_type,
        )

        battery_health = update_battery_health(
            battery_health=state.battery_health,
            action_type=action_type,
            battery_issue_active=state.failures.battery_issue,
        )

        failures = infer_failure_state(
            current_failures=state.failures,
            engine_temp=engine_temp,
            oil_level=oil_level,
            battery_health=battery_health,
        )

        collision_risk = estimate_collision_risk(
            speed=powertrain["speed"],
            distance_to_obstacle=distance_to_obstacle,
            road_condition=state.road_condition,
            brake_failure=failures.brake_failure,
        )

        is_collision = distance_to_obstacle <= 0.0 or collision_risk >= 0.97
        engine_failure = is_engine_failure(
            engine_temp=engine_temp,
            oil_level=oil_level,
        )

        heading = (state.heading + self.rng.uniform(-4.0, 4.0)) % 360.0
        latitude, longitude = update_gps(
            lat=state.latitude,
            lon=state.longitude,
            speed_kmph=powertrain["speed"],
            heading_deg=heading,
            dt_seconds=self.dt_seconds,
        )

        next_state = TelemetryState(
            speed=round(powertrain["speed"], 2),
            rpm=round(powertrain["rpm"], 2),
            throttle=round(powertrain["throttle"], 2),
            gear=powertrain["gear"],
            engine_load=round(powertrain["engine_load"], 2),
            transmission_load=round(powertrain["transmission_load"], 2),
            fuel_rate=round(powertrain["fuel_rate"], 2),
            acceleration=round(powertrain["acceleration"], 2),
            engine_temp=round(clamp(engine_temp, 0.0, 150.0), 2),
            distance_to_obstacle=round(clamp(distance_to_obstacle, 0.0, 300.0), 2),
            road_condition=state.road_condition,
            drive_mode=powertrain["drive_mode"],
            oil_level=round(clamp(oil_level, 0.0, 100.0), 2),
            battery_health=round(clamp(battery_health, 0.0, 100.0), 2),
            latitude=round(latitude, 6),
            longitude=round(longitude, 6),
            heading=round(heading, 2),
            failures=failures,
        )

        observed_speed = add_sensor_noise(
            rng=self.rng,
            value=next_state.speed,
            std_dev=1.2,
            low=0.0,
            high=220.0,
        )

        observed_rpm = add_sensor_noise(
            rng=self.rng,
            value=next_state.rpm,
            std_dev=45.0,
            low=0.0,
            high=8000.0,
        )

        observed_engine_temp = add_sensor_noise(
            rng=self.rng,
            value=next_state.engine_temp,
            std_dev=1.0,
            low=0.0,
            high=150.0,
        )

        observed_distance = maybe_corrupt_distance(
            rng=self.rng,
            value=next_state.distance_to_obstacle,
            sensor_failure=failures.sensor_failure,
        )

        updated_history = list(previous_observation.history)
        updated_history.append(
            HistoryItem(
                state_summary={
                    "speed": round(previous_observation.speed, 2),
                    "rpm": round(previous_observation.rpm, 2),
                    "engine_temp": round(previous_observation.engine_temp, 2),
                    "oil_level": round(previous_observation.oil_level, 2),
                    "battery_health": round(previous_observation.battery_health, 2),
                },
                action_taken=action_type,
            )
        )

        observation = Observation(
            speed=round(observed_speed, 2),
            rpm=round(observed_rpm, 2),
            throttle=next_state.throttle,
            gear=next_state.gear,
            engine_load=next_state.engine_load,
            transmission_load=next_state.transmission_load,
            fuel_rate=next_state.fuel_rate,
            acceleration=next_state.acceleration,
            engine_temp=round(observed_engine_temp, 2),
            distance_to_obstacle=round(observed_distance, 2),
            road_condition=next_state.road_condition,
            drive_mode=next_state.drive_mode,
            oil_level=next_state.oil_level,
            battery_health=next_state.battery_health,
            latitude=next_state.latitude,
            longitude=next_state.longitude,
            heading=next_state.heading,
            failures=next_state.failures,
            history=updated_history[-8:],
        )

        return {
            "true_state": next_state,
            "observation": observation,
            "collision_risk": round(collision_risk, 3),
            "traffic_pressure": round(traffic_pressure, 3),
            "is_collision": is_collision,
            "is_engine_failure": engine_failure,
        }
