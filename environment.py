from __future__ import annotations

import random
from typing import Optional

from models import (
    Observation,
    FailureState,
    EpisodeState,
    Action,
    StepResult,
    Metrics,
    TelemetryState,
)
from simulator import AutoMindSimulator
from service_engine import find_nearest_service
from tasks import grade_driving_decision, grade_fault_diagnosis


class AutoMindEnv:
    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 20,
        update_interval_seconds: int = 10,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.update_interval_seconds = update_interval_seconds

        self.simulator = AutoMindSimulator(seed=seed, dt_seconds=float(update_interval_seconds))

        self.current_task: Optional[str] = None
        self.current_difficulty: Optional[str] = None

        self.episode_state: EpisodeState = EpisodeState(max_steps=max_steps)
        self.current_observation: Optional[Observation] = None
        self.current_true_state: Optional[TelemetryState] = None

        self.last_action = Action(action_type="continue", value=0.25, reason="background cruise")
        self.override_active = False
        self.override_count = 0

        self.last_metrics = Metrics(
            safety_score=1.0,
            efficiency_score=1.0,
            diagnosis_score=0.5,
            sequence_score=0.0,
        )
        self.last_reward = 0.0
        self.last_done = False
        self.last_info: dict = {
            "outcome": "not_initialized",
            "collision_risk": 0.0,
            "override_active": False,
            "override_count": 0,
            "step_count": 0,
            "health_score": 100,
            "alerts": [],
            "service_recommended": None,
            "service_booking": None,
            "task_score": 0.0,
        }



    def is_initialized(self) -> bool:
        return self.current_observation is not None

    def _build_initial_telemetry_state(self, difficulty: str) -> TelemetryState:
        if difficulty == "easy":
            return TelemetryState(
                speed=18.0,
                rpm=1200.0,
                throttle=18.0,
                gear=1,
                engine_load=24.0,
                transmission_load=18.0,
                fuel_rate=1.4,
                acceleration=0.6,
                engine_temp=86.0,
                distance_to_obstacle=85.0,
                road_condition="dry",
                drive_mode="city",
                oil_level=82.0,
                battery_health=88.0,
                latitude=28.613900,
                longitude=77.209000,
                heading=0.0,
                failures=FailureState(),
            )

        if difficulty == "medium":
            return TelemetryState(
                speed=42.0,
                rpm=2200.0,
                throttle=34.0,
                gear=3,
                engine_load=48.0,
                transmission_load=42.0,
                fuel_rate=2.6,
                acceleration=0.3,
                engine_temp=97.0,
                distance_to_obstacle=48.0,
                road_condition="wet",
                drive_mode="cruise",
                oil_level=31.0,
                battery_health=73.0,
                latitude=28.613900,
                longitude=77.209000,
                heading=22.0,
                failures=FailureState(low_oil=True),
            )

        if difficulty == "hard":
            return TelemetryState(
                speed=44.0,
                rpm=2580.0,
                throttle=36.0,
                gear=3,
                engine_load=58.0,
                transmission_load=46.0,
                fuel_rate=2.8,
                acceleration=0.1,
                engine_temp=109.0,
                distance_to_obstacle=68.0,
                road_condition="rain",
                drive_mode="cruise",
                oil_level=24.0,
                battery_health=62.0,
                latitude=28.613900,
                longitude=77.209000,
                heading=38.0,
                failures=FailureState(
                    brake_failure=True,
                    sensor_failure=True,
                    engine_overheating=True,
                    low_oil=True,
                ),
            )

        raise ValueError("Invalid difficulty")

    def _telemetry_to_observation(self, telemetry: TelemetryState) -> Observation:
        return Observation(
            speed=telemetry.speed,
            rpm=telemetry.rpm,
            throttle=telemetry.throttle,
            gear=telemetry.gear,
            engine_load=telemetry.engine_load,
            transmission_load=telemetry.transmission_load,
            fuel_rate=telemetry.fuel_rate,
            acceleration=telemetry.acceleration,
            engine_temp=telemetry.engine_temp,
            distance_to_obstacle=telemetry.distance_to_obstacle,
            road_condition=telemetry.road_condition,
            drive_mode=telemetry.drive_mode,
            oil_level=telemetry.oil_level,
            battery_health=telemetry.battery_health,
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            heading=telemetry.heading,
            failures=telemetry.failures,
            history=[],
        )

    def reset(self, task_name: str = "fault_diagnosis", difficulty: str = "easy") -> Observation:
        self.current_task = task_name
        self.current_difficulty = difficulty

        self.episode_state = EpisodeState(
            step_count=0,
            max_steps=self.max_steps,
            is_collision=False,
            is_engine_failure=False,
            is_safe_stop=False,
        )

        self.current_true_state = self._build_initial_telemetry_state(difficulty=difficulty)
        self.current_observation = self._telemetry_to_observation(self.current_true_state)

        self.override_active = False
        self.override_count = 0
        self.last_action = Action(action_type="continue", value=0.25, reason="background cruise")
        self.last_done = False
        self.last_reward = 0.0
        self.last_info = {
            "outcome": "in_progress",
            "collision_risk": 0.0,
            "override_active": False,
            "override_count": 0,
            "step_count": 0,
            "health_score": 100,
            "alerts": [],
            "service_recommended": None,
            "service_booking": None,
            "task_score": 0.0,
        }

        self.last_metrics = self._compute_metrics(
            collision_risk=0.02,
            observation=self.current_observation,
        )
        self.last_info = self._build_info(
            observation=self.current_observation,
            collision_risk=0.02,
            action_type="continue",
        )

        return self.current_observation

    def state(self) -> Observation:
        if self.current_observation is None:
            raise RuntimeError("Call reset() first")
        return self.current_observation

    def compute_health(self, obs: Observation, collision_risk: float) -> int:
        score = 100.0

        if obs.engine_temp > 110:
            score -= 22
        elif obs.engine_temp > 100:
            score -= 10

        if obs.oil_level < 20:
            score -= 24
        elif obs.oil_level < 35:
            score -= 10

        if obs.battery_health < 30:
            score -= 18
        elif obs.battery_health < 50:
            score -= 8



        if obs.failures.brake_failure:
            score -= 12
        if obs.failures.sensor_failure:
            score -= 8
        if obs.failures.engine_overheating:
            score -= 14

        return max(0, min(100, int(round(score))))

    def get_alerts(self, obs: Observation, collision_risk: float) -> list[str]:
        alerts: list[str] = []

        if obs.engine_temp > 110 or obs.failures.engine_overheating:
            alerts.append("ENGINE OVERHEATING")
        if obs.oil_level < 25 or obs.failures.low_oil:
            alerts.append("LOW OIL")
        if obs.battery_health < 30 or obs.failures.battery_issue:
            alerts.append("BATTERY ISSUE")
        if collision_risk > 0.70:
            alerts.append("COLLISION RISK")
        if obs.failures.brake_failure:
            alerts.append("BRAKE FAILURE")
        if obs.failures.sensor_failure:
            alerts.append("SENSOR FAILURE")

        return alerts

    def _build_service_payload(
        self,
        observation: Observation,
        health_score: int,
        alerts: list[str],
        action_type: str,
    ) -> tuple[Optional[dict], Optional[dict]]:
        service_recommended = None
        service_booking = None

        severe = (
            health_score < 55
            or "ENGINE OVERHEATING" in alerts
            or "BRAKE FAILURE" in alerts
            or "BATTERY ISSUE" in alerts
        )

        if severe:
            service_recommended = find_nearest_service(
                observation.latitude,
                observation.longitude,
            )

        should_book = action_type == "request_service" or health_score < 35

        if service_recommended and should_book:
            service_booking = {
                "status": "booked",
                "center_name": service_recommended["name"],
                "distance_km": service_recommended["distance_km"],
                "eta_minutes": service_recommended["eta_minutes"],
                "slots_available": service_recommended["slots_available"],
                "urgency": "HIGH" if health_score < 35 else "MEDIUM",
            }

        return service_recommended, service_booking

    def _build_info(self, observation: Observation, collision_risk: float, action_type: str) -> dict:
        health_score = self.compute_health(observation, collision_risk)
        alerts = self.get_alerts(observation, collision_risk)
        service_recommended, service_booking = self._build_service_payload(
            observation=observation,
            health_score=health_score,
            alerts=alerts,
            action_type=action_type,
        )

        return {
            "outcome": self.get_episode_outcome(),
            "collision_risk": round(collision_risk, 3),
            "override_active": self.override_active,
            "override_count": self.override_count,
            "step_count": self.episode_state.step_count,
            "health_score": health_score,
            "alerts": alerts,
            "service_recommended": service_recommended,
            "service_booking": service_booking,
            "task_score": self.last_info.get("task_score", 0.0),
        }

    def _finish_task_episode(
        self,
        reward: float,
        info_updates: dict,
        metrics: Metrics,
    ) -> StepResult:
        self.last_reward = max(0.0, min(1.0, round(reward, 3)))
        self.last_done = True
        self.last_metrics = metrics
        self.last_info = {
            **self.last_info,
            **info_updates,
            "override_active": self.override_active,
            "override_count": self.override_count,
            "step_count": self.episode_state.step_count,
            "task_score": round(reward, 3),
        }

        return StepResult(
            observation=self.current_observation,
            reward=self.last_reward,
            done=self.last_done,
            info=self.last_info,
            metrics=self.last_metrics,
        )

    def _compute_reward(
        self,
        action: Action,
        collision_risk: float,
        observation: Observation,
        is_collision: bool,
    ) -> float:
        if is_collision:
            return -1.0

        safety = 1.0 - collision_risk
        efficiency = max(0.0, min(1.0, observation.speed / 90.0))
        action_quality = 1.0 if action.action_type in ["brake", "stop", "continue", "request_service"] else 0.6
        sequence_quality = min(1.0, max(0.2, len(observation.history) / 6.0))

        reward = (
            0.4 * safety
            + 0.2 * efficiency
            + 0.2 * action_quality
            + 0.2 * sequence_quality
        )

        return max(-1.0, min(1.0, round(reward, 3)))

    def _compute_metrics(self, collision_risk: float, observation: Observation) -> Metrics:
        safety = 0.0 if collision_risk >= 0.97 else max(0.0, 1.0 - collision_risk)
        efficiency = max(0.0, min(1.0, observation.speed / 90.0))

        diagnosis = 0.5
        if (
            observation.engine_temp > 105
            or observation.oil_level < 25
            or observation.battery_health < 30
            or observation.failures.engine_overheating
            or observation.failures.low_oil
            or observation.failures.battery_issue
        ):
            diagnosis = 1.0

        sequence = min(1.0, self.episode_state.step_count / 10.0)

        return Metrics(
            safety_score=round(safety, 3),
            efficiency_score=round(efficiency, 3),
            diagnosis_score=round(diagnosis, 3),
            sequence_score=round(sequence, 3),
        )

    def _check_done(self) -> bool:
        return self.episode_state.check_done()

    def get_episode_outcome(self) -> str:
        if self.episode_state.is_collision:
            return "failure_collision"
        if self.episode_state.is_engine_failure:
            return "failure_engine"
        if self.episode_state.is_safe_stop:
            return "success_safe_stop"
        if self.episode_state.step_count >= self.max_steps:
            return "episode_timeout"
        return "in_progress"

    def step(self, action: Action) -> StepResult:
        if self.current_observation is None or self.current_true_state is None:
            raise RuntimeError("Call reset() before step()")
        if self.current_difficulty is None:
            raise RuntimeError("Call reset() first.")

        if self.current_task == "fault_diagnosis":
            self.last_action = action
            self.episode_state.step_count += 1
            score = grade_fault_diagnosis(action, self.current_observation)
            self.last_info = self._build_info(
                observation=self.current_observation,
                collision_risk=0.02,
                action_type=action.action_type,
            )
            return self._finish_task_episode(
                reward=score,
                info_updates={
                    "outcome": "success_diagnosis" if score >= 1.0 else "failure_diagnosis",
                },
                metrics=Metrics(
                    safety_score=1.0,
                    efficiency_score=0.0,
                    diagnosis_score=score,
                    sequence_score=1.0,
                ),
            )

        pre_action_observation = self.current_observation

        self.override_active = self.episode_state.step_count > 0 and self.episode_state.step_count % 3 == 0

        applied_action = action
        if (
            self.override_active
            and action.action_type in ["stop", "brake"]
            and self.current_observation.distance_to_obstacle > 25
        ):
            self.override_count += 1
            applied_action = Action(
                action_type="accelerate",
                value=min(1.0, max(0.2, action.value)),
                reason="Human override",
            )

        transition = self.simulator.transition(
            state=self.current_true_state,
            previous_observation=self.current_observation,
            action_type=applied_action.action_type,
            action_value=applied_action.value,
            difficulty=self.current_difficulty,
        )

        self.current_true_state = transition["true_state"]
        self.current_observation = transition["observation"]
        self.last_action = applied_action

        self.episode_state.step_count += 1
        self.episode_state.is_collision |= transition["is_collision"]
        self.episode_state.is_engine_failure |= transition["is_engine_failure"]

        if applied_action.action_type == "stop" and self.current_observation.speed <= 3.0:
            self.episode_state.is_safe_stop = True

        self.last_metrics = self._compute_metrics(
            collision_risk=transition["collision_risk"],
            observation=self.current_observation,
        )
        self.last_reward = self._compute_reward(
            action=applied_action,
            collision_risk=transition["collision_risk"],
            observation=self.current_observation,
            is_collision=transition["is_collision"],
        )
        self.last_done = self._check_done()
        self.last_info = self._build_info(
            observation=self.current_observation,
            collision_risk=transition["collision_risk"],
            action_type=applied_action.action_type,
        )
        self.last_info["task_score"] = round(self.last_reward, 3)

        if self.current_task == "driving_decision":
            decision_score = grade_driving_decision(action, pre_action_observation)
            decision_reward = 0.75 * decision_score + 0.25 * self.last_metrics.safety_score
            return self._finish_task_episode(
                reward=decision_reward,
                info_updates={
                    "outcome": (
                        "success_safe_decision"
                        if decision_score >= 0.7
                        and not transition["is_collision"]
                        and transition["collision_risk"] < 0.7
                        else "failure_unsafe_decision"
                    ),
                    "decision_score": round(decision_score, 3),
                },
                metrics=Metrics(
                    safety_score=self.last_metrics.safety_score,
                    efficiency_score=self.last_metrics.efficiency_score,
                    diagnosis_score=decision_score,
                    sequence_score=1.0,
                ),
            )

        return StepResult(
            observation=self.current_observation,
            reward=self.last_reward,
            done=self.last_done,
            info=self.last_info,
            metrics=self.last_metrics,
        )

    def get_full_state(self) -> dict:
        if self.current_observation is None:
            raise RuntimeError("Call reset() first")

        return {
            "observation": self.current_observation.model_dump(),
            "reward": self.last_reward,
            "done": self.last_done,
            "metrics": self.last_metrics.model_dump(),
            "info": self.last_info,
        }
