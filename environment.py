# ==============================
# AutoMind OpenEnv - Environment (FINAL FIXED)
# ==============================

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
)

from simulator import AutoMindSimulator


class AutoMindEnv:

    def __init__(self, seed: int = 42, max_steps: int = 20) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps

        self.simulator = AutoMindSimulator(seed=seed)

        self.current_task: Optional[str] = None
        self.current_difficulty: Optional[str] = None

        self.episode_state: EpisodeState = EpisodeState(max_steps=max_steps)
        self.current_observation: Optional[Observation] = None

        self.override_active = False
        self.override_count = 0

    # ---------------------------------
    # INITIAL STATE
    # ---------------------------------
    def _build_initial_observation(self, task_name: str, difficulty: str) -> Observation:

        if difficulty == "easy":
            return Observation(
                speed=35.0,
                engine_temp=88.0,
                distance_to_obstacle=60.0,
                road_condition="dry",
                oil_level=78.0,
                battery_health=84.0,
                failures=FailureState(),
                history=[],
            )

        if difficulty == "medium":
            return Observation(
                speed=48.0,
                engine_temp=98.0,
                distance_to_obstacle=28.0,
                road_condition="wet",
                oil_level=32.0,
                battery_health=70.0,
                failures=FailureState(low_oil=True),
                history=[],
            )

        if difficulty == "hard":
            return Observation(
                speed=62.0,
                engine_temp=114.0,
                distance_to_obstacle=14.0,
                road_condition="rain",
                oil_level=18.0,
                battery_health=61.0,
                failures=FailureState(
                    brake_failure=True,
                    sensor_failure=True,
                    engine_overheating=True,
                    low_oil=True,
                ),
                history=[],
            )

        raise ValueError("Invalid difficulty")

    # ---------------------------------
    # RESET
    # ---------------------------------
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

        self.current_observation = self._build_initial_observation(task_name, difficulty)

        self.override_active = False
        self.override_count = 0

        return self.current_observation

    # ---------------------------------
    # STATE (OLD - KEEP)
    # ---------------------------------
    def state(self) -> Observation:
        if self.current_observation is None:
            raise RuntimeError("Call reset() first")
        return self.current_observation

    # ---------------------------------
    # 🔥 NEW: FULL STATE (FIX FOR UI)
    # ---------------------------------
    def get_full_state(self):

        if self.current_observation is None:
            raise RuntimeError("Call reset() first")

        sim_result = self.simulator.transition(
            observation=self.current_observation,
            action_type="continue",
            action_value=0.0,
            difficulty=self.current_difficulty,
        )

        new_observation = sim_result["observation"]

        self.current_observation = new_observation

        metrics = self._compute_metrics(sim_result)

        return {
            "observation": new_observation.model_dump(),
            "metrics": metrics.model_dump(),
            "info": {
                "collision_risk": sim_result["collision_risk"],
                "override_active": self.override_active,
            }
        }

    # ---------------------------------
    # STEP (NO CHANGE)
    # ---------------------------------
    def step(self, action: Action) -> StepResult:

        if self.current_observation is None:
            raise RuntimeError("Call reset() before step()")

        if self.current_difficulty is None:
            raise RuntimeError("Call reset() first.")

        if self.episode_state.step_count % 3 == 0:
            self.override_active = True
        else:
            self.override_active = False

        if self.override_active and action.action_type in ["stop", "brake"]:
            action = Action(
                action_type="accelerate",
                value=action.value,
                reason="Human override",
            )

        sim_result = self.simulator.transition(
            observation=self.current_observation,
            action_type=action.action_type,
            action_value=action.value,
            difficulty=self.current_difficulty,
        )

        new_observation = sim_result["observation"]

        self.episode_state.step_count += 1
        self.episode_state.is_collision |= sim_result["is_collision"]
        self.episode_state.is_engine_failure = sim_result["is_engine_failure"]

        if action.action_type == "stop" and new_observation.speed <= 3:
            self.episode_state.is_safe_stop = True

        reward = self._compute_reward(action, sim_result, new_observation)
        done = self._check_done()
        metrics = self._compute_metrics(sim_result)

        self.current_observation = new_observation

        return StepResult(
            observation=new_observation,
            reward=reward,
            done=done,
            info={
                "outcome": self.get_episode_outcome(),
                "collision_risk": sim_result["collision_risk"],
                "override_active": self.override_active,
            },
            metrics=metrics,
        )

    # ---------------------------------
    def _compute_reward(self, action, sim_result, new_observation):
        if sim_result["is_collision"]:
            return -1.0
        return 0.4 * (1 - sim_result["collision_risk"])

    def _compute_metrics(self, sim_result):
        return Metrics(
            safety_score=1.0 if not sim_result["is_collision"] else 0.0,
            efficiency_score=max(0.0, 1 - sim_result["collision_risk"]),
            diagnosis_score=1.0,
            sequence_score=1.0,
        )

    def _check_done(self):
        return (
            self.episode_state.is_collision or
            self.episode_state.is_engine_failure or
            self.episode_state.is_safe_stop or
            self.episode_state.step_count >= self.max_steps
        )

    def get_episode_outcome(self):
        if self.episode_state.is_collision:
            return "failure_collision"
        if self.episode_state.is_engine_failure:
            return "failure_engine"
        if self.episode_state.is_safe_stop:
            return "success_safe_stop"
        return "in_progress"