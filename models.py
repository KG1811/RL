from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class FailureState(BaseModel):
    brake_failure: bool = False
    sensor_failure: bool = False
    engine_overheating: bool = False
    low_oil: bool = False
    battery_issue: bool = False


class HistoryItem(BaseModel):
    state_summary: Dict[str, float]
    action_taken: Optional[str] = None


class Observation(BaseModel):
    speed: float = Field(..., ge=0, le=220)
    rpm: float = Field(..., ge=0, le=8000)
    throttle: float = Field(..., ge=0, le=100)
    gear: int = Field(..., ge=0, le=6)
    engine_load: float = Field(..., ge=0, le=100)
    transmission_load: float = Field(..., ge=0, le=100)
    fuel_rate: float = Field(..., ge=0, le=40)
    acceleration: float = Field(..., ge=-12, le=12)

    engine_temp: float = Field(..., ge=0, le=150)
    distance_to_obstacle: float = Field(..., ge=0, le=300)
    road_condition: str
    drive_mode: str

    oil_level: float = Field(..., ge=0, le=100)
    battery_health: float = Field(..., ge=0, le=100)

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    heading: float = Field(..., ge=0, le=360)

    failures: FailureState
    history: List[HistoryItem]


class TelemetryState(BaseModel):
    speed: float = Field(..., ge=0, le=220)
    rpm: float = Field(..., ge=0, le=8000)
    throttle: float = Field(..., ge=0, le=100)
    gear: int = Field(..., ge=0, le=6)
    engine_load: float = Field(..., ge=0, le=100)
    transmission_load: float = Field(..., ge=0, le=100)
    fuel_rate: float = Field(..., ge=0, le=40)
    acceleration: float = Field(..., ge=-12, le=12)

    engine_temp: float = Field(..., ge=0, le=150)
    distance_to_obstacle: float = Field(..., ge=0, le=300)
    road_condition: str
    drive_mode: str

    oil_level: float = Field(..., ge=0, le=100)
    battery_health: float = Field(..., ge=0, le=100)

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    heading: float = Field(..., ge=0, le=360)

    failures: FailureState


class Action(BaseModel):
    action_type: str
    value: float = Field(..., ge=0, le=1)
    reason: str


class Metrics(BaseModel):
    safety_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    diagnosis_score: float = Field(..., ge=0.0, le=1.0)
    sequence_score: float = Field(..., ge=0.0, le=1.0)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
    metrics: Metrics


class EpisodeState(BaseModel):
    step_count: int = 0
    max_steps: int = 20
    is_collision: bool = False
    is_engine_failure: bool = False
    is_safe_stop: bool = False

    def check_done(self) -> bool:
        return (
            self.is_collision
            or self.is_engine_failure
            or self.is_safe_stop
            or self.step_count >= self.max_steps
        )
