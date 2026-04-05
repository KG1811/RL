from fastapi import FastAPI, Body, Request
from models import Action, Observation, StepResult
from environment import AutoMindEnv
import threading

app = FastAPI(title="AutoMind OpenEnv Fleet Benchmark", version="1.0.0")

envs: dict[str, AutoMindEnv] = {}
envs_lock = threading.Lock()

def get_env(car_id: str) -> AutoMindEnv:
    with envs_lock:
        if car_id not in envs:
            envs[car_id] = AutoMindEnv()
            envs[car_id].reset()
        return envs[car_id]

@app.get("/")
def root():
    return {
        "status": "AutoMind OpenEnv fleet maintenance benchmark running",
        "active_cars": list(envs.keys()),
    }

@app.get("/health")
def health(car_id: str = "default"):
    env = get_env(car_id)
    return {
        "status": "healthy",
        "initialized": env.is_initialized(),
        "task": env.current_task,
        "difficulty": env.current_difficulty,
        "update_interval_seconds": env.update_interval_seconds,
        "car_id": car_id,
    }

@app.post("/reset")
def reset(payload: dict = Body(default_factory=dict), car_id: str = "default"):
    cid = payload.get("car_id", car_id)
    task_name = payload.get("task_name", "fault_diagnosis")
    difficulty = payload.get("difficulty", "easy")
    env = get_env(cid)
    obs = env.reset(task_name=task_name, difficulty=difficulty)
    return {"observation": obs.model_dump(), "car_id": cid}

@app.post("/step")
def step(action: Action, car_id: str = "default"):
    env = get_env(car_id)
    if not env.is_initialized():
        env.reset()
    result = env.step(action)
    return result.model_dump()

@app.get("/state")
def state(car_id: str = "default"):
    env = get_env(car_id)
    if not env.is_initialized():
        env.reset()
    return env.get_full_state()

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "fault_diagnosis",
                "difficulty": "easy",
                "goal": "Identify the active fleet maintenance issue from telemetry before dispatching service",
            },
            {
                "name": "driving_decision",
                "difficulty": "medium",
                "goal": "Choose the safest immediate roadside maneuver from current vehicle context",
            },
            {
                "name": "autonomous_control",
                "difficulty": "hard",
                "goal": "Recover the vehicle safely while handling overrides and coordinating roadside service",
            },
        ]
    }

@app.get("/schema")
def schema():
    return {
        "Observation": Observation.model_json_schema(),
        "Action": Action.model_json_schema(),
        "StepResult": StepResult.model_json_schema(),
    }
