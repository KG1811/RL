from fastapi import FastAPI, Body
from models import Action, Observation, StepResult
from environment import AutoMindEnv

app = FastAPI(title="AutoMind OpenEnv", version="1.0.0")
env = AutoMindEnv()


@app.get("/")
def root():
    return {
        "status": "AutoMind OpenEnv running",
        "initialized": env.is_initialized(),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "initialized": env.is_initialized(),
        "task": env.current_task,
        "difficulty": env.current_difficulty,
        "update_interval_seconds": env.update_interval_seconds,
    }


@app.post("/reset")
def reset(payload: dict = Body(default_factory=dict)):
    task_name = payload.get("task_name", "fault_diagnosis")
    difficulty = payload.get("difficulty", "easy")
    obs = env.reset(task_name=task_name, difficulty=difficulty)
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(action: Action):
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state():
    return env.get_full_state()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "fault_diagnosis",
                "difficulty": "easy",
                "goal": "Detect overheating, low oil, or battery issue correctly",
            },
            {
                "name": "driving_decision",
                "difficulty": "medium",
                "goal": "Choose the safest immediate action from current vehicle context",
            },
            {
                "name": "autonomous_control",
                "difficulty": "hard",
                "goal": "Handle diagnosis, safety, override, GPS-aware service recommendation, and control",
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