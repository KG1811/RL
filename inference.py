import json
import os
from typing import Optional

import requests
from openai import OpenAI

from agent import agent_step
from environment import AutoMindEnv
from models import Action, Metrics, Observation
from tasks import evaluate_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
MAX_STEPS = 20
TEMPERATURE = 0.1


def build_prompt(observation: dict, task_name: str) -> str:
    if task_name == "fault_diagnosis":
        instruction = (
            'Choose exactly one action: "diagnose". Put the predicted fault in "reason". '
            'Valid fault labels: engine_overheating, low_oil, battery_issue, no_fault.'
        )
    elif task_name == "driving_decision":
        instruction = (
            'Choose the single safest next action. Valid actions: brake, accelerate, '
            'turn_left, turn_right, continue, stop.'
        )
    else:
        instruction = (
            "Control the vehicle safely over the episode. Valid actions: brake, accelerate, "
            "turn_left, turn_right, continue, stop, request_service."
        )

    return f"""
You are controlling an automotive agent for the task "{task_name}".
Return JSON only in this format:
{{
  "action_type": "string",
  "value": 0.0,
  "reason": "short reason"
}}

{instruction}

Observation:
{json.dumps(observation, indent=2)}
""".strip()


def get_model_action(observation: dict, task_name: str) -> Optional[Action]:
    if not OPENAI_API_KEY:
        return None

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(observation, task_name)}],
            temperature=TEMPERATURE,
            max_tokens=120,
        )
        text = completion.choices[0].message.content or ""
        payload = json.loads(text)
        return Action(
            action_type=str(payload["action_type"]).strip(),
            value=float(payload.get("value", 1.0)),
            reason=str(payload.get("reason", "")).strip(),
        )
    except Exception:
        return None


class EnvClient:
    def __init__(self) -> None:
        self.local_env: Optional[AutoMindEnv] = None
        self.remote_available = self._check_remote()

    def _check_remote(self) -> bool:
        try:
            response = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            return response.ok
        except Exception:
            return False

    def reset(self, task_name: str, difficulty: str) -> dict:
        if self.remote_available:
            response = requests.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_name": task_name, "difficulty": difficulty},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["observation"]

        if self.local_env is None:
            self.local_env = AutoMindEnv()
        return self.local_env.reset(task_name=task_name, difficulty=difficulty).model_dump()

    def step(self, action: Action) -> dict:
        if self.remote_available:
            response = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action.model_dump(),
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        if self.local_env is None:
            raise RuntimeError("Local environment is not initialized")
        return self.local_env.step(action).model_dump()

    def mode(self) -> str:
        return "http" if self.remote_available else "local"


def run_episode(client: EnvClient, task_name: str, difficulty: str) -> float:
    print("[START]")
    print(json.dumps({"task": task_name, "difficulty": difficulty, "runner": client.mode()}))

    obs = client.reset(task_name=task_name, difficulty=difficulty)
    last_action = None
    last_metrics = None
    last_info = None

    for step_idx in range(1, MAX_STEPS + 1):
        observation_obj = Observation(**obs)
        action = get_model_action(obs, task_name=task_name) or agent_step(
            observation_obj,
            task_name=task_name,
        )

        result = client.step(action)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        metrics = result["metrics"]
        info = result["info"]

        print("[STEP]")
        print(
            json.dumps(
                {
                    "task": task_name,
                    "difficulty": difficulty,
                    "step": step_idx,
                    "observation": obs,
                    "action": action.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )
        )

        last_action = action
        last_metrics = Metrics(**metrics)
        last_info = info

        if done:
            break

    final_score = evaluate_task(
        task_name=task_name,
        action=last_action,
        observation=Observation(**obs),
        metrics=last_metrics,
        info=last_info,
    )

    print("[END]")
    print(
        json.dumps(
            {
                "task": task_name,
                "difficulty": difficulty,
                "final_score": final_score,
                "outcome": (last_info or {}).get("outcome"),
            }
        )
    )
    return final_score


if __name__ == "__main__":
    client = EnvClient()
    for task_name in ["fault_diagnosis", "driving_decision", "autonomous_control"]:
        for difficulty in ["easy", "medium", "hard"]:
            run_episode(client=client, task_name=task_name, difficulty=difficulty)
