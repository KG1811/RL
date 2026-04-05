# AutoMind OpenEnv

AutoMind OpenEnv is a fleet maintenance and roadside decision benchmark for AI agents. It simulates the kind of telemetry-driven decisions that fleet operators, roadside support teams, and maintenance copilots make every day: diagnose a failing vehicle, choose the safest immediate maneuver, and coordinate recovery actions before the incident turns into downtime or a roadside emergency.

## Table of Contents
- [Tasks & Grading](#tasks--grading)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Installation & Local Setup](#installation--local-setup)
- [Deploying to HuggingFace / Docker](#deploying-to-huggingface--docker)
- [Baseline Inference & Scores](#baseline-inference--scores)

---

## Tasks & Grading

AutoMind evaluates agents across three increasing difficulty tiers:

1. **`fault_diagnosis`**: Given raw ECU and vehicle health telemetry, identify the active roadside issue quickly enough for dispatch or maintenance triage. Target action: `{"action_type": "diagnose", "reason": "<fault>"}`. (Difficulty: Easy).
2. **`driving_decision`**: Choose the safest immediate roadside maneuver when a vehicle is trending toward a hazardous condition. This mirrors the first decision an on-vehicle support agent or ADAS supervisor must make. (Difficulty: Medium).
3. **`autonomous_control`**: Full 20-step recovery loop. The agent must balance safety, degradation management, service escalation, and operator override while preserving the vehicle and avoiding roadside failure. (Difficulty: Hard).

All graders emit deterministic `0.0` to `1.0` scores. The benchmark rewards early diagnosis, safe intervention, and correct roadside escalation rather than only rewarding end-of-episode survival.

### Real-World Utility

This environment is designed to represent a real operational workflow instead of a toy driving task:

- **Fleet maintenance triage**: decide whether a vehicle can continue, should stop, or needs immediate service.
- **Roadside assistance dispatching**: recommend and book the nearest support center when a vehicle becomes unsafe.
- **Predictive maintenance evaluation**: test whether an agent can act before low oil, overheating, or brake degradation turn into a costly breakdown.
- **Human override handling**: evaluate robustness when a driver or operator partially overrides the agent's intended recovery behavior.

---

## Action Space

The server receives JSON matching the following schema:

```json
{
  "action_type": "string (enum)",
  "value": "float [0.0, 1.0]",
  "reason": "string"
}
```

**Allowed Actions:**
- `diagnose` (for task 1)
- `brake`
- `accelerate`
- `turn_left`
- `turn_right`
- `continue`
- `stop`
- `request_service`

---

## Observation Space

The `Observation` received back maps directly to vehicle telemetry and situational context so that the tasks remain learnable and programmatically gradable.

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `speed` | Float | 0 - 220 km/h | Absolute velocity. |
| `rpm` | Float | 0 - 8000 | Current engine RPM. |
| `throttle` | Float | 0 - 100% | Pedal depress percentage. |
| `gear` | Int | 0 - 6 | Current transmission gear. |
| `engine_load`| Float | 0 - 100% | System load calculated by logic. |
| `transmission_load`| Float | 0 - 100% | Transmission stress under current maneuver. |
| `fuel_rate` | Float | 0 - 40 L/hr| Current consumption. |
| `acceleration`| Float| -12 - +12 | Derivative of speed over time. |
| `engine_temp` | Float | 0 - 150 C | Live temperature tracking. |
| `distance_to_obstacle` | Float | 0 - 300 m | Perceived obstacle distance from onboard sensors. |
| `road_condition` | String | dry/wet/rain | Traction context for the policy. |
| `drive_mode` | String | idle/city/cruise/sport | Powertrain mode derived from telemetry. |
| `oil_level` | Float | 0 - 100% | Remaining healthy oil %. |
| `battery_health`| Float | 0 - 100% | High Voltage Battery %. |
| `failures` | Object | boolean flags | Detected catastrophic faults. |
| `history` | Array | past 8 steps | Short-term memory stream. |

---

## Installation & Local Setup

### Python Virtual Environment

```bash
# Clone the repo and enter directory
git clone <repo>
cd automind-openenv

# Install requirements
pip install -r requirements.txt

# Launch FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server natively exposes standard OpenEnv hooks: `/reset`, `/step`, `/state`, `/schema`, `/tasks`.

---

## Deploying to HuggingFace / Docker

You can containerize this environment using the included Docker deployment, which wraps the FastAPI interface for HuggingFace Spaces.

```dockerfile
# Build image
docker build -t automind-env .

# Run container
docker run -p 7860:7860 automind-env
```

**HuggingFace Deployment Instructions:**
1. Create a new "Docker" Space on HuggingFace.
2. Link the repository GitHub branch.
3. HuggingFace will natively run the `Dockerfile` and expose the API via the standard Space URL (e.g., `https://your-user-automind.hf.space`).

---

## Baseline Inference & Scores

You can execute a full loop benchmark using the bundled baseline evaluator. Define the required environment variables before running:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="<your_openai_key>"
export HF_TOKEN="<optional_fallback_token>"
export ENV_BASE_URL="http://127.0.0.1:8000"

python inference.py
```

`inference.py` emits only structured `[START]`, `[STEP]`, and `[END]` stdout logs and falls back to a seeded local environment when the HTTP server is unavailable, making baseline runs reproducible for local validation.

### Reproducible Baseline Results
Measured locally with the bundled heuristic agent and the seeded fallback runner:

```text
==============================
      BASELINE RESULTS
==============================

[FAULT_DIAGNOSIS]
  Easy: 1.000
  Medium: 1.000
  Hard: 1.000
  --> Average: 1.000

[DRIVING_DECISION]
  Easy: 1.000
  Medium: 1.000
  Hard: 0.700
  --> Average: 0.900

[AUTONOMOUS_CONTROL]
  Easy: 0.731
  Medium: 0.749
  Hard: 0.481
  --> Average: 0.654
```
