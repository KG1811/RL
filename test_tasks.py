# ==============================
# TEST TASKS + GRADERS
# ==============================

from environment import AutoMindEnv
from models import Action
from tasks import evaluate_task, detect_true_fault


def test_fault_diagnosis():
    print("\n=== TASK 1: FAULT DIAGNOSIS ===")

    env = AutoMindEnv()
    obs = env.reset("fault_diagnosis", "medium")

    true_fault = detect_true_fault(obs)
    print("True Fault:", true_fault)

    action = Action(action_type="diagnose", value=1.0, reason=true_fault)

    score = evaluate_task(
        task_name="fault_diagnosis",
        action=action,
        observation=obs,
        metrics=None,
    )

    print("Score:", score)


def test_driving_decision():
    print("\n=== TASK 2: DRIVING DECISION ===")

    env = AutoMindEnv()
    obs = env.reset("driving_decision", "medium")

    action = Action(
        action_type="brake",
        value=0.8,
        reason="Obstacle ahead",
    )

    result = env.step(action)

    score = evaluate_task(
        task_name="driving_decision",
        action=action,
        observation=obs,
        metrics=result.metrics,
    )

    print("Action:", action.action_type)
    print("Score:", score)


def test_autonomous_control():
    print("\n=== TASK 3: AUTONOMOUS CONTROL ===")

    env = AutoMindEnv()
    obs = env.reset("autonomous_control", "hard")

    result = None
    action = None

    for _ in range(5):
        action = Action(
            action_type="brake",
            value=0.7,
            reason="Test control",
        )
        result = env.step(action)

        if result.done:
            break

    if result is None or action is None:
        print("Autonomous control test did not run.")
        return

    final_score = evaluate_task(
        task_name="autonomous_control",
        action=action,
        observation=result.observation,
        metrics=result.metrics,
        info=result.info,
    )

    print("Final Score:", final_score)


if __name__ == "__main__":
    test_fault_diagnosis()
    test_driving_decision()
    test_autonomous_control()
