# ==============================
# AutoMind OpenEnv - Traffic Engine
# ==============================

from __future__ import annotations

import random


def get_obstacle_relative_motion(
    rng: random.Random,
    difficulty: str,
) -> float:
    """
    Positive means obstacle moves away.
    Negative means obstacle gets closer faster.
    """
    if difficulty == "easy":
        return rng.uniform(1.0, 4.0)

    if difficulty == "medium":
        return rng.uniform(-1.0, 2.0)

    if difficulty == "hard":
        return rng.uniform(-0.5, 2.0)

    return 0.0


def get_traffic_pressure(
    rng: random.Random,
    difficulty: str,
) -> float:
    """
    Traffic pressure in [0, 1].
    """
    if difficulty == "easy":
        return rng.uniform(0.0, 0.2)

    if difficulty == "medium":
        return rng.uniform(0.2, 0.6)

    if difficulty == "hard":
        return rng.uniform(0.2, 0.6)

    return 0.0
