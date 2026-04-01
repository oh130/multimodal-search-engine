from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence


@dataclass(frozen=True)
class ABTestResult:
    control_mean: float
    treatment_mean: float
    absolute_diff: float
    relative_lift: float
    p_value: float
    confidence_level: float
    confidence_interval: tuple[float, float]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty.")
    return sum(values) / len(values)


def _bootstrap_confidence_interval(
    control: Sequence[float],
    treatment: Sequence[float],
    confidence_level: float,
    num_bootstrap: int,
    rng: Random,
) -> tuple[float, float]:
    bootstrap_diffs: list[float] = []

    for _ in range(num_bootstrap):
        control_sample = [control[rng.randrange(len(control))] for _ in range(len(control))]
        treatment_sample = [treatment[rng.randrange(len(treatment))] for _ in range(len(treatment))]
        bootstrap_diffs.append(_mean(treatment_sample) - _mean(control_sample))

    bootstrap_diffs.sort()
    alpha = 1.0 - confidence_level
    lower_index = int((alpha / 2.0) * num_bootstrap)
    upper_index = int((1.0 - alpha / 2.0) * num_bootstrap) - 1
    lower_index = max(0, min(lower_index, num_bootstrap - 1))
    upper_index = max(0, min(upper_index, num_bootstrap - 1))
    return bootstrap_diffs[lower_index], bootstrap_diffs[upper_index]


def _permutation_p_value(
    control: Sequence[float],
    treatment: Sequence[float],
    observed_diff: float,
    num_permutations: int,
    rng: Random,
) -> float:
    pooled = list(control) + list(treatment)
    control_size = len(control)
    extreme_count = 0

    for _ in range(num_permutations):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        perm_control = shuffled[:control_size]
        perm_treatment = shuffled[control_size:]
        perm_diff = _mean(perm_treatment) - _mean(perm_control)
        if abs(perm_diff) >= abs(observed_diff):
            extreme_count += 1

    return (extreme_count + 1) / (num_permutations + 1)


def compare_group_means(
    control: Sequence[float],
    treatment: Sequence[float],
    confidence_level: float = 0.95,
    num_bootstrap: int = 5000,
    num_permutations: int = 5000,
    seed: int = 42,
) -> ABTestResult:
    if not control or not treatment:
        raise ValueError("control and treatment must not be empty.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if num_bootstrap <= 0 or num_permutations <= 0:
        raise ValueError("num_bootstrap and num_permutations must be positive.")

    rng = Random(seed)
    control_mean = _mean(control)
    treatment_mean = _mean(treatment)
    absolute_diff = treatment_mean - control_mean
    relative_lift = 0.0 if control_mean == 0 else absolute_diff / control_mean

    ci = _bootstrap_confidence_interval(
        control=control,
        treatment=treatment,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
        rng=rng,
    )
    p_value = _permutation_p_value(
        control=control,
        treatment=treatment,
        observed_diff=absolute_diff,
        num_permutations=num_permutations,
        rng=rng,
    )

    return ABTestResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        absolute_diff=absolute_diff,
        relative_lift=relative_lift,
        p_value=p_value,
        confidence_level=confidence_level,
        confidence_interval=ci,
    )


if __name__ == "__main__":
    control = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    treatment = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]

    result = compare_group_means(control, treatment, confidence_level=0.95)

    print("Control mean:", round(result.control_mean, 4))
    print("Treatment mean:", round(result.treatment_mean, 4))
    print("Absolute diff:", round(result.absolute_diff, 4))
    print("Relative lift:", round(result.relative_lift, 4))
    print("p-value:", round(result.p_value, 4))
    print(
        f"{int(result.confidence_level * 100)}% CI:",
        tuple(round(bound, 4) for bound in result.confidence_interval),
    )
