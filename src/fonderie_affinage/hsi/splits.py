"""Grouped splits: every case of an animal or patient stays on one side."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from fonderie_affinage.hsi.data import Case, DataError


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in self.__dataclass_fields__}


def grouped_split(
    cases: Sequence[Case],
    test_groups: Sequence[str],
    validation_fraction: float,
    seed: int,
    max_attempts: int = 1000,
) -> Split:
    """Hold out ``test_groups``, then draw validation groups from the rest.

    The draw is seeded and repeated until the training part contains every
    class of the development set, so no class is learned from validation only.
    An empty ``test_groups`` gives a split without a held-out part.
    """
    groups = sorted({case.group for case in cases})
    unknown = sorted(set(test_groups) - set(groups))
    if unknown:
        raise DataError(f"Test groups not in the manifest: {', '.join(unknown)}")
    development = [group for group in groups if group not in set(test_groups)]
    n_validation = max(1, round(len(development) * validation_fraction))
    if len(development) - n_validation < 1:
        raise DataError("Not enough groups left for training after the split.")

    classes = {case.label for case in cases if case.group in set(development)}
    rng = random.Random(seed)
    for _ in range(max_attempts):
        validation = set(rng.sample(development, n_validation))
        train_classes = {
            case.label
            for case in cases
            if case.group in set(development) and case.group not in validation
        }
        if train_classes == classes:
            break
    else:
        raise DataError("No validation draw leaves every class in the training part.")

    def ids(selected: set[str]) -> tuple[str, ...]:
        return tuple(sorted(case.case_id for case in cases if case.group in selected))

    train = set(development) - validation
    return Split(
        train=ids(train),
        validation=ids(validation),
        test=ids(set(test_groups)),
        train_groups=tuple(sorted(train)),
        validation_groups=tuple(sorted(validation)),
        test_groups=tuple(sorted(test_groups)),
    )
