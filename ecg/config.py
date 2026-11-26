from dataclasses import dataclass
from math import ceil, floor
from typing import Literal, overload

import numpy as np

from .morphology import MORPHOLOGY_PRESETS


@overload
def parameter_bounds(
    value: float,
    variation_pct: float,
    *,
    integer: Literal[True],
) -> tuple[int, int]: ...


@overload
def parameter_bounds(
    value: float,
    variation_pct: float,
    *,
    integer: Literal[False] = False,
) -> tuple[float, float]: ...


def parameter_bounds(
    value: float,
    variation_pct: float,
    *,
    integer: bool = False,
) -> tuple[int, int] | tuple[float, float]:
    delta = abs(value) * variation_pct / 100
    low, high = value - delta, value + delta
    return (ceil(low), floor(high)) if integer else (low, high)


def validate_intervals(
    *,
    p_ms: tuple[int, int],
    pq_ms: tuple[int, int],
    qrs_ms: tuple[int, int],
    qt_ms: tuple[int, int],
    t_ms: tuple[int, int],
    hr_bpm: int,
    variation_pct: float,
) -> None:
    bounds = {}
    for name, (low, high) in (
        ("P", p_ms),
        ("PQ", pq_ms),
        ("QRS", qrs_ms),
        ("QT", qt_ms),
        ("T", t_ms),
    ):
        bounds[name] = (
            parameter_bounds(low, variation_pct, integer=True)[0],
            parameter_bounds(high, variation_pct, integer=True)[1],
        )
        if bounds[name][0] <= 0:
            raise ValueError(
                f"{name}: разброс должен оставлять длительность положительной",
            )
    if bounds["P"][1] >= bounds["PQ"][0]:
        raise ValueError(
            "Максимальная P должна быть строго меньше минимального PQ "
            "с учётом разброса",
        )
    if bounds["QT"][0] < bounds["QRS"][1] + bounds["T"][1]:
        raise ValueError(
            "Минимальный QT должен быть не меньше максимальных QRS + T "
            "с учётом разброса",
        )
    if bounds["PQ"][1] + bounds["QT"][1] >= int(60 / hr_bpm * 1000):
        raise ValueError("Максимальные PQ + QT должны помещаться в RR")


@dataclass(frozen=True)
class EcgConfig:
    seed: int | None = None
    duration_s: float = 10.0
    sample_rate: int = 1000
    random_start: bool = True
    variation_pct: float = 5.0
    amplitude_variation_pct: float = 20.0
    rr_std_ms: float = 0.0  # Std распределения RR; 0 — постоянный RR

    pq_ms: tuple[int, int] = (135, 160)
    qtc_ms: tuple[int, int] | None = (380, 430)
    qt_ms: tuple[int, int] | None = None  # Если задан, выбираем QT напрямую вместо QTc
    qrs_ms: tuple[int, int] = (80, 100)
    hr_bpm: tuple[int, int] = (35, 110)
    shapes: tuple[str, ...] = tuple(MORPHOLOGY_PRESETS)
    p_ms: tuple[int, int] = (100, 120)
    t_ms: tuple[int, int] = (150, 180)

    p_voltage: tuple[float, float] = (0.15, 0.15)
    t_height: tuple[float, float] = (5.50, 5.50)
    p_lean: tuple[float, float] = (1.0, 1.0)
    t_lean: tuple[float, float] = (0.0, 0.0)
    # None — брать значение из выбранной формы QRS
    q_depth: tuple[float, float] | None = None
    r_height: tuple[float, float] | None = None
    s_depth: tuple[float, float] | None = None
    s_to_qrs_duration_ratio: tuple[float, float] | None = None

    noise_amplitude: tuple[float, float] = (0.020, 0.035)  # Шаг 0.001
    baseline_wander_amplitude: tuple[float, float] = (0.0, 1.0)
    baseline_wander_frequency: tuple[float, float] = (0.25, 0.35)
    baseline_sine_amplitude: tuple[float, float] = (0.0, 2.0)
    baseline_sine_frequency: tuple[float, float] = (0.15, 0.25)

    no_p: float = 10.0
    negative_p: float = 10.0
    negative_qrs: float = 10.0
    negative_t: float = 10.0

    def validate(self) -> None:
        for name in ("p_lean", "s_to_qrs_duration_ratio"):
            bounds = getattr(self, name)
            if (
                bounds is not None
                and parameter_bounds(bounds[0], self.variation_pct)[0] < 0
            ):
                raise ValueError(
                    f"{name}: разброс не должен давать отрицательные значения",
                )
        # Проверяем крайние сочетания с учётом разброса внутри записи:
        # максимальные P/QRS/T с минимальными PQ/QT, затем максимальные PQ/QT
        for hr in range(self.hr_bpm[1], self.hr_bpm[0] - 1, -1):
            if self.qt_ms is not None:
                qt_bounds = self.qt_ms
            else:
                assert self.qtc_ms is not None
                qt_bounds = tuple(round(qtc * np.cbrt(60 / hr)) for qtc in self.qtc_ms)
            try:
                validate_intervals(
                    p_ms=self.p_ms,
                    pq_ms=self.pq_ms,
                    qrs_ms=self.qrs_ms,
                    qt_ms=qt_bounds,
                    t_ms=self.t_ms,
                    hr_bpm=hr,
                    variation_pct=self.variation_pct,
                )
            except ValueError as exc:
                raise ValueError(f"При HR={hr}: {exc}") from exc
