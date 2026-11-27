from math import ceil
from typing import TypedDict

import numpy as np

from .config import EcgConfig, parameter_bounds


class SampledIntervals(TypedDict):
    p_ms: int
    pq_ms: int
    qrs_ms: int
    qt_ms: int
    qtc_ms: int
    t_ms: int
    hr_bpm: int


def _varied_bounds(
    values: np.ndarray, variation_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.abs(values) * variation_pct / 100
    return np.ceil(values - delta).astype(int), np.floor(values + delta).astype(int)


def _candidates(
    bounds: tuple[int, int], variation_pct: float, name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.arange(bounds[0], bounds[1] + 1)
    lower, upper = _varied_bounds(values, variation_pct)
    allowed = (lower > 0) & (lower <= upper)
    if not allowed.any():
        raise ValueError(f"{name}: нет положительной длительности с учётом разброса")
    return values[allowed], lower[allowed], upper[allowed]


def sample_intervals(
    config: EcgConfig, rng: np.random.Generator,
) -> SampledIntervals:
    """Выбирает HR → QT/QTc → PQ → пару (QRS, T) → P из совместимых значений"""
    variation = config.variation_pct
    p_values, _, p_upper = _candidates(config.p_ms, variation, "P")
    pq_values, pq_lower, pq_upper = _candidates(config.pq_ms, variation, "PQ")
    qrs_values, _, qrs_upper = _candidates(config.qrs_ms, variation, "QRS")
    t_values, _, t_upper = _candidates(config.t_ms, variation, "T")
    # Запас на смещение R внутри QRS между циклами и округление двух R-меток
    r_margin = np.ceil(2 * qrs_values * variation / 100).astype(int)
    if variation > 0 or config.sample_rate != 1000:
        r_margin += ceil(2000 / config.sample_rate)
    qt_bounds = config.qt_ms if config.qt_ms is not None else config.qtc_ms
    assert qt_bounds is not None
    qt_values = np.arange(qt_bounds[0], qt_bounds[1] + 1)

    def to_qt(values: np.ndarray, hr: int) -> np.ndarray:
        if config.qt_ms is not None:
            return values
        return np.asarray(np.rint(values * np.cbrt(60 / hr)), dtype=int)

    allowed_pq = pq_lower > p_upper[0]
    if not allowed_pq.any():
        raise ValueError("В заданных диапазонах P не помещается в PQ с учётом разброса")
    pq_values, pq_upper = pq_values[allowed_pq], pq_upper[allowed_pq]

    # Сначала ограничиваем QT по RR, оставляя место для минимального PQ
    options: dict[int, np.ndarray] = {}
    for hr in range(max(1, config.hr_bpm[0]), config.hr_bpm[1] + 1):
        qt_lower, qt_upper = _varied_bounds(to_qt(qt_values, hr), variation)
        rr_ms = int(60 / hr * 1000)
        allowed_qt = (
            (qt_lower >= qrs_upper[0] + t_upper[0])
            & (qt_lower <= qt_upper)
            & (pq_upper[0] + qt_upper + r_margin[0] < rr_ms)
        )
        if allowed_qt.any():
            options[hr] = qt_values[allowed_qt]
    if not options:
        raise ValueError(
            "В заданных диапазонах нет совместимых HR, P, PQ, QRS, QT и T "
            "с учётом разброса",
        )

    hr = int(rng.choice(list(options)))
    rr_ms = int(60 / hr * 1000)
    qt_or_qtc = int(rng.choice(options[hr]))
    if config.qt_ms is None:
        qtc = qt_or_qtc
        qt = round(qtc * np.cbrt(60 / hr))
    else:
        qt = qt_or_qtc
        qtc = round(qt / np.cbrt(60 / hr))

    qt_min, qt_max = parameter_bounds(qt, variation, integer=True)
    pq = int(rng.choice(pq_values[pq_upper + qt_max + r_margin[0] < rr_ms]))
    pq_min, pq_max = parameter_bounds(pq, variation, integer=True)

    # Каждая допустимая пара QRS/T равновероятна; приоритета между ними нет
    qrs_indices, t_indices = np.nonzero(
        (qrs_upper[:, None] + t_upper <= qt_min)
        & (pq_max + qt_max + r_margin[:, None] < rr_ms),
    )
    pair_index = int(rng.integers(len(qrs_indices)))
    qrs = int(qrs_values[qrs_indices[pair_index]])
    t = int(t_values[t_indices[pair_index]])
    p = int(rng.choice(p_values[p_upper < pq_min]))
    return {
        "p_ms": p,
        "pq_ms": pq,
        "qrs_ms": qrs,
        "qt_ms": qt,
        "qtc_ms": qtc,
        "t_ms": t,
        "hr_bpm": hr,
    }
