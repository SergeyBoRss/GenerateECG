from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from physiokit.ecg.synthetic.defines import EcgPreset

from .morphology import MORPHOLOGY_PRESETS


@dataclass(frozen=True)
class CommonEcgConfig:
    sample_rate: int = 1000  # Гц
    heart_rate: float = 70  # уд/мин
    seed: int = 42

    # Интервалы и длительности волн, мс.
    pq_interval_ms: int = 150
    qrs_duration_ms: int = 90
    qt_interval_ms: int = 400
    p_duration_ms: int = 100
    t_duration_ms: int = 210

    # Амплитуды и форма первого отведения в единицах модели PhysioKit.
    p_voltage: float = 0.15
    q_depth: float = 0.10
    r_height: float = 3.00
    s_depth: float = 0.20
    t_height: float = 5.50
    t_lean: float = 0.0  # Симметричная форма T
    s_to_qrs_duration_ratio: float = 0.15

    # Шум NeuroKit
    noise_amplitude: float = 0.03
    # Амплитуды базовой линии; 0 отключает
    baseline_wander_amplitude: float = 0.0  # В долях std чистого сигнала
    baseline_wander_frequency: float = 0.3  # Гц, частота изменений случайного дрейфа
    baseline_sine_amplitude: float = 0.0  # В долях std чистого сигнала
    baseline_sine_frequency: float = 0.2  # Гц, частота синусоиды

    @property
    def st_duration_ms(self) -> int:
        return self.qt_interval_ms - self.qrs_duration_ms - self.t_duration_ms


@dataclass(frozen=True)
class EcgConfig(CommonEcgConfig):
    duration_s: float = 3
    preset: EcgPreset = EcgPreset.SR
    p_biphasic: bool = False

    @property
    def rr_interval_ms(self) -> float:
        return 60000 / self.heart_rate

    @property
    def signal_length(self) -> int:
        return round(self.duration_s * self.sample_rate)

    def validate(self) -> None:
        if not 0 < self.p_duration_ms <= self.pq_interval_ms:
            raise ValueError("Длительность P должна быть положительной и <= PQ")
        if self.qrs_duration_ms <= 0 or self.t_duration_ms <= 0:
            raise ValueError("Длительности QRS и T должны быть положительными")
        if self.st_duration_ms < 0:
            raise ValueError("QT должен быть не меньше QRS + T")
        if self.pq_interval_ms + self.qt_interval_ms >= self.rr_interval_ms:
            raise ValueError("PQ + QT не помещаются в RR")


@dataclass(frozen=True)
class StitchedEcgConfig(CommonEcgConfig):
    beats: int = 4  # Длина окна в циклах; крайние циклы могут быть неполными.
    random_start: bool = True  # False — начинать с начала первого P.
    variation_pct: float = 5.0  # Разброс длительностей и параметров формы; 0 отключает.
    amplitude_variation_pct: float = 5.0  # Отдельный разброс амплитуд P, Q, R, S, T.
    p_lean: float = 1.0

    def parameter_bounds(self, name: str) -> tuple[int, int] | tuple[float, float]:
        value = getattr(self, name)
        variation_pct = (
            self.amplitude_variation_pct
            if name in ("p_voltage", "q_depth", "r_height", "s_depth", "t_height")
            else self.variation_pct
        )
        delta = abs(value) * variation_pct / 100
        low, high = value - delta, value + delta
        if name.endswith("_ms"):
            return ceil(low), floor(high)
        return low, high

    def validate(self) -> None:
        if not np.isfinite(self.variation_pct) or self.variation_pct < 0:
            raise ValueError("variation_pct должен быть конечным и неотрицательным")
        if not np.isfinite(self.amplitude_variation_pct) or not 0 <= self.amplitude_variation_pct <= 100:
            raise ValueError("amplitude_variation_pct должен быть от 0 до 100, чтобы не менять полярность зубцов")
        for name in (
            "pq_interval_ms", "qrs_duration_ms", "qt_interval_ms",
            "p_duration_ms", "t_duration_ms", "p_lean", "s_to_qrs_duration_ratio",
        ):
            if name.endswith("_ms"):
                if self.parameter_bounds(name)[0] <= 0:
                    raise ValueError(f"{name}: разброс должен оставлять длительность положительной")
            elif self.parameter_bounds(name)[0] < 0:
                raise ValueError(f"{name}: разброс не должен давать отрицательные значения")

        # Проверяем крайние значения: любая выбранная комбинация должна подходить.
        if self.parameter_bounds("p_duration_ms")[1] >= self.parameter_bounds("pq_interval_ms")[0]:
            raise ValueError("Максимальная P должна быть строго меньше минимального PQ с учётом разброса")
        min_qt = self.parameter_bounds("qt_interval_ms")[0]
        max_qrs = self.parameter_bounds("qrs_duration_ms")[1]
        if min_qt < max_qrs + self.parameter_bounds("t_duration_ms")[1]:
            raise ValueError("Минимальный QT должен быть не меньше максимальных QRS + T с учётом разброса")
        rr_ms = int(60 / self.heart_rate * 1000)
        if self.parameter_bounds("pq_interval_ms")[1] + self.parameter_bounds("qt_interval_ms")[1] >= rr_ms:
            raise ValueError("Максимальные PQ + QT должны помещаться в RR")


@dataclass(frozen=True)
class SamplingPreset:
    pq_ms: tuple[int, int] = (135, 160)
    qtc_ms: tuple[int, int] = (380, 430)
    qrs_ms: tuple[int, int] = (80, 100)
    hr_bpm: tuple[int, int] = (50, 110)
    seed: tuple[int, int] = (0, 2**32 - 1)
    shapes: tuple[str, ...] = tuple(MORPHOLOGY_PRESETS)
    p_ms: tuple[int, int] = (100, 120)
    t_ms: tuple[int, int] = (150, 180)

    def validate(self, variation_pct: float = StitchedEcgConfig().variation_pct) -> None:
        # Проверяем крайние сочетания с учётом разброса внутри записи:
        # максимальные P/QRS/T с минимальными PQ/QT, затем максимальные PQ/QT.
        for hr in range(self.hr_bpm[1], self.hr_bpm[0] - 1, -1):
            for pq, qtc, qrs in (
                (self.pq_ms[0], self.qtc_ms[0], self.qrs_ms[1]),
                (self.pq_ms[1], self.qtc_ms[1], self.qrs_ms[0]),
            ):
                config = StitchedEcgConfig(
                    heart_rate=hr,
                    p_duration_ms=self.p_ms[1],
                    t_duration_ms=self.t_ms[1],
                    pq_interval_ms=pq,
                    qt_interval_ms=round(qtc * np.cbrt(60 / hr)),
                    qrs_duration_ms=qrs,
                    variation_pct=variation_pct,
                )
                try:
                    config.validate()
                except ValueError as exc:
                    raise ValueError(f"При HR={hr}: {exc}") from exc
