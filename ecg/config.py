from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from physiokit.ecg.synthetic.defines import EcgPreset

from .morphology import MORPHOLOGY_PRESETS


@dataclass(frozen=True)
class EcgConfig:
    sample_rate: int = 1000  # Гц
    duration_s: float = 3
    heart_rate: float = 70  # уд/мин
    seed: int = 42
    preset: EcgPreset = EcgPreset.SR

    # Интервалы и длительности волн, мс.
    pq_interval_ms: int = 150
    qrs_duration_ms: int = 90
    qt_interval_ms: int = 400
    p_duration_ms: int = 100
    st_duration_ms: int = 100

    # Амплитуды и форма первого отведения в единицах модели PhysioKit.
    p_voltage: float = 0.15
    q_depth: float = 0.10
    r_height: float = 3.00
    s_depth: float = 0.20
    t_height: float = 5.00
    p_biphasic: bool = False
    t_lean: float = 0  # Симметричная форма T
    s_to_qrs_duration_ratio: float = 0.15

    # Шум NeuroKit
    noise_amplitude: float = 0.03
    # Амплитуды базовой линии; 0 отключает
    baseline_wander_amplitude: float = 0.0  # В долях std чистого сигнала
    baseline_wander_frequency: float = 0.3  # Гц, частота изменений случайного дрейфа
    baseline_sine_amplitude: float = 0.0  # В долях std чистого сигнала
    baseline_sine_frequency: float = 0.2  # Гц, частота синусоиды

    @property
    def t_duration_ms(self) -> int:
        return self.qt_interval_ms - self.qrs_duration_ms - self.st_duration_ms

    @property
    def rr_interval_ms(self) -> float:
        return 60000 / self.heart_rate

    @property
    def signal_length(self) -> int:
        return round(self.duration_s * self.sample_rate)

    def validate(self) -> None:
        if not 0 < self.p_duration_ms <= self.pq_interval_ms:
            raise ValueError("Длительность P должна быть положительной и <= PQ")
        if self.qrs_duration_ms <= 0 or self.st_duration_ms < 0:
            raise ValueError("QRS должен быть положительным, ST — неотрицательным")
        if self.t_duration_ms <= 0:
            raise ValueError("QT должен быть больше QRS + ST")
        if self.pq_interval_ms + self.qt_interval_ms >= self.rr_interval_ms:
            raise ValueError("PQ + QT не помещаются в RR")


@dataclass(frozen=True)
class StitchedEcgConfig:
    beats: int = 4  # Длина окна в циклах; крайние циклы могут быть неполными.
    sample_rate: int = 1000  # Частота генерации и готовой записи, Гц.
    heart_rate: int = 70  # Заданная ЧСС генератора; фактические R–R могут различаться.
    seed: int = 42
    random_start: bool = True  # False — начинать с начала первого P.
    noise_amplitude: float = 0.03
    variation_pct: float = 5.0  # Общий разброс ±5%; 0 отключает разброс.

    # Амплитуды базовой линии
    baseline_wander_amplitude: float = 0.0  # В долях std чистого сигнала; 0 отключает
    baseline_wander_frequency: float = 0.3  # Гц, частота изменений случайного дрейфа.
    baseline_sine_amplitude: float = 0.0  # В долях std чистого сигнала; 0 отключает
    baseline_sine_frequency: float = 0.2  # Гц, частота синусоиды.

    # Базовые длительности
    pq_interval_ms: int = 150
    qrs_duration_ms: int = 90
    qt_interval_ms: int = 400
    p_duration_ms: int = 100
    st_duration_ms: int = 100

    # Амплитуды в единицах модели PhysioKit
    p_voltage: float = 0.15
    q_depth: float = 0.10
    r_height: float = 3.00
    s_depth: float = 0.20
    t_height: float = 5.00
    p_lean: float = 1.0
    t_lean: float = 0.0  # Постоянный наклон T
    s_to_qrs_duration_ratio: float = 0.15

    def parameter_bounds(self, name: str) -> tuple[int, int] | tuple[float, float]:
        value = getattr(self, name)
        delta = abs(value) * self.variation_pct / 100
        low, high = value - delta, value + delta
        if name.endswith("_ms"):
            return ceil(low), floor(high)
        return low, high

    def validate(self) -> None:
        if not np.isfinite(self.variation_pct) or self.variation_pct < 0:
            raise ValueError("variation_pct должен быть конечным и неотрицательным")
        for name in (
            "pq_interval_ms", "qrs_duration_ms", "qt_interval_ms",
            "p_duration_ms", "st_duration_ms", "p_lean", "s_to_qrs_duration_ratio",
        ):
            if name.endswith("_ms"):
                if self.parameter_bounds(name)[0] <= 0:
                    raise ValueError(f"{name}: разброс должен оставлять длительность положительной")
            elif self.parameter_bounds(name)[0] < 0:
                raise ValueError(f"{name}: разброс не должен давать отрицательные значения")

        # Проверяем крайние значения: любая выбранная комбинация должна подходить.
        if self.parameter_bounds("p_duration_ms")[1] > self.parameter_bounds("pq_interval_ms")[0]:
            raise ValueError("Максимальная P должна быть <= минимального PQ")
        if self.parameter_bounds("qt_interval_ms")[0] <= (
            self.parameter_bounds("qrs_duration_ms")[1] + self.parameter_bounds("st_duration_ms")[1]
        ):
            raise ValueError("Минимальный QT должен быть больше максимальных QRS + ST")
        rr_ms = int(60 / self.heart_rate * 1000)
        if self.parameter_bounds("pq_interval_ms")[1] + self.parameter_bounds("qt_interval_ms")[1] >= rr_ms:
            raise ValueError("Максимальные PQ + QT должны помещаться в RR")


@dataclass(frozen=True)
class SamplingPreset:
    pq_ms: tuple[int, int] = (135, 160)
    qtc_ms: tuple[int, int] = (380, 430)
    qrs_ms: tuple[int, int] = (80, 100)
    hr_bpm: tuple[int, int] = (50, 120)
    seed: tuple[int, int] = (0, 2**32 - 1)
    shapes: tuple[str, ...] = tuple(MORPHOLOGY_PRESETS)

    def validate(self, variation_pct: float = StitchedEcgConfig().variation_pct) -> None:
        # Проверяем крайние сочетания с учётом разброса внутри записи:
        # минимальный PQ/QT с максимальным QRS, затем максимальные PQ/QT.
        for hr in range(self.hr_bpm[1], self.hr_bpm[0] - 1, -1):
            for pq, qtc, qrs in (
                (self.pq_ms[0], self.qtc_ms[0], self.qrs_ms[1]),
                (self.pq_ms[1], self.qtc_ms[1], self.qrs_ms[0]),
            ):
                config = StitchedEcgConfig(
                    heart_rate=hr,
                    pq_interval_ms=pq,
                    qt_interval_ms=round(qtc * np.cbrt(60 / hr)),
                    qrs_duration_ms=qrs,
                    variation_pct=variation_pct,
                )
                try:
                    config.validate()
                except ValueError as exc:
                    raise ValueError(f"При HR={hr}: {exc}") from exc
