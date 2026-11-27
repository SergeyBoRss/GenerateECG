from dataclasses import dataclass

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment
from physiokit.ecg.synthetic.defines import EcgPresetParameters


@dataclass(frozen=True)
class BeatInfo:
    start: int  # Начало цикла относительно записи; может быть отрицательным.
    stop: int  # Конец цикла; может выходить за правый край записи.
    heart_rate: int
    parameters: EcgPresetParameters


@dataclass
class EcgResult:
    clean_signal: np.ndarray
    signal: np.ndarray
    segments: np.ndarray
    fiducials: np.ndarray
    beats: list[BeatInfo]
    start_offset_samples: int = 0
    segment_before_start: int = int(EcgSegment.tp_segment)

    @property
    def r_peak_samples(self) -> np.ndarray:
        """Позиции меток R относительно начала готовой записи, в отсчётах"""
        return np.flatnonzero(self.fiducials == int(EcgFiducial.r_peak))


@dataclass(frozen=True)
class SampledParameters:
    pq_ms: int
    qtc_ms: int
    qrs_ms: int
    hr_bpm: int
    seed: int
    shape: str
    t_ms: int
    qt_ms: int
    p_ms: int
    no_p: bool
    negative_p: bool
    negative_qrs: bool
    negative_t: bool
    p_voltage: float
    q_depth: float
    r_height: float
    s_depth: float
    t_height: float
    p_lean: float
    t_lean: float
    s_to_qrs_duration_ratio: float
    noise_amplitude: float
    baseline_wander_amplitude: float
    baseline_wander_frequency: float
    baseline_sine_amplitude: float
    baseline_sine_frequency: float
