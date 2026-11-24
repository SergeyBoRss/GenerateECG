from dataclasses import dataclass

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment
from physiokit.ecg.synthetic.defines import EcgPresetParameters


@dataclass
class EcgResult:
    clean_signal: np.ndarray
    signal: np.ndarray
    fiducials: np.ndarray
    segments: np.ndarray


@dataclass(frozen=True)
class BeatInfo:
    start: int  # Начало цикла относительно записи; может быть отрицательным.
    stop: int  # Конец цикла; может выходить за правый край записи.
    heart_rate: int
    parameters: EcgPresetParameters


@dataclass
class StitchedEcgResult:
    clean_signal: np.ndarray
    signal: np.ndarray
    segments: np.ndarray
    fiducials: np.ndarray
    beats: list[BeatInfo]
    start_offset_samples: int = 0
    segment_before_start: int = int(EcgSegment.tp_segment)

    @property
    def r_peak_samples(self) -> np.ndarray:
        """Позиции меток R относительно начала готовой записи, в отсчётах."""
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
    p_ms: int = 100
    no_p: bool = False
    negative_p: bool = False
    negative_t: bool = False

    @property
    def qt_ms(self) -> int:
        return round(self.qtc_ms * np.cbrt(60 / self.hr_bpm))
