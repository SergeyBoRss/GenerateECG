import numpy as np
from physiokit.ecg.defines import EcgSegment

from .models import StitchedEcgResult


WAVE_STYLES = {
    "P": (EcgSegment.p_wave, "green"),
    "QRS": (EcgSegment.qrs_complex, "red"),
    "T": (EcgSegment.t_wave, "blue"),
}


def extract_boundaries(segments: np.ndarray) -> dict[str, np.ndarray]:
    boundaries = {}
    for name, (segment_code, _) in WAVE_STYLES.items():
        in_wave = segments == int(segment_code)
        boundaries[f"{name} onset"] = np.flatnonzero(~in_wave[:-1] & in_wave[1:]) + 1
        # Конец — первый отсчёт после участка волны.
        boundaries[f"{name} offset"] = np.flatnonzero(in_wave[:-1] & ~in_wave[1:]) + 1
    return boundaries


def make_ecg_lines(boundaries: dict[str, np.ndarray]) -> list[tuple[int, str]]:
    return sorted(
        (int(position), color)
        for name, (_, color) in WAVE_STYLES.items()
        for edge in ("onset", "offset")
        for position in boundaries[f"{name} {edge}"]
    )


def get_boundaries(result: StitchedEcgResult) -> dict[str, np.ndarray]:
    with_previous = np.concatenate(([result.segment_before_start], result.segments))
    return {
        name: positions - 1
        for name, positions in extract_boundaries(with_previous).items()
    }
