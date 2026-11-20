import random
from dataclasses import dataclass

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment
from physiokit.ecg.synthetic import presets, rhythm_generator
from physiokit.ecg.synthetic.defines import EcgPreset, EcgPresetParameters

from ecg_noise import add_ecg_noise


@dataclass(frozen=True)
class EcgConfig:
    sample_rate: int = 1000  # Гц
    duration_s: float = 3
    heart_rate: float = 70  # уд/мин
    seed: int = 43
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

    # Шум NeuroKit.
    noise_amplitude: float = 0.03

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


@dataclass
class EcgResult:
    clean_signal: np.ndarray
    signal: np.ndarray
    fiducials: np.ndarray
    segments: np.ndarray


WAVE_STYLES = {
    "P": (EcgSegment.p_wave, "green"),
    "QRS": (EcgSegment.qrs_complex, "red"),
    "T": (EcgSegment.t_wave, "blue"),
}


def build_parameters(config: EcgConfig) -> EcgPresetParameters:
    params = presets.generate_preset_parameters(config.preset, config.heart_rate)
    params.p_length = int(config.p_duration_ms)
    params.pr_interval = int(config.pq_interval_ms)
    params.qrs_duration = int(config.qrs_duration_ms)
    params.st_length = int(config.st_duration_ms)
    params.t_length = int(config.t_duration_ms)

    params.p_voltages[0] = config.p_voltage
    params.q_depths[0] = config.q_depth
    params.r_heights[0] = config.r_height
    params.s_depths[0] = config.s_depth
    params.t_heights[0] = config.t_height
    params.p_biphasics[0] = config.p_biphasic
    params.t_leans[0] = config.t_lean
    params.s_to_qrs_duration_ratio[0] = config.s_to_qrs_duration_ratio
    return params


def _no_smoothing(y, noise_multiplier=1.0, impedance=1.0):
    return np.asarray(y, dtype=float) / impedance


def generate_ecg(config: EcgConfig) -> EcgResult:
    config.validate()

    random.seed(config.seed)
    np.random.seed(config.seed)
    rhythm_generator.smooth_and_noise = _no_smoothing
    params = build_parameters(config)
    _, ecg, segments, fiducials, _ = rhythm_generator.simulate_brisk(
        signal_length=config.signal_length,
        sample_rate=config.sample_rate,
        heart_rate=config.heart_rate,
        leads=1,
        preset=config.preset,
        parameters=params,
        noise_multiplier=0.0,  # Шум добавляется отдельно
    )

    clean_signal = ecg[0].copy()
    signal = add_ecg_noise(
        clean_signal,
        sampling_rate=config.sample_rate,
        noise_amplitude=config.noise_amplitude,
        random_state=config.seed,
    )
    return EcgResult(
        clean_signal=clean_signal,
        signal=signal,
        fiducials=fiducials[0],
        segments=segments[0],
    )


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


def print_summary(
    config: EcgConfig,
    result: EcgResult,
    boundaries: dict[str, np.ndarray],
) -> None:
    print("Параметры генерации:")
    print(f"Длительность: {config.duration_s} с")
    print(f"Частота дискретизации: {config.sample_rate} Гц")
    print(f"ЧСС: {config.heart_rate} уд/мин")
    for name, value in (
        ("PQ", config.pq_interval_ms),
        ("QRS", config.qrs_duration_ms),
        ("QT", config.qt_interval_ms),
        ("P", config.p_duration_ms),
        ("ST", config.st_duration_ms),
        ("T", config.t_duration_ms),
    ):
        print(f"{name}: {value} мс")
    print(f"RR (расчётный): {config.rr_interval_ms:.1f} мс")
    print(f"Фактическая длина: {len(result.signal) / config.sample_rate:.3f} с")
    r_peaks = np.count_nonzero(result.fiducials == int(EcgFiducial.r_peak))
    print(f"Количество R-пиков: {r_peaks}")
    print("\nГраницы волн PhysioKit:")
    for name, positions in boundaries.items():
        print(f"{name}:", positions)


def main() -> None:
    import matplotlib.pyplot as plt
    from ml_validation.plotter import plot_ecg

    config = EcgConfig()
    result = generate_ecg(config)
    boundaries = extract_boundaries(result.segments)
    print_summary(config, result, boundaries)
    plot_ecg(
        result.signal,
        sample_rate=config.sample_rate,
        ecg_lines=make_ecg_lines(boundaries),
    )
    plt.show()


if __name__ == "__main__":
    main()
