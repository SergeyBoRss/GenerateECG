import random
from dataclasses import asdict, replace
from math import ceil, floor

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment
from physiokit.ecg.synthetic import rhythm_generator
from physiokit.ecg.synthetic.defines import EcgPreset, EcgPresetParameters

from ecg_noise import add_ecg_noise

from .config import EcgConfig, SamplingPreset, StitchedEcgConfig
from .models import BeatInfo, EcgResult, SampledParameters, StitchedEcgResult
from .morphology import MORPHOLOGY_PRESETS
from .parameters import build_parameters, sample_beat_parameters


def _no_smoothing(y, noise_multiplier=0.0, impedance=1.0):
    return np.asarray(y, dtype=float) / impedance


def generate_base_ecg(config: EcgConfig) -> EcgResult:
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
        baseline_wander_amplitude=config.baseline_wander_amplitude,
        baseline_wander_frequency=config.baseline_wander_frequency,
        baseline_sine_amplitude=config.baseline_sine_amplitude,
        baseline_sine_frequency=config.baseline_sine_frequency,
    )
    return EcgResult(
        clean_signal=clean_signal,
        signal=signal,
        fiducials=fiducials[0],
        segments=segments[0],
    )


def generate_cycle(
    heart_rate: int, params: EcgPresetParameters, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_cycle_samples = 60 / heart_rate * sample_rate
    rhythm_generator.smooth_and_noise = _no_smoothing
    _, ecg, segments, fiducials, _ = rhythm_generator.simulate_brisk(
        # Целое число секунд избегает ошибки округления длины внутри PhysioKit.
        signal_length=ceil(3 * 60 / heart_rate) * sample_rate,
        sample_rate=sample_rate,
        heart_rate=heart_rate,
        leads=1,
        preset=EcgPreset.SR,
        parameters=params,
        noise_multiplier=0.0,
    )

    p_onsets = np.flatnonzero(fiducials[0] == int(EcgFiducial.p_wave))
    if p_onsets.size < 2:
        raise RuntimeError("PhysioKit не вернул две границы P для полного цикла")
    start, stop = p_onsets[:2]
    signal = ecg[0, start:stop].copy()
    segments = segments[0, start:stop].copy()
    fiducials = fiducials[0, start:stop].copy()

    segments[segments == int(EcgSegment.tp_overlap)] = int(EcgSegment.p_wave)

    # Округление разметки может занести начало следующей P в последний отсчёт.
    if segments[-1] == int(EcgSegment.p_wave):
        tail_start = np.flatnonzero(segments != int(EcgSegment.p_wave))[-1] + 1
        segments[tail_start:] = int(EcgSegment.tp_segment)

    # Фактические R–R считаются по меткам в готовой записи, где есть соседние R.
    if abs(len(signal) - expected_cycle_samples) >= 2 or np.count_nonzero(fiducials == int(EcgFiducial.r_peak)) != 1:
        raise RuntimeError("Не удалось выделить один полный цикл с одним R-пиком")

    return signal, segments, fiducials


def generate_stitched_ecg(config: StitchedEcgConfig) -> StitchedEcgResult:
    config.validate()
    # Один генератор выбора параметров на всю запись, без сброса между циклами.
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)
    signals, segment_arrays, fiducial_arrays = [], [], []
    beats = []
    offset = 0

    for _ in range(config.beats + int(config.random_start)):
        heart_rate, params = sample_beat_parameters(config, rng)
        signal, segments, fiducials = generate_cycle(heart_rate, params, config.sample_rate)
        if config.no_p:
            # Метки P нужны только для выделения цикла внутри generate_cycle.
            segments[segments == int(EcgSegment.p_wave)] = int(EcgSegment.tp_segment)
            fiducials[fiducials == int(EcgFiducial.p_wave)] = 0
        signals.append(signal)
        segment_arrays.append(segments)
        fiducial_arrays.append(fiducials)
        beats.append(BeatInfo(offset, offset + len(signal), heart_rate, params))
        offset += len(signal)

    clean_signal = np.concatenate(signals)
    segments = np.concatenate(segment_arrays)
    fiducials = np.concatenate(fiducial_arrays)

    # Дополнительный цикл позволяет сдвинуть начало, сохранив исходную длину.
    window_length = beats[config.beats - 1].stop
    start = 0
    if config.random_start:
        max_offset = min(len(signals[0]) - 1, len(clean_signal) - window_length)
        start = int(rng.integers(0, max_offset + 1))
    stop = start + window_length
    previous_segment = int(segments[start - 1]) if start > 0 else int(EcgSegment.tp_segment)
    clean_signal = clean_signal[start:stop].copy()
    segments = segments[start:stop].copy()
    fiducials = fiducials[start:stop].copy()
    beats = [
        replace(beat, start=beat.start - start, stop=beat.stop - start)
        for beat in beats
        if beat.stop > start and beat.start < stop
    ]

    signal = add_ecg_noise(
        clean_signal,
        sampling_rate=config.sample_rate,
        noise_amplitude=config.noise_amplitude,
        random_state=config.seed,
        baseline_wander_amplitude=config.baseline_wander_amplitude,
        baseline_wander_frequency=config.baseline_wander_frequency,
        baseline_sine_amplitude=config.baseline_sine_amplitude,
        baseline_sine_frequency=config.baseline_sine_frequency,
    )
    return StitchedEcgResult(
        clean_signal, signal, segments, fiducials, beats,
        start_offset_samples=start,
        segment_before_start=previous_segment,
    )


def generate_record(
    parameters: SampledParameters,
    *,
    duration_s: float = 10.0,
    sample_rate: int = 1000,
    noise_amplitude: float = 0.03,
    random_start: bool = True,
    variation_pct: float = StitchedEcgConfig().variation_pct,
    amplitude_variation_pct: float = StitchedEcgConfig().amplitude_variation_pct,
    baseline_wander_amplitude: float = StitchedEcgConfig().baseline_wander_amplitude,
    baseline_wander_frequency: float = StitchedEcgConfig().baseline_wander_frequency,
    baseline_sine_amplitude: float = StitchedEcgConfig().baseline_sine_amplitude,
    baseline_sine_frequency: float = StitchedEcgConfig().baseline_sine_frequency,
) -> tuple[StitchedEcgConfig, StitchedEcgResult]:
    target_samples = round(duration_s * sample_rate)

    # Проверка нужна и для прямого вызова с параметрами, без случайного выбора.
    SamplingPreset(
        p_ms=(parameters.p_ms, parameters.p_ms),
        t_ms=(parameters.t_ms, parameters.t_ms),
        pq_ms=(parameters.pq_ms, parameters.pq_ms),
        qtc_ms=(parameters.qtc_ms, parameters.qtc_ms),
        qrs_ms=(parameters.qrs_ms, parameters.qrs_ms),
        hr_bpm=(parameters.hr_bpm, parameters.hr_bpm),
        seed=(parameters.seed, parameters.seed),
        shapes=(parameters.shape,),
    ).validate(variation_pct=variation_pct)

    min_cycle_samples = floor(int(60 / parameters.hr_bpm * 1000) * sample_rate / 1000)

    config = StitchedEcgConfig(
        beats=ceil(target_samples / min_cycle_samples),
        sample_rate=sample_rate,
        heart_rate=parameters.hr_bpm,
        seed=parameters.seed,
        random_start=random_start,
        noise_amplitude=noise_amplitude,
        variation_pct=variation_pct,
        amplitude_variation_pct=amplitude_variation_pct,
        baseline_wander_amplitude=baseline_wander_amplitude,
        baseline_wander_frequency=baseline_wander_frequency,
        baseline_sine_amplitude=baseline_sine_amplitude,
        baseline_sine_frequency=baseline_sine_frequency,
        p_duration_ms=parameters.p_ms,
        t_duration_ms=parameters.t_ms,
        pq_interval_ms=parameters.pq_ms,
        qt_interval_ms=parameters.qt_ms,
        qrs_duration_ms=parameters.qrs_ms,
        no_p=parameters.no_p,
        negative_p=parameters.negative_p,
        negative_t=parameters.negative_t,
        **asdict(MORPHOLOGY_PRESETS[parameters.shape]),
    )
    result = generate_stitched_ecg(config)
    if len(result.signal) < target_samples:
        raise RuntimeError("Генератор вернул недостаточно отсчётов для заданной длительности")
    result = replace(
        result,
        **{
            name: getattr(result, name)[:target_samples].copy()
            for name in ("signal", "clean_signal", "segments", "fiducials")
        },
        beats=[beat for beat in result.beats if beat.stop > 0 and beat.start < target_samples],
    )
    return config, result
