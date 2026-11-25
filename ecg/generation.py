import random
from dataclasses import replace
from math import ceil, floor

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment
from physiokit.ecg.synthetic import rhythm_generator
from physiokit.ecg.synthetic.defines import EcgPreset, EcgPresetParameters

from ecg_noise import add_ecg_noise

from .config import EcgConfig, validate_intervals
from .models import BeatInfo, SampledParameters, EcgResult
from .parameters import sample_beat_parameters


def _no_smoothing(y, noise_multiplier=0.0, impedance=1.0):
    return np.asarray(y, dtype=float) / impedance


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


def generate_record(parameters: SampledParameters, config: EcgConfig) -> EcgResult:
    config.validate()
    validate_intervals(
        p_ms=(parameters.p_ms, parameters.p_ms),
        pq_ms=(parameters.pq_ms, parameters.pq_ms),
        qrs_ms=(parameters.qrs_ms, parameters.qrs_ms),
        qt_ms=(parameters.qt_ms, parameters.qt_ms),
        t_ms=(parameters.t_ms, parameters.t_ms),
        hr_bpm=parameters.hr_bpm, variation_pct=config.variation_pct,
    )
    target_samples = round(config.duration_s * config.sample_rate)
    min_cycle_samples = floor(int(60 / parameters.hr_bpm * 1000) * config.sample_rate / 1000)
    beat_count = ceil(target_samples / min_cycle_samples)

    # Один генератор выбора параметров на всю запись, без сброса между циклами.
    rng = np.random.default_rng(parameters.seed)
    random.seed(parameters.seed)
    signals, segment_arrays, fiducial_arrays = [], [], []
    beats = []
    offset = 0

    for _ in range(beat_count + int(config.random_start)):
        heart_rate, params = sample_beat_parameters(parameters, config, rng)
        signal, segments, fiducials = generate_cycle(heart_rate, params, config.sample_rate)
        if parameters.no_p:
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
    window_length = beats[beat_count - 1].stop
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
        noise_amplitude=parameters.noise_amplitude,
        random_state=parameters.seed,
        baseline_wander_amplitude=parameters.baseline_wander_amplitude,
        baseline_wander_frequency=parameters.baseline_wander_frequency,
        baseline_sine_amplitude=parameters.baseline_sine_amplitude,
        baseline_sine_frequency=parameters.baseline_sine_frequency,
    )
    result = EcgResult(
        clean_signal, signal, segments, fiducials, beats,
        start_offset_samples=start,
        segment_before_start=previous_segment,
    )
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
    return result
