import random
from dataclasses import replace
from math import ceil

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment
from physiokit.ecg.synthetic import rhythm_generator
from physiokit.ecg.synthetic.defines import EcgPreset, EcgPresetParameters

from ecg_noise import add_ecg_noise

from .config import EcgConfig, validate_intervals
from .models import BeatInfo, SampledParameters, EcgResult
from .parameters import sample_beat_parameters
from .rr import RrDistribution


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

    # Один генератор выбора параметров на всю запись, без сброса между циклами.
    rng = np.random.default_rng(parameters.seed)
    random.seed(parameters.seed)
    signals, segment_arrays, fiducial_arrays = [], [], []
    beats = []
    r_offsets = []
    covered_samples = 0

    while covered_samples < target_samples:
        heart_rate, params = sample_beat_parameters(parameters, config, rng)
        signal, segments, fiducials = generate_cycle(heart_rate, params, config.sample_rate)
        r_offsets.append(int(np.flatnonzero(fiducials == int(EcgFiducial.r_peak))[0]))

        occupied = (segments != int(EcgSegment.tp_segment)) | (fiducials != 0)
        core_stop = int(np.flatnonzero(occupied)[-1]) + 1
        if parameters.no_p:
            segments[segments == int(EcgSegment.p_wave)] = int(EcgSegment.tp_segment)
            fiducials[fiducials == int(EcgFiducial.p_wave)] = 0
        signals.append(signal[:core_stop].copy())
        segment_arrays.append(segments[:core_stop].copy())
        fiducial_arrays.append(fiducials[:core_stop].copy())
        beats.append(BeatInfo(0, core_stop, heart_rate, params))
        if len(beats) > 2:
            covered_samples += beats[-2].stop

    minimum_rr = max(
        beat.stop - current_r + next_r
        for beat, current_r, next_r in zip(beats, r_offsets, r_offsets[1:])
    )
    mean_rr = 60 / parameters.hr_bpm * config.sample_rate

    distribution = RrDistribution(
        mean=mean_rr, std=config.rr_std_ms * config.sample_rate / 1000,
        minimum=minimum_rr,
    )
    rr_samples = distribution.sample(rng, len(beats) - 1)
    cycle_starts = [0]
    for rr, current_r, next_r in zip(rr_samples, r_offsets, r_offsets[1:]):
        cycle_starts.append(cycle_starts[-1] + round(float(rr)) + current_r - next_r)

    start = int(rng.integers(cycle_starts[1])) if config.random_start else 0
    previous_segment = (
        int(segment_arrays[0][start - 1]) if 0 < start <= len(segment_arrays[0])
        else int(EcgSegment.tp_segment)
    )

    clean_signal = np.zeros(target_samples)
    segments = np.full(target_samples, int(EcgSegment.tp_segment), dtype=int)
    fiducials = np.zeros(target_samples, dtype=int)
    placed_beats = []
    for i, beat in enumerate(beats[:-1]):
        beat_start, beat_stop = cycle_starts[i] - start, cycle_starts[i + 1] - start
        if beat_start >= target_samples:
            break
        placed_beats.append(replace(beat, start=beat_start, stop=beat_stop))
        left = max(0, beat_start)
        right = min(target_samples, beat_start + len(signals[i]))
        if left < right:
            source = slice(left - beat_start, right - beat_start)
            clean_signal[left:right] = signals[i][source]
            segments[left:right] = segment_arrays[i][source]
            fiducials[left:right] = fiducial_arrays[i][source]

    if parameters.negative_qrs:
        clean_signal[segments == int(EcgSegment.qrs_complex)] *= -1

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
    return EcgResult(
        clean_signal, signal, segments, fiducials, placed_beats,
        start_offset_samples=start,
        segment_before_start=previous_segment,
    )
