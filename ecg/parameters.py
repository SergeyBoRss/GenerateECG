from math import ceil, floor

import numpy as np
from physiokit.ecg.synthetic.defines import EcgPresetParameters

from .config import EcgConfig, parameter_bounds
from .models import SampledParameters
from .morphology import MORPHOLOGY_PRESETS


def sample_beat_parameters(
    parameters: SampledParameters, config: EcgConfig, rng: np.random.Generator,
) -> tuple[int, EcgPresetParameters]:
    def integer(name: str) -> int:
        low, high = parameter_bounds(getattr(parameters, name), config.variation_pct, integer=True)
        return int(rng.integers(low, high + 1))

    def floating(name: str) -> float:
        variation_pct = (
            config.amplitude_variation_pct
            if name in ("p_voltage", "q_depth", "r_height", "s_depth", "t_height")
            else config.variation_pct
        )
        return float(rng.uniform(*parameter_bounds(getattr(parameters, name), variation_pct)))

    heart_rate = parameters.hr_bpm
    p = integer("p_ms")
    pq = integer("pq_ms")
    qrs = integer("qrs_ms")
    qt = integer("qt_ms")
    t = integer("t_ms")
    st = qt - qrs - t
    p_voltage = floating("p_voltage")
    if parameters.no_p:
        p_voltage = 0.0
    elif parameters.negative_p:
        p_voltage *= -1

    # Все параметры одного отведения заданы явно, без случайного пресета.
    params = EcgPresetParameters(
        p_length=p,
        pr_interval=pq,
        qrs_duration=qrs,
        st_length=st,
        t_length=t,
        qt=qt,
        qtc=qt / np.cbrt(60 / heart_rate),
        flippers=[1],
        p_voltages=[p_voltage],
        p_biphasics=[False],
        p_leans=[floating("p_lean")],
        q_depths=[floating("q_depth")],
        r_heights=[floating("r_height")],
        r_prime_presents=[False],
        r_prime_heights=[0.0],
        r_to_r_prime_duration_ratio=[1.0],
        # Нулевая глубина убирает и зубец S, и его метку.
        s_presents=[parameters.s_depth > 0],
        s_depths=[floating("s_depth")],
        s_prime_heights=[0.0],
        s_to_qrs_duration_ratio=[floating("s_to_qrs_duration_ratio")],
        st_deltas=[0.0],
        j_points=[0.0],
        t_heights=[floating("t_height") * (-1 if parameters.negative_t else 1)],
        t_leans=[parameters.t_lean],
    )
    return heart_rate, params


def sample_record_parameters(
    config: EcgConfig, rng: np.random.Generator,
) -> SampledParameters:
    config.validate()

    def draw(bounds: tuple[int, int]) -> int:
        return int(rng.integers(bounds[0], bounds[1] + 1))

    def floating(bounds: tuple[float, float]) -> float:
        low, high = bounds
        return float(low) if low == high else float(rng.uniform(low, high))

    def chance(percent: float) -> bool:
        return percent > 0 and bool(rng.random() < percent / 100)

    # Одно решение на всю запись; отсутствие P важнее её полярности.
    no_p = chance(config.no_p)
    negative_p = chance(config.negative_p) and not no_p
    negative_qrs = chance(config.negative_qrs)
    negative_t = chance(config.negative_t)

    pq = draw(config.pq_ms)
    qt_or_qtc = draw(config.qt_ms if config.qt_ms is not None else config.qtc_ms)
    qrs = draw(config.qrs_ms)
    hr = draw(config.hr_bpm)
    shape = config.shapes[int(rng.integers(len(config.shapes)))]
    p = draw(config.p_ms)
    t = draw(config.t_ms)
    if config.qt_ms is None:
        qtc = qt_or_qtc
        qt = round(qtc * np.cbrt(60 / hr))
    else:
        qt = qt_or_qtc
        qtc = float(qt / np.cbrt(60 / hr))

    morphology = MORPHOLOGY_PRESETS[shape]
    if any(getattr(config, name) is not None for name in (
        "q_depth", "r_height", "s_depth", "s_to_qrs_duration_ratio",
    )):
        shape = "custom"

    def morphology_value(name: str) -> float:
        bounds = getattr(config, name)
        return getattr(morphology, name) if bounds is None else floating(bounds)

    noise_low, noise_high = config.noise_amplitude

    return SampledParameters(
        pq_ms=pq, qtc_ms=qtc, qt_ms=qt, qrs_ms=qrs, hr_bpm=hr,
        seed=config.seed, shape=shape, p_ms=p, t_ms=t,
        no_p=no_p,
        negative_p=negative_p,
        negative_qrs=negative_qrs,
        negative_t=negative_t,
        p_voltage=floating(config.p_voltage),
        q_depth=morphology_value("q_depth"),
        r_height=morphology_value("r_height"),
        s_depth=morphology_value("s_depth"),
        t_height=floating(config.t_height),
        p_lean=floating(config.p_lean),
        t_lean=floating(config.t_lean),
        s_to_qrs_duration_ratio=morphology_value("s_to_qrs_duration_ratio"),
        noise_amplitude=(
            float(noise_low) if noise_low == noise_high
            else draw((ceil(noise_low * 1000), floor(noise_high * 1000))) / 1000
        ),
        baseline_wander_amplitude=floating(config.baseline_wander_amplitude),
        baseline_wander_frequency=floating(config.baseline_wander_frequency),
        baseline_sine_amplitude=floating(config.baseline_sine_amplitude),
        baseline_sine_frequency=floating(config.baseline_sine_frequency),
    )
