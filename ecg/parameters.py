import numpy as np
from physiokit.ecg.synthetic import presets
from physiokit.ecg.synthetic.defines import EcgPresetParameters

from .config import EcgConfig, SamplingPreset, StitchedEcgConfig
from .models import SampledParameters


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


def sample_beat_parameters(
    config: StitchedEcgConfig, rng: np.random.Generator
) -> tuple[int, EcgPresetParameters]:
    def integer(name: str) -> int:
        low, high = config.parameter_bounds(name)
        return int(rng.integers(low, high + 1))

    def floating(name: str) -> float:
        return float(rng.uniform(*config.parameter_bounds(name)))

    heart_rate = config.heart_rate
    p = integer("p_duration_ms")
    pq = integer("pq_interval_ms")
    qrs = integer("qrs_duration_ms")
    qt = integer("qt_interval_ms")
    t = integer("t_duration_ms")
    st = qt - qrs - t

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
        p_voltages=[floating("p_voltage")],
        p_biphasics=[False],
        p_leans=[floating("p_lean")],
        q_depths=[floating("q_depth")],
        r_heights=[floating("r_height")],
        r_prime_presents=[False],
        r_prime_heights=[0.0],
        r_to_r_prime_duration_ratio=[1.0],
        # Нулевая глубина убирает и зубец S, и его метку.
        s_presents=[config.s_depth > 0],
        s_depths=[floating("s_depth")],
        s_prime_heights=[0.0],
        s_to_qrs_duration_ratio=[floating("s_to_qrs_duration_ratio")],
        st_deltas=[0.0],
        j_points=[0.0],
        t_heights=[floating("t_height")],
        t_leans=[config.t_lean],
    )
    return heart_rate, params


def sample_record_parameters(
    preset: SamplingPreset, rng: np.random.Generator,
    *, variation_pct: float = StitchedEcgConfig().variation_pct,
) -> SampledParameters:
    preset.validate(variation_pct=variation_pct)

    def draw(bounds: tuple[int, int]) -> int:
        return int(rng.integers(bounds[0], bounds[1] + 1))

    return SampledParameters(
        pq_ms=draw(preset.pq_ms),
        qtc_ms=draw(preset.qtc_ms),
        qrs_ms=draw(preset.qrs_ms),
        hr_bpm=draw(preset.hr_bpm),
        seed=draw(preset.seed),
        shape=preset.shapes[int(rng.integers(len(preset.shapes)))],
        p_ms=draw(preset.p_ms),
        t_ms=draw(preset.t_ms),
    )
