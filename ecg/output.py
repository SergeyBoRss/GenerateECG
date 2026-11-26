from pathlib import Path

import numpy as np

from .annotations import get_boundaries, make_ecg_lines
from .config import EcgConfig
from .models import EcgResult, SampledParameters


def _print_rr_summary(config: EcgConfig, result: EcgResult) -> None:
    rr_samples = np.diff(result.r_peak_samples)
    if rr_samples.size:
        rr_ms = rr_samples * 1000 / config.sample_rate
        print("Фактические RR по меткам R, мс: " + ", ".join(f"{value:.3f}" for value in rr_ms))
        print(f"Среднее RR: {rr_ms.mean():.3f} мс; std: {rr_ms.std():.3f} мс")
    else:
        print("Фактический RR не определён: в записи меньше двух меток R.")


def print_random_summary(
    parameters: SampledParameters,
    config: EcgConfig,
    result: EcgResult,
    sampler_seed: int,
) -> None:
    print(f"sampler_seed: {sampler_seed}")
    print(f"P={parameters.p_ms} мс; T={parameters.t_ms} мс; "
          f"PQ={parameters.pq_ms} мс; QTc={parameters.qtc_ms} мс (Fridericia); "
          f"QT={parameters.qt_ms} мс; QRS={parameters.qrs_ms} мс")
    print(f"HR={parameters.hr_bpm} уд/мин; seed={parameters.seed}; форма={parameters.shape}; "
          f"разброс амплитуд=±{config.amplitude_variation_pct}%; "
          f"остальных параметров=±{config.variation_pct}%; "
          f"заданный std RR={config.rr_std_ms} мс")
    if parameters.no_p:
        print("P отсутствует")
    elif parameters.negative_p:
        print("P инвертирована")
    if parameters.negative_qrs:
        print("QRS инвертирован")
    if parameters.negative_t:
        print("T инвертирована")
    print(f"Длительность: {len(result.signal) / config.sample_rate:g} с; "
          f"отсчётов: {len(result.signal)}; частота: {config.sample_rate} Гц")
    _print_rr_summary(config, result)


def plot_random_result(
    config: EcgConfig,
    result: EcgResult,
    parameters: SampledParameters,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    p_label = "absent" if parameters.no_p else f"{parameters.p_ms} ms"
    if parameters.negative_p and not parameters.no_p:
        p_label += " (inverted)"
    qrs_label = f"{parameters.qrs_ms} ms" + (" (inverted)" if parameters.negative_qrs else "")
    t_label = f"{parameters.t_ms} ms" + (" (inverted)" if parameters.negative_t else "")
    plot_result(
        result.signal, config.sample_rate, get_boundaries(result),
        save_path=save_path, show=show,
        title=(f"{parameters.shape} | P={p_label} | T={t_label} | "
               f"PQ={parameters.pq_ms} | QTcF={parameters.qtc_ms} | "
               f"QRS={qrs_label} | HR={parameters.hr_bpm} | "
               f"RR std={config.rr_std_ms:g} ms | "
               f"variation=±{config.variation_pct:g}% | "
               f"amplitude variation=±{config.amplitude_variation_pct:g}% | seed={parameters.seed}"),
    )


def plot_result(
    signal: np.ndarray,
    sample_rate: int,
    boundaries: dict[str, np.ndarray],
    *,
    save_path: Path | None = None,
    show: bool = True,
    title: str = "",
    amplitude: float | tuple[float, float] = 1.5,
) -> None:
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    from ml_validation.plotter import plot_ecg

    plot_ecg(
        signal,
        sample_rate=sample_rate,
        ecg_lines=make_ecg_lines(boundaries),
        save_path=save_path,
        show=show,
        title=title,
        amplitude=amplitude,
    )
