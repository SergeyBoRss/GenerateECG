import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment

from .annotations import extract_annotations, get_boundaries, make_ecg_lines
from .config import EcgConfig, StitchedEcgConfig
from .models import EcgResult, SampledParameters, StitchedEcgResult


def print_base_summary(
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


def _print_rr_summary(config: StitchedEcgConfig, result: StitchedEcgResult) -> None:
    rr_samples = np.diff(result.r_peak_samples)
    if rr_samples.size:
        rr_ms = rr_samples * 1000 / config.sample_rate
        print("Фактические RR по меткам R, мс: " + ", ".join(f"{value:.3f}" for value in rr_ms))
    else:
        print("Фактический RR не определён: в записи меньше двух меток R.")


def print_stitched_summary(config: StitchedEcgConfig, result: StitchedEcgResult) -> None:
    print(f"Длительность: {len(result.signal) / config.sample_rate:.3f} с; задано циклов: {config.beats}")
    print(f"Циклов в окне, включая неполные: {len(result.beats)}")
    print(f"Частота дискретизации: {config.sample_rate} Гц, seed: {config.seed}")
    print(f"Разброс амплитуд: ±{config.amplitude_variation_pct:g}%; "
          f"остальных параметров: ±{config.variation_pct:g}%")
    print(f"Смещение начала: {result.start_offset_samples / config.sample_rate * 1000:.1f} мс")
    _print_rr_summary(config, result)
    if result.beats[0].start < 0:
        print("Отрицательный старт в таблице означает, что цикл начался до начала записи.")
    print("Выбранные параметры циклов (длительности в мс):")
    print(" N   Старт, с   ЧСС   PQ  QRS   QT    P   ST    T")
    for number, beat in enumerate(result.beats, start=1):
        p = beat.parameters
        print(
            f"{number:2d} {beat.start / config.sample_rate:10.3f} {beat.heart_rate:5d}"
            f" {p.pr_interval:4d} {p.qrs_duration:4d} {p.qt:4d}"
            f" {p.p_length:4d} {p.st_length:4d} {p.t_length:4d}"
        )


def print_random_summary(
    parameters: SampledParameters,
    config: StitchedEcgConfig,
    result: StitchedEcgResult,
    sampler_seed: int,
) -> None:
    print(f"sampler_seed: {sampler_seed}")
    print(f"PQ={parameters.pq_ms} мс; QTc={parameters.qtc_ms} мс (Fridericia); "
          f"QT={parameters.qt_ms} мс; QRS={parameters.qrs_ms} мс")
    print(f"HR={parameters.hr_bpm} уд/мин; seed={parameters.seed}; форма={parameters.shape}; "
          f"разброс амплитуд=±{config.amplitude_variation_pct:g}%; "
          f"остальных параметров=±{config.variation_pct:g}%")
    print(f"Длительность: {len(result.signal) / config.sample_rate:g} с; "
          f"отсчётов: {len(result.signal)}; частота: {config.sample_rate} Гц")
    _print_rr_summary(config, result)


def build_record_metadata(
    config: StitchedEcgConfig, result: StitchedEcgResult
) -> dict:
    r_peak_samples = result.r_peak_samples
    rr_samples = np.diff(r_peak_samples)
    rr_ms = rr_samples * 1000 / config.sample_rate
    return {
        "schema_version": 1,
        "config": asdict(config),
        "sample_count": len(result.signal),
        "duration_s": len(result.signal) / config.sample_rate,
        "start_offset_samples": result.start_offset_samples,
        "segment_before_start": result.segment_before_start,
        "rr_source": "r_peak fiducials in the final cropped signal",
        "r_peak_samples": r_peak_samples.tolist(),
        "rr_samples": rr_samples.tolist(),
        "rr_ms": rr_ms.tolist(),
        "beats": [asdict(beat) for beat in result.beats],
        "annotations": extract_annotations(result.segments),
        "boundaries_samples": {
            name: positions.tolist() for name, positions in get_boundaries(result).items()
        },
        "segment_codes": {label.name: int(label) for label in EcgSegment},
        "fiducial_codes": {label.name: int(label) for label in EcgFiducial},
    }


def save_result(
    config: StitchedEcgConfig,
    result: StitchedEcgResult,
    output_prefix: Path,
    *,
    sampling: dict[str, object] | None = None,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_record_metadata(config, result)
    r_peak_samples = result.r_peak_samples
    rr_samples = np.diff(r_peak_samples)
    np.save(f"{output_prefix}.npy", result.signal, allow_pickle=False)
    np.savez_compressed(
        f"{output_prefix}.npz",
        time_s=np.arange(len(result.signal)) / config.sample_rate,
        signal=result.signal,
        clean_signal=result.clean_signal,
        segments=result.segments,
        fiducials=result.fiducials,
        sample_rate=config.sample_rate,
        r_peak_samples=r_peak_samples,
        rr_samples=rr_samples,
        rr_ms=rr_samples * 1000 / config.sample_rate,
    )
    if sampling is not None:
        metadata["sampling"] = sampling
    Path(f"{output_prefix}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def plot_random_result(
    config: StitchedEcgConfig,
    result: StitchedEcgResult,
    parameters: SampledParameters,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    plot_result(
        result.signal, config.sample_rate, get_boundaries(result),
        save_path=save_path, show=show,
        title=(f"{parameters.shape} | PQ={parameters.pq_ms} | QTcF={parameters.qtc_ms} | "
               f"QRS={parameters.qrs_ms} ms | HR={parameters.hr_bpm} | "
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


def show_plot() -> None:
    import matplotlib.pyplot as plt
    plt.show()
