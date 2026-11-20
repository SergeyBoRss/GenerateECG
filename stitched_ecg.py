import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from math import ceil, floor
from pathlib import Path

import numpy as np
from physiokit.ecg.defines import EcgFiducial, EcgSegment
from physiokit.ecg.synthetic import rhythm_generator
from physiokit.ecg.synthetic.defines import EcgPreset, EcgPresetParameters

from base import extract_boundaries, make_ecg_lines
from ecg_noise import add_ecg_noise


@dataclass(frozen=True)
class StitchedEcgConfig:
    beats: int = 4  # Длина окна в циклах; крайние циклы могут быть неполными.
    sample_rate: int = 1000  # Частота генерации и готовой записи, Гц.
    heart_rate: int = 70  # Заданная ЧСС генератора; фактические R–R могут различаться.
    seed: int = 42
    random_start: bool = True  # False — начинать с начала первого P.
    noise_amplitude: float = 0.03
    variation_pct: float = 5.0  # Общий разброс ±5%; 0 отключает разброс.

    # Базовые длительности; разброс задаётся одним variation_pct.
    pq_interval_ms: int = 150
    qrs_duration_ms: int = 90
    qt_interval_ms: int = 400
    p_duration_ms: int = 100
    st_duration_ms: int = 100

    # Амплитуды в единицах модели PhysioKit
    p_voltage: float = 0.15
    q_depth: float = 0.10
    r_height: float = 3.00
    s_depth: float = 0.20
    t_height: float = 5.00
    p_lean: float = 1.0
    t_lean: float = 0.0  # Постоянный наклон T
    s_to_qrs_duration_ratio: float = 0.15

    def parameter_bounds(self, name: str) -> tuple[int, int] | tuple[float, float]:
        value = getattr(self, name)
        delta = abs(value) * self.variation_pct / 100
        low, high = value - delta, value + delta
        if name.endswith("_ms"):
            return ceil(low), floor(high)
        return low, high

    def validate(self) -> None:
        for name in (
            "pq_interval_ms", "qrs_duration_ms", "qt_interval_ms",
            "p_duration_ms", "st_duration_ms", "p_lean", "s_to_qrs_duration_ratio",
        ):
            if name.endswith("_ms"):
                if self.parameter_bounds(name)[0] <= 0:
                    raise ValueError(f"{name}: разброс должен оставлять длительность положительной")
            elif self.parameter_bounds(name)[0] < 0:
                raise ValueError(f"{name}: разброс не должен давать отрицательные значения")

        # Проверяем крайние значения: любая выбранная комбинация должна подходить.
        if self.parameter_bounds("p_duration_ms")[1] > self.parameter_bounds("pq_interval_ms")[0]:
            raise ValueError("Максимальная P должна быть <= минимального PQ")
        if self.parameter_bounds("qt_interval_ms")[0] <= (
            self.parameter_bounds("qrs_duration_ms")[1] + self.parameter_bounds("st_duration_ms")[1]
        ):
            raise ValueError("Минимальный QT должен быть больше максимальных QRS + ST")
        rr_ms = int(60 / self.heart_rate * 1000)
        if self.parameter_bounds("pq_interval_ms")[1] + self.parameter_bounds("qt_interval_ms")[1] >= rr_ms:
            raise ValueError("Максимальные PQ + QT должны помещаться в RR")


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

    @property
    def rr_samples(self) -> np.ndarray:
        """Фактические интервалы между соседними метками R после склейки и обрезки."""
        return np.diff(self.r_peak_samples)


@dataclass(frozen=True)
class CliOptions:
    config: StitchedEcgConfig
    output_prefix: Path | None = None
    show: bool = True


def sample_parameters(
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
    st = integer("st_duration_ms")

    # Все параметры одного отведения заданы явно, без случайного пресета.
    params = EcgPresetParameters(
        p_length=p,
        pr_interval=pq,
        qrs_duration=qrs,
        st_length=st,
        t_length=qt - qrs - st,
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


def generate_ecg(config: StitchedEcgConfig) -> StitchedEcgResult:
    config.validate()
    # Один генератор выбора параметров на всю запись, без сброса между циклами.
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)
    signals, segment_arrays, fiducial_arrays = [], [], []
    beats = []
    offset = 0

    for _ in range(config.beats + int(config.random_start)):
        heart_rate, params = sample_parameters(config, rng)
        signal, segments, fiducials = generate_cycle(heart_rate, params, config.sample_rate)
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
    )
    return StitchedEcgResult(
        clean_signal, signal, segments, fiducials, beats,
        start_offset_samples=start,
        segment_before_start=previous_segment,
    )


def get_boundaries(result: StitchedEcgResult) -> dict[str, np.ndarray]:
    with_previous = np.concatenate(([result.segment_before_start], result.segments))
    return {
        name: positions - 1
        for name, positions in extract_boundaries(with_previous).items()
    }


def print_summary(config: StitchedEcgConfig, result: StitchedEcgResult) -> None:
    print(f"Длительность: {len(result.signal) / config.sample_rate:.3f} с; задано циклов: {config.beats}")
    print(f"Циклов в окне, включая неполные: {len(result.beats)}")
    print(f"Частота дискретизации: {config.sample_rate} Гц, seed: {config.seed}")
    print(f"Смещение начала: {result.start_offset_samples / config.sample_rate * 1000:.1f} мс")
    rr_samples = result.rr_samples
    if rr_samples.size:
        rr_ms = rr_samples * 1000 / config.sample_rate
        print(f"Фактические RR по меткам R, отсчёты: {rr_samples.tolist()}")
        print("Фактические RR по меткам R, мс: " + ", ".join(f"{value:.3f}" for value in rr_ms))
    else:
        print("Фактический RR не определён: в записи меньше двух меток R.")
    if result.beats[0].start < 0:
        print("Отрицательный старт в таблице означает, что цикл начался до начала записи.")
    print("Выбранные параметры циклов (длительности в мс):")
    print(" №   Старт, с   ЧСС   PQ  QRS   QT    P   ST    T")
    for number, beat in enumerate(result.beats, start=1):
        p = beat.parameters
        print(
            f"{number:2d} {beat.start / config.sample_rate:10.3f} {beat.heart_rate:5d}"
            f" {p.pr_interval:4d} {p.qrs_duration:4d} {p.qt:4d}"
            f" {p.p_length:4d} {p.st_length:4d} {p.t_length:4d}"
        )


def save_result(
    config: StitchedEcgConfig, result: StitchedEcgResult, output_prefix: Path
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f"{output_prefix}.npz",
        time_s=np.arange(len(result.signal)) / config.sample_rate,
        signal=result.signal,
        clean_signal=result.clean_signal,
        segments=result.segments,
        fiducials=result.fiducials,
        sample_rate=config.sample_rate,
        r_peak_samples=result.r_peak_samples,
        rr_samples=result.rr_samples,
        rr_ms=result.rr_samples * 1000 / config.sample_rate,
    )
    metadata = {
        "schema_version": 1,
        "config": asdict(config),
        "sample_count": len(result.signal),
        "duration_s": len(result.signal) / config.sample_rate,
        "start_offset_samples": result.start_offset_samples,
        "segment_before_start": result.segment_before_start,
        "rr_source": "r_peak fiducials in the final cropped signal",
        "r_peak_samples": result.r_peak_samples.tolist(),
        "rr_samples": result.rr_samples.tolist(),
        "rr_ms": (result.rr_samples * 1000 / config.sample_rate).tolist(),
        "beats": [asdict(beat) for beat in result.beats],
        "boundaries_samples": {
            name: positions.tolist() for name, positions in get_boundaries(result).items()
        },
        "segment_codes": {label.name: int(label) for label in EcgSegment},
        "fiducial_codes": {label.name: int(label) for label in EcgFiducial},
    }
    Path(f"{output_prefix}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> CliOptions:
    defaults = StitchedEcgConfig()
    parser = argparse.ArgumentParser(
        description="Генерация ЭКГ из отдельных циклов с заданными параметрами.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--beats", type=int, default=defaults.beats,
                        help="Длина окна в циклах")
    parser.add_argument("--sample-rate", type=int, default=defaults.sample_rate,
                        help="Частота дискретизации, Гц")
    parser.add_argument("--heart-rate", type=int, default=defaults.heart_rate,
                        help="Постоянная ЧСС, уд/мин")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Начальное значение генератора случайных чисел")
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction,
                        default=defaults.random_start,
                        help="Начинать запись со случайной позиции внутри цикла")
    parser.add_argument("--noise-amplitude", type=float, default=defaults.noise_amplitude,
                        help="Амплитуда шума; 0 отключает шум")
    parser.add_argument("--variation-pct", type=float, default=defaults.variation_pct,
                        help="Разброс параметров в процентах; 0 отключает разброс")

    durations = parser.add_argument_group("Интервалы и длительности, мс")
    durations.add_argument("--pq-interval-ms", type=int, default=defaults.pq_interval_ms,
                           help="Интервал PQ")
    durations.add_argument("--qrs-duration-ms", type=int, default=defaults.qrs_duration_ms,
                           help="Длительность QRS")
    durations.add_argument("--qt-interval-ms", type=int, default=defaults.qt_interval_ms,
                           help="Интервал QT")
    durations.add_argument("--p-duration-ms", type=int, default=defaults.p_duration_ms,
                           help="Длительность P")
    durations.add_argument("--st-duration-ms", type=int, default=defaults.st_duration_ms,
                           help="Длительность ST")

    morphology = parser.add_argument_group("Амплитуды и форма волн")
    morphology.add_argument("--p-voltage", type=float, default=defaults.p_voltage,
                            help="Амплитуда P")
    morphology.add_argument("--q-depth", type=float, default=defaults.q_depth,
                            help="Глубина Q")
    morphology.add_argument("--r-height", type=float, default=defaults.r_height,
                            help="Высота R")
    morphology.add_argument("--s-depth", type=float, default=defaults.s_depth,
                            help="Глубина S")
    morphology.add_argument("--t-height", type=float, default=defaults.t_height,
                            help="Высота T")
    morphology.add_argument("--p-lean", type=float, default=defaults.p_lean,
                            help="Наклон P")
    morphology.add_argument("--t-lean", type=float, default=defaults.t_lean,
                            help="Постоянный наклон T")
    morphology.add_argument("--s-to-qrs-duration-ratio", type=float,
                            default=defaults.s_to_qrs_duration_ratio,
                            help="Отношение длительности S к длительности QRS")

    output = parser.add_argument_group("Сохранение результатов")
    output.add_argument("--output-prefix", type=Path,
                        help="Путь без расширения для файлов PNG, NPZ и JSON")
    output.add_argument("--no-show", action="store_true",
                        help="Не открывать окно графика (для пакетной генерации)")

    values = vars(parser.parse_args(argv))
    output_prefix = values.pop("output_prefix")
    show = not values.pop("no_show")
    config = StitchedEcgConfig(**values)
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return CliOptions(config=config, output_prefix=output_prefix, show=show)


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    config = options.config
    result = generate_ecg(config)
    print_summary(config, result)

    if options.show or options.output_prefix is not None:
        if not options.show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ml_validation.plotter import plot_ecg

        save_path = None
        title = ""
        if options.output_prefix is not None:
            options.output_prefix.parent.mkdir(parents=True, exist_ok=True)
            save_path = Path(f"{options.output_prefix}.png")
            title = (
                f"{options.output_prefix.parent.name} | "
                f"QRS={config.qrs_duration_ms} ms | seed={config.seed}"
            )
        plot_ecg(
            result.signal,
            sample_rate=config.sample_rate,
            ecg_lines=make_ecg_lines(get_boundaries(result)),
            save_path=save_path,
            show=options.show,
            title=title,
        )
        if options.output_prefix is not None:
            save_result(config, result, options.output_prefix)
            print(f"Сохранено: {options.output_prefix}.{{png,npz,json}}")
        if options.show:
            plt.show()


if __name__ == "__main__":
    main()
