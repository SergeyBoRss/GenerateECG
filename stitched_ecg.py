import argparse
from dataclasses import dataclass
from pathlib import Path

from ecg.annotations import extract_boundaries, get_boundaries, make_ecg_lines
from ecg.config import StitchedEcgConfig
from ecg.generation import generate_cycle, generate_stitched_ecg as generate_ecg
from ecg.models import BeatInfo, StitchedEcgResult
from ecg.output import plot_result, print_stitched_summary as print_summary, save_result, show_plot
from ecg.parameters import sample_beat_parameters as sample_parameters


@dataclass(frozen=True)
class CliOptions:
    config: StitchedEcgConfig
    output_prefix: Path | None = None
    show: bool = True


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
                        help="Амплитуда обычного шума; 0 отключает")
    parser.add_argument("--baseline-wander-amplitude", type=float, default=defaults.baseline_wander_amplitude,
                        help="Масштаб случайного дрейфа в долях std чистого сигнала; 0 отключает")
    parser.add_argument("--baseline-wander-frequency", type=float, default=defaults.baseline_wander_frequency,
                        help="Частота изменений случайного дрейфа, Гц")
    parser.add_argument("--baseline-sine-amplitude", type=float, default=defaults.baseline_sine_amplitude,
                        help="Амплитуда синусоидального дрейфа в долях std чистого сигнала; 0 отключает")
    parser.add_argument("--baseline-sine-frequency", type=float, default=defaults.baseline_sine_frequency,
                        help="Частота синусоидального дрейфа, Гц")
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
                        help="Путь без расширения для файлов PNG, NPY, NPZ и JSON")
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
        save_path = None
        title = ""
        if options.output_prefix is not None:
            options.output_prefix.parent.mkdir(parents=True, exist_ok=True)
            save_path = Path(f"{options.output_prefix}.png")
            title = (
                f"{options.output_prefix.parent.name} | "
                f"QRS={config.qrs_duration_ms} ms | seed={config.seed}"
            )
        plot_result(
            result.signal, config.sample_rate, get_boundaries(result),
            save_path=save_path, show=options.show, title=title,
        )
        if options.output_prefix is not None:
            save_result(config, result, options.output_prefix)
            print(f"Сохранено: {options.output_prefix}.{{png,npy,npz,json}}")
        if options.show:
            show_plot()


if __name__ == "__main__":
    main()
