import argparse
from pathlib import Path

from ecg.batch import generate_dataset
from ecg.config import EcgConfig
from ecg.morphology import MORPHOLOGY_PRESETS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = EcgConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Генерация ЭКГ из диапазонов параметров; "
            "MIN = MAX фиксирует значение на запись."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    range_arguments = (
        ("p", "p_ms", int, "Длительность P, мс"),
        ("t", "t_ms", int, "Длительность T, мс"),
        ("pq", "pq_ms", int, "PQ, мс"),
        ("qrs", "qrs_ms", int, "QRS, мс"),
        ("hr", "hr_bpm", int, "HR, уд/мин"),
        ("p-voltage", "p_voltage", float, "Амплитуда P"),
        ("q-depth", "q_depth", float, "Глубина Q; без аргумента — из формы QRS"),
        ("r-height", "r_height", float, "Высота R; без аргумента — из формы QRS"),
        ("s-depth", "s_depth", float, "Глубина S; без аргумента — из формы QRS"),
        ("t-height", "t_height", float, "Высота T"),
        ("p-lean", "p_lean", float, "Наклон P"),
        ("t-lean", "t_lean", float, "Наклон T"),
        (
            "s-to-qrs-duration-ratio",
            "s_to_qrs_duration_ratio",
            float,
            "Отношение длительности S к QRS; без аргумента — из формы QRS",
        ),
        (
            "noise-amplitude",
            "noise_amplitude",
            float,
            "Мелкий шум, шаг 0.001; 0 0 отключает",
        ),
        (
            "baseline-wander-amplitude",
            "baseline_wander_amplitude",
            float,
            "Амплитуда дрейфа в долях std чистого сигнала; 0 0 отключает",
        ),
        (
            "baseline-wander-frequency",
            "baseline_wander_frequency",
            float,
            "Частота дрейфа, Гц",
        ),
        (
            "baseline-sine-amplitude",
            "baseline_sine_amplitude",
            float,
            "Амплитуда синусоиды в долях std чистого сигнала; 0 0 отключает",
        ),
        (
            "baseline-sine-frequency",
            "baseline_sine_frequency",
            float,
            "Частота синусоиды, Гц",
        ),
    )
    for flag, dest, value_type, help_text in range_arguments:
        parser.add_argument(
            f"--{flag}-range",
            dest=dest,
            type=value_type,
            nargs=2,
            default=getattr(defaults, dest),
            metavar=("MIN", "MAX"),
            help=help_text,
        )
    qt = parser.add_mutually_exclusive_group()
    qt.add_argument(
        "--qtc-range",
        dest="qtc_ms",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help=(
            "QTc Fridericia, мс; без QT/QTc аргументов "
            "используются диапазоны из конфига"
        ),
    )
    qt.add_argument(
        "--qt-range",
        dest="qt_ms",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="QT напрямую, мс; вместо выбора QTc",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=tuple(MORPHOLOGY_PRESETS),
        default=defaults.shapes,
        help="Формы QRS; одна форма фиксирует выбор",
    )
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=defaults.random_start,
        help="Случайное начало внутри цикла",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=defaults.duration_s,
        help="Длительность записи, с",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=defaults.sample_rate,
        help="Частота дискретизации, Гц (100–1000)",
    )
    parser.add_argument(
        "--variation-pct",
        type=float,
        default=defaults.variation_pct,
        help="Разброс длительностей и параметров формы каждого цикла, ±%%; 0 отключает",
    )
    parser.add_argument(
        "--amplitude-variation-pct",
        type=float,
        default=defaults.amplitude_variation_pct,
        help="Разброс амплитуд P, Q, R, S, T каждого цикла, ±%% (0–100); 0 отключает",
    )
    parser.add_argument(
        "--rr-std-ms",
        type=float,
        default=defaults.rr_std_ms,
        help="Стандартное отклонение распределения RR, мс; 0 — постоянный RR",
    )
    parser.add_argument(
        "--no-p",
        type=float,
        default=defaults.no_p,
        metavar="PERCENT",
        help="Вероятность отсутствия P на всей записи, %% (0–100)",
    )
    parser.add_argument(
        "--negative-p",
        type=float,
        default=defaults.negative_p,
        metavar="PERCENT",
        help="Вероятность инверсии P на всей записи, если P присутствует, %% (0–100)",
    )
    parser.add_argument(
        "--negative-qrs",
        type=float,
        default=defaults.negative_qrs,
        metavar="PERCENT",
        help="Вероятность инверсии всего QRS на всей записи, %% (0–100)",
    )
    parser.add_argument(
        "--negative-t",
        type=float,
        default=defaults.negative_t,
        metavar="PERCENT",
        help="Вероятность инверсии T на всей записи, %% (0–100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/random_ecg"),
        help="Папка набора; новые записи и строки meta.csv добавляются к существующим",
    )
    parser.add_argument("--count", type=int, default=1, help="Количество новых записей")
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help=(
            "Общий seed генератора сидов записей; "
            "если не задан в конфиге, выбирается случайно"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Количество параллельных процессов; 1 — последовательная генерация",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Сохранять signal.png для каждой записи",
    )
    args = parser.parse_args(argv)
    overrides = {
        dest: tuple(getattr(args, dest))
        for _, dest, _, _ in range_arguments
        if getattr(args, dest) is not None
    }
    if args.qt_ms is not None:
        qt_bounds, qtc_bounds = tuple(args.qt_ms), None
    elif args.qtc_ms is not None:
        qt_bounds, qtc_bounds = None, tuple(args.qtc_ms)
    else:
        qt_bounds = defaults.qt_ms
        qtc_bounds = defaults.qtc_ms if qt_bounds is None else None
    args.config = EcgConfig(
        **overrides,
        seed=args.seed,
        duration_s=args.duration,
        sample_rate=args.sample_rate,
        random_start=args.random_start,
        variation_pct=args.variation_pct,
        amplitude_variation_pct=args.amplitude_variation_pct,
        rr_std_ms=args.rr_std_ms,
        qt_ms=qt_bounds,
        qtc_ms=qtc_bounds,
        shapes=tuple(args.shapes),
        no_p=args.no_p,
        negative_p=args.negative_p,
        negative_qrs=args.negative_qrs,
        negative_t=args.negative_t,
    )
    try:
        args.config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        meta_path = generate_dataset(
            args.config,
            args.output_dir,
            count=args.count,
            workers=args.workers,
            save_images=args.save_images,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Общая таблица: {meta_path}")


if __name__ == "__main__":
    main()
