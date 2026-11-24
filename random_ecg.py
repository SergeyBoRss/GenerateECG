import argparse
import secrets
from pathlib import Path

import numpy as np

from ecg.config import SamplingPreset, StitchedEcgConfig
from ecg.dataset import RandomDatasetWriter
from ecg.generation import generate_record
from ecg.models import SampledParameters
from ecg.morphology import MORPHOLOGY_PRESETS, Morphology
from ecg.output import plot_random_result, print_random_summary, show_plot
from ecg.parameters import sample_record_parameters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = StitchedEcgConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for flag, dest, help_text in (
        ("pq", "pq_ms", "PQ, мс"),
        ("qtc", "qtc_ms", "QTc Fridericia, мс"),
        ("qrs", "qrs_ms", "QRS, мс"),
        ("hr", "hr_bpm", "HR, уд/мин"),
        ("seed", "seed", "Seed генерации ЭКГ"),
    ):
        parser.add_argument(f"--{flag}-range", dest=dest, type=int, nargs=2,
                            metavar=("MIN", "MAX"), help=f"{help_text}; обе границы включены")
    parser.add_argument("--shapes", nargs="+", choices=MORPHOLOGY_PRESETS,
                        help="Равновероятные формы; по умолчанию все семь")
    parser.add_argument("--sampler-seed", type=int,
                        help="Seed выбора параметров; при пропуске выбирается случайно и печатается")
    parser.add_argument("--duration", type=float, default=10.0, help="Длительность записи, с")
    parser.add_argument("--sample-rate", type=int, default=1000, help="Частота дискретизации, Гц (100–1000)")
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
                        help="Разброс параметров каждого цикла относительно базовых, ±%; 0 отключает")
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True,
                        help="Начинать со случайной позиции внутри цикла")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/random_ecg"),
                        help="Папка набора: meta.csv и отдельный каталог на каждую запись")
    parser.add_argument("--count", type=int, default=1, help="Количество записей")
    parser.add_argument("--save-images", action="store_true", help="Сохранять signal.png для каждой записи")
    parser.add_argument("--no-show", action="store_true",
                        help="Не открывать график; при count > 1 окно не открывается автоматически")
    args = parser.parse_args(argv)
    overrides = {
        name: tuple(getattr(args, name))
        for name in ("pq_ms", "qtc_ms", "qrs_ms", "hr_bpm", "seed", "shapes")
        if getattr(args, name) is not None
    }
    args.sampling_ranges = SamplingPreset(**overrides)
    try:
        args.sampling_ranges.validate(variation_pct=args.variation_pct)
        if args.sampler_seed is not None and args.sampler_seed < 0:
            raise ValueError("sampler-seed должен быть неотрицательным")
        if args.count < 1:
            raise ValueError("count должен быть положительным")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(sampler_seed)
    show = args.count == 1 and not args.no_show
    try:
        writer = RandomDatasetWriter(args.output_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for record_index in range(1, args.count + 1):
        try:
            parameters = sample_record_parameters(
                args.sampling_ranges, rng, variation_pct=args.variation_pct,
            )
            config, result = generate_record(
                parameters, duration_s=args.duration, sample_rate=args.sample_rate,
                noise_amplitude=args.noise_amplitude, random_start=args.random_start,
                variation_pct=args.variation_pct,
                baseline_wander_amplitude=args.baseline_wander_amplitude,
                baseline_wander_frequency=args.baseline_wander_frequency,
                baseline_sine_amplitude=args.baseline_sine_amplitude,
                baseline_sine_frequency=args.baseline_sine_frequency,
            )
            record_dir = writer.save(
                config, result, parameters, args.sampling_ranges, sampler_seed,
                record_index=record_index, save_image=args.save_images, show_image=show,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print_random_summary(parameters, config, result, sampler_seed)
        print(f"Сохранено: {record_dir}")
        if show:
            if not args.save_images:
                plot_random_result(config, result, parameters)
            show_plot()
    print(f"Общая таблица: {writer.meta_path}")


if __name__ == "__main__":
    main()
