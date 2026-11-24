import argparse
import secrets
from pathlib import Path

import numpy as np

from ecg.config import SamplingPreset, StitchedEcgConfig
from ecg.dataset import RandomDatasetWriter
from ecg.generation import generate_record
from ecg.output import print_random_summary
from ecg.parameters import sample_record_parameters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = StitchedEcgConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for flag, dest, help_text in (
        ("p", "p_ms", "Длительность P, мс"),
        ("t", "t_ms", "Длительность T, мс"),
        ("pq", "pq_ms", "PQ, мс"),
        ("qtc", "qtc_ms", "QTc Fridericia, мс"),
        ("qrs", "qrs_ms", "QRS, мс"),
        ("hr", "hr_bpm", "HR, уд/мин"),
    ):
        parser.add_argument(f"--{flag}-range", dest=dest, type=int, nargs=2,
                            metavar=("MIN", "MAX"), help=f"{help_text}; обе границы включены")
    parser.add_argument("--duration", type=float, default=10.0, help="Длительность записи, с")
    parser.add_argument("--sample-rate", type=int, default=1000, help="Частота дискретизации, Гц (100–1000)")
    parser.add_argument("--variation-pct", type=float, default=defaults.variation_pct,
                        help="Разброс длительностей и параметров формы каждого цикла, ±%%; 0 отключает")
    parser.add_argument("--amplitude-variation-pct", type=float, default=defaults.amplitude_variation_pct,
                        help="Разброс амплитуд P, Q, R, S, T каждого цикла, ±%% (0–100); 0 отключает")
    parser.add_argument("--no-p", type=float, default=0.0, metavar="PERCENT",
                        help="Вероятность отсутствия P на всей записи, %% (0–100)")
    parser.add_argument("--negative-p", type=float, default=0.0, metavar="PERCENT",
                        help="Вероятность инверсии P на всей записи, если P присутствует, %% (0–100)")
    parser.add_argument("--negative-t", type=float, default=0.0, metavar="PERCENT",
                        help="Вероятность инверсии T на всей записи, %% (0–100)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/random_ecg"),
                        help="Папка набора: meta.csv и отдельный каталог на каждую запись")
    parser.add_argument("--count", type=int, default=1, help="Количество записей")
    parser.add_argument("--save-images", action="store_true", help="Сохранять signal.png для каждой записи")
    args = parser.parse_args(argv)
    overrides = {
        name: tuple(getattr(args, name))
        for name in ("p_ms", "t_ms", "pq_ms", "qtc_ms", "qrs_ms", "hr_bpm")
        if getattr(args, name) is not None
    }
    args.sampling_ranges = SamplingPreset(
        **overrides, no_p=args.no_p,
        negative_p=args.negative_p, negative_t=args.negative_t,
    )
    try:
        args.sampling_ranges.validate(variation_pct=args.variation_pct)
        if args.count < 1:
            raise ValueError("count должен быть положительным")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sampler_seed = secrets.randbits(32)
    rng = np.random.default_rng(sampler_seed)
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
                noise_amplitude=int(rng.integers(20, 36)) / 1000,  # 0.010–0.035, шаг 0.001
                random_start=True,
                variation_pct=args.variation_pct,
                amplitude_variation_pct=args.amplitude_variation_pct,
                baseline_wander_amplitude=float(rng.uniform(0.0, 1.0)),
                baseline_wander_frequency=float(rng.uniform(0.25, 0.35)),
                baseline_sine_amplitude=float(rng.uniform(0.0, 2.0)),
                baseline_sine_frequency=float(rng.uniform(0.15, 0.25)),
            )
            record_dir = writer.save(
                config, result, parameters, args.sampling_ranges, sampler_seed,
                record_index=record_index, save_image=args.save_images, show_image=False,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print_random_summary(parameters, config, result, sampler_seed)
        print(f"Сохранено: {record_dir}")
    print(f"Общая таблица: {writer.meta_path}")


if __name__ == "__main__":
    main()
