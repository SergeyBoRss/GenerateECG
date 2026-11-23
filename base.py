from ecg.annotations import extract_boundaries
from ecg.config import EcgConfig
from ecg.generation import generate_base_ecg as generate_ecg
from ecg.output import plot_result, print_base_summary as print_summary, show_plot


def main() -> None:
    config = EcgConfig()
    result = generate_ecg(config)
    boundaries = extract_boundaries(result.segments)
    print_summary(config, result, boundaries)
    plot_result(result.signal, config.sample_rate, boundaries)
    show_plot()


if __name__ == "__main__":
    main()
