import numpy as np
import neurokit2 as nk


def add_ecg_noise(
    signal,
    sampling_rate=1000,
    noise_amplitude=0.05,
    noise_frequency=(5, 10, 100),
    noise_shape="laplace",
    random_state=None,
    *,
    baseline_wander_amplitude=0.0,
    baseline_wander_frequency=0.3,
    baseline_sine_amplitude=0.0,
    baseline_sine_frequency=0.2,
):
    signal = np.asarray(signal, dtype=float)
    for name, amplitude, frequency in (
        ("baseline_wander", baseline_wander_amplitude, baseline_wander_frequency),
        ("baseline_sine", baseline_sine_amplitude, baseline_sine_frequency),
    ):
        if amplitude > 0 and (not np.isfinite(frequency) or not 0 < frequency <= sampling_rate / 10):
            raise ValueError(f"{name}_frequency должна быть > 0 и <= {sampling_rate / 10} Гц")

    noisy_signal = signal.copy()
    if noise_amplitude > 0:
        noisy_signal = nk.signal_distort(
            signal,
            sampling_rate=sampling_rate,
            noise_amplitude=noise_amplitude,
            noise_frequency=list(noise_frequency),
            noise_shape=noise_shape,
            random_state=random_state,
            silent=False,
        )

    signal_sd = np.std(signal, ddof=1) if signal.size > 1 else 0.0
    if baseline_wander_amplitude > 0:
        baseline = nk.signal_distort(
            np.zeros(len(signal)),
            sampling_rate=sampling_rate,
            noise_amplitude=baseline_wander_amplitude * signal_sd,
            noise_frequency=baseline_wander_frequency,
            noise_shape=noise_shape,
            random_state=random_state,
        )
        noisy_signal += baseline

    if baseline_sine_amplitude > 0:
        baseline = nk.signal_simulate(
            duration=len(signal) / sampling_rate,
            sampling_rate=sampling_rate,
            frequency=baseline_sine_frequency,
            amplitude=baseline_sine_amplitude * signal_sd,
        )
        noisy_signal += baseline

    return noisy_signal
