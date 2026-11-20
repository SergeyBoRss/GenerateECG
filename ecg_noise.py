
import numpy as np
import neurokit2 as nk


def add_ecg_noise(
    signal,
    sampling_rate=1000,
    noise_amplitude=0.05,
    noise_frequency=(5, 10, 100),
    noise_shape="laplace",
    random_state=None,
):
    signal = np.asarray(signal, dtype=float)

    if noise_amplitude <= 0:
        return signal.copy()

    return nk.signal_distort(
        signal,
        sampling_rate=sampling_rate,
        noise_amplitude=noise_amplitude,
        noise_frequency=list(noise_frequency),
        noise_shape=noise_shape,
        random_state=random_state,
        silent=False,
    )
