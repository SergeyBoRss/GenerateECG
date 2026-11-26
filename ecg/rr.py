from math import erf, expm1, isfinite, sqrt
from typing import overload

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfinv, gammainc


def _relative_moments(limit: float) -> tuple[float, float]:
    squared = limit * limit
    mass = erf(limit / sqrt(2))
    mean_ratio = -expm1(-squared / 2) / mass
    second_moment = gammainc(1.5, squared / 2) / mass
    return mean_ratio, second_moment


class RrDistribution:
    def __init__(self, mean: float, std: float, minimum: float):
        self.mean = float(mean)
        self.std = float(std)
        self.minimum = float(minimum)
        if self.mean < self.minimum or (self.std > 0 and self.mean == self.minimum):
            raise ValueError(
                "Для разброса RR среднее должно быть больше нижней границы",
            )
        if self.std == 0:
            return

        distance = self.mean - self.minimum
        relative_distance = distance / self.std

        def relative_std(scale_ratio: float) -> float:
            mean_ratio, second_moment = _relative_moments(
                relative_distance / scale_ratio,
            )
            return scale_ratio * sqrt((mean_ratio + second_moment) / (1 + mean_ratio))

        upper = 2.0
        while relative_std(upper) < 1:
            upper *= 2
            if not isfinite(upper):
                raise ValueError(
                    "Заданный разброс RR не представим с этой нижней границей",
                )

        self._scale = self.std * brentq(
            lambda value: relative_std(value) - 1, 1.0, upper,
        )
        if not isfinite(self._scale):
            raise ValueError("Масштаб распределения RR выходит за диапазон float")
        limit = distance / self._scale
        mean_ratio, _ = _relative_moments(limit)
        self._positive_probability = mean_ratio / (1 + mean_ratio)
        self._left_mass = erf(limit / sqrt(2))

    @overload
    def sample(
        self,
        rng: np.random.Generator,
        size: None = None,
    ) -> float: ...

    @overload
    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray: ...

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        if self.std == 0:
            return self.mean if size is None else np.full(size, self.mean)

        positive = rng.random(size) < self._positive_probability
        # Обратная CDF выбирает укорочение сразу из допустимого интервала.
        shortening = sqrt(2) * erfinv(rng.random(size) * self._left_mass) * self._scale
        lengthening = np.abs(rng.normal(0.0, self._scale, size))
        values = np.where(positive, self.mean + lengthening, self.mean - shortening)
        values = np.maximum(values, self.minimum)  # Погрешность округления у границы.
        return float(values) if size is None else values
