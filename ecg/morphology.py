from dataclasses import dataclass


@dataclass(frozen=True)
class Morphology:
    q_depth: float
    r_height: float
    s_depth: float
    s_to_qrs_duration_ratio: float = 0.15


MORPHOLOGY_PRESETS = {
    "qRs": Morphology(0.30, 3.00, 0.60),
    "qR": Morphology(0.30, 3.00, 0.00),
    "Rs": Morphology(0.00, 3.00, 0.60),
    "R": Morphology(0.00, 3.00, 0.00),
    "RS": Morphology(0.00, 3.00, 2.00, 0.35),
    "QR": Morphology(1.50, 3.00, 0.00),
    "rS": Morphology(0.00, 0.45, 3.00, 0.35),
}
