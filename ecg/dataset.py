import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import SamplingPreset, StitchedEcgConfig
from .models import SampledParameters, StitchedEcgResult
from .output import plot_random_result


META_FIELDS = (
    "record_id", "signal_path", "json_path", "image_path",
    "sample_rate", "sample_count", "duration_s",
    "pq_ms", "qtc_ms", "qt_ms", "qrs_ms", "hr_bpm", "shape",
    "seed", "sampler_seed", "record_index", "noise_amplitude",
    "random_start", "start_offset_samples",
)


def _record_number(record_id: str) -> int:
    suffix = record_id.removeprefix("record_")
    return int(suffix) if record_id.startswith("record_") and suffix.isdecimal() else 0


class RandomDatasetWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_dir / "meta.csv"
        last_number = max(
            (_record_number(path.name) for path in self.output_dir.glob("record_*")),
            default=0,
        )
        self.next_number = last_number + 1

    def save(
        self,
        config: StitchedEcgConfig,
        result: StitchedEcgResult,
        parameters: SampledParameters,
        ranges: SamplingPreset,
        sampler_seed: int,
        *,
        record_index: int = 1,
        save_image: bool = False,
        show_image: bool = False,
    ) -> Path:
        while True:
            record_id = f"record_{self.next_number:06d}"
            self.next_number += 1
            record_dir = self.output_dir / record_id
            try:
                record_dir.mkdir()
                break
            except FileExistsError:
                continue

        metadata = {
            "config": asdict(config),
            "duration_s": len(result.signal) / config.sample_rate,
            "sampling": {
                "ranges": asdict(ranges),
                "sampler_seed": sampler_seed,
                "record_index": record_index,
                "selected": asdict(parameters),
            },
        }
        np.save(record_dir / "signal.npy", result.signal, allow_pickle=False)
        (record_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if save_image:
            plot_random_result(config, result, parameters,
                               save_path=record_dir / "signal.png", show=show_image)

        row = {
            "record_id": record_id,
            "signal_path": f"{record_id}/signal.npy",
            "json_path": f"{record_id}/metadata.json",
            "image_path": f"{record_id}/signal.png" if save_image else "",
            "sample_rate": config.sample_rate,
            "sample_count": len(result.signal),
            "duration_s": len(result.signal) / config.sample_rate,
            **asdict(parameters),
            "qt_ms": parameters.qt_ms,
            "sampler_seed": sampler_seed,
            "record_index": record_index,
            "noise_amplitude": config.noise_amplitude,
            "random_start": config.random_start,
            "start_offset_samples": result.start_offset_samples,
        }

        write_header = not self.meta_path.exists() or self.meta_path.stat().st_size == 0
        needs_newline = False
        if not write_header:
            with self.meta_path.open("rb") as stream:
                stream.seek(-1, 2)
                needs_newline = stream.read(1) not in (b"\n", b"\r")
        with self.meta_path.open("a", encoding="utf-8", newline="") as stream:
            if needs_newline:
                stream.write("\n")
            writer = csv.DictWriter(stream, fieldnames=META_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return record_dir
