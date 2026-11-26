import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .annotations import extract_annotations
from .config import EcgConfig
from .models import EcgResult, SampledParameters
from .output import plot_random_result


def save_record(
    output_dir: Path,
    config: EcgConfig,
    result: EcgResult,
    parameters: SampledParameters,
    *,
    sampler_seed: int,
    record_index: int = 1,
    save_image: bool = False,
    show_image: bool = False,
) -> dict[str, object]:
    record_id = f"record_{record_index:06d}"
    record_dir = Path(output_dir) / record_id

    row = {
        "record_id": record_id,
        "signal_path": f"{record_id}/signal.npy",
        "json_path": f"{record_id}/metadata.json",
        "image_path": f"{record_id}/signal.png" if save_image else "",
        "sample_rate": config.sample_rate,
        "sample_count": len(result.signal),
        "duration_s": len(result.signal) / config.sample_rate,
        **asdict(parameters),
        "sampler_seed": sampler_seed,
        "record_index": record_index,
        "random_start": config.random_start,
        "start_offset_samples": result.start_offset_samples,
        "variation_pct": config.variation_pct,
        "amplitude_variation_pct": config.amplitude_variation_pct,
        "rr_std_ms": config.rr_std_ms,
    }

    record_dir.mkdir()
    metadata = {
        "config": asdict(config),
        "duration_s": len(result.signal) / config.sample_rate,
        "sampling": {
            "sampler_seed": sampler_seed,
            "record_index": record_index,
            "selected": asdict(parameters),
        },
        "annotations": extract_annotations(result.segments),
    }
    np.save(record_dir / "signal.npy", result.signal, allow_pickle=False)
    (record_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if save_image:
        plot_random_result(
            config,
            result,
            parameters,
            save_path=record_dir / "signal.png",
            show=show_image,
        )
    return row


class RandomDatasetWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_dir / "meta.csv"

    def append(self, row: dict[str, object]) -> None:
        write_header = not self.meta_path.exists() or self.meta_path.stat().st_size == 0
        if not write_header:
            with self.meta_path.open(encoding="utf-8", newline="") as stream:
                header = stream.readline().rstrip("\r\n")
            if header != ",".join(row):
                raise ValueError(
                    f"В {self.meta_path} другой набор колонок. "
                    "Укажите новую --output-dir.",
                )

        needs_newline = False
        if not write_header:
            with self.meta_path.open("rb") as stream:
                stream.seek(-1, 2)
                needs_newline = stream.read(1) not in (b"\n", b"\r")
        with self.meta_path.open("a", encoding="utf-8", newline="") as stream:
            if needs_newline:
                stream.write("\n")
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
