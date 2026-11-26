import secrets
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .config import EcgConfig
from .dataset import RandomDatasetWriter, save_record
from .generation import generate_record
from .output import print_random_summary
from .parameters import sample_record_parameters


def _generate_and_save_record(
    config: EcgConfig,
    output_dir: Path,
    record_index: int,
    save_image: bool,
    sampler_seed: int,
) -> tuple[dict[str, object], str]:
    rng = np.random.default_rng(config.seed)
    parameters = sample_record_parameters(config, rng)
    result = generate_record(parameters, config)
    row = save_record(
        output_dir,
        config,
        result,
        parameters,
        sampler_seed=sampler_seed,
        record_index=record_index,
        save_image=save_image,
    )
    summary = StringIO()
    with redirect_stdout(summary):
        print_random_summary(parameters, config, result)
        print(f"Сохранено: {output_dir / str(row['record_id'])}")
    return row, summary.getvalue()


def generate_dataset(
    config: EcgConfig,
    output_dir: Path,
    *,
    count: int = 1,
    workers: int = 1,
    save_images: bool = False,
) -> Path:
    writer = RandomDatasetWriter(output_dir)
    sampler_seed = config.seed if config.seed is not None else secrets.randbits(32)
    seed_rng = np.random.default_rng(sampler_seed)
    tasks = (
        (
            replace(config, seed=int(seed_rng.integers(0, 2**32))),
            writer.output_dir,
            record_index,
            save_images,
            sampler_seed,
        )
        for record_index in range(1, count + 1)
    )
    workers = min(workers, count)
    if workers == 1:
        for task in tasks:
            row, summary = _generate_and_save_record(*task)
            writer.append(row)
            print(summary, end="")
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
        ) as executor:
            pending: deque[Future[tuple[dict[str, object], str]]] = deque()
            try:
                for task in islice(tasks, workers * 2):
                    pending.append(executor.submit(_generate_and_save_record, *task))
                while pending:
                    row, summary = pending.popleft().result()
                    writer.append(row)
                    print(summary, end="")
                    next_task = next(tasks, None)
                    if next_task is not None:
                        pending.append(
                            executor.submit(_generate_and_save_record, *next_task),
                        )
            finally:
                for future in pending:
                    future.cancel()
    return writer.meta_path
