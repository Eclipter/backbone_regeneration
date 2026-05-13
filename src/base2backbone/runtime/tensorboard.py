"""TensorBoard scalar loading helpers for analysis scripts."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def load_event_accumulator(path: str):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={'scalars': 0, 'histograms': 0, 'images': 0},
    )
    ea.Reload()
    return ea


def scalars_to_dataframe(ea: Any, tag: str) -> pd.DataFrame:
    rows = [(scalar.step, scalar.value) for scalar in ea.Scalars(tag)]
    return pd.DataFrame(rows, columns=['epoch', 'value'])


def collect_scalar_history(
    event_files: Sequence[str],
    metric_tags: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    tracked_tags = set(metric_tags)
    dfs: list[pd.DataFrame] = []
    for event_file in event_files:
        ea = load_event_accumulator(event_file)
        for tag in ea.Tags()['scalars']:  # type: ignore[index]
            if tag not in tracked_tags:
                continue
            df = scalars_to_dataframe(ea, tag)
            mode, metric = metric_tags[tag]
            df['mode'] = mode
            df['metric'] = metric
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['epoch', 'mode', 'metric', 'value'])
    return pd.concat(dfs, ignore_index=True)[['epoch', 'mode', 'metric', 'value']].reset_index(
        drop=True
    )
