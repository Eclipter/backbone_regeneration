"""Training LR logging: slash-prefixed LR tags (lr/<Optimizer>) and plateau num_bad_epochs."""

from __future__ import annotations

import itertools
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import override


def _with_slash_lr_prefix(names: list[list[str]]) -> tuple[list[list[str]], bool]:
    """Map default Lightning tags `lr-<Class>` to `optimizer/<Class>`. Returns (names, any_changed)."""
    out: list[list[str]] = []
    changed = False
    for group in names:
        row: list[str] = []
        for n in group:
            if n.startswith('lr-'):
                row.append('optimizer/' + n[3:])
                changed = True
            else:
                row.append(n)
        out.append(row)
    return out, changed


class TrainingLrMonitor(LearningRateMonitor):
    """LearningRateMonitor but uses optimizer/<Optimizer> when Lightning would use lr-<Optimizer>.

    Logs ReduceLROnPlateau num_bad_epochs as optimizer/num_bad_epochs when slash names apply, else
    optimizer-num_bad_epochs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_bad_epochs_tag = 'optimizer/num_bad_epochs'

    @override
    def on_train_start(self, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        self._num_bad_epochs_tag = 'optimizer/num_bad_epochs'
        super().on_train_start(trainer, *args, **kwargs)

    @override
    def _find_names_from_schedulers(self, lr_scheduler_configs):  # type: ignore[no-untyped-def]
        names, seen_opt, seen_types = super()._find_names_from_schedulers(lr_scheduler_configs)
        flat_len = len(list(itertools.chain.from_iterable(names)))
        names, ok = _with_slash_lr_prefix(names)
        if flat_len > 0 and not ok:
            self._num_bad_epochs_tag = 'optimizer-num_bad_epochs'
        return names, seen_opt, seen_types

    @override
    def _find_names_from_optimizers(self, optimizers, seen_optimizers, seen_optimizer_types):  # type: ignore[no-untyped-def]
        names, opts = super()._find_names_from_optimizers(
            optimizers, seen_optimizers, seen_optimizer_types,
        )
        flat_len = len(list(itertools.chain.from_iterable(names)))
        names, ok = _with_slash_lr_prefix(names)
        if flat_len > 0 and not ok:
            self._num_bad_epochs_tag = 'optimizer-num_bad_epochs'
        return names, opts

    @staticmethod
    def _plateau_num_bad_epochs(trainer: pl.Trainer) -> float | None:
        for cfg in trainer.lr_scheduler_configs:
            if isinstance(cfg.scheduler, ReduceLROnPlateau):
                return float(cfg.scheduler.num_bad_epochs)
        return None

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        super().on_train_epoch_start(trainer, *args, **kwargs)
        pl_module = trainer.lightning_module
        if pl_module.hparams.get('lr_scheduler') is None:
            return
        if trainer.current_epoch == 0:
            return
        val = self._plateau_num_bad_epochs(trainer)
        if val is None:
            return
        tag = self._num_bad_epochs_tag
        step = trainer.global_step  # align with LearningRateMonitor's own step axis
        tval = torch.tensor(val, device=trainer.strategy.root_device)
        for logger in trainer.loggers:
            logger.log_metrics({tag: val}, step=step)
        trainer.callback_metrics.update({tag: tval})
