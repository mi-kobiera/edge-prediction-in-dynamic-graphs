import numpy as np
import torch
import logging
from pathlib import Path

from .base import Callback
from utils.config import ExperimentConfig

try:
    from hydra.core.hydra_config import HydraConfig

    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

logger = logging.getLogger(__name__)


class EarlyStopping(Callback):
    def __init__(self, cfg: ExperimentConfig):
        self.monitor = cfg.training.early_stopping.metric_name
        self.patience = cfg.training.early_stopping.patience
        self.mode = cfg.training.early_stopping.mode
        self.delta = cfg.training.early_stopping.delta
        self.verbose = cfg.training.early_stopping.verbose
        self.path = Path(cfg.training.early_stopping.path)

        if HAS_HYDRA and HydraConfig.initialized() and not self.path.is_absolute():
            hydra_output_dir = HydraConfig.get().runtime.output_dir
            self.path = Path(hydra_output_dir) / self.path

        if self.verbose:
            logger.info(f"Checkpoint path: {self.path.absolute()}")

        self.counter = 0
        self.best_score = None

        if self.mode == "min":
            self.monitor_op = np.less
            self.min_delta = -self.delta
        elif self.mode == "max":
            self.monitor_op = np.greater
            self.min_delta = self.delta
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def on_epoch_end(self, trainer, metrics: dict):
        if self.monitor not in metrics:
            return

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(trainer, current_score)
            return

        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self._save_checkpoint(trainer, current_score)
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Patience counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                trainer.should_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered.")

    def on_train_end(self, trainer):
        checkpoint = torch.load(self.path, weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        if self.verbose:
            logger.info("Best model loaded.")

    def _save_checkpoint(self, trainer, score):
        checkpoint = {
            "epoch": trainer.current_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": trainer.cfg,
            "metric": score,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, self.path)

        if self.verbose:
            logger.info(f"Saved best model.")
