from pathlib import Path
from typing import Optional
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import confusion_matrix, roc_curve, auc

from .base import Callback
from utils.config import ExperimentConfig


class TensorBoardLogger(Callback):
    def __init__(self, config: ExperimentConfig):
        self.config = config

        self.writer: Optional[SummaryWriter] = None
        self.run_dir: Optional[Path] = None

    def on_train_start(self, trainer):
        self.run_dir = Path(HydraConfig.get().runtime.output_dir)

        tb_dir = self.run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(tb_dir))

        # log full config
        config_json = json.dumps(self.config.model_dump(), indent=2, default=str)

        self.writer.add_text("config/full_config", config_json)

    def on_epoch_end(self, trainer, metrics: dict):
        if self.writer is None:
            return

        epoch = trainer.current_epoch

        # scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        # learning rate
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            self.writer.add_scalar(f"lr/group_{i}", param_group["lr"], epoch)

        # weights histograms
        if self.config.tensorboard.log_weights:
            for name, param in trainer.model.named_parameters():
                self.writer.add_histogram(f"weights/{name}", param, epoch)

        # confusion matrix
        if "y_true" in metrics and "y_pred" in metrics:
            self.log_confusion_matrix(metrics["y_true"], metrics["y_pred"], epoch)

        # log curve
        if "y_true" in metrics and "y_scores" in metrics:
            self.log_roc_curve(metrics["y_true"], metrics["y_scores"], epoch)

        self.writer.flush()

    def log_confusion_matrix(self, y_true, y_pred, epoch):
        if self.writer is None:
            return

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        plt.close(fig)

        self.writer.add_figure("confusion_matrix", fig, epoch)

    def log_roc_curve(self, y_true, y_scores, epoch):
        if self.writer is None:
            return

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_title(f"ROC Curve (AUC = {roc_auc:.4f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        plt.close(fig)

        self.writer.add_figure("roc_curve", fig, epoch)

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()
