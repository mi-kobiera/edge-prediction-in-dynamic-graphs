from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
from hydra.core.hydra_config import HydraConfig

from .base import Callback
from utils.config import ExperimentConfig


class TensorBoardLogger(Callback):
    def __init__(self, config: ExperimentConfig):
        self.config = config

        self.writer: SummaryWriter | None = None
        self.run_dir: Path | None = None

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

        self.writer.flush()

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()
