import logging
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ----- Data -----


class DatasetSplitConfig(BaseModel):
    val_ratio: float = Field(..., ge=0.0)
    test_ratio: float = Field(..., gt=0.0)


class DatasetConfig(BaseModel):
    path: Path = Field(..., description="Path to the processed dataset file")
    batch_size: int = Field(..., gt=0.0)
    split: DatasetSplitConfig


# ----- Model -----


class ModelConfig(BaseModel):
    dropout: float = Field(0.0, ge=0.0, lt=1.0)


class TGNConfig(ModelConfig):
    num_neighbors: int = Field(10, description="Number of neigbors for neigbor_loader")
    memory_dim: int = Field(..., gt=0)
    embedding_dim: int = Field(..., gt=0)
    time_dim: int = Field(..., gt=0)


# ----- Node labeling -----


class NodeLabelingConfig(BaseModel):
    label_dim: int = Field(1, ge=0)


# ----- Negative sampling -----


class NegativeSamplingConfig(BaseModel):
    pass


# ----- Training -----


class EarlyStoppingConfig(BaseModel):
    metric_name: str = Field("val/ap", description="Early stopping metric name")
    patience: int = Field(10, gt=0)
    mode: Literal["min", "max"] = Field("max")
    delta: float = Field(0.0, ge=0.0)
    verbose: bool = Field(True)
    path: str = Field("best_model.pt")


class TrainingConfig(BaseModel):
    epochs: int = Field(30, gt=0)
    lr: float = Field(0.001, gt=0)
    early_stopping: EarlyStoppingConfig
    weight_decay: float


class EvalConfig(BaseModel):
    eval_every: int = Field(1, ge=1, description="Run validation every X epochs")


# ----- Logging and tracking -----


class TensorBoardConfig(BaseModel):
    enabled: bool = True
    log_model: bool = False
    log_weights: bool = True


# ----- Root -----


class ExperimentConfig(BaseModel):
    experiment_name: str | None = None
    seed: int | None = Field(default=None)
    device: Literal["cpu", "cuda", "mps"] = Field(
        "cuda"
    )  # TODO dla mps: RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
    dataset: DatasetConfig
    model: TGNConfig
    negative_sampling: NegativeSamplingConfig
    labeling: NodeLabelingConfig | None = None
    training: TrainingConfig
    eval: EvalConfig
    tensorboard: TensorBoardConfig


def load_config(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    exp_cfg = ExperimentConfig(**dict_cfg)

    logger.info("Configuration loaded succesfully.")

    return exp_cfg
