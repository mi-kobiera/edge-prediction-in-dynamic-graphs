import logging
from pathlib import Path
from typing import Annotated, Literal
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

# ==========================================
#  BASE CONFIGURATION
# ==========================================


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ==========================================
#  DATA CONFIGURATION
# ==========================================


class DataSplit(BaseConfig):
    val_ratio: float = Field(
        ..., gt=0.0, description="Validation set ratio (e.g., 0.15)"
    )
    test_ratio: float = Field(..., gt=0.0, description="Test set ratio (e.g., 0.15)")


class NegativeSamplingConfig(BaseConfig):
    train_ratio: int = Field(
        1,
        ge=1,
        description="Number of negative samples per positive edge during training",
    )
    eval_ratio: int = Field(
        100,
        ge=1,
        description="Number of negative samples per positive edge during evaluation",
    )


class DataConfig(BaseConfig):
    dataset_name: Literal["wikipedia", "reddit"]
    path: Path = Field(..., description="Path to the dataset directory")
    batch_size: int = Field(..., gt=0, description="Number of edges per batch")
    num_neighbors: int = Field(
        ...,
        ge=0,
        description="Number of temporal neighbors to sample for the memory/attention module",
    )
    data_split: DataSplit
    negative_sampling: NegativeSamplingConfig


# ==========================================
#  MODEL CONFIGURATION
# ==========================================


class BaseArchitectureConfig(BaseConfig):
    name: str
    target: str = Field(
        ..., alias="_target_", description="Python class path for Hydra"
    )
    dropout: float = Field(0.1, ge=0.0, lt=1.0, description="Dropout probability")


class TGNConfig(BaseArchitectureConfig):
    name: Literal["TGN"]
    memory_dim: int = Field(..., gt=0)
    embedding_dim: int = Field(..., gt=0)
    time_dim: int = Field(..., gt=0)


class GCNConfig(BaseArchitectureConfig):
    name: Literal["GCN"]
    hidden_dim: int = Field(..., gt=0, description="Hidden layer dimension")
    num_layers: int = Field(2, gt=0, description="Number of message passing layers")


# Based on the value of the 'name' field in the YAML file.
type AnyModelConfig = Annotated[TGNConfig | GCNConfig, Field(discriminator="name")]


# ==========================================
#  TRAINING & EVALUATION CONFIGURATION
# ==========================================


class TrainingConfig(BaseConfig):
    epochs: int = Field(..., gt=0, description="Maximum number of training epochs")
    lr: float = Field(..., gt=0.0, description="Learning rate for the optimizer")
    weight_decay: float = Field(
        0.0, ge=0.0, description="L2 regularization penalty (weight decay)"
    )
    patience: int = Field(
        ...,
        ge=0,
        description="Early stopping patience (number of epochs without improvement)",
    )


class EvalConfig(BaseConfig):
    eval_every: int = Field(1, gt=0, description="Run validation every X epochs")
    inductive_setting: bool = Field(
        True,
        description="Evaluate on new, previously unseen nodes (inductive link prediction)",
    )


# ==========================================
#  LOGGING & TRACKING CONFIG
# ==========================================


class TensorBoardConfig(BaseConfig):
    enabled: bool = True
    log_model: bool = False
    log_weights: bool = True


# ==========================================
#  ROOT CONFIGURATION
# ==========================================


class ExperimentConfig(BaseConfig):
    project_name: str = Field(..., description="Name of the project")
    experiment_name: str = Field(..., description="Unique name for the specific run")

    seed: int | None = Field(
        default=None, description="Random seed for reproducibility across runs"
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        "cuda", description="Hardware accelerator to use"
    )

    data: DataConfig
    model: AnyModelConfig
    training: TrainingConfig
    eval: EvalConfig

    tensorboard: TensorBoardConfig


def load_config(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    exp_cfg = ExperimentConfig(**dict_cfg)

    if exp_cfg.seed is not None:
        exp_cfg.experiment_name += f"_seed{cfg.seed}"

    logger.info(
        "Konfiguracja załadowana i zwalidowana pomyślnie przez Hydra & Pydantic."
    )

    return exp_cfg
