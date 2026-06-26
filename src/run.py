import torch
import logging
import hydra
from omegaconf import DictConfig
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from training.trainer import Trainer
from training.callbacks.early_stopping import EarlyStopping
from training.callbacks.tensorboard import TensorBoardLogger
from utils.config import load_config
from data.loader import load_dataset, get_dataloader
from utils.random import set_random_seed


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    exp_cfg = load_config(cfg)
    set_random_seed(exp_cfg.seed)

    data_dir = getattr(exp_cfg.dataset, "path")
    network_name = getattr(exp_cfg.dataset, "name")

    logger.info(f"Loading dataset: {network_name} from {data_dir}...")

    if network_name == "tgbl-wiki":
        dataset = PyGLinkPropPredDataset(name=network_name, root="../datasets")
        data = dataset.get_TemporalData()
        if hasattr(dataset, "train_mask"):
            data.train_mask = dataset.train_mask
            data.val_mask = dataset.val_mask
            data.test_mask = dataset.test_mask
    else:
        data = load_dataset(
            data_dir=data_dir,
            network_name=network_name,
            val_ratio=exp_cfg.dataset.split.val_ratio
            if hasattr(exp_cfg.dataset, "split")
            else 0.15,
            test_ratio=exp_cfg.dataset.split.test_ratio
            if hasattr(exp_cfg.dataset, "split")
            else 0.15,
        )

    for key in data.keys():
        val = getattr(data, key)
        if hasattr(val, "dtype") and val.dtype == torch.float64:
            setattr(data, key, val.to(torch.float32))

    data = data.to(exp_cfg.device)

    train_data = data[data.train_mask]
    val_data = data[data.val_mask]
    test_data = data[data.test_mask]

    batch_size = exp_cfg.dataset.batch_size
    train_loader = get_dataloader(train_data, batch_size)
    val_loader = get_dataloader(val_data, batch_size)
    test_loader = get_dataloader(test_data, batch_size)

    labeling_cfg = cfg.get("labeling")
    labeling = (
        hydra.utils.instantiate(labeling_cfg) if labeling_cfg is not None else None
    )

    negative_sampler = hydra.utils.instantiate(cfg.negative_sampling, data)

    model = hydra.utils.instantiate(
        cfg.model,
        data=data,
        negative_sampler=negative_sampler,
        labeling=labeling,
        device=exp_cfg.device,
    )

    early_stopping = EarlyStopping(exp_cfg)
    callbacks = [early_stopping]

    if exp_cfg.tensorboard.enabled:
        tb_logger = TensorBoardLogger(config=exp_cfg)
        callbacks.append(tb_logger)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg.training.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    trainer = Trainer(
        cfg=exp_cfg,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        negative_sampler=negative_sampler,
        device=exp_cfg.device,
        callbacks=callbacks,
    )

    trainer.train(train_loader, val_loader)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
