import torch
import logging
import hydra
from omegaconf import DictConfig

from training.trainer import Trainer
from training.callbacks.early_stopping import EarlyStopping
from training.callbacks.tensorboard import TensorBoardLogger
from utils.config import load_config
from data.loader import load_data, get_dataloader
from utils.random import set_random_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    exp_cfg = load_config(cfg)
    print(exp_cfg)

    set_random_seed(exp_cfg.seed)

    # ------- 1
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="../datasets")
    data = dataset.get_TemporalData()
    if hasattr(dataset, 'train_mask'):
        data.train_mask = dataset.train_mask
        data.val_mask = dataset.val_mask
        data.test_mask = dataset.test_mask

    train_data = data[data.train_mask]
    val_data = data[data.val_mask]
    test_data = data[data.test_mask]

    # -------

    data = load_data(exp_cfg.dataset.path)

    train_data, val_data, test_data = data.train_val_test_split(
        exp_cfg.dataset.split.val_ratio, exp_cfg.dataset.split.test_ratio
    )

    #inductive sampler
    from utils.negative_sampling import InductiveNegativeSampler


    negative_sampler = InductiveNegativeSampler(data, train_data)
    ######


    batch_size = exp_cfg.dataset.batch_size
    train_loader = get_dataloader(train_data, batch_size)
    val_loader = get_dataloader(val_data, batch_size)
    test_loader = get_dataloader(test_data, batch_size)

    labeling_cfg = cfg.get("labeling")
    labeling = (
        hydra.utils.instantiate(labeling_cfg)
        if labeling_cfg is not None
        else None
    )
    # negative_sampler = hydra.utils.instantiate(cfg.negative_sampling, data=data)

    # labeling = (
    #     hydra.utils.get_class(exp_cfg.labeling.target)()
    #     if exp_cfg.labeling and exp_cfg.labeling.target
    #     else None
    # )

    # labeling = (
    #     hydra.utils.get_class(cfg.target)()
    #     if cfg and getattr(cfg, "target", None)
    #     else None
    # )

    # negative_sampler = (
    #     hydra.utils.get_class(exp_cfg.negative_sampling.target)()
    #     if exp_cfg.labeling.target
    #     else None
    # )

    # ModelClass = hydra.utils.get_class(exp_cfg.model.target)
    # model = ModelClass(data, negative_sampler=negative_sampler, labeling=labeling, cfg=exp_cfg)

    model = hydra.utils.instantiate(cfg.model, data=data, negative_sampler=negative_sampler, labeling=labeling, device=exp_cfg.device)

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
