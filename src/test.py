import torch
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from training.trainer import Trainer
from training.evaluator import Evaluator
from training.callbacks.early_stopping import EarlyStopping
from training.callbacks.tensorboard import TensorBoardLogger
from utils.config import load_config
from data.loader import load_data, get_dataloader
from utils.random import set_random_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    exp_cfg = load_config(cfg)

    set_random_seed(exp_cfg.seed)
    data = load_data(exp_cfg.dataset.path)
    train_data, val_data, test_data = data.train_val_test_split(
        exp_cfg.dataset.split.val_ratio, exp_cfg.dataset.split.test_ratio
    )

    batch_size = exp_cfg.dataset.batch_size
    train_loader = get_dataloader(
        train_data, batch_size, exp_cfg.dataset.negative_sampling_ratio
    )
    val_loader = get_dataloader(
        val_data, batch_size, exp_cfg.dataset.negative_sampling_ratio
    )
    test_loader = get_dataloader(
        test_data, batch_size, exp_cfg.dataset.negative_sampling_ratio
    )

    ModelClass = hydra.utils.get_class(exp_cfg.model.target)
    model = ModelClass(data, exp_cfg)
    # model = hydra.utils.instantiate(exp_cfg.model, data, exp_cfg)


    early_stopping = EarlyStopping(exp_cfg)
    callbacks = [early_stopping]

    if exp_cfg.tensorboard.enabled:
        tb_logger = TensorBoardLogger(config=exp_cfg)
        callbacks.append(tb_logger)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg.training.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    evaluator = Evaluator(model, exp_cfg.device, exp_cfg)
    trainer = Trainer(
        cfg=exp_cfg,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        evaluator=evaluator,
        device=exp_cfg.device,
        callbacks=callbacks,
    )

    trainer.train(train_loader, val_loader)

    # test po treningu
    # TODO dodać tu wczytanie modelu, przenieść do train
    logger.info("Evaluating on test set...")
    test_metrics = evaluator.evaluate(test_loader)
    logger.info(f"Test results: {test_metrics}")


if __name__ == "__main__":
    main()
