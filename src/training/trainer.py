import logging

from models.base import BaseModel
from training.callbacks.base import Callback
from training.evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterion,
        evaluator,
        device,
        callbacks: list[Callback] | None = None,
    ):
        self.cfg = cfg
        self.model: BaseModel = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluator: Evaluator = evaluator
        self.device = device

        self.callbacks = callbacks or []
        self.should_stop = False
        self.current_epoch = 0

    def train(self, train_loader, val_loader):
        logger.info(f"Starting training on device: {self.device}")
        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(1, self.cfg.training.epochs + 1):
            self.current_epoch = epoch

            self.model.train()
            self.model.on_epoch_start()
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.train_step(batch, self.criterion)
                loss.backward()
                self.optimizer.step()
                self.model.after_optimizer_step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluation
            if epoch % self.cfg.eval.eval_every == 0:
                metrics = self.evaluator.evaluate(val_loader, "val")
                metrics["train/loss"] = avg_loss
                metrics["epoch"] = epoch

                logger.info(
                    f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                    f"Val AUC: {metrics.get('val/auc', 0):.4f} | "
                    f"Val AP: {metrics.get('val/ap', 0):.4f}"
                )

                for cb in self.callbacks:
                    cb.on_epoch_end(self, metrics)

                if self.should_stop:
                    logger.info("Early stopping triggered.")
                    break

        for cb in self.callbacks:
            cb.on_train_end(self)
        logger.info("Training finished.")
