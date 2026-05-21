import logging
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from models.base import BaseModel
from training.callbacks.base import Callback

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg,
        model: BaseModel,
        optimizer,
        criterion,
        negative_sampler,
        device,
        callbacks: list[Callback] | None = None,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.negative_sampler = negative_sampler
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
                metrics = self._evaluate(val_loader, "val")
                metrics["train/loss"] = avg_loss

                logger.info(
                    f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                    f"Val AUC: {metrics.get('val/auc', 0):.4f} | "
                    f"Val AP: {metrics.get('val/ap', 0):.4f}"
                )

                for cb in self.callbacks:
                    cb.on_epoch_end(self, metrics)

                if self.should_stop:
                    break

        for cb in self.callbacks:
            cb.on_train_end(self)
        logger.info("Training finished.")

    def test(self, test_loader):
        logger.info("Evaluating on test set...")
        self.negative_sampler.reset()
        test_metrics = self._evaluate(test_loader)
        logger.info(f"Test results: {test_metrics}")
        return test_metrics

    @torch.no_grad()
    def _evaluate(self, loader, prefix="test"):
        self.model.eval()
        y_pred, y_true = [], []

        for batch in loader:
            batch = batch.to(self.device)

            pos_out, neg_out = self.model.test_step(batch)

            y_pred.append(torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu())
            y_true.append(
                torch.cat(
                    [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
                )
            )

            # y_pred.append(pos_out.cpu())
            # y_true.append(torch.ones_like(pos_out).cpu())

            # y_pred.append(neg_out.cpu())
            # y_true.append(torch.zeros_like(neg_out).cpu())

        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()

        return {
            f"{prefix}/auc": roc_auc_score(y_true, y_pred),
            f"{prefix}/ap": average_precision_score(y_true, y_pred),
        }

    # @torch.no_grad()
    # def _evaluate(self, loader, prefix="test"):
    #     self.model.eval()
    #     y_pred, y_true = [], []

    #     for batch in loader:
    #         batch = batch.to(self.device)

    #         out = self.model.test_step(batch)  # [N, 1]
    #         out = out.squeeze(-1)              # [N]

    #         y_pred.append(out.sigmoid().cpu())
    #         y_true.append(batch.y.float().cpu())

    #     y_pred = torch.cat(y_pred).numpy()
    #     y_true = torch.cat(y_true).numpy()

    #     return {
    #         f"{prefix}/auc": roc_auc_score(y_true, y_pred),
    #         f"{prefix}/ap": average_precision_score(y_true, y_pred),
    #     }
