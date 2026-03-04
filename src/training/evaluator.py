import torch
from sklearn.metrics import roc_auc_score, average_precision_score


class Evaluator:
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg = cfg

    @torch.no_grad()
    def evaluate(self, loader, prefix="val"):
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
