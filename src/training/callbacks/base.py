import abc


class Callback(abc.ABC):
    def on_train_start(self, trainer):
        pass

    def on_epoch_end(self, trainer, metrics: dict):
        pass

    def on_train_end(self, trainer):
        pass
