import abc
from pathlib import Path

import loguru
import torch

from segmentation.network import Network, TrainCallback
from segmentation.settings import ModelConfiguration, ModelFile

class ModelWriterCallback(TrainCallback):
    def __init__(self, network: Network, model_configuration: ModelConfiguration, save_path: Path, save_all: bool = False):
        self.highest_accuracy = -1
        self.network = network
        self.save_path = save_path
        self.model_config = model_configuration
        self.save_all = save_all

    def on_train_epoch_end(self, epoch, acc):
        pass

    def save(self, path: Path):
        torch.save(self.network.model.state_dict(), path.with_suffix(".torch"))
        loguru.logger.info(f'Saving model to {path.with_suffix(".torch")}')
        with open(path.with_suffix(".json"), "w") as f:
            f.write(ModelFile(self.model_config,{}).to_json())  # TODO: write json

    def on_val_epoch_end(self, epoch, acc):
        # TODO: ajust paths
        if self.save_all:
            self.save(self.save_path / f"epoch{epoch}")

        if acc > self.highest_accuracy:
            loguru.logger.info(f"New best accuracy: {acc:.4f}, Before: {self.highest_accuracy:.4f}")
            best_path = self.save_path / "best"
            self.save(best_path)
            self.highest_accuracy = acc

    def on_batch_end(self, batch, loss, acc, logs=None):
        pass

"""
class TrainProgressCallbackWrapper:

    def __init__(self,
                 n_iters_per_epoch: int,
                 train_callback: TrainProgressCallback):
        super().__init__()
        self.train_callback = train_callback
        self.n_iters_per_epoch = n_iters_per_epoch
        self.epoch = 0
        self.iter = 0

    def on_batch_end(self, batch, loss, acc, logs=None):
        self.iter = batch + self.epoch * self.n_iters_per_epoch
        self.train_callback.update_loss(self.iter,
                                        loss=loss,
                                        acc=acc)

    def on_epoch_end(self, epoch, acc, wait=0):
        self.epoch = epoch + 1
        self.train_callback.next_best(self.iter, acc, wait)
"""