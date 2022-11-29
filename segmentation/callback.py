import abc
import json
from pathlib import Path

import loguru
import torch

from segmentation.network import Network, TrainCallback
from segmentation.settings import ModelConfiguration, ModelFile

class ModelWriterCallback(TrainCallback):
    def __init__(self, network: Network, model_configuration: ModelConfiguration, save_path: Path, prefix: str="", save_all: bool = False):
        assert not save_all, "Not implemented"
        self.highest_accuracy = -1
        self.best_loss = -1
        self.network = network
        self.save_path = save_path
        self.model_config = model_configuration
        self.save_all = save_all
        self.prefix = prefix
        if len(self.prefix) > 1 and not self.prefix.endswith("_"):
            self.prefix = f"{self.prefix}_"

    def on_train_epoch_end(self, epoch, acc, loss):
        pass

    def save(self, path: Path):
        torch.save(self.network.model.state_dict(), path.with_suffix(".torch"))
        loguru.logger.info(f'Saving model to {path.with_suffix(".torch")}')
        with open(path.with_suffix(".json"), "w") as f:
            f.write(json.dumps(ModelFile(
                self.model_config,
                {"accuracy": self.highest_accuracy, "loss": self.best_loss}).to_dict(), indent=4))  # TODO: write json

    def on_val_epoch_end(self, epoch, acc, loss):
        if self.save_all:
            self.save(self.get_epoch_path(epoch))

        if acc > self.highest_accuracy:
            accuracy_before = self.highest_accuracy
            self.highest_accuracy = acc
            self.best_loss = loss
            loguru.logger.info(f"New best accuracy: {acc:.4f}, Before: {accuracy_before:.4f}")
            best_path = self.get_best_model_path()
            self.save(best_path)

    def on_batch_end(self, batch, loss, acc, logs=None):
        pass

    def get_best_model_path(self):
        return (self.save_path / f"{self.prefix}best").with_suffix(".torch")

    def get_epoch_path(self, e:int):
        return (self.save_path / f"{self.prefix}e{e}").with_suffix(".torch")

    def get_best_json_path(self):
        return self.get_best_model_path().with_suffix(".json")

    def get_epoch_json_path(self, e:int):
        return self.get_epoch_path(e).with_suffix(".json")
