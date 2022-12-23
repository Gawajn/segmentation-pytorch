import abc
import json
from pathlib import Path

import loguru
import torch

from segmentation.network import Network, TrainCallback
from segmentation.settings import ModelConfiguration, ModelFile
from segmentation.stats import EpochStats


class ModelWriterCallback(TrainCallback):
    def on_train_epoch_start(self):
        # if -1 returned than epoch is skipped
        return 0

    def __init__(self, network: Network, model_configuration: ModelConfiguration, save_path: Path, prefix: str = "",
                 metric_watcher_index=0,
                 save_all: bool = False):
        assert not save_all, "Not implemented"
        self.stats: EpochStats = None
        self.metric_watcher_index = metric_watcher_index
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
                self.stats.to_dict()).to_dict(), indent=4))  # TODO: write json

    def on_val_epoch_end(self, epoch, acc, loss):
        if self.save_all:
            self.save(self.get_epoch_path(epoch))
        acc: EpochStats = acc
        if self.stats is None or acc.stats[self.metric_watcher_index].value() > self.stats.stats[
            self.metric_watcher_index].value():
            accuracy_before = 0 if self.stats is None else self.stats.stats[self.metric_watcher_index].value()
            self.stats = acc
            self.best_loss = loss
            loguru.logger.info(
                f"New best {self.stats.stats[self.metric_watcher_index].name}: {self.stats.stats[self.metric_watcher_index].value():.4f}, Before: {accuracy_before:.4f}")
            best_path = self.get_best_model_path()
            self.save(best_path)

    def on_batch_end(self, batch, loss, acc, logs=None):
        pass

    def get_best_model_path(self):
        return (self.save_path / f"{self.prefix}best").with_suffix(".torch")

    def get_epoch_path(self, e: int):
        return (self.save_path / f"{self.prefix}e{e}").with_suffix(".torch")

    def get_best_json_path(self):
        return self.get_best_model_path().with_suffix(".json")

    def get_epoch_json_path(self, e: int):
        return self.get_epoch_path(e).with_suffix(".json")


class EarlyStoppingCallback(TrainCallback):
    def __init__(self, patience: int = 5, metric_watcher_index=0):
        self.current_patience = 0
        self.patience: int = patience
        self.stats: EpochStats = None
        self.metric_watcher_index = metric_watcher_index
        self.best_loss = -1

        pass

    def on_train_epoch_start(self):
        if self.current_patience >= self.patience:
            return -1
        else:
            return 0

    def on_val_epoch_end(self, epoch, acc, loss):
        acc: EpochStats = acc
        if self.metric_watcher_index > 0:
            if self.stats is None or acc.stats[self.metric_watcher_index].value() > self.stats.stats[
                    self.metric_watcher_index].value():
                self.stats = acc
                self.best_loss = loss
                self.current_patience = 0
            else:
                self.current_patience += 1
        else:
            if self.best_loss > loss:
                self.best_loss = loss
                self.current_patience = 0
            else:
                self.current_patience += 1
