import abc
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Any

import PIL.Image
import loguru
import matplotlib
import ttach
import gc
from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import segmentation_models_pytorch as smp

from segmentation.losses import Losses
from segmentation.metrics import Metrics
from segmentation.preprocessing.source_image import SourceImage
from segmentation.preprocessing.workflow import PreprocessingTransforms
from segmentation.settings import NetworkTrainSettings, ProcessingSettings, ModelConfiguration, ClassSpec, ColorMap
import numpy as np

from segmentation.stats import MetricStats, EpochStats
from segmentation.util import PerformanceCounter


def pad(tensor, factor=32):
    shape = list(tensor.shape)[2:]
    h_dif = factor - (shape[0] % factor)
    x_dif = factor - (shape[1] % factor)
    x_dif = x_dif if factor != x_dif else 0
    h_dif = h_dif if factor != h_dif else 0
    augmented_image = tensor
    if h_dif != 0 or x_dif != 0:
        augmented_image = torch.nn.functional.pad(input=tensor, pad=[0, x_dif, 0, h_dif])
    return augmented_image


def unpad(tensor, o_shape):
    output = tensor[:, :, :o_shape[0], :o_shape[1]]
    return output


def test(model, device, test_loader, criterion, classes, metrics: List[Metrics], metric_reduction,
         metric_watcher_index=0, class_weights=None, padding_value=32, debug_color_map=None, ):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    metric_stats = EpochStats([MetricStats(name=i.name) for i in metrics])

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader))
        for idx, (data, target, id) in progress_bar:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            shape = list(data.shape)[2:]
            padded = pad(data, padding_value)

            input = padded.float()

            output = model(input)
            output = unpad(output, shape)
            # if batch_idx % 250 == 0:
            if debug_color_map: debug_img(output, target, data, debug_color_map)
            test_loss += criterion(output, target)
            # _, predicted = torch.max(output.data, 1)
            predicted = torch.argmax(output.data, 1)

            tp, fp, fn, tn = smp.metrics.get_stats(predicted, target,
                                                   num_classes=classes,
                                                   mode='multiclass', threshold=None)

            for metric, stats in zip(metrics, metric_stats):
                acc = metric.get_metric()(tp, fp, fn, tn, class_weights=class_weights,
                                          reduction=metric_reduction.value)
                stats.values.append(acc * 100)

            metric_string = " ".join(f"{i.name}: {i.value():.2f}%" for i in metric_stats)

            progress_bar.set_description(
                f"Testing. Loss: {test_loss / (idx + 1) : .4f} {metric_string}")

    test_loss /= len(test_loader.dataset)
    metric_string = " ".join(f"{i.name}: {i.value():.2f}%" for i in metric_stats)

    loguru.logger.info(
        f'Test set: Average loss: {test_loss:.4f}, Length of Test Set: {len(test_loader.dataset)} {metric_string}')
    loguru.logger.info(
        f'Metric used for model saving: {metric_stats.stats[metric_watcher_index].name} {metric_stats.stats[metric_watcher_index].value():.2f}%')

    return metric_stats, test_loss.data.cpu().numpy()


class NetworkBase:
    @abc.abstractmethod
    def predict(self, data: torch.Tensor, tta_aug: ttach.Compose = None) -> np.ndarray:
        pass


"""
class EnsembleNetwork(NetworkBase):
    def __init__(self, nets: List[NetworkBase]):
        assert len(nets) > 0, "Must get at least one network"
        self.nets = nets

    def predict(self, data: torch.Tensor, tta_aug: ttach.Compose = None):
        res = [x.predict(data, tta_aug) for x in self.nets]
        if len(res) == 1:
            return res[0]
        else:
            res = np.stack(res, axis=0)
            return np.mean(res, axis=0)
"""


class Network(NetworkBase):
    def __init__(self, model: nn.Module, proc_settings: ProcessingSettings, device):
        self.proc_settings = proc_settings
        self.model = model
        self.device = device

        self.model.to(self.device)

    def predict(self, data: torch.Tensor, tta_aug: ttach.Compose = None):
        self.model.eval()

        with torch.no_grad():
            data = data.to(self.device)
            output = None
            o_shape = data.shape
            if tta_aug:
                outputs = []
                for transformer in tta_aug:
                    augmented_image = transformer.augment_image(data)
                    shape = list(augmented_image.shape)[2:]
                    padded = pad(augmented_image, self.proc_settings.input_padding_value)  ## 2**5

                    input = padded.float()
                    output = self.model(input)
                    output = unpad(output, shape)
                    reversed = transformer.deaugment_mask(output)
                    reversed = torch.nn.functional.interpolate(reversed, size=list(o_shape)[2:], mode="nearest")
                    # loguru.logger.info("original: {} input: {}, padded: {} unpadded {} output {}".format(str(o_shape), str(shape),str(list(augmented_image.shape)),str(list(output.shape)),str(list(reversed.shape))))
                    outputs.append(reversed)
                    stacked = torch.stack(outputs)
                    output = torch.mean(stacked, dim=0)
            else:
                shape = list(data.shape)[2:]
                padded = pad(data, self.proc_settings.input_padding_value)  ## 2**5

                input = padded.float()
                output = self.model(input)
                output = unpad(output, shape)

            out = output.data.cpu().numpy()
            out = np.transpose(out, (0, 2, 3, 1))
            out = np.squeeze(out)
            return out


class TrainMetrics:
    pass


@dataclass
class TrainProgress:
    step: int
    total_steps: int


class TrainCallback(abc.ABC):
    @abc.abstractmethod
    def on_batch_end(self, batch, loss, acc, logs=None):
        pass

    @abc.abstractmethod
    def on_train_epoch_start(self):
        # if -1 returned than epoch is skipped
        return 0

    @abc.abstractmethod
    def on_train_epoch_end(self, epoch, acc, loss):
        pass

    @abc.abstractmethod
    def on_val_epoch_end(self, epoch, acc, loss):
        pass


def debug_img(mask, target, original, color_map: ColorMap):
    if color_map is not None:
        try:
            import PyQt6
        except:
            pass
        from matplotlib import pyplot as plt
        #matplotlib.use('TkAgg')
        # mean = [0.485, 0.456, 0.406]
        # stds = [0.229, 0.224, 0.225]

        mask = torch.argmax(mask, dim=1)
        mask = torch.squeeze(mask).cpu()
        original = original.permute(0, 2, 3, 1)
        original = torch.squeeze(original).cpu().numpy()
        # original = original * stds
        # original = original + mean
        original = original * 255
        original = original.astype(int)
        f, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        target = torch.squeeze(target).cpu()
        ax[0].imshow(NewImageReconstructor.label_to_colors(target, color_map))
        ax[1].imshow(NewImageReconstructor.label_to_colors(mask, color_map))
        ax[2].imshow(original)
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()


class NetworkTrainer(object):
    def __init__(self, network: Network, settings: NetworkTrainSettings, device, callbacks: List[TrainCallback] = None,
                 debug_color_map: ColorMap = None):
        self.network = network
        self.train_settings = settings
        self.device = device
        self.debug_color_map = debug_color_map

        opt = settings.optimizer.getOptimizer()
        try:
            optimizer1 = opt(self.network.model.encoder.parameters(), lr=self.train_settings.learningrate_encoder)
            optimizer2 = opt(self.network.model.decoder.parameters(), lr=self.train_settings.learningrate_decoder)
            optimizer3 = opt(self.network.model.segmentation_head.parameters(),
                             lr=self.train_settings.learningrate_seghead)
            optimizer = [optimizer1, optimizer2, optimizer3]
        except:
            optimizer = opt(self.network.model.parameters(), lr=self.train_settings.learningrate_seghead)
        self.optimizer = optimizer

        self.criterion = settings.loss.get_loss()() if settings.loss == Losses.cross_entropy_loss else settings.loss.get_loss()(mode='multiclass')
        self.callbacks: List[TrainCallback] = callbacks if callbacks is not None else []

    def train_epoch(self, train_loader: data.DataLoader, current_epoch: int = None):
        #print(torch.is_grad_enabled())
        model = self.network.model
        device = self.device

        model.train()
        progress_bar = tqdm(enumerate(train_loader), desc="Training", total=len(train_loader))
        metric_stats = EpochStats([MetricStats(name=i.name) for i in self.train_settings.metrics])

        acc_loss = 0
        for batch_idx, (data, target, id) in progress_bar:
            data, target = data.to(device), target.to(device, dtype=torch.int64)

            shape = list(data.shape)[2:]
            padded = pad(data, self.network.proc_settings.input_padding_value)

            input = padded.float()

            output = model(input)


            output = unpad(output, shape)
            loss = self.criterion(output, target)

            loss = loss / self.train_settings.batch_accumulation
            acc_loss += float(loss)
            model.zero_grad()  # Reset gradients tensors

            loss.backward()
            predicted = torch.argmax(output.data, 1)
            #if batch_idx % 1 == 0:
            #    debug_img(output, target, data, self.debug_color_map)
            tp, fp, fn, tn = smp.metrics.get_stats(predicted, target,
                                                   num_classes=self.train_settings.classes,
                                                   mode='multiclass', threshold=None)
            for metric, stats in zip(self.train_settings.metrics, metric_stats):
                acc = metric.get_metric()(tp, fp, fn, tn, class_weights=self.train_settings.class_weights,
                                          reduction=self.train_settings.metric_reduction.value)
                stats.values.append(acc * 100)

            if (batch_idx + 1) % self.train_settings.batch_accumulation == 0:  # Wait for several backward steps

                if isinstance(self.optimizer, Iterable):  # Now we can do an optimizer step
                    for opt in self.optimizer:
                        opt.step()
                else:
                    self.optimizer.step()

            for cb in self.callbacks:
                cb.on_batch_end(batch_idx, loss=loss.item(),
                                acc=metric_stats.stats[self.train_settings.watcher_metric_index].value())
            metric_string = " ".join(f"{i.name}: {i.value():.2f}%" for i in metric_stats)
            progress_bar.set_description(
                desc=f"Train E {current_epoch} Loss: {acc_loss / (batch_idx + 1):.4f} {metric_string}",
                refresh=False)
            #gc.collect()

        for cb in self.callbacks:
            cb.on_train_epoch_end(current_epoch,
                                  acc=metric_stats.stats[self.train_settings.watcher_metric_index].value(),
                                  loss=acc_loss / len(train_loader))

    def train_epochs(self, train_loader: data.DataLoader, val_loader: data.DataLoader, n_epoch: int, lr_schedule=None):
        #criterion = nn.CrossEntropyLoss()
        self.network.model.float()
        loguru.logger.info('Training started ...')
        for epoch in tqdm(range(0, n_epoch)):
            train_epoch = True
            for cb in self.callbacks:
                cb_val = cb.on_train_epoch_start()
                if cb_val == -1:
                    train_epoch = False

            if train_epoch:
                self.train_epoch(train_loader, epoch)
                accuracy, loss = test(self.network.model, self.device, val_loader, criterion=self.criterion,
                                      padding_value=self.network.proc_settings.input_padding_value,
                                      metrics=self.train_settings.metrics,
                                      metric_watcher_index=self.train_settings.watcher_metric_index,
                                      classes=self.train_settings.classes, class_weights=self.train_settings.class_weights,
                                      metric_reduction=self.train_settings.metric_reduction)
                # debug_color_map=self.debug_color_map)

                for cb in self.callbacks:
                    cb.on_val_epoch_end(epoch=epoch, acc=accuracy, loss=loss)


@dataclass
class PredictionResult:
    source_image: SourceImage
    preprocessed_image: SourceImage
    network_input: np.ndarray
    probability_map: np.ndarray
    other: Optional[Any] = None


class NetworkPredictorBase(abc.ABC):
    @abc.abstractmethod
    def predict_image(self, img: SourceImage) -> PredictionResult:
        pass

    def get_color_map(self) -> ColorMap:
        pass


class NetworkPredictor(NetworkPredictorBase):

    @classmethod
    def from_model_config(cls, network: NetworkBase, mc: ModelConfiguration, tta_aug: ttach.Compose = None):
        return cls(network=network, processing_settings=mc.preprocessing_settings, tta_aug=tta_aug)

    def __init__(self, network: NetworkBase, processing_settings: ProcessingSettings, tta_aug: ttach.Compose = None):
        self.network = network
        self.proc_settings = processing_settings
        self.tta_aug = tta_aug
        self.transforms = PreprocessingTransforms.from_dict(processing_settings.transforms)


    def predict_image(self, img: SourceImage) -> PredictionResult:
        if self.proc_settings.scale_predict:
            scaled_image = img.scale_area(self.proc_settings.scale_max_area)
        else:
            scaled_image = img

        input_img = self.transforms.transform_predict(scaled_image.array())["image"]

        input_img = input_img.unsqueeze(0)

        prediction = self.network.predict(input_img, tta_aug=self.tta_aug)

        return PredictionResult(source_image=img, preprocessed_image=scaled_image, network_input=input_img,
                                probability_map=prediction)


class EnsemblePredictor(NetworkPredictorBase):

    @classmethod
    def from_model_config(cls, networks: List[NetworkBase], mcs: List[ModelConfiguration],
                          tta_aug: ttach.Compose = None):
        return cls(networks=networks, processing_settings=[mc.preprocessing_settings for mc in mcs], tta_aug=tta_aug)

    def __init__(self, networks: List[NetworkBase], processing_settings: List[ProcessingSettings],
                 tta_aug: ttach.Compose = None):
        self.networks = networks
        self.proc_settings = processing_settings
        self.tta_aug = tta_aug
        self.transforms = [PreprocessingTransforms.from_dict(tr.transforms) for tr in processing_settings]

    def predict_image(self, img: SourceImage) -> PredictionResult:
        single_network_prediction_result: List[PredictionResult] = []

        for network, config, transforms in zip(self.networks, self.proc_settings, self.transforms):
            if config.scale_predict:
                scaled_image = img.scale_area(config.scale_max_area)
            else:
                scaled_image = img
            input_img = transforms.transform_predict(scaled_image.array())["image"]

            input_img = input_img.unsqueeze(0)

            prediction = network.predict(input_img, self.tta_aug)
            single_network_prediction_result.append(PredictionResult(
                source_image=img,
                preprocessed_image=scaled_image,
                network_input=input_img,
                probability_map=prediction,
                other=single_network_prediction_result))

        res = np.stack([i.probability_map for i in single_network_prediction_result], axis=0)
        prediction = np.mean(res, axis=0)
        return PredictionResult(source_image=img, preprocessed_image=single_network_prediction_result[0].preprocessed_image, network_input=single_network_prediction_result[0].network_input,
                                probability_map=prediction, other=single_network_prediction_result)  #  TODO: Bugfix: preprocessed image must not necessarily be equal


class NewImageReconstructor:
    def __init__(self, labeled_image, total_labels=None, background_color=(0, 0, 0), undefined_color=(255, 255, 255)):
        if total_labels is None or total_labels == 0:
            total_labels = int(np.max(labeled_image)) + 1
        self.color_keys = np.tile(np.array(undefined_color, dtype=np.uint8), (total_labels, 1))
        self.labeled_image = labeled_image
        # set label 0 to white
        self.label(0, background_color)

    def label(self, label, color):
        self.color_keys[label, 0] = color[0]
        self.color_keys[label, 1] = color[1]
        self.color_keys[label, 2] = color[2]

    def get_image(self):
        return self.color_keys[self.labeled_image]

    @staticmethod
    def reconstructed_to_binary(reconstructed, background_color=(0, 0, 0)):
        img = np.array(np.where(np.all(reconstructed == background_color, axis=2), 255, 0), dtype=np.uint8)
        assert img.dtype == np.uint8
        return img

    @staticmethod
    def reconstructed_where(reconstructed, positive):
        return np.where(np.all(reconstructed == positive, axis=2), 255, 0, dtype=np.uint8)

    @staticmethod
    def label_to_colors(labeled_image, color_map: ColorMap):
        nr = NewImageReconstructor(total_labels=len(color_map), labeled_image=labeled_image)

        for cls in color_map:
            nr.label(cls.label, cls.color)
        return nr.get_image()
