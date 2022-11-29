import abc
from dataclasses import dataclass, asdict
from typing import List

import PIL.Image
import loguru
import ttach
import gc
from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm

from segmentation.dataset import process, get_rescale_factor, rescale_pil, label_to_colors
from segmentation.preprocessing.source_image import SourceImage
from segmentation.settings import NetworkTrainSettings, ProcessingSettings, ModelConfiguration, ClassSpec, ColorMap
import numpy as np

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


def test(model, device, test_loader, criterion, padding_value=32):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader))
        for idx, (data, target, id) in progress_bar:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            shape = list(data.shape)[2:]
            padded = pad(data, padding_value)

            input = padded.float()

            output = model(input)
            output = unpad(output, shape)
            test_loss += criterion(output, target)
            _, predicted = torch.max(output.data, 1)

            total += target.nelement()
            correct += predicted.eq(target.data).sum().item()
            # loguru.logger.info('\r Image [{}/{}'.format(idx * len(data), len(test_loader.dataset)))
            progress_bar.set_description(
                f"Testing. Loss: {test_loss / (idx + 1) : .4f} Accuracy: {correct / total :.4f}")

    test_loss /= len(test_loader.dataset)

    loguru.logger.info('Test set: Average loss: {:.4f}, Length of Test Set: {} (Accuracy{:.6f}%)'.format(
        test_loss, len(test_loader.dataset),
        100. * correct / total))

    return 100. * correct / total, test_loss.data.cpu().numpy()


class NetworkBase:
    @abc.abstractmethod
    def predict(self, data: torch.Tensor, tta_aug: ttach.Compose = None) -> np.ndarray:
        pass


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
    def on_train_epoch_end(self, epoch, acc):
        pass

    @abc.abstractmethod
    def on_val_epoch_end(self, epoch, acc):
        pass

def debug_img(mask, target, original, color_map: ColorMap):
    if color_map is not None:
        from matplotlib import pyplot as plt
        #mean = [0.485, 0.456, 0.406]
        #stds = [0.229, 0.224, 0.225]

        mask = torch.argmax(mask, dim=1)
        mask = torch.squeeze(mask).cpu()
        original = original.permute(0, 2, 3, 1)
        original = torch.squeeze(original).cpu().numpy()
        #original = original * stds
        #original = original + mean
        original = original * 255
        original = original.astype(int)
        f, ax = plt.subplots(1, 3, True, True)
        target = torch.squeeze(target).cpu()
        ax[0].imshow(NewImageReconstructor.label_to_colors(target, color_map))
        ax[1].imshow(NewImageReconstructor.label_to_colors(mask, color_map))
        ax[2].imshow(original)

        plt.show()

class NetworkTrainer(object):
    def __init__(self, network: Network, settings: NetworkTrainSettings, device, callbacks: List[TrainCallback] = None, debug_color_map: ColorMap = None):
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

        self.criterion = nn.CrossEntropyLoss()
        self.callbacks: List[TrainCallback] = callbacks if callbacks is not None else []

    def train_epoch(self, train_loader: data.DataLoader, current_epoch: int = None):
        model = self.network.model
        device = self.device

        model.train()
        total_train = 0
        correct_train = 0
        train_accuracy = 0
        progress_bar = tqdm(enumerate(train_loader), desc="Training", total=len(train_loader))

        acc_loss = 0
        for batch_idx, (data, target, id) in progress_bar:
            data, target = data.to(device), target.to(device, dtype=torch.int64)

            shape = list(data.shape)[2:]
            padded = pad(data, self.network.proc_settings.input_padding_value)

            input = padded.float()

            output = model(input)
            if batch_idx % 250 == 0:
                debug_img(output, target, data, self.debug_color_map)

            output = unpad(output, shape)
            loss = self.criterion(output, target)
            loss = loss / self.train_settings.batch_accumulation
            acc_loss += float(loss)
            loss.backward()

            # _, predicted = torch.max(output.data, 1)
            predicted = torch.argmax(output.data, 1)
            total_train += target.nelement()
            correct_train += predicted.eq(target.data).sum().item()
            train_accuracy = 100 * correct_train / total_train

            if (batch_idx + 1) % self.train_settings.batch_accumulation == 0:  # Wait for several backward steps
                if isinstance(self.optimizer, Iterable):  # Now we can do an optimizer step
                    for opt in self.optimizer:
                        opt.step()
                else:
                    self.optimizer.step()
                model.zero_grad()  # Reset gradients tensors

            for cb in self.callbacks:
                cb.on_batch_end(batch_idx, loss=loss.item(), acc=train_accuracy)

            progress_bar.set_description(
                desc=f"Train E {current_epoch} Loss: {acc_loss / (batch_idx + 1):.4f} Accuracy: {train_accuracy:.2f}%", refresh=False)
            gc.collect()

        for cb in self.callbacks:
            cb.on_train_epoch_end(current_epoch, acc=train_accuracy)

    def train_epochs(self, train_loader: data.DataLoader, val_loader: data.DataLoader, n_epoch: int, lr_schedule=None):
        criterion = nn.CrossEntropyLoss()
        self.network.model.float()
        loguru.logger.info('Training started ...')
        for epoch in tqdm(range(0, n_epoch)):
            self.train_epoch(train_loader, epoch)
            accuracy, loss = test(self.network.model, self.device, val_loader, criterion=criterion,
                                  padding_value=self.network.proc_settings.input_padding_value)

            for cb in self.callbacks:
                cb.on_val_epoch_end(epoch=epoch, acc=accuracy)


@dataclass
class PredictionResult:
    source_image: SourceImage
    preprocessed_image: SourceImage
    network_input: np.ndarray
    probability_map: np.ndarray


class NetworkPredictor:

    @classmethod
    def from_model_config(cls, network: NetworkBase, mc: ModelConfiguration):
        return cls(network=network, processing_settings=mc.preprocessing_settings)

    def __init__(self, network: NetworkBase, processing_settings: ProcessingSettings):
        self.network = network
        self.proc_settings = processing_settings

    def predict_image(self, img: SourceImage) -> PredictionResult:
        if self.proc_settings.scale_predict:
            scaled_image = img.scale_area(self.proc_settings.scale_max_area)
        else:
            scaled_image = img

        input, _ = process(image=scaled_image.array(), mask=scaled_image.array(), rgb=self.proc_settings.rgb,
                           preprocessing=self.proc_settings.preprocessing.get_preprocessing_function(),
                           apply_preprocessing=True, augmentation=None, color_map=None,
                           binary_augmentation=False)
        input = input.unsqueeze(0)

        prediction = self.network.predict(input)

        return PredictionResult(source_image=img, preprocessed_image=scaled_image, network_input=input,
                                probability_map=prediction)


class NewImageReconstructor:
    def __init__(self, labeled_image, total_labels=None, background_color=(0,0,0), undefined_color=(255,255,255)):
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


@dataclass
class MaskPredictionResult:
    prediction_result: PredictionResult
    generated_mask: PIL.Image


class NetworkMaskPredictor:
    def __init__(self, network: NetworkBase, model_config: ModelConfiguration, overwrite_color_map: ColorMap = None):
        self.predictor = NetworkPredictor(network, model_config.preprocessing_settings)
        self.model_config = model_config
        if overwrite_color_map:
            self.color_map = overwrite_color_map
        else:
            self.color_map = self.model_config.color_map

    def predict_image(self, img: SourceImage, keep_dim: bool = True) -> PIL.Image:
        res = self.predictor.predict_image(img)

        # create labeled image from probability map
        lmap = np.argmax(res.probability_map, axis=-1)
        mask = NewImageReconstructor.label_to_colors(lmap, self.color_map)

        outimg = PIL.Image.fromarray(mask, mode="RGB")

        if keep_dim:
            mask = outimg
        else:
            mask = outimg.resize(size=(img.get_width(), img.get_height()), resample=PIL.Image.NEAREST)

        mpr = MaskPredictionResult(prediction_result=res, generated_mask=mask)
        return mpr



