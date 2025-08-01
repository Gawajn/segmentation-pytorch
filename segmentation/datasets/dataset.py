import PIL.Image
import skimage
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch.transforms import ToTensorV2
import json
from ast import literal_eval
import random
from matplotlib import pyplot as plt
from typing import List
from skimage.morphology import remove_small_holes
import albumentations as albu
import gc

from segmentation.preprocessing.workflow import PreprocessingTransforms
from segmentation.settings import ColorMap
from segmentation.util import gray_to_rgb, rgb2gray, show_images
from pagexml_mask_converter.pagexml_to_mask import MaskGenerator, MaskSetting, BaseMaskGenerator, MaskType, PCGTSVersion
import math
from segmentation.preprocessing.basic_binarizer import gauss_threshold
from segmentation.util import logger

from segmentation.preprocessing.ocrupus import binarize


# When training/testing/evaluationg on cpu set environment varialbe LRU_CACHE_CAPACITY=1
# Needed for dynamicaly sized inputs, else memory leak
def to_categorical(y, num_classes, torch=True):
    """ 1-hot encodes a tensor """
    one_hot = np.eye(num_classes, dtype='uint8')[y]
    if torch:
        one_hot = np.transpose(one_hot, (2, 0, 1))
    return one_hot


def color_to_label(mask, colormap: ColorMap):
    out = np.zeros(mask.shape[0:2], dtype=np.int32)

    if mask.ndim == 2:
        return mask.astype(np.int32) / 255

    if mask.shape[2] == 2:
        return mask[:, :, 0].astype(np.int32) / 255
    mask = mask.astype(np.uint32)
    mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]
    for cspec in colormap:
        color, label = cspec.color, cspec.label
        color_1d = 256 * 256 * color[0] + 256 * color[1] + color[2]
        out += np.int32((mask == color_1d) * label)
    return out


def label_to_colors(mask, colormap: dict):
    out = np.zeros(mask.shape + (3,), dtype=np.int64)
    for color, label in colormap.items():
        trues = np.stack([(mask == label[0])] * 3, axis=-1)
        out += np.tile(color, mask.shape + (1,)) * trues

    out = np.ndarray.astype(out, dtype=np.uint8)
    return out


def rescale_pil(image, scale, order=1):
    return image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), order)


def default_preprocessing(x):
    return x / 255.


# def process(image, mask, rgb, preprocessing, apply_preprocessing, augmentation, color_map=None,
#             binary_augmentation=True, ocropy=True, crop=False, crop_x=512, crop_y=512):
#     if rgb:
#         image = gray_to_rgb(image)
#     result = {"image": image}
#     if color_map:
#         if mask.ndim == 3:
#             result["mask"] = color_to_label(mask, color_map)
#         elif mask.ndim == 2:
#             u_values = np.unique(mask)
#             mask = result["mask"]
#             for ind, x in enumerate(u_values):
#                 mask[mask == x] = ind
#             result["mask"] = mask
#     else:
#         result["mask"] = mask if mask is not None else image
#
#     if augmentation is not None:
#         result = augmentation(**result)
#
#     if augmentation is not None and binary_augmentation:
#         from segmentation.preprocessing.basic_binarizer import gauss_threshold
#         from segmentation.preprocessing.ocrupus import binarize
#         ran = np.random.randint(1, 5)
#         if ran == 1:
#             if ocropy:
#                 binary = binarize(result["image"].astype("float64")).astype("uint8") * 255
#                 gray = gray_to_rgb(binary)
#                 result["image"] = gray_to_rgb(gray)
#         if ran == 2:
#             image = rgb2gray(result["image"]).astype(np.uint8)
#             result["image"] = gray_to_rgb(gauss_threshold(image))
#     if crop:
#         result = compose([[albu.RandomCrop(
#             crop_y, crop_x, p=1
#         )]])(**result)
#         ## albumentations.augmentations.crops.transforms.RandomResizedCrop
#     if apply_preprocessing is not None and apply_preprocessing:
#         result["image"] = preprocessing(result["image"])
#     result = compose([post_transforms()])(**result)
#     return result["image"], result["mask"]"

def extract_data_from_pd(df, item):
    image_id, mask_id = df.get('images')[item], df.get('masks')[item]
    mask2_id = None
    try:
        mask2_id = df.get('add_symbols_mask')[item]
    except:
        pass
    return image_id, mask_id, mask2_id

class MaskDataset(Dataset):
    def __init__(self, df, transforms: PreprocessingTransforms = None, scale_area=1000000):
        self.df = df
        self.transforms = transforms
        self.index = self.df.index.tolist()
        self.scale_area = scale_area


    def __getitem__(self, item, apply_preprocessing=True):
        image_id, mask_id = self.df.get('images')[item], self.df.get('masks')[item]

        image = Image.open(image_id)
        mask = Image.open(mask_id)
        rescale_factor = get_rescale_factor(image, scale_area=self.scale_area)

        mask = np.array(rescale_pil(mask, rescale_factor, 0))
        image = np.array(rescale_pil(image, rescale_factor, 1))
        if image.dtype == bool:
            image = image.astype("uint8") * 255

        transformed = self.transforms.transform_train(image, mask)

        return transformed["image"], transformed["mask"], torch.tensor(item)

    def __len__(self):
        return len(self.index)


class MemoryDataset(Dataset):
    def __init__(self, df, transforms: PreprocessingTransforms = None,
                 scale_area=1000000):
        self.df = df
        self.index = self.df.index.tolist()
        self.transforms = transforms
        self.scale_area = scale_area

    def __getitem__(self, item, apply_preprocessing=True):
        image_id, mask_id, mask_id2 = extract_data_from_pd(self.df, item)

        image = image_id
        mask = mask_id
        pil_image = Image.fromarray(image)
        rescale_factor = get_rescale_factor(pil_image, scale_area=self.scale_area)
        image = np.array(rescale_pil(pil_image, rescale_factor, 1))
        if image.dtype == bool:
            image = image.astype("uint8") * 255

        if mask_id2 is not None:
            transformed = self.transforms.transform_train(image, mask, mask2=mask_id2)
            # show_images([image,transformed["image"].cpu().numpy().transpose([1,2,0])],["Original","Augmented"])

            return transformed["image"], transformed["mask"], transformed["add_symbols_mask"], torch.tensor(item)

        else:
            transformed = self.transforms.transform_train(image, mask)
            return transformed["image"], transformed["mask"],  torch.tensor(item)

    def __len__(self):
        return len(self.index)


class XMLDataset(Dataset):
    def __init__(self, df, mask_generator: BaseMaskGenerator,
                 transforms: PreprocessingTransforms = None, scale_area=1000000):
        self.df = df
        self.transforms = transforms
        self.index = self.df.index.tolist()
        self.mask_generator = mask_generator
        self.scale_area = scale_area

    def __getitem__(self, item, apply_preprocessing=True):
        image_id, mask_id, mask_id2 = extract_data_from_pd(self.df, item)

        image = Image.open(image_id)
        rescale_factor = get_rescale_factor(image, scale_area=self.scale_area)

        mask = self.mask_generator.get_mask(mask_id, rescale_factor)

        image = np.array(rescale_pil(image, rescale_factor, 1))
        #from  matplotlib import pyplot as plt
        #fix, ax = plt.subplots(1, 2, True, True)
        #ax[0].imshow(mask)
        #ax[1].imshow(image)
        #plt.show()
        if image.dtype == bool:
            image = image.astype("uint8") * 255
        if mask_id2 is not None:
            mask2 = self.mask_generator.get_mask(mask_id2, rescale_factor)
            transformed = self.transforms.transform_train(image, mask, mask2=mask2)
            # show_images([image,transformed["image"].cpu().numpy().transpose([1,2,0])],["Original","Augmented"])

            return transformed["image"], transformed["mask"], transformed["add_symbols_mask"], torch.tensor(item)

        else:
            transformed = self.transforms.transform_train(image, mask)
            return transformed["image"], transformed["mask"],  torch.tensor(item)

        # TODO: switch between modes based on parameter


    def __len__(self):
        return len(self.index)


class PredictDataset(Dataset):
    def __init__(self, df,  transform=None, scale_area=1000000):
        self.df = df
        self.index = self.df.index.tolist()
        self.scale_area = scale_area
        self.transforms = transform

    def __getitem__(self, item, apply_preprocessing=True):
        image_id, mask_id, mask_id2 = extract_data_from_pd(self.df, item)
        image = Image.open(image_id)
        rescale_factor = get_rescale_factor(image, self.scale_area)
        image = np.array(rescale_pil(image, rescale_factor, 1))
        mask = image
        transformed = self.transforms.transform_train(image, mask)  # TODO: switch between modes based on parameter
        return transformed["image"], transformed["mask"], torch.tensor(item)

    def __len__(self):
        return len(self.index)


def get_rescale_factor(pil_image, scale_area=1000000):
    rescale_factor = 1.0
    if (pil_image.size[1] * pil_image.size[0]) >= scale_area:
        rescale_factor = math.sqrt(scale_area / (pil_image.size[1] * pil_image.size[0]))
    return rescale_factor


def listdir(dir, postfix="", not_postfix=False):
    if dir is None:
        return None
    if len(postfix) > 0 and not_postfix:
        return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if not f.endswith(postfix)]
    else:
        return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if f.endswith(postfix)]


def dirs_to_pandaframe(images_dir: List[str], masks_dir: List[str], verify_filenames: bool = True):
    img = []
    m = []
    for img_d, mask_d in zip(images_dir, masks_dir):
        img += listdir(img_d)
        m += listdir(mask_d)
    if verify_filenames:
        def filenames(fn, postfix=None):
            if postfix and len(postfix) > 0:
                fn = [f[:-len(postfix)] if f.endswith(postfix) else f for f in fn]

            x = {os.path.basename(f).split('.')[0]: f for f in fn}
            return x

        img_dir = filenames(img)
        mask_dir = filenames(m)
        base_names = sorted(set(img_dir.keys()).intersection(set(mask_dir.keys())))

        img = [img_dir.get(basename) for basename in base_names]
        m = [mask_dir.get(basename) for basename in base_names]

    else:
        base_names = None
    df = pd.DataFrame(data={'images': img, 'masks': m, 'add_symbols_mask': m})

    return df


def load_image_map_from_file(path):
    if not os.path.exists(path):
        raise Exception("Cannot open {}".format(path))

    with open(path) as f:
        data = json.load(f)
    color_map = {literal_eval(k): v for k, v in data.items()}
    return color_map


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        # albu.RandomRotate90(),
        albu.CoarseDropout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3, border_mode=0, value=255, mask_value=[255, 255, 255]),
    ]

    return result


def default_transform():
    result = [
        albu.HorizontalFlip(),
        albu.RandomGamma(),
        albu.RandomBrightnessContrast(),
        albu.OneOf([
            albu.ToGray(),
            albu.CLAHE()]),
        albu.RandomScale(),

    ]
    return result


def resize_transforms(image_size=512):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(image_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size * 2, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            # rescale,
            # random_crop_big
        ], p=1)
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    # convenient if ypu want to add extra targets, e.g. binary input
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def show_examples(name: str, image: np.ndarray, binary: np.ndarray, mask: np.ndarray):
    foreground = np.stack([(binary)] * 3, axis=-1)
    inv_binary = 1 - binary
    inv_binary = np.stack([inv_binary] * 3, axis=-1)
    overlay_mask = mask.copy()
    overlay_mask[foreground == 0] = 0
    inverted_overlay_mask = mask.copy()
    inverted_overlay_mask[inv_binary == 0] = 0

    plt.figure(figsize=(10, 14))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")

    plt.subplot(1, 3, 3)
    plt.imshow(inverted_overlay_mask)
    plt.title(f"Binary: {name}")


def show(index: int, image, mask, transforms=None) -> None:
    image = Image.open(image)
    image = np.asarray(image)
    mask = np.array(Image.open(mask))

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp['image']
        mask = temp['mask']
    bin_og = Image.fromarray(image)
    bin_og = bin_og.convert('1')
    binary = np.array(bin_og)
    binary = np.asarray(binary)
    binary = remove_small_holes(binary, 3, True)
    show_examples(index, image, binary, mask)


def show_random(df, transforms=None) -> None:
    length = len(df)
    index = random.randint(0, length - 1)
    image = df.get('images')[index]
    mask = df.get('masks')[index]
    show(index, image, mask, transforms)
    plt.show()

def show_image_batch(img_list, mask_list):
    num = len(img_list)
    f, ax = plt.subplots(2, num, True, True)
    for i in range(num):
        ax[0][i].imshow(img_list[i].numpy().transpose([2,1,0]))
        ax[1][i].imshow(mask_list[i].numpy().transpose([1,0]))

    plt.show()
def train(model, device, train_loader, optimizer, epoch, accumulation_steps=1):
    model.train()
    total_train = 0
    correct_train = 0
    for batch_idx, (data, target, id) in enumerate(train_loader):
        show_image_batch(data, target)

        data, target = data.to(device), target.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data.float())
        loss = F.nll_loss(output, target)
        loss = loss / accumulation_steps
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        total_train += target.nelement()
        correct_train += predicted.eq(target.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        print(
            '\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                          len(train_loader.dataset),
                                                                                          100. * batch_idx / len(
                                                                                              train_loader),
                                                                                          loss.item(),
                                                                                          train_accuracy), end="",
            flush=True)
        if (batch_idx + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            model.zero_grad()  # Reset gradients tensors
        # track()
        gc.collect()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, id in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def track():
    print((len(gc.get_objects())))
    '''
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
    '''

def collate_fn_variable_sized_images(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]
if __name__ == '__main__':
    'https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb'
    a = dirs_to_pandaframe(
        ['/home/alexanderh/Documents/datasets/baselines/train/image/'],
        ['/home/alexanderh/Documents/datasets/baselines/train/page/'])

    b = dirs_to_pandaframe(
        ['/home/alexanderh/Documents/datasets/baselines/test/image/'],
        ['/home/alexanderh/Documents/datasets/baselines/test/page/'])

    map = load_image_map_from_file(
        '/home/alexanderh/Documents/datasets/baselines/image_map.json')

    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    dt = XMLDataset(a, map, transform=compose([default_transform(), resize_transforms()]), mask_generator=MaskGenerator(settings=settings))
    d_test = XMLDataset(b, map, transform=compose([default_transform()]), mask_generator=MaskGenerator(settings=settings))



    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from segmentation.custom_model import UNet, AttentionUnet

    model = UNet(in_channels=3,
                  out_channels=16,
                  n_class=len(map),
                  kernel_size=3,
                  padding=1,
                  stride=1)

    model1 = AttentionUnet(
        in_channels=3,
        out_channels=16,
        n_class=len(map),
        kernel_size=3,
        padding=1,
        stride=1,
        attention=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.NLLLoss()
    model.float()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    from torch.utils import data

    train_loader = data.DataLoader(dataset=dt, batch_size=5, shuffle=True, num_workers=5)
    test_loader = data.DataLoader(dataset=d_test, batch_size=1, shuffle=False)

    for epoch in range(1, 3):
        print('Training started ...')
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
