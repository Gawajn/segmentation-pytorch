import glob
import itertools
import os

import numpy as np
from matplotlib import pyplot as plt

from segmentation.binarization.doxapy_bin import binarize, BinarizationAlgorithm
from segmentation.datasets.network_postprocessor import NetworkTablePostProcessor, TableResult
from segmentation.model_builder import ModelBuilderLoad
from segmentation.network import EnsemblePredictor
from segmentation.network_postprocessor import NetworkMaskPostProcessor, MaskPredictionResult
from segmentation.preprocessing.source_image import SourceImage
from segmentation.scripts.train import get_default_device
from PIL import Image, ImageDraw

colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (0, 255, 255),
          (255, 0, 255)]


def show_result(result: TableResult):
    pil_image = result.prediction_result.source_image.pil_image.convert('RGB')
    draw = ImageDraw.Draw(pil_image)

    if True:
        for ind, x in enumerate(result.horizontal_lines):
            t = list(itertools.chain.from_iterable(x))
            a = t[::]
            draw.line(a, fill=colors[ind % len(colors)], width=4)
    for ind, x in enumerate(result.vertical_lines):
        t = list(itertools.chain.from_iterable(x))
        a = t[::]
        draw.line(a, fill=colors[ind % len(colors)], width=4)
    f, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(result.prediction_result.source_image.pil_image.convert('RGB'))
    ax[1].imshow(np.array(pil_image))
    ax[2].imshow(np.array(result.mask))
    plt.show()


if __name__ == "__main__":
    modelbuilder = ModelBuilderLoad.from_disk(
        model_weights=os.path.join("/home/alexanderh/Documents/datasets/table/models_f", 'best.torch'),
        device=get_default_device())

    base_model = modelbuilder.get_model()
    config = modelbuilder.get_model_configuration()
    preprocessing_settings = modelbuilder.get_model_configuration().preprocessing_settings
    predictor = EnsemblePredictor([base_model], [preprocessing_settings])
    ntablepredictor = NetworkTablePostProcessor(predictor, config.color_map)
    path = "/home/alexanderh/mount/uni/scratch/tables/nextgentmf/visual_test/*png"
    path2 = "/home/alexanderh/Documents/datasets/table/doc/*jpg"
    path3 = "/home/alexanderh/mount/uni/scratch/gehrke/tables/original/*.png"
    path4 = "/home/alexanderh/mount/uni/scratch/tables/nextgentmf/hard_scans/*.png"

    for i in glob.glob(path4):
        pil_image = Image.open(i)
        image = np.array(pil_image)
        # image = binarize(image, algorithm=BinarizationAlgorithm("isauvola")).astype("uint8") * 255

        source_image = SourceImage.from_numpy(image)
        output: TableResult = ntablepredictor.predict_image(source_image)
        # from matplotlib import pyplot as plt

        # f, ax = plt.subplots(1, 2, sharey=True, sharex=True)
        # ax[0].imshow(np.array(image))
        # ax[1].imshow(np.array(output.generated_mask))
        # plt.show()
        show_result(result=output)
