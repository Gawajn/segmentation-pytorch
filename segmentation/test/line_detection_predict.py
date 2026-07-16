import glob
import itertools

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from segmentation.model_builder import ModelBuilderLoad
from segmentation.network import NetworkPredictor
from segmentation.network_postprocessor import NetworkMaskPostProcessor, NetworkBaselinePostProcessor, BaselineResult
from segmentation.preprocessing.source_image import SourceImage
from segmentation.settings import ColorMap, ClassSpec




colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (0, 255, 255),
          (255, 0, 255)]

def show_result(result: BaselineResult):
    pil_image = result.prediction_result.source_image.pil_image.convert('RGB')
    draw = ImageDraw.Draw(pil_image)

    for ind, x in enumerate(result.base_lines):
        t = list(itertools.chain.from_iterable(x))
        a = t[::]
        draw.line(a, fill=colors[ind % len(colors)], width=4)
    f, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(result.prediction_result.source_image.pil_image.convert('RGB'))
    ax[1].imshow(np.array(pil_image))
    ax[2].imshow(np.array(result.mask))
    plt.show()





if __name__ == "__main__":

    model_path = "/tmp/best.torch"
    model_path = "/home/alexanderh/projects/segmentation-pytorch/models/model_93.torch"
    #model_path = "/home/alexanderh/projects/segmentation-pytorch/logs/logs_converted.torch"
    model_path = "/home/alexanderh/projects/segmentation-pytorch/segmentation/scripts/best.torch"

    mb = ModelBuilderLoad.from_disk(model_weights=model_path, device="cuda")
    config = mb.get_model_configuration()
    net = mb.get_model()
    predictor = NetworkPredictor.from_model_config(net,mb.get_model_configuration())
    image_list = sorted(glob.glob('/home/alexanderh/PycharmProjects/pythonProject3/todo/Köln_Dombibl_1001b/*'))

    #cmap = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
    #                 ClassSpec(label=1, name="Baseline", color=[255, 0, 255]),
    #                 ClassSpec(label=2, name="BaselineBorder", color=[255, 255, 0])])
    cmap = config.color_map
    bmaskpred = NetworkBaselinePostProcessor(predictor, cmap)
    for img_path in tqdm(image_list):

        simg = SourceImage.load(img_path)
        result = bmaskpred.predict_image(simg)

        show_result(result)










