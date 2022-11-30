import glob

from PIL import Image
from tqdm import tqdm

from segmentation.model_builder import ModelBuilderLoad
from segmentation.network import NetworkMaskPostProcessor
from segmentation.preprocessing.source_image import SourceImage
from segmentation.settings import ColorMap, ClassSpec

if __name__ == "__main__":

    model_path = "/tmp/best.torch"
    mb = ModelBuilderLoad.from_disk(model_weights="/tmp/best.torch", device="cuda")
    config = mb.get_model_configuration()
    net = mb.get_model()
    image_list = sorted(glob.glob('/home/alexanderh/Documents/datasets/baselines/train/image/*jpg')[:5])
    cmap = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                     ClassSpec(label=1, name="Baseline", color=[255, 0, 255]),
                     ClassSpec(label=2, name="BaselineBorder", color=[255, 255, 0])])

    nmaskpred = NetworkMaskPostProcessor(net, config, cmap)

    for img_path in tqdm(image_list):
        simg = SourceImage.load(img_path)
        mask = nmaskpred.predict_image(simg)
        mask.generated_mask.show()









