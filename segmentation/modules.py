from segmentation_models_pytorch import encoders
import segmentation_models_pytorch as smp
from enum import Enum

ENCODERS = smp.encoders.get_encoder_names()


class Architecture(Enum):
    FPN = 'fpn'
    UNET = 'unet'
    PSPNET = 'pspnet'
    LINKNET = 'linknet'
    DeepLabV3 = 'deeplabV3'
    DeepLabV3Plus = 'deeplabV3plus'
    PAN = 'pan'
    UnetPlusPlus = 'UnetPlusPlus'
    MAnet = 'MAnet'

    def get_architecture(self):
        return {'fpn': smp.FPN,
                'unet': smp.Unet,
                'pspnet': smp.PSPNet,
                'linknet': smp.Linknet,
                'MAnet': smp.MAnet,
                'UnetPlusPlus': smp.UnetPlusPlus,
                'pan': smp.PAN,
                'deeplabV3': smp.DeepLabV3,
                'deeplabV3plus': smp.DeepLabV3Plus,

                }[self.value]

    @staticmethod
    def get_all_architectures():
        return [smp.FPN, smp.Unet, smp.PSPNet, smp.Linknet, smp.DeepLabV3, smp.DeepLabV3Plus, smp.UnetPlusPlus, smp.PAN,
                smp.MAnet]

    def get_architecture_params(self):
        import inspect
        t = self.get_architecture()
        signature = inspect.signature(t.__init__)
        a = dict()
        for name, parameter in signature.parameters.items():
            if name == 'self':
                continue
            a[name] = parameter.default
        return a
