import torch
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, initialization


class MultiHeadNetwork(torch.nn.Module):
    def __init__(self, model, number_of_heads: int, activation=None, upsampling=1, add_classes=None, out_channels=None):
        super().__init__()
        if add_classes is None:
            add_classes = []
        self.model = model

        self.number_of_heads = number_of_heads
        self.heads = []
        for i in range(self.number_of_heads-1):
            head = SegmentationHead(
                in_channels=out_channels,
                out_channels=add_classes[i],
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )
            setattr(self, f'head_{i}', head)
        self.initialize()

    def initialize(self):
        for i in self.heads:
            initialization.initialize_head(i)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.model.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        add_masks = []

        self.check_input_shape(x)

        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)
        for i in self.heads:
            add_masks.append(i(*features))
        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
            return masks, labels, add_masks

        return masks, add_masks


    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
