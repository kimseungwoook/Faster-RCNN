from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn

class VGG16Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        # max pooling 전까지만 사용
        self.features = model.features[:30]

    def forward(self, x):
        output = self.features(x)
        return output