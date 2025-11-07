import torch.nn as nn
import torchvision.models as tv

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool=False):
        super().__init__()
        self.backbone = tv.efficientnet_b4(weights=tv.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
