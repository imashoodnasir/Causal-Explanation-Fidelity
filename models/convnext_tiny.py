import torch.nn as nn
import torchvision.models as tv

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool=False):
        super().__init__()
        self.backbone = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
