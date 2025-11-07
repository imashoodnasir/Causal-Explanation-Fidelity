import torch.nn as nn
import torchvision.models as tv

class ViTB16(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool=False):
        super().__init__()
        self.backbone = tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
