import torch
import torch.nn as nn
from torchvision.models import resnet50


class Backbone(nn.Module):
    """Same structure as RadImageNet's ResNet50 weights"""
    def __init__(self):
        super().__init__()
        layers = list(resnet50(weights=None).children())
        self.backbone = nn.Sequential(*layers[:9])

    def forward(self, x):
        return self.backbone(x)


class Classifier(nn.Module):
    """RadImageNet backbone plus a new one-logit head for benign vs malignant."""
    def __init__(self, backbone: Backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1))

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(weights_path) -> Classifier:
    backbone = Backbone()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    result = backbone.load_state_dict(state, strict=False)
    print("missing keys:", result.missing_keys[:5], "| unexpected:", result.unexpected_keys[:5])
    assert not result.missing_keys and not result.unexpected_keys, "weights did not match the architecture"
    return Classifier(backbone)