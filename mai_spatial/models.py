"""Model definitions for single-patch regression."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


def _weights(enum_cls, pretrained: bool):
    return enum_cls.DEFAULT if pretrained else None


def make_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Create an image backbone and return it with its output feature dimension."""
    name = name.lower()

    if name == "resnet18":
        model = tv_models.resnet18(
            weights=_weights(tv_models.ResNet18_Weights, pretrained)
        )
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "densenet121":
        model = tv_models.densenet121(
            weights=_weights(tv_models.DenseNet121_Weights, pretrained)
        )
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif name == "mobilenet_v2":
        model = tv_models.mobilenet_v2(
            weights=_weights(tv_models.MobileNet_V2_Weights, pretrained)
        )
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == "efficientnet_b7":
        model = tv_models.efficientnet_b7(
            weights=_weights(tv_models.EfficientNet_B7_Weights, pretrained)
        )
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(
            weights=_weights(tv_models.EfficientNet_B0_Weights, pretrained)
        )
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return model, feature_dim


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 3000) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features):
        return self.net(features)


class SinglePatchRegressor(nn.Module):
    """Backbone plus MLP head for spot-level gene-expression regression."""

    def __init__(
        self,
        out_dim: int,
        backbone_name: str = "efficientnet_b0",
        hidden_dim: int = 3000,
        unfreeze_from: int | None = -9,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = make_backbone(backbone_name, pretrained=pretrained)
        self._configure_trainable_layers(unfreeze_from)
        self.head = RegressionHead(feature_dim, out_dim, hidden_dim)

    def _configure_trainable_layers(self, unfreeze_from: int | None) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if unfreeze_from is None:
            return

        if hasattr(self.backbone, "features"):
            for parameter in self.backbone.features[unfreeze_from:].parameters():
                parameter.requires_grad = True
        else:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True

    def forward(self, images):
        features = self.backbone(images)
        if isinstance(features, (list, tuple)):
            features = features[0]
        return self.head(features)
