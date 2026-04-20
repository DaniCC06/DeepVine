from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torch import Tensor
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

BackboneName = Literal["resnet50", "efficientnet_b0"]


class VineLeafClassifier(nn.Module):
    """Transfer learning classifier for grapevine leaf recognition.

    The model uses an ImageNet pretrained backbone and replaces its classifier
    head with a task-specific output layer. Feature extractor parameters are
    frozen by default to reduce overfitting on low-sample classes.

    Args:
        num_classes: Number of grapevine classes in the dataset.
        backbone: Backbone architecture to use.
        dropout_p: Dropout probability applied in the custom classifier head.
        freeze_backbone: Whether to freeze pretrained feature extractor weights.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: BackboneName = "resnet50",
        dropout_p: float = 0.2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            msg = "num_classes must be >= 2 for multiclass classification."
            raise ValueError(msg)

        self.backbone_name: BackboneName = backbone

        if backbone == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )

        if freeze_backbone:
            self.freeze_feature_extractor()

    def freeze_feature_extractor(self) -> None:
        """Freeze all backbone parameters except the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False

        if self.backbone_name == "resnet50":
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze entire backbone for fine-tuning stages."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x: Batch tensor with shape (N, 3, H, W).

        Returns:
            Logits tensor with shape (N, num_classes).
        """
        return self.model(x)
