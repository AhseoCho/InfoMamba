"""Food checkpoint factory."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ReleasedFoodClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        try:
            from mambavision import create_model
        except ImportError as exc:
            raise ImportError("Install the Food evaluation extra first") from exc
        self.backbone = create_model("mamba_vision_B", pretrained=False, num_classes=num_classes)
        dim = int(self.backbone.head.in_features)
        self.info_norm = nn.LayerNorm(dim)
        self.info_down = nn.Linear(dim, max(dim // 16, 32), bias=False)
        self.info_up = nn.Linear(max(dim // 16, 32), dim, bias=False)
        self.info_gate = nn.Parameter(torch.tensor(1.0e-3))

    def forward(self, images: Tensor) -> Tensor:
        x = self.backbone.forward_features(images)
        info_state = self.info_up(torch.nn.functional.gelu(self.info_down(self.info_norm(x))))
        return self.backbone.head(x + self.info_gate * info_state)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        own_keys = {"info_norm", "info_down", "info_up", "info_gate"}
        remapped = {}
        for key, value in state_dict.items():
            prefix = key.split(".", 1)[0]
            remapped[key if prefix in own_keys or key.startswith("backbone.") else f"backbone.{key}"] = value
        return super().load_state_dict(remapped, strict=strict, assign=assign)


def build_food_model(num_classes: int) -> ReleasedFoodClassifier:
    return ReleasedFoodClassifier(num_classes)
