"""InfoMamba Food classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class InfoSummaryAdapter(nn.Module):
    """Low-rank information summary adapter."""

    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = max(dim // 16, 32)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(1.0e-3))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        summary = self.up(torch.nn.functional.gelu(self.down(self.norm(x))))
        return x + self.gate * summary


class InfoMambaFoodClassifier(nn.Module):
    """Pretrained visual backbone with the Info summary adapter."""

    def __init__(self, backbone_name: str, num_classes: int,
                 pretrained: bool = True, checkpoint_path: str | None = None):
        super().__init__()
        try:
            from mambavision import create_model
        except ImportError as exc:
            raise ImportError("Install mambavision before training") from exc
        options: dict[str, object] = {"pretrained": pretrained}
        if checkpoint_path:
            options["model_path"] = checkpoint_path
        self.backbone = create_model(backbone_name, **options)
        feature_dim = int(self.backbone.head.in_features)
        self.backbone.head = nn.Identity()
        self.info = InfoSummaryAdapter(feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.info(self.backbone.forward_features(images)))
