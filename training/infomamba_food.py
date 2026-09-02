"""InfoMamba classifier for Food fine-tuning."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConceptReadout(nn.Module):
    """Concept readout over spatial features."""

    def __init__(self, dim: int, num_concepts: int = 32, temperature: float = 1.0):
        super().__init__()
        if num_concepts < 1 or temperature <= 0:
            raise ValueError("num_concepts and temperature must be positive")
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.empty(num_concepts, dim))
        self.mixer = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, pooled: Tensor, spatial: Tensor) -> Tensor:
        tokens = spatial.flatten(2).transpose(1, 2)
        scale = tokens.shape[-1] ** -0.5
        assignment = torch.softmax(
            (tokens @ self.prototypes.transpose(0, 1)) * scale / self.temperature,
            dim=-1,
        )
        concepts = assignment.transpose(1, 2) @ tokens
        concepts = concepts / assignment.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)
        mixed, _ = self.mixer(concepts, concepts, concepts, need_weights=False)
        context = self.norm((concepts + mixed).mean(dim=1))
        return pooled + self.gate(torch.cat((pooled, context), dim=-1)) * context


class InfoMambaFoodClassifier(nn.Module):
    """Pretrained visual backbone with a concept readout."""

    def __init__(self, backbone_name: str, num_classes: int, num_concepts: int = 32,
                 temperature: float = 1.0, pretrained: bool = True,
                 checkpoint_path: str | None = None):
        super().__init__()
        try:
            from mambavision import create_model
        except ImportError as exc:
            raise ImportError("Install the external backbone first: pip install mambavision") from exc
        options: dict[str, object] = {"pretrained": pretrained}
        if checkpoint_path:
            options["model_path"] = checkpoint_path
        self.backbone = create_model(backbone_name, **options)
        try:
            feature_dim = int(self.backbone.head.in_features)
        except AttributeError as exc:
            raise RuntimeError("The selected external backbone lacks head.in_features") from exc
        self.readout = ConceptReadout(feature_dim, num_concepts, temperature)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        spatial: Tensor | None = None

        def capture(_: nn.Module, __: tuple[Tensor, ...], output: Tensor) -> None:
            nonlocal spatial
            spatial = output

        handle = self.backbone.norm.register_forward_hook(capture)
        try:
            pooled = self.backbone.forward_features(images)
        finally:
            handle.remove()
        if spatial is None or spatial.ndim != 4:
            raise RuntimeError("The external backbone did not expose final spatial features")
        return self.classifier(self.readout(pooled, spatial))
