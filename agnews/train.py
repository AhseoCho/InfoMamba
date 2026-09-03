"""Train or evaluate InfoMamba on AG News."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, Mamba2Model, MambaModel


@dataclass
class RunConfig:
    base_model: str
    seed: int
    max_length: int
    concepts: int
    active_mass_ratio: float
    inject_layers: list[int]
    beta: float
    gamma: float
    position_aware_routing: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConceptRepair(nn.Module):
    def __init__(self, width: int, concepts: int, dropout: float, max_positions: int, position_aware: bool) -> None:
        super().__init__()
        self.assignment = nn.Linear(width, concepts, bias=False)
        self.position_aware = position_aware
        if position_aware:
            self.position_embedding = nn.Embedding(max_positions, width)
            self.position_assignment = nn.Linear(width, concepts, bias=False)
        self.concept_norm = nn.LayerNorm(width)
        self.mixer = nn.MultiheadAttention(width, 8 if width % 8 == 0 else 4, dropout=dropout, batch_first=True)
        self.mixer_norm = nn.LayerNorm(width)
        self.readout_norm = nn.LayerNorm(width)
        self.state_projection = nn.Linear(width, width, bias=False)
        self.output_projection = nn.Linear(width, width, bias=False)
        self.state_gate = nn.Parameter(torch.tensor(-2.0))
        self.output_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor, active_mass_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.assignment(hidden)
        if self.position_aware:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            logits = logits + self.position_assignment(self.position_embedding(positions)).unsqueeze(0)
        routing = torch.softmax(logits, dim=-1) * attention_mask.unsqueeze(-1).to(hidden.dtype)
        mass = routing.sum(dim=1)
        active = mass >= attention_mask.sum(dim=1, keepdim=True).to(mass.dtype) * active_mass_ratio
        active.scatter_(1, mass.argmax(dim=1, keepdim=True), True)
        routing = routing * active.unsqueeze(1).to(routing.dtype)
        routing = routing / routing.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        concepts = torch.einsum("blk,bld->bkd", routing, hidden) / mass.unsqueeze(-1).clamp_min(1e-6)
        mixed, _ = self.mixer(self.concept_norm(concepts), self.concept_norm(concepts), self.concept_norm(concepts), key_padding_mask=~active)
        concepts = self.mixer_norm(concepts + mixed)
        readout = F.gelu(self.readout_norm(torch.einsum("blk,bkd->bld", routing, concepts)))
        return torch.sigmoid(self.state_gate) * self.state_projection(readout), torch.sigmoid(self.output_gate) * self.output_projection(readout), concepts


class InfoMambaClassifier(nn.Module):
    def __init__(self, base_model: str, concepts: int = 32, inject_layers: list[int] | None = None, dropout: float = 0.1, max_positions: int = 160, position_aware: bool = True) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(base_model)
        model_class = Mamba2Model if config.model_type == "mamba2" else MambaModel
        self.backbone = model_class.from_pretrained(base_model)
        self.inject_layers = set(inject_layers or [5, 11, 17, 23])
        width = self.backbone.config.hidden_size
        self.repairs = nn.ModuleDict({str(index): ConceptRepair(width, concepts, dropout, max_positions, position_aware) for index in self.inject_layers})
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 4))
        self.concept_classifier = nn.Linear(width, 4)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, active_mass_ratio: float = 0.01) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone.embeddings(input_ids)
        all_concepts = []
        for index, block in enumerate(self.backbone.layers):
            if index in self.inject_layers:
                state_delta, output_delta, concepts = self.repairs[str(index)](hidden, attention_mask, active_mass_ratio)
                hidden = block(hidden + state_delta, attention_mask=attention_mask) + output_delta
                all_concepts.append(concepts.mean(dim=1))
            else:
                hidden = block(hidden, attention_mask=attention_mask)
        recurrent = self.masked_mean(self.backbone.norm_f(hidden), attention_mask)
        concept = torch.stack(all_concepts).mean(dim=0)
        return self.classifier(recurrent), self.concept_classifier(concept), recurrent, concept


def make_loader(dataset, tokenizer, batch_size: int, shuffle: bool, max_length: int) -> DataLoader:
    def collate(rows):
        texts = [row["text"] for row in rows]
        batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        return batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, collate_fn=collate)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, active_mass_ratio: float) -> tuple[float, float]:
    model.eval(); correct = total = 0; loss_sum = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        logits, _, _, _ = model(**{key: value.to(device) for key, value in batch.items()}, active_mass_ratio=active_mass_ratio)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item(); total += labels.numel()
    return loss_sum / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--concepts", type=int, default=32)
    parser.add_argument("--active-mass-ratio", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--position-aware-routing", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only and args.checkpoint is None:
        parser.error("--evaluate-only requires --checkpoint")
    args.output.mkdir(parents=True, exist_ok=True); seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("fancyzhx/ag_news")
    split = data["train"].train_test_split(test_size=6000, seed=args.seed, stratify_by_column="label")
    layers = [5, 11, 17, 23]
    config = RunConfig(args.base_model, args.seed, args.max_length, args.concepts, args.active_mass_ratio, layers, args.beta, args.gamma, args.position_aware_routing)
    model = InfoMambaClassifier(args.base_model, args.concepts, layers, max_positions=args.max_length, position_aware=args.position_aware_routing).to(device)
    test_loader = make_loader(data["test"], tokenizer, args.batch_size * 2, False, args.max_length)
    if args.evaluate_only:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"], strict=not state.get("adapter_only", False))
        loss, accuracy = evaluate(model, test_loader, device, args.active_mass_ratio)
        print(json.dumps({"test_loss": loss, "test_accuracy": accuracy}, indent=2)); return
    for parameter in model.backbone.parameters(): parameter.requires_grad_(False)
    for block in model.backbone.layers[-args.unfreeze_last_n:]:
        for parameter in block.parameters(): parameter.requires_grad_(True)
    for parameter in model.backbone.norm_f.parameters(): parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = make_loader(split["train"], tokenizer, args.batch_size, True, args.max_length)
    validation_loader = make_loader(split["test"], tokenizer, args.batch_size * 2, False, args.max_length)
    best_accuracy = -1.0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device); batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, concept_logits, recurrent, concept = model(**batch, active_mass_ratio=args.active_mass_ratio)
                redundancy = (F.normalize(recurrent - recurrent.mean(0), dim=0).T @ F.normalize(concept - concept.mean(0), dim=0) / recurrent.shape[0]).square().mean()
                loss = (F.cross_entropy(logits, labels) + args.beta * F.cross_entropy(concept_logits, labels) + args.gamma * redundancy) / args.grad_accum
            scaler.scale(loss).backward(); total_loss += loss.item() * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        val_loss, val_accuracy = evaluate(model, validation_loader, device, args.active_mass_ratio)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_loader), "val_loss": val_loss, "val_accuracy": val_accuracy}; history.append(row); print(json.dumps(row), flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"state_dict": model.state_dict(), "config": asdict(config), "epoch": epoch, "val_accuracy": val_accuracy}, args.output / "best.pt")
    state = torch.load(args.output / "best.pt", map_location=device, weights_only=False); model.load_state_dict(state["state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, device, args.active_mass_ratio)
    (args.output / "report.json").write_text(json.dumps({"best_epoch": state["epoch"], "best_val_accuracy": state["val_accuracy"], "test_loss": test_loss, "test_accuracy": test_accuracy, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
