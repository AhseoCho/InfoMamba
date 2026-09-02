"""Evaluation-only AG News runner for the released InfoMamba checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class AGNewsDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_length: int):
        with csv_path.open(encoding="utf-8") as handle:
            self.rows = [{"label": int(row["Class Index"]) - 1, "text": f"{row['Title']} [SEP] {row['Description']}"} for row in csv.DictReader(handle)]
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(row["text"], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {key: value[0] for key, value in encoded.items()} | {"labels": torch.tensor(row["label"], dtype=torch.long)}


class InfoGateClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        states = self.encoder(**kwargs).last_hidden_state
        cls = states[:, 0]
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        mean = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        gate = self.gate(torch.cat([cls, mean], dim=-1))
        return self.classifier(self.dropout(self.norm(gate * cls + (1.0 - gate) * mean)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Official AG News test CSV")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=None, help="Optional override; by default the checkpoint metadata is used")
    parser.add_argument("--max-length", default=256, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_name = args.model or checkpoint.get("model")
    if not model_name:
        raise ValueError("Checkpoint does not specify an inference backbone")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = AGNewsDataset(args.data, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = InfoGateClassifier(model_name)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            logits = model(**{key: value.to(device) for key, value in batch.items()})
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.numel()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {"samples": total, "accuracy": 100.0 * correct / total, "checkpoint": args.checkpoint.name}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
