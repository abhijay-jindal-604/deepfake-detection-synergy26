"""timm model factory. Input: uint8 (B,3,H,W) tensor; ImageNet normalisation is done inside forward()."""
from __future__ import annotations

import timm
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Classifier(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.net = timm.create_model(
            name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() - self.mean) / self.std
        return self.net(x).squeeze(1)  # (B,) logits


def build_model(cfg: dict, pretrained: bool = True) -> Classifier:
    m = cfg["model"]
    return Classifier(m["name"], pretrained, m.get("drop_rate", 0.0), m.get("drop_path_rate", 0.0))


def param_groups(model: Classifier, lr_backbone: float, lr_head: float, wd: float) -> list[dict]:
    """Head gets a higher LR (it starts from scratch); no weight decay on biases/norm weights."""
    head_ids = {id(p) for p in model.net.get_classifier().parameters()}
    groups = {k: [] for k in ("bb_decay", "bb_no", "head_decay", "head_no")}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "bb"
        groups[f"{part}_{'no' if p.ndim <= 1 else 'decay'}"].append(p)
    return [
        {"params": groups["bb_decay"], "lr": lr_backbone, "weight_decay": wd, "name": "backbone"},
        {"params": groups["bb_no"], "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_nd"},
        {"params": groups["head_decay"], "lr": lr_head, "weight_decay": wd, "name": "head"},
        {"params": groups["head_no"], "lr": lr_head, "weight_decay": 0.0, "name": "head_nd"},
    ]
