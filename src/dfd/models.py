"""timm model factory. Input: uint8 (B,3,H,W) tensor; ImageNet normalisation (and optional upsampling to
`input_size`) is done inside forward(), so data loading, val-C and inference always feed 224x224.

Auxiliary mask head (PHASE4 §2, `model.mask_head: true`): a small conv head on the backbone's stride-16
feature map predicts the manipulated-pixel mask (14x14 at 224, 20x20 for B4@320). It runs ONLY in train
mode; eval-mode forward is the plain classifier and never touches it.

Bundling / inference contract (PHASE5): every mask-head parameter and buffer lives under the state_dict
prefix `mask_head.` (MASK_HEAD_PREFIX). The head is training-only, so bundle.py strips those keys
(`strip_mask_head(sd)`), and a `Classifier(mask_head=False)` loads a state_dict with or without them
(they are dropped silently, see Classifier.load_state_dict). With mask_head off the module tree,
state_dict keys and forward are exactly the pre-Phase-4 ones.
"""
from __future__ import annotations

import math

import timm
import timm.optim
import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _zero_head(net: nn.Module) -> nn.Linear:
    """Zero-init the 1-output classifier. timm's heads (EfficientNet `classifier`, ConvNeXt/Swin `head.fc`,
    ViT `head`) can start with large weights -> huge initial logits and oversized gradients into the
    pretrained backbone. Zero-init: start at p=0.5 exactly (Phase 1 lesson 1)."""
    head = net.get_classifier()
    if not isinstance(head, nn.Linear):  # fallback: last 1-output Linear in the module tree
        lins = [m for m in net.modules() if isinstance(m, nn.Linear) and m.out_features == 1]
        head = lins[-1] if lins else head
    assert isinstance(head, nn.Linear) and head.out_features == 1, f"1-output Linear head not found: {head!r}"
    nn.init.zeros_(head.weight)
    if head.bias is not None:
        nn.init.zeros_(head.bias)
    assert not head.weight.any() and (head.bias is None or not head.bias.any()), "head not zeroed"
    return head


MASK_HEAD_PREFIX = "mask_head."


def strip_mask_head(state_dict: dict) -> dict:
    """Drop the training-only mask-head entries (bundling for inference, PHASE5 §4)."""
    return {k: v for k, v in state_dict.items() if not k.startswith(MASK_HEAD_PREFIX)}


def stride16_index(net: nn.Module) -> int:
    """Index (into net.feature_info, as used by forward_intermediates) of the LAST stride-16 feature map:
    EfficientNet blocks.4, ConvNeXt stages.2, Swin layers.2, ViT the last block (all blocks are /16)."""
    red = [f["reduction"] for f in net.feature_info]
    idx = [i for i, r in enumerate(red) if r == 16]
    if not idx:
        raise ValueError(f"{type(net).__name__}: no stride-16 feature map (reductions {red})")
    return idx[-1]


class MaskHead(nn.Module):
    """Conv3x3(C->64) -> BN -> ReLU -> Conv1x1(64->1): per-location manipulation logits."""

    def __init__(self, in_chs: int, mid: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chs, mid, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, 1, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.bn(self.conv1(f)))).squeeze(1)  # (B, h, w)


class Classifier(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        input_size: int = 224,
        mask_head: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.net = timm.create_model(
            name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        _zero_head(self.net)  # no alias attribute: it would add duplicate keys to the state_dict
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0, persistent=False)
        if mask_head:  # only registered when on: keeps mask_head=False state_dicts identical to pre-Phase-4
            self.mask_index = stride16_index(self.net)
            self.mask_head = MaskHead(self.net.feature_info[self.mask_index]["num_chs"])
        else:
            self.mask_head = None

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() - self.mean) / self.std
        if x.shape[-2:] != (self.input_size, self.input_size):
            # upsample on the GPU after normalisation (e.g. B4 at 320); no-op path at 224 = bit-identical
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        """eval (or no mask head): (B,) logits. train with mask head: ((B,) logits, (B,h,w) mask logits)."""
        x = self._prep(x)
        if self.mask_head is None or not self.training:
            return self.net(x).squeeze(1)  # (B,) logits
        # one backbone pass: final features for the classifier + the stride-16 map for the mask head
        final, (feat,) = self.net.forward_intermediates(x, indices=[self.mask_index], output_fmt="NCHW")
        return self.net.forward_head(final).squeeze(1), self.mask_head(feat)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.mask_head is None:  # weights trained with the (training-only) mask head load fine
            state_dict = strip_mask_head(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def build_model(cfg: dict, pretrained: bool = True) -> Classifier:
    m = cfg["model"]
    return Classifier(
        m["name"], pretrained, m.get("drop_rate", 0.0), m.get("drop_path_rate", 0.0), m.get("input_size", 224),
        mask_head=bool(m.get("mask_head", False)),
    )


def param_groups(
    model: Classifier, lr_backbone: float, lr_head: float, wd: float, layer_decay: float | None = None
) -> list[dict]:
    """Head gets a higher LR (it starts from scratch); no weight decay on biases/norm weights.

    With `layer_decay` (LLRD): timm's BEiT-style grouping (model.group_matcher) assigns each backbone layer
    k of L the LR lr_backbone * layer_decay^(L-k). The classifier params (net.get_classifier()) are split
    out into `head`/`head_nd` groups at lr_head. timm puts the classifier (and any un-matched neck, e.g.
    ConvNeXt head.norm) in the top layer (scale 1.0), so for ViT/EfficientNet/Swin the top backbone block
    is at layer_decay * lr_backbone (BEiT convention). Backbone groups are ordered by descending lr_scale,
    so group 0 (what train.py logs as lr_backbone) is the highest-LR backbone group.

    The optional mask head (model.mask_head) joins the head groups (lr_head; no WD on its biases / BN).
    """
    head_ids = {id(p) for p in model.net.get_classifier().parameters()}
    # mask head (PHASE4 §2) starts from scratch too -> head groups (lr_head); it lives outside model.net
    mask_params = [p for p in model.mask_head.parameters() if p.requires_grad] if model.mask_head is not None else []
    head_ids |= {id(p) for p in mask_params}
    if not layer_decay:
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

    nwd = model.net.no_weight_decay() if hasattr(model.net, "no_weight_decay") else ()
    tg = timm.optim.param_groups_layer_decay(
        model.net, weight_decay=wd, no_weight_decay_list=nwd, layer_decay=float(layer_decay)
    )
    bb, head = [], {"decay": [], "no": []}
    for i, g in enumerate(tg):
        keep = [p for p in g["params"] if id(p) not in head_ids]
        for p in g["params"]:
            if id(p) in head_ids:
                head["no" if g["weight_decay"] == 0.0 else "decay"].append(p)
        if keep:
            bb.append({**g, "params": keep, "_i": i})
    for p in mask_params:  # not part of model.net, so timm's grouping never saw them
        head["no" if p.ndim <= 1 else "decay"].append(p)
    bb.sort(key=lambda g: (-g["lr_scale"], g["_i"]))
    out = []
    for g in bb:
        g.pop("_i")
        tag = "nd" if g["weight_decay"] == 0.0 else "d"
        depth = round(math.log(g["lr_scale"]) / math.log(layer_decay)) if g["lr_scale"] > 0 else -1
        out.append({**g, "lr": lr_backbone * g["lr_scale"], "name": f"bb_top-{depth}_{tag}"})
    out.append({"params": head["decay"], "lr": lr_head, "weight_decay": wd, "lr_scale": 1.0, "name": "head"})
    out.append({"params": head["no"], "lr": lr_head, "weight_decay": 0.0, "lr_scale": 1.0, "name": "head_nd"})
    n_all = sum(1 for p in model.parameters() if p.requires_grad)
    assert sum(len(g["params"]) for g in out) == n_all, "LLRD groups do not cover every trainable param once"
    return out
