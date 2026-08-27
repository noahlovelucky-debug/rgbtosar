"""Minimal HiViT loader compatible with the supplied SARATR-X checkpoint.

Architecture reference: https://github.com/waterdisappear/SARATR-X
The local ``SOC_40classes.pth`` contains the linear-probe HiViT weights with
``embed_dim=512``, ``depths=(2, 2, 20)``, no relative position embedding, and
a non-affine BatchNorm classifier head.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# Class order recovered by running the supplied checkpoint on real
# ``SOC_40classes/test`` samples (38/40 correct on the first sample/class).
SOC40_CLASSES = (
    "Buick_Excelle_GT", "Buick_GL8", "CNHTC_HOWO", "Chang'an_CS75_Plus",
    "Chang'an_Starlight_4500", "Changfeng_Cheetah_CFA6473C", "Changlin_8228-5",
    "Chery_Arrizo 5", "Chery_qq3", "Chevrolet_Blazer_1998", "Dongfeng_Duolika",
    "Dongfeng_EQ6608LTV", "Dongfeng_Forthing_Lingzhi", "Dongfeng_Tianjin_DFH2200B",
    "Dongfeng_Tianjin_KR230", "FAW_J6P", "FAW_Jiabao_T51", "Foton_BJ1045V9JB5-54",
    "Great_Wall_Voleex_C50", "Great_Wall_poer", "Hawtai_EV160B", "Hongqi_CA7180A3E",
    "Hongqi_h5", "Huanghai_N1", "Hyundai_HLF25_II", "JAC_Junling",
    "JINBEI_SY5033XJH", "Jeep_Patriot", "Lincoln_MKC", "Lveco_Proud_2009", "MAXUS_V80",
    "Mitsubishi_Outlander_2003", "SDLG_ZL40F", "SHACMAN_DeLong_M3000",
    "SHACMAN_DeLong_X3000", "WAW_Aochi_1800", "WAW_Aochi_Hongrui",
    "Wuling_Rongguang_V", "Yangzi_YZK6590XCA", "Yutong_ZK6120HY1",
)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps) if heads else None
        self.attn = Attention(dim, heads) if heads else None
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None and self.norm1 is not None:
            x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 128, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(128, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        if height % 16 or width % 16:
            raise ValueError(f"SARATR-X input size must be divisible by 16, got {height}x{width}")
        patch_h, patch_w = height // 16, width // 16
        x = self.proj(x).view(batch, -1, patch_h, 4, patch_w, 4)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch, patch_h * patch_w, 4, 4, 128)
        return self.norm(x)


class PatchMerge(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4, eps=1e-6)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_tokens = x.ndim == 3
        if is_tokens:
            batch, tokens, channels = x.shape
            side = math.isqrt(tokens)
            if side * side != tokens or side % 2:
                raise ValueError(f"cannot merge {tokens} tokens")
            x = x.reshape(batch, side // 2, 2, side // 2, 2, channels)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, -1, 2, 2, channels)
        x0, x1 = x[..., 0::2, 0::2, :], x[..., 1::2, 0::2, :]
        x2, x3 = x[..., 0::2, 1::2, :], x[..., 1::2, 1::2, :]
        x = self.reduction(self.norm(torch.cat((x0, x1, x2, x3), dim=-1)))
        return x[:, :, 0, 0, :] if is_tokens else x


class SARATRX(nn.Module):
    """Checkpoint-compatible 40-class HiViT with accessible 512-D features."""
    feature_dim = 512

    def __init__(self, num_classes: int = 40, input_size: int = 224) -> None:
        super().__init__()
        if input_size % 16:
            raise ValueError("SARATR-X input size must be divisible by 16")
        self.input_size = input_size
        self.patch_embed = PatchEmbed()
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, (input_size // 16) ** 2, 512))
        self.pos_drop = nn.Dropout(0.0)
        blocks: list[nn.Module] = []
        blocks.extend(Block(128, 0, 3.0) for _ in range(4))
        blocks.append(PatchMerge(128))
        blocks.extend(Block(256, 0, 3.0) for _ in range(4))
        blocks.append(PatchMerge(256))
        blocks.extend(Block(512, 8, 4.0) for _ in range(20))
        self.blocks = nn.ModuleList(blocks)
        self.fc_norm = nn.LayerNorm(512, eps=1e-6)
        self.head = nn.Sequential(
            nn.BatchNorm1d(512, affine=False, eps=1e-6),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks[:10]:
            x = block(x)
        x = x[..., 0, 0, :]
        pos = self.absolute_pos_embed
        if pos.shape[1] != x.shape[1]:
            old_side, new_side = math.isqrt(pos.shape[1]), math.isqrt(x.shape[1])
            pos = pos.reshape(1, old_side, old_side, 512).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, (new_side, new_side), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.pos_drop(x + pos)
        for block in self.blocks[10:]:
            x = block(x)
        return self.fc_norm(x.mean(dim=1))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.head(features)
        return (logits, features) if return_features else logits


def _resize_absolute_position(state: dict[str, torch.Tensor], target_tokens: int) -> None:
    position = state.get("absolute_pos_embed")
    if position is None or position.shape[1] == target_tokens:
        return
    old_side, new_side = math.isqrt(position.shape[1]), math.isqrt(target_tokens)
    if old_side * old_side != position.shape[1] or new_side * new_side != target_tokens:
        raise ValueError(f"cannot resize absolute position {position.shape[1]} -> {target_tokens}")
    position = position.reshape(1, old_side, old_side, 512).permute(0, 3, 1, 2)
    position = F.interpolate(position, (new_side, new_side), mode="bicubic", align_corners=False)
    state["absolute_pos_embed"] = position.permute(0, 2, 3, 1).flatten(1, 2)


def load_saratrx(checkpoint: Path, device: torch.device | str = "cpu",
                  freeze: bool = True, input_size: int | None = None) -> SARATRX:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if input_size is None:
        input_size = int(payload.get("input_size", 224)) if isinstance(payload, dict) else 224
    model = SARATRX(len(SOC40_CLASSES), input_size=input_size)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported SARATR-X checkpoint object: {type(state)!r}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    _resize_absolute_position(state, model.absolute_pos_embed.shape[1])
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"head.0.num_batches_tracked"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def saratrx_input(roi: torch.Tensor, input_size: int = 64) -> torch.Tensor:
    """Convert a Bx1xHxW [-1,1] cut ROI to the classifier's direct input."""
    if roi.ndim != 4 or roi.shape[1] != 1:
        raise ValueError(f"expected Bx1xHxW ROI, got {tuple(roi.shape)}")
    image = F.interpolate((roi + 1.0) * 0.5, (input_size, input_size), mode="bilinear", align_corners=False)
    return image.repeat(1, 3, 1, 1)

