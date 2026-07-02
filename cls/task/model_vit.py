"""
32x32 tour — VISION TRANSFORMER (CLIP-style, reused as the CLIP image encoder).

A plain ViT built the way CLIP's vision tower is: patchify -> prepend a learned CLS token
-> add a LEARNED positional embedding -> pre-LN transformer blocks -> take the CLS token
-> project. Attention is written by hand (no nn.MultiheadAttention / no fused SDPA) so the
QK^T-softmax-V is fully visible.

`VisionTransformer` is CLIP-compatible (same attribute names: class_embedding,
positional_embedding, ln_pre/ln_post, proj) and is imported directly by the CLIP model
(clip/model.py) as its `vision="vit"` encoder. For the tour, `output_dim` is the number of
classes, so the projection doubles as the classifier.

    from expirements.solution.model_vit import model_vit, VisionTransformer
    model = model_vit(size="resnet18")            # 32x32 -> 4x4 patches (8x8 grid + cls = 65)
"""
import torch
import torch.nn as nn

try:
    from train import NUM_CLASSES
except Exception:
    NUM_CLASSES = 100

# "resnet18/34" keys are only for interface parity with the rest of the tour.
CONFIGS = {
    "resnet18": dict(width=384, layers=9, heads=6),     # ~16M params (macro reference)
    "resnet34": dict(width=384, layers=12, heads=6),    # deeper variant
}
PATCH = 4
IMG = 32
MLP_RATIO = 4


class QuickGELU(nn.Module):
    """CLIP's activation: x * sigmoid(1.702 x)."""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    """Multi-head self-attention, written out (softmax(QK^T / sqrt(d)) V)."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape                                       # [batch, tokens, dim]
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)          # each [B, heads, L, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale          # [B, heads, L, L]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)       # merge heads -> [B, L, C]
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, dim, heads, mlp_ratio=MLP_RATIO):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), QuickGELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    """CLIP-style ViT. forward(image [N,3,H,W]) -> [N, output_dim]."""

    def __init__(self, image_size=IMG, patch_size=PATCH, width=384, layers=9, heads=6,
                 output_dim=NUM_CLASSES, mlp_ratio=MLP_RATIO):
        super().__init__()
        grid = image_size // patch_size
        n_tokens = grid * grid + 1                              # +1 for the CLS token
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_tokens, width))   # LEARNED
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, heads, mlp_ratio) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.patch_embed(x)                                # [N, width, grid, grid]
        x = x.flatten(2).transpose(1, 2)                       # [N, grid^2, width]
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)                         # [N, grid^2 + 1, width]
        x = x + self.positional_embedding                      # learned absolute positions
        x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_post(x[:, 0])                              # take the CLS token
        return x @ self.proj                                  # [N, output_dim]


def model_vit(size="resnet18", num_classes=None):
    """Tour ViT: the projection maps the CLS token straight to `num_classes` logits."""
    if num_classes is None:                      # resolve live (after init_train set it)
        try:
            from train import NUM_CLASSES as num_classes
        except Exception:
            num_classes = 100
    assert size in CONFIGS, f"size must be one of {set(CONFIGS)}"
    return VisionTransformer(image_size=IMG, patch_size=PATCH, output_dim=num_classes, **CONFIGS[size])


if __name__ == "__main__":
    for s in ("resnet18", "resnet34"):
        m = model_vit(s)
        p = sum(x.numel() for x in m.parameters()) / 1e6
        y = m(torch.zeros(2, 3, 32, 32))
        print(f"{s:9s}: {p:5.2f}M params, out {tuple(y.shape)}")