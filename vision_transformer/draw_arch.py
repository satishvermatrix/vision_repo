"""Overview ViT figure matching vision_transformer/vit.py (MNIST)."""

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt

OUT = "vit_arch.png"

EDGE = "#1C2833"
MUTED = "#5D6D7E"
C = {
    "in": "#E8F1FF",
    "embed": "#D6F0E0",
    "pos": "#FFF3D6",
    "cls": "#F3E6FF",
    "enc": "#FDE2E2",
    "attn": "#E2EEFD",
    "mlp": "#E8E8E8",
    "head": "#D9F7F1",
}


def box(ax, x, y, w, h, text, fc, fs=9, sub=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            linewidth=1.2, edgecolor=EDGE, facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2 + (0.08 if sub else 0), text,
            ha="center", va="center", fontsize=fs, fontweight="bold", color=EDGE)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.16, sub,
                ha="center", va="center", fontsize=7.4, color=MUTED)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.15, color=EDGE))


def main():
    fig, ax = plt.subplots(figsize=(14.6, 9.0), dpi=160)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(7.3, 8.65, "Vision Transformer (ViT)  —  MNIST from-scratch toy model",
            ha="center", fontsize=16, fontweight="bold", color=EDGE)
    ax.text(7.3, 8.28,
            "Matches vit.py   ·   28×28 grayscale   ·   patch 7   ·   N=16 patches   ·   D=64   ·   4 heads   ·   4 encoder blocks   ·   10 classes",
            ha="center", fontsize=8.5, color=MUTED)

    # 1 input
    ax.text(1.85, 7.88, "1. Image → patches", ha="center", fontsize=10.5, fontweight="bold")
    ax.add_patch(Rectangle((1.15, 7.15), 0.7, 0.7, facecolor="#111", edgecolor=EDGE, lw=1))
    ax.text(1.5, 6.98, "28×28×1", ha="center", fontsize=7.5, color=MUTED)
    # 4x4 patch grid
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((2.15 + j * 0.22, 7.18 + i * 0.16), 0.2, 0.14,
                                   facecolor="#dfe6e9", edgecolor=EDGE, lw=0.6))
    ax.text(2.6, 6.98, "4×4 = 16 patches", ha="center", fontsize=7.5, color=MUTED)

    box(ax, 0.55, 6.05, 2.6, 0.72, "PatchEmbedding", C["embed"],
        sub="Conv2d(1, 64, k=7, s=7)")
    box(ax, 0.55, 5.1, 2.6, 0.72, "tokens", C["in"],
        sub="flatten+transpose → (B, 16, 64)")
    arrow(ax, (1.85, 7.12), (1.85, 6.79))
    arrow(ax, (1.85, 6.05), (1.85, 5.84))

    # 2 cls + pos
    ax.text(5.15, 7.88, "2. CLS + position", ha="center", fontsize=10.5, fontweight="bold")
    box(ax, 3.85, 6.95, 2.6, 0.7, "cls_token", C["cls"],
        sub="Parameter (1, 1, 64)  expand B")
    box(ax, 3.85, 6.05, 2.6, 0.7, "cat on seq dim", C["cls"],
        sub="(B, 16, 64) → (B, 17, 64)")
    box(ax, 3.85, 5.1, 2.6, 0.72, "+ position_embedding", C["pos"],
        sub="Parameter (1, 17, 64)  learned 1D pos")
    arrow(ax, (5.15, 6.95), (5.15, 6.77))
    arrow(ax, (5.15, 6.05), (5.15, 5.84))
    arrow(ax, (3.17, 5.46), (3.83, 5.46))

    # 3 encoder
    ax.text(9.15, 7.88, "3. Encoder block  (×4)", ha="center", fontsize=10.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((6.75, 5.0), 4.8, 2.7,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                lw=1.35, edgecolor=EDGE, facecolor="#FAFBFC"))
    box(ax, 6.95, 6.85, 4.4, 0.62, "Pre-norm + MHA (4 heads)", C["attn"],
        sub="x = x + MHA(LN1(x))     Q=K=V")
    box(ax, 6.95, 5.95, 4.4, 0.62, "Pre-norm + MLP", C["mlp"],
        sub="Linear(64→128) → GELU → Linear(128→64)")
    box(ax, 6.95, 5.15, 4.4, 0.55, "output still (B, 17, 64)", C["enc"],
        sub="seq = CLS + 16 patches")
    arrow(ax, (9.15, 6.85), (9.15, 6.59))
    arrow(ax, (9.15, 5.95), (9.15, 5.72))
    arrow(ax, (6.47, 5.46), (6.73, 5.46))

    ax.annotate("PRE-norm\n(LN before attn/MLP)\n≠ TimeSformer post-norm",
                xy=(11.65, 6.55), fontsize=7.2, color="#922B21", ha="left")

    # 4 head
    ax.text(13.15, 7.88, "4. Classify", ha="center", fontsize=10.5, fontweight="bold")
    box(ax, 11.85, 6.85, 2.5, 0.7, "take CLS", C["cls"],
        sub="x[:, 0]  → (B, 64)")
    box(ax, 11.85, 5.95, 2.5, 0.7, "MLP_Head", C["head"],
        sub="LayerNorm → Linear(64, 10)")
    box(ax, 11.85, 5.1, 2.5, 0.7, "logits (B, 10)", C["head"],
        sub="CrossEntropy  digits 0–9")
    arrow(ax, (13.1, 6.85), (13.1, 6.67))
    arrow(ax, (13.1, 5.95), (13.1, 5.82))
    arrow(ax, (11.57, 5.46), (11.83, 5.46))

    # bottom notes
    ax.add_patch(FancyBboxPatch((0.45, 2.55), 13.7, 2.2,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                lw=1.1, edgecolor=EDGE, facecolor="#F4F6F7"))
    ax.text(7.3, 4.48, "How this ViT differs from the TimeSformer in video_timesformer/",
            ha="center", fontsize=10, fontweight="bold", color=EDGE)
    ax.text(7.3, 3.95,
            "One image, not T frames.  One attention (space only).  One position table (not time+space).  Trained from scratch on MNIST — no ImageNet copy.",
            ha="center", fontsize=8.3, color=MUTED)
    ax.text(7.3, 3.5,
            "Encoder is PRE-norm:  residual = x;  x = LN(x);  x = module(x);  x = x + residual.   TimeSformer is POST-norm:  x = x + LN(module(x)).",
            ha="center", fontsize=8.3, color="#1A5276")
    ax.text(7.3, 3.05,
            "MLP width is 128 (2×D), not 4×D.  CLS is sliced in VisionTransformer.forward before MLP_Head (the commented x[:,0] inside the head is unused).",
            ha="center", fontsize=8.3, color=MUTED)
    ax.text(7.3, 2.68,
            "Optimizer Adam lr=3e-4, 5 epochs, batch 64.  Val set is MNIST test split.  predictions.png shows 10 val digits.",
            ha="center", fontsize=8.3, color=MUTED)

    ax.text(0.5, 2.25, "Tensor shapes  (B=64)", ha="left", fontsize=10, fontweight="bold")
    rows = [
        ("MNIST image", "(B, 1, 28, 28)"),
        ("Conv2d patch embed", "(B, 64, 4, 4)"),
        ("tokens", "(B, 16, 64)"),
        ("after CLS + pos", "(B, 17, 64)"),
        ("MHA sees", "(B, 17, 64)  seq=17"),
        ("CLS for class", "(B, 64)"),
        ("logits", "(B, 10)"),
    ]
    y = 1.95
    for name, shp in rows:
        ax.text(0.55, y, name, fontsize=8, color=EDGE)
        ax.text(4.3, y, shp, fontsize=8, family="monospace", color=MUTED)
        y -= 0.22

    ax.text(8.4, 1.95, "Sizes in vit.py", fontsize=10, fontweight="bold", color=EDGE)
    specs = [
        "img_size=28   patch_size=7   patch_num=16",
        "embed_dim=64   attention_heads=4",
        "transformer_blocks=4   mlp_nodes=128",
        "num_classes=10   num_channels=1",
        "head dim per attn head = 64/4 = 16",
    ]
    y = 1.65
    for s in specs:
        ax.text(8.4, y, s, fontsize=8, family="monospace", color=MUTED)
        y -= 0.22

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
