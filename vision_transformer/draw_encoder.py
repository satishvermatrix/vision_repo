"""Detailed ViT encoder-block figure matching vision_transformer/vit.py."""

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.pyplot as plt

OUT = "vit_encoder.png"
EDGE = "#1C2833"
MUTED = "#5D6D7E"
WHITE = "#FFFFFF"
PAL = {
    "token": "#D6EAF8",
    "pos": "#FCF3CF",
    "cls": "#E8DAEF",
    "attn": "#D4E6F1",
    "mha": "#AED6F1",
    "mlp": "#E5E8E8",
    "ln": "#FDEBD0",
    "add": "#D5F5E3",
    "head": "#D5F5E3",
}


def rbox(ax, x, y, w, h, text, fc, fs=8.2, sub=None):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.07",
                       linewidth=1.15, edgecolor=EDGE, facecolor=fc)
    )
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=EDGE)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=6.6, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=EDGE)


def arr(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.15, color=EDGE))


def plus(ax, x, y, r=0.13):
    ax.add_patch(Circle((x, y), r, facecolor=PAL["add"], edgecolor=EDGE, lw=1.2, zorder=5))
    ax.text(x, y, "+", ha="center", va="center", fontsize=11, fontweight="bold",
            color=EDGE, zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(16.2, 11.4), dpi=160)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 11.4)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    ax.text(8.1, 11.12, "ViT encoder internals  —  CLS, positional embedding, TransformerEncoder, MLP head",
            ha="center", fontsize=14.5, fontweight="bold", color=EDGE)
    ax.text(8.1, 10.78,
            "Matches vit.py exactly: PRE-norm residual   x ← x + module(LayerNorm(x))   ·   4 heads   ·   D=64   ·   N=16   ·   4 blocks   ·   MNIST 10-way",
            ha="center", fontsize=8.3, color=MUTED)

    # A stem
    ax.add_patch(FancyBboxPatch((0.22, 0.28), 4.55, 10.25,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                lw=1.3, edgecolor=EDGE, facecolor="#F8FBFF"))
    ax.text(2.5, 10.28, "A. Patches + CLS + position", ha="center", fontsize=11, fontweight="bold")

    rbox(ax, 0.5, 9.42, 4.0, 0.62, "MNIST image", PAL["token"],
         sub="(B, 1, 28, 28)  grayscale digit")
    rbox(ax, 0.5, 8.58, 4.0, 0.62, "PatchEmbedding  Conv2d", PAL["token"],
         sub="in=1, out=D=64, kernel=7, stride=7")
    rbox(ax, 0.5, 7.74, 4.0, 0.62, "flatten(2) + transpose(1,2)", PAL["token"],
         sub="(B, 64, 4, 4) → (B, N=16, 64)")
    rbox(ax, 0.5, 6.85, 4.0, 0.68, "cls_token", PAL["cls"],
         sub="Parameter (1, 1, 64)  expand → (B, 1, 64)")
    rbox(ax, 0.5, 5.95, 4.0, 0.68, "torch.cat([cls, patches], dim=1)", PAL["cls"],
         sub="seq dim: 1 + 16 = 17 tokens")
    rbox(ax, 0.5, 5.0, 4.0, 0.72, "position_embedding", PAL["pos"],
         sub="Parameter (1, 17, 64)  learned, one vector per index")
    rbox(ax, 0.5, 4.05, 4.0, 0.72, "x ← x + position_embedding", PAL["pos"],
         sub="index 0 = CLS position, 1..16 = patch grid")
    rbox(ax, 0.5, 3.1, 4.0, 0.7, "into encoder  (B, 17, 64)", PAL["token"],
         sub="same shape through all 4 blocks")

    for a, b in ((9.42, 9.22), (8.58, 8.38), (7.74, 7.55), (6.85, 6.65),
                 (5.95, 5.74), (5.0, 4.79), (4.05, 3.82)):
        arr(ax, (2.5, a), (2.5, b))

    ax.text(2.5, 2.75, "Positional embedding is 1D", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(2.5, 1.85,
            "Not 2D row/col tables.\nToken 5 is “the 5th patch\nin raster order”, not (row, col).\nCLS is always sequence index 0.\nNo time embedding — one image.",
            ha="center", va="top", fontsize=7.6, color=MUTED)

    # B encoder
    ax.add_patch(FancyBboxPatch((5.0, 0.28), 6.55, 10.25,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                lw=1.3, edgecolor=EDGE, facecolor="#FFF9F8"))
    ax.text(8.28, 10.28, "B. One TransformerEncoder  (stacked ×4)", ha="center", fontsize=11, fontweight="bold")

    rbox(ax, 5.85, 9.48, 4.4, 0.52, "x  in   (B, 17, 64)", PAL["token"])

    ax.text(8.28, 9.18, "①  Multi-head self-attention  (PRE-norm)", ha="center",
            fontsize=8.5, fontweight="bold", color="#1A5276")
    rbox(ax, 5.85, 8.42, 1.9, 0.55, "residual1 = x", PAL["add"], sub="save skip")
    rbox(ax, 8.0, 8.42, 2.25, 0.55, "LayerNorm  D", PAL["ln"], sub="layer_norm1")
    rbox(ax, 5.85, 7.55, 4.4, 0.65, "MultiheadAttention  (4 heads)", PAL["mha"],
         sub="Q = K = V = x     batch_first=True     head dim = 64/4 = 16")
    plus(ax, 8.55, 7.12)
    ax.text(9.05, 7.12, "residual-add", fontsize=7, color="#196F3D", va="center")
    rbox(ax, 9.55, 6.88, 1.55, 0.48, "x'", PAL["add"], sub="(B, 17, 64)")

    arr(ax, (8.05, 9.48), (8.05, 9.0))
    arr(ax, (7.75, 8.7), (8.0, 8.7))
    arr(ax, (8.05, 8.42), (8.05, 8.22))
    arr(ax, (8.05, 7.55), (8.05, 7.25))
    arr(ax, (8.68, 7.12), (9.52, 7.12))
    # skip residual1 to plus
    ax.annotate("", xy=(8.42, 7.12), xytext=(6.8, 8.42),
                arrowprops=dict(arrowstyle="-|>", color="#1E8449", lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(6.35, 7.7, "skip", fontsize=7, color="#196F3D")

    ax.text(8.28, 6.62, "x  ←  residual1  +  MHA( LN1(x) )", ha="center",
            fontsize=7.5, family="monospace", color="#1A5276")
    ax.text(8.28, 6.38,
            "inside MHA: 4 heads · softmax(QKᵀ / √16) V · concat heads · out_proj",
            ha="center", fontsize=6.6, color=MUTED)

    ax.text(8.28, 6.08, "②  MLP  (PRE-norm, channel mix)", ha="center",
            fontsize=8.5, fontweight="bold", color="#4D5656")
    rbox(ax, 5.85, 5.32, 1.9, 0.52, "residual2 = x", PAL["add"], sub="save skip")
    rbox(ax, 8.0, 5.32, 2.25, 0.52, "LayerNorm  D", PAL["ln"], sub="layer_norm2")
    rbox(ax, 5.85, 4.45, 1.35, 0.55, "Linear", PAL["mlp"], sub="64 → 128")
    rbox(ax, 7.35, 4.45, 1.15, 0.55, "GELU", PAL["mlp"])
    rbox(ax, 8.65, 4.45, 1.6, 0.55, "Linear", PAL["mlp"], sub="128 → 64")
    plus(ax, 8.55, 3.95)
    ax.text(9.05, 3.95, "residual-add", fontsize=7, color="#196F3D", va="center")
    rbox(ax, 9.55, 3.7, 1.55, 0.5, "x out", PAL["add"], sub="→ next block")

    arr(ax, (8.05, 6.88), (8.05, 5.86))
    arr(ax, (7.75, 5.58), (8.0, 5.58))
    arr(ax, (8.05, 5.32), (8.05, 5.02))
    arr(ax, (7.2, 4.72), (7.35, 4.72))
    arr(ax, (8.5, 4.72), (8.65, 4.72))
    arr(ax, (8.05, 4.45), (8.05, 4.1))
    arr(ax, (8.68, 3.95), (9.52, 3.95))
    ax.annotate("", xy=(8.42, 3.95), xytext=(6.8, 5.32),
                arrowprops=dict(arrowstyle="-|>", color="#1E8449", lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(6.35, 4.55, "skip", fontsize=7, color="#196F3D")

    ax.text(8.28, 3.42, "x  ←  residual2  +  MLP( LN2(x) )     then repeat ×4",
            ha="center", fontsize=7.5, family="monospace", color="#4D5656")

    ax.add_patch(FancyBboxPatch((5.85, 0.48), 4.4, 2.7,
                                boxstyle="round,pad=0.015,rounding_size=0.08",
                                lw=1.0, edgecolor=EDGE, facecolor="#F8F9F9"))
    ax.text(8.05, 2.92, "PRE-norm vs TimeSformer POST-norm", ha="center",
            fontsize=8.2, fontweight="bold", color="#922B21")
    ax.text(8.05, 1.55,
            "This ViT (vit.py):\n  residual = x\n  x = LayerNorm(x)\n  x = Attn/MLP(x)\n  x = x + residual\n\nTimeSformer:\n  x = x + LayerNorm(Attn/MLP(x))\n\nLN is BEFORE the module here.",
            ha="center", va="top", fontsize=7.4, color=MUTED)

    # C head
    ax.add_patch(FancyBboxPatch((11.75, 0.28), 4.2, 10.25,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                lw=1.3, edgecolor=EDGE, facecolor="#F4FBF8"))
    ax.text(13.85, 10.28, "C. Classification  (MLP_Head)", ha="center", fontsize=11, fontweight="bold")

    rbox(ax, 12.0, 9.35, 3.7, 0.7, "after 4 encoder blocks", PAL["token"],
         sub="x still (B, 17, 64)")
    rbox(ax, 12.0, 8.32, 3.7, 0.78, "take CLS  x[:, 0]", PAL["cls"],
         sub="in VisionTransformer.forward  → (B, 64)")
    rbox(ax, 12.0, 7.28, 3.7, 0.72, "MLP_Head.layernorm1", PAL["ln"],
         sub="LayerNorm(embed_dim=64)")
    rbox(ax, 12.0, 6.22, 3.7, 0.78, "MLP_Head.mlp_head", PAL["head"],
         sub="Linear(64, 10)   the class layer")
    rbox(ax, 12.0, 5.18, 3.7, 0.7, "logits  (B, 10)", PAL["head"],
         sub="CrossEntropyLoss vs digit label")
    rbox(ax, 12.0, 4.18, 3.7, 0.7, "argmax  → predicted digit", PAL["head"],
         sub="0 … 9     val accuracy on MNIST test")

    for a, b in ((9.35, 9.12), (8.32, 8.02), (7.28, 7.02), (6.22, 5.90), (5.18, 4.90)):
        arr(ax, (13.85, a), (13.85, b))

    ax.text(13.85, 3.78, "Note on x[:, 0]", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(13.85, 2.85,
            "MLP_Head.forward has a\ncommented  # x = x[:,0].\nThe slice already happened\nin VisionTransformer, so the\nhead receives (B, 64), not\n(B, 17, 64).",
            ha="center", va="top", fontsize=7.6, color=MUTED)

    ax.text(13.85, 1.55, "What is NOT here", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(13.85, 0.7,
            "No dropout. No time axis.\nNo divided attention.\nNo ImageNet init — random\nweights, trained 5 epochs\nAdam 3e-4 on MNIST.",
            ha="center", va="top", fontsize=7.6, color=MUTED)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
