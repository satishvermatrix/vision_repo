"""Detailed TimeSformer encoder-block figure matching timesformer.py."""

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.pyplot as plt

OUT = "timesformer_encoder.png"

EDGE = "#1C2833"
MUTED = "#5D6D7E"
WHITE = "#FFFFFF"

PAL = {
    "token": "#D6EAF8",
    "pos": "#FCF3CF",
    "cls": "#E8DAEF",
    "temp": "#FADBD8",
    "spat": "#D4E6F1",
    "mlp": "#E5E8E8",
    "ln": "#FDEBD0",
    "add": "#D5F5E3",
    "head": "#D5F5E3",
    "panel": "#FBFCFC",
    "mha": "#F5B7B1",
    "mha2": "#AED6F1",
}


def rbox(ax, x, y, w, h, text, fc, fs=8.2, sub=None, lw=1.15):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.07",
        linewidth=lw,
        edgecolor=EDGE,
        facecolor=fc,
    )
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=EDGE)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.6, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=EDGE)
    return x + w / 2, y + h / 2


def arr(ax, a, b, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=EDGE,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def skip(ax, x, y1, y2, label, side=-0.55):
    """Vertical residual skip on the left or right of a column."""
    ax.add_patch(
        FancyArrowPatch(
            (x, y1),
            (x, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.35,
            color="#1E8449",
            connectionstyle="arc3,rad=0",
        )
    )
    ax.text(x + side, (y1 + y2) / 2, label, ha="center", va="center", fontsize=7, color="#196F3D", rotation=90)


def plus(ax, x, y, r=0.13):
    c = Circle((x, y), r, facecolor=PAL["add"], edgecolor=EDGE, linewidth=1.2, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=11, fontweight="bold", color=EDGE, zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(16.2, 11.4), dpi=160)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 11.4)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    ax.text(
        8.1,
        11.12,
        "TimeSformer encoder internals  —  CLS, positional embeddings, one block, classification head",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=EDGE,
    )
    ax.text(
        8.1,
        10.78,
        "Matches timesformer.py exactly: post-norm residual  x ← x + LayerNorm(module(x))   ·   12 heads   ·   D=768   ·   T=10   ·   N=196",
        ha="center",
        fontsize=8.3,
        color=MUTED,
    )

    # =====================================================================
    # LEFT: stem — patches, pos embeds, CLS
    # =====================================================================
    stem = FancyBboxPatch((0.22, 0.28), 4.55, 10.25, boxstyle="round,pad=0.02,rounding_size=0.1", lw=1.3, edgecolor=EDGE, facecolor="#F8FBFF")
    ax.add_patch(stem)
    ax.text(2.5, 10.28, "A. Input + embeddings + CLS", ha="center", fontsize=11, fontweight="bold", color=EDGE)

    rbox(ax, 0.5, 9.45, 4.0, 0.62, "Video clip", PAL["token"], sub="(B, T, 3, 224, 224)   T=10 @ 1 FPS")
    rbox(ax, 0.5, 8.62, 4.0, 0.62, "view → (B·T, 3, 224, 224)", PAL["token"], sub="merge time into batch for Conv2d")
    rbox(ax, 0.5, 7.78, 4.0, 0.62, "patch_embed  Conv2d", PAL["token"], sub="in=3, out=D=768, kernel=16, stride=16")
    rbox(ax, 0.5, 6.95, 4.0, 0.62, "flatten + transpose", PAL["token"], sub="(B·T, 768, 14, 14) → (B·T, N=196, 768)")
    rbox(ax, 0.5, 6.12, 4.0, 0.62, "restore time axis", PAL["token"], sub="tokens x :  (B, T, N, D)")
    for y1, y2 in ((9.45, 9.24), (8.62, 8.42), (7.78, 7.57), (6.95, 6.74)):
        arr(ax, (2.5, y1), (2.5, y2))

    ax.text(2.5, 5.88, "separate learned positional tables (added, not concat)", ha="center", fontsize=7, color=MUTED)
    rbox(ax, 0.5, 5.08, 4.0, 0.62, "space_embed", PAL["pos"], sub="Parameter (1, N, D)  —  which patch on the frame")
    rbox(ax, 0.5, 4.25, 4.0, 0.62, "time_embed", PAL["pos"], sub="Parameter (1, T+1, D)  —  which frame, + slot 0 for CLS")
    rbox(ax, 0.5, 3.38, 4.0, 0.68, "x ← x + time[:, 1:T+1] + space", PAL["pos"], sub="broadcast: time over N, space over T")
    arr(ax, (2.5, 6.12), (2.5, 5.72))
    arr(ax, (2.5, 5.08), (2.5, 4.89))
    arr(ax, (2.5, 4.25), (2.5, 4.08))

    ax.text(2.5, 3.12, "CLS is an extra time step, copied across all N patches", ha="center", fontsize=7, color=MUTED)
    rbox(ax, 0.5, 2.28, 4.0, 0.68, "cls_token", PAL["cls"], sub="Parameter (1, 1, 1, D)  expand → (B, 1, N, D)")
    rbox(ax, 0.5, 1.42, 4.0, 0.62, "cls ← cls + time_embed[:, 0]", PAL["cls"], sub="CLS lives at time index 0")
    rbox(ax, 0.5, 0.48, 4.0, 0.72, "cat([cls, x], dim=1)", PAL["cls"], sub="(B, T+1, N, D) = (B, 11, 196, 768)")
    arr(ax, (2.5, 3.38), (2.5, 2.98))
    arr(ax, (2.5, 2.28), (2.5, 2.06))
    arr(ax, (2.5, 1.42), (2.5, 1.22))

    # =====================================================================
    # CENTER: one encoder block fully exploded
    # =====================================================================
    blk = FancyBboxPatch((5.0, 0.28), 6.55, 10.25, boxstyle="round,pad=0.02,rounding_size=0.1", lw=1.3, edgecolor=EDGE, facecolor="#FFF9F8")
    ax.add_patch(blk)
    ax.text(8.28, 10.28, "B. One TimeSformerBlock  (stacked ×12)", ha="center", fontsize=11, fontweight="bold", color=EDGE)

    rbox(ax, 5.85, 9.48, 4.4, 0.55, "x  in   (B, T+1, N, D)", PAL["token"], fs=8.5)
    arr(ax, (8.05, 9.48), (8.05, 9.28))

    # --- temporal ---
    ax.text(8.28, 9.12, "①  Temporal attention  —  same patch, all frames", ha="center", fontsize=8.5, fontweight="bold", color="#922B21")
    rbox(ax, 5.85, 8.38, 4.4, 0.55, "rearrange", PAL["temp"], sub="(B, T, N, D) → (B·N, T, D)     seq length = T+1 ≈ 11")
    rbox(ax, 5.85, 7.58, 4.4, 0.62, "MultiheadAttention  (12 heads)", PAL["mha"], sub="Q = K = V = xt     self-attn     out_proj back to D")
    rbox(ax, 5.85, 6.82, 4.4, 0.52, "rearrange back", PAL["temp"], sub="(B·N, T, D) → (B, T, N, D)")
    rbox(ax, 5.85, 6.12, 2.05, 0.48, "LayerNorm  D", PAL["ln"], sub="norm1")
    plus(ax, 8.55, 6.36)
    ax.text(9.05, 6.36, "residual-add", fontsize=7, color="#196F3D", va="center")
    rbox(ax, 9.55, 6.12, 1.55, 0.48, "x'", PAL["add"], sub="same shape")

    arr(ax, (8.05, 8.38), (8.05, 8.22))
    arr(ax, (8.05, 7.58), (8.05, 7.36))
    arr(ax, (8.05, 6.82), (8.05, 6.62))
    arr(ax, (7.9, 6.36), (8.42, 6.36))
    arr(ax, (8.68, 6.36), (9.52, 6.36))

    # skip from x in to plus
    ax.annotate(
        "",
        xy=(8.42, 6.48),
        xytext=(10.95, 9.55),
        arrowprops=dict(arrowstyle="-|>", color="#1E8449", lw=1.25, connectionstyle="angle,angleA=180,angleB=-90,rad=0"),
    )
    ax.text(11.05, 7.95, "skip\nx", fontsize=7, color="#196F3D", ha="left")

    ax.text(8.28, 5.92, "x  ←  x  +  norm1( temporal_attn(x) )", ha="center", fontsize=7.4, family="monospace", color="#922B21")

    # --- spatial ---
    ax.text(8.28, 5.68, "②  Spatial attention  —  same frame, all patches", ha="center", fontsize=8.5, fontweight="bold", color="#1A5276")
    rbox(ax, 5.85, 4.95, 4.4, 0.55, "rearrange", PAL["spat"], sub="(B, T, N, D) → (B·T, N, D)     seq length = N = 196")
    rbox(ax, 5.85, 4.15, 4.4, 0.62, "MultiheadAttention  (12 heads)", PAL["mha2"], sub="Q = K = V = xs     self-attn     (ViT attention, ImageNet weights)")
    rbox(ax, 5.85, 3.42, 4.4, 0.5, "rearrange back", PAL["spat"], sub="(B·T, N, D) → (B, T, N, D)")
    rbox(ax, 5.85, 2.72, 2.05, 0.48, "LayerNorm  D", PAL["ln"], sub="norm2")
    plus(ax, 8.55, 2.96)
    ax.text(9.05, 2.96, "residual-add", fontsize=7, color="#196F3D", va="center")
    rbox(ax, 9.55, 2.72, 1.55, 0.48, "x''", PAL["add"], sub="same shape")

    arr(ax, (8.05, 6.12), (8.05, 5.52))
    arr(ax, (8.05, 4.95), (8.05, 4.79))
    arr(ax, (8.05, 4.15), (8.05, 3.94))
    arr(ax, (8.05, 3.42), (8.05, 3.22))
    arr(ax, (7.9, 2.96), (8.42, 2.96))
    arr(ax, (8.68, 2.96), (9.52, 2.96))

    ax.text(8.28, 2.52, "x  ←  x  +  norm2( spatial_attn(x) )", ha="center", fontsize=7.4, family="monospace", color="#1A5276")

    # --- mlp ---
    ax.text(8.28, 2.28, "③  MLP  (channel mix, no attention)", ha="center", fontsize=8.5, fontweight="bold", color="#4D5656")
    rbox(ax, 5.85, 1.55, 1.35, 0.52, "Linear", PAL["mlp"], sub="D → 4D")
    rbox(ax, 7.32, 1.55, 1.2, 0.52, "GELU", PAL["mlp"])
    rbox(ax, 8.64, 1.55, 1.6, 0.52, "Linear", PAL["mlp"], sub="4D → D")
    arr(ax, (7.2, 1.81), (7.32, 1.81))
    arr(ax, (8.52, 1.81), (8.64, 1.81))

    rbox(ax, 5.85, 0.72, 2.05, 0.52, "LayerNorm  D", PAL["ln"], sub="norm3")
    plus(ax, 8.55, 0.98)
    ax.text(9.05, 0.98, "residual-add", fontsize=7, color="#196F3D", va="center")
    rbox(ax, 9.55, 0.72, 1.55, 0.52, "x out", PAL["add"], sub="→ next block")

    arr(ax, (8.05, 1.55), (8.05, 1.26))
    arr(ax, (7.9, 0.98), (8.42, 0.98))
    arr(ax, (8.68, 0.98), (9.52, 0.98))

    ax.text(8.28, 0.48, "x  ←  x  +  norm3( MLP(x) )     then repeat block ×12", ha="center", fontsize=7.4, family="monospace", color="#4D5656")

    # MHA callout
    ax.text(
        8.28,
        7.22,
        "inside MHA: split D into 12 heads of 64  ·  softmax(QKᵀ / √64) V  ·  concat heads  ·  out_proj",
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )

    # =====================================================================
    # RIGHT: classification head
    # =====================================================================
    headp = FancyBboxPatch((11.75, 0.28), 4.2, 10.25, boxstyle="round,pad=0.02,rounding_size=0.1", lw=1.3, edgecolor=EDGE, facecolor="#F4FBF8")
    ax.add_patch(headp)
    ax.text(13.85, 10.28, "C. Classification head", ha="center", fontsize=11, fontweight="bold", color=EDGE)

    rbox(ax, 12.0, 9.35, 3.7, 0.7, "after 12 blocks", PAL["token"], sub="x still (B, T+1, N, D)")
    rbox(ax, 12.0, 8.35, 3.7, 0.78, "take CLS token", PAL["cls"], sub="x[:, 0, 0]   time=0, patch=0   → (B, D)")
    rbox(ax, 12.0, 7.35, 3.7, 0.72, "final LayerNorm", PAL["ln"], sub="self.norm  over last dim D")
    rbox(ax, 12.0, 6.28, 3.7, 0.78, "head  Linear(D, C)", PAL["head"], sub="C = num_classes = 4     this is the class MLP")
    rbox(ax, 12.0, 5.22, 3.7, 0.72, "logits  (B, 4)", PAL["head"], sub="CrossEntropyLoss vs label")
    rbox(ax, 12.0, 4.22, 3.7, 0.7, "softmax  (eval only)", PAL["head"], sub="pred = argmax    conf = max prob")

    for a, b in ((9.35, 9.15), (8.35, 8.09), (7.35, 7.08), (6.28, 5.96), (5.22, 4.94)):
        arr(ax, (13.85, a), (13.85, b))

    ax.text(13.85, 3.88, "Why [:, 0, 0]?", ha="center", fontsize=8.5, fontweight="bold", color=EDGE)
    ax.text(
        13.85,
        3.15,
        "CLS was prepended as time index 0\nand expanded over every patch.\nThe classifier reads only patch 0\nof that CLS time step — one vector\nthat has mixed space and time\nthrough 12 encoder blocks.",
        ha="center",
        va="top",
        fontsize=7.6,
        color=MUTED,
    )

    ax.text(13.85, 1.85, "What is NOT in the block", ha="center", fontsize=8.5, fontweight="bold", color=EDGE)
    ax.text(
        13.85,
        1.15,
        "No dropout in this file.\nNo pre-norm (ViT often does\nx + attn(LN(x)); we do\nx + LN(attn(x)) — post-norm).\nTemporal attn is random init.\nSpatial attn + MLP copied from ViT.",
        ha="center",
        va="top",
        fontsize=7.6,
        color=MUTED,
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
