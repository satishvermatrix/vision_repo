"""Render a detailed TimeSformer architecture figure matching timesformer.py."""

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt

OUT = "timesformer_arch.png"

# paper-style palette
C = {
    "input": "#E8F1FF",
    "embed": "#D6F0E0",
    "pos": "#FFF3D6",
    "cls": "#F3E6FF",
    "temp": "#FDE2E2",
    "spat": "#E2EEFD",
    "mlp": "#E8E8E8",
    "head": "#D9F7F1",
    "edge": "#2C3E50",
    "muted": "#5D6D7E",
}


def box(ax, x, y, w, h, text, facecolor, fontsize=9, subtitle=None):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=C["edge"],
        facecolor=facecolor,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2 + (0.07 if subtitle else 0),
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=C["edge"],
        wrap=True,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.14,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
            color=C["muted"],
        )
    return x + w / 2, y


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=C["edge"],
        )
    )


def mini_frames(ax, x, y):
    """Stack of T RGB frames."""
    for i, shift in enumerate((0.0, 0.12, 0.24, 0.36)):
        r = Rectangle(
            (x + shift, y + shift * 0.35),
            0.55,
            0.42,
            linewidth=1.0,
            edgecolor=C["edge"],
            facecolor=["#f8d7da", "#d4edda", "#cce5ff", "#fff3cd"][i],
        )
        ax.add_patch(r)
    ax.text(x + 0.55, y - 0.12, "T frames  (B, T, 3, 224, 224)", ha="center", fontsize=8, color=C["muted"])


def main():
    fig, ax = plt.subplots(figsize=(14.5, 9.2), dpi=160)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        7.25,
        8.85,
        "TimeSformer  —  divided space-time video transformer",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=C["edge"],
    )
    ax.text(
        7.25,
        8.48,
        "This figure matches timesformer.py   ·   T=10 frames @ 1 FPS   ·   224×224   ·   patch 16   ·   N=196   ·   D=768   ·   12 heads   ·   12 blocks",
        ha="center",
        va="center",
        fontsize=8.5,
        color=C["muted"],
    )

    # ---- column 1: input + patch ----
    ax.text(1.7, 8.05, "1. Input video", ha="center", fontsize=10, fontweight="bold", color=C["edge"])
    mini_frames(ax, 1.15, 7.25)
    box(ax, 0.55, 6.35, 2.3, 0.7, "Patch embed", C["embed"], subtitle="Conv2d(3, 768, k=16, s=16)")
    arrow(ax, 1.7, 7.22, 1.7, 7.08)
    ax.text(1.7, 6.18, "flatten each frame → 14×14 = 196 patches", ha="center", fontsize=7.5, color=C["muted"])

    box(ax, 0.45, 5.25, 2.5, 0.75, "Tokens", C["input"], subtitle="(B, T, N, D) = (B, 10, 196, 768)")
    arrow(ax, 1.7, 6.35, 1.7, 6.02)

    # ---- column 2: embeddings + cls ----
    ax.text(4.7, 8.05, "2. Add identity", ha="center", fontsize=10, fontweight="bold", color=C["edge"])
    box(ax, 3.5, 7.15, 2.4, 0.7, "space_embed", C["pos"], subtitle="(1, 196, 768)  per patch location")
    box(ax, 3.5, 6.25, 2.4, 0.7, "time_embed", C["pos"], subtitle="(1, T+1, 768)  per frame + CLS")
    box(ax, 3.5, 5.25, 2.4, 0.75, "x = x + time + space", C["pos"], subtitle="broadcast over T and N")
    arrow(ax, 4.7, 7.15, 4.7, 6.97)
    arrow(ax, 4.7, 6.25, 4.7, 6.02)

    box(ax, 3.5, 4.15, 2.4, 0.85, "Prepend CLS", C["cls"], subtitle="cat on time axis → (B, T+1, N, D)")
    arrow(ax, 4.7, 5.25, 4.7, 5.02)

    # flow from tokens into add
    arrow(ax, 2.95, 5.62, 3.48, 5.62)

    # ---- column 3: one block exploded ----
    ax.text(8.55, 8.05, "3. One TimeSformerBlock  (×12)", ha="center", fontsize=10, fontweight="bold", color=C["edge"])

    block = FancyBboxPatch(
        (6.2, 3.95),
        4.7,
        3.95,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.4,
        edgecolor=C["edge"],
        facecolor="#FAFBFC",
        linestyle="-",
    )
    ax.add_patch(block)

    box(ax, 6.45, 6.95, 4.2, 0.72, "Temporal attention", C["temp"], subtitle="rearrange (B,T,N,D) → (B·N, T, D)   same patch, all frames")
    box(ax, 6.45, 5.95, 4.2, 0.72, "Spatial attention", C["spat"], subtitle="rearrange (B,T,N,D) → (B·T, N, D)   same frame, all patches")
    box(ax, 6.45, 4.95, 4.2, 0.72, "MLP", C["mlp"], subtitle="Linear(D, 4D) → GELU → Linear(4D, D)   residual + LayerNorm")
    arrow(ax, 8.55, 6.95, 8.55, 6.69)
    arrow(ax, 8.55, 5.95, 8.55, 5.69)
    ax.text(8.55, 4.72, "each step:  x ← x + LayerNorm(module(x))", ha="center", fontsize=7.5, color=C["muted"])

    arrow(ax, 5.92, 4.55, 6.18, 4.55)

    # residual notes on the right of the block
    ax.annotate(
        "seq length T+1 ≈ 11\ncheap motion path",
        xy=(10.85, 7.3),
        fontsize=7,
        color="#922B21",
        ha="left",
    )
    ax.annotate(
        "seq length N = 196\nViT-like appearance",
        xy=(10.85, 6.3),
        fontsize=7,
        color="#1A5276",
        ha="left",
    )

    # ---- column 4: head ----
    ax.text(12.7, 8.05, "4. Classify", ha="center", fontsize=10, fontweight="bold", color=C["edge"])
    box(ax, 11.45, 6.85, 2.55, 0.85, "Take CLS", C["cls"], subtitle="x[:, 0, 0]  →  (B, 768)")
    box(ax, 11.45, 5.75, 2.55, 0.8, "LayerNorm + head", C["head"], subtitle="Linear(768, num_classes)")
    box(ax, 11.45, 4.55, 2.55, 0.85, "Logits", C["head"], subtitle="(B, 4)  softmax → class")
    arrow(ax, 12.72, 6.85, 12.72, 6.57)
    arrow(ax, 12.72, 5.75, 12.72, 5.42)
    arrow(ax, 10.92, 5.3, 11.42, 5.3)

    # ---- bottom: ViT transfer + shapes ----
    transfer = FancyBboxPatch(
        (0.45, 2.55),
        13.6,
        1.2,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.1,
        edgecolor=C["edge"],
        facecolor="#F4F6F7",
    )
    ax.add_patch(transfer)
    ax.text(7.25, 3.48, "Weight transfer from ImageNet ViT-B/16  (timm vit_base_patch16_224)", ha="center", fontsize=10, fontweight="bold", color=C["edge"])
    ax.text(
        7.25,
        3.05,
        "COPIED into TimeSformer:  patch_embed    ·    spatial_attn QKV + out_proj    ·    MLP fc1/fc2",
        ha="center",
        fontsize=8.5,
        color="#196F3D",
    )
    ax.text(
        7.25,
        2.72,
        "LEFT RANDOM (video-specific):  temporal_attn    ·    time_embed    ·    classification head    ·    then fine-tune all weights at lr=1e-5",
        ha="center",
        fontsize=8.5,
        color="#922B21",
    )

    # shape table
    ax.text(0.55, 2.25, "Tensor shapes through a forward pass  (B=2, T=10)", ha="left", fontsize=10, fontweight="bold", color=C["edge"])

    rows = [
        ("input clip", "(B, T, C, H, W)", "(2, 10, 3, 224, 224)"),
        ("merge time into batch", "(B·T, C, H, W)", "(20, 3, 224, 224)"),
        ("Conv2d patch embed", "(B·T, D, 14, 14)", "(20, 768, 14, 14)"),
        ("tokens", "(B, T, N, D)", "(2, 10, 196, 768)"),
        ("after CLS", "(B, T+1, N, D)", "(2, 11, 196, 768)"),
        ("temporal attn sees", "(B·N, T+1, D)", "(392, 11, 768)"),
        ("spatial attn sees", "(B·(T+1), N, D)", "(22, 196, 768)"),
        ("CLS used for class", "(B, D)", "(2, 768)"),
        ("logits", "(B, classes)", "(2, 4)"),
    ]
    col_x = [0.55, 4.3, 9.3]
    y = 1.95
    ax.text(col_x[0], y, "stage", fontsize=8, fontweight="bold", color=C["muted"])
    ax.text(col_x[1], y, "layout", fontsize=8, fontweight="bold", color=C["muted"])
    ax.text(col_x[2], y, "example", fontsize=8, fontweight="bold", color=C["muted"])
    y -= 0.18
    for name, layout, ex in rows:
        ax.text(col_x[0], y, name, fontsize=8, color=C["edge"])
        ax.text(col_x[1], y, layout, fontsize=8, family="monospace", color=C["edge"])
        ax.text(col_x[2], y, ex, fontsize=8, family="monospace", color=C["muted"])
        y -= 0.17

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
