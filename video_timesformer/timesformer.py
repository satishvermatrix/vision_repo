import os
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_PATH = "data/frames"


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=10, image_size=224):
        self.root_dir = root_dir
        self.num_frames = num_frames

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"Dataset not found at {root_dir}. Expected:\n"
                "  root_dir/\n"
                "    class_a/\n"
                "      video_1/  frame0.jpg, frame1.jpg, ...\n"
                "    class_b/\n"
                "      ..."
            )

        self.classes = sorted(
            c
            for c in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, c)) and not c.startswith(".")
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for video in os.listdir(cls_path):
                video_path = os.path.join(cls_path, video)
                if os.path.isdir(video_path):
                    self.samples.append((video_path, self.class_to_idx[cls]))

        if not self.samples:
            raise FileNotFoundError(
                f"No video folders found under {root_dir}."
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = sorted(
            f
            for f in os.listdir(video_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not frames:
            raise FileNotFoundError(f"No image frames in {video_path}")

        if len(frames) < self.num_frames:
            frames = frames * (self.num_frames // len(frames) + 1)

        idxs = torch.linspace(0, len(frames) - 1, self.num_frames).long()
        selected = [frames[i] for i in idxs]

        imgs = []
        for f in selected:
            img = Image.open(os.path.join(video_path, f)).convert("RGB")
            imgs.append(self.transform(img))

        video = torch.stack(imgs)  # (T, C, H, W)
        return video, label


class TimeSformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        self.temporal_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        # Temporal attention (across frames for same patch)
        xt = rearrange(x, "b t n d -> (b n) t d")
        xt = self.temporal_attn(xt, xt, xt)[0]
        xt = rearrange(xt, "(b n) t d -> b t n d", b=B, n=N)
        x = x + self.norm1(xt)

        # Spatial attention (across patches within same frame)
        xs = rearrange(x, "b t n d -> (b t) n d")
        xs = self.spatial_attn(xs, xs, xs)[0]
        xs = rearrange(xs, "(b t) n d -> b t n d", b=B, t=T)
        x = x + self.norm2(xs)

        # MLP
        x = x + self.norm3(self.mlp(x))
        return x


class TimeSformer(nn.Module):
    def __init__(
        self,
        num_classes,
        num_frames=10,
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        heads=12,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.time_embed = nn.Parameter(torch.randn(1, num_frames + 1, embed_dim))
        self.space_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList(
            [TimeSformerBlock(embed_dim, heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D)
        x = x.view(B, T, -1, self.embed_dim)  # (B, T, N, D)

        x = x + self.time_embed[:, 1 : T + 1, None, :] + self.space_embed[:, None, :, :]

        cls = self.cls_token.expand(B, -1, self.num_patches, -1)
        cls = cls + self.time_embed[:, :1, None, :]
        x = torch.cat([cls, x], dim=1)  # (B, T+1, N, D)

        for block in self.blocks:
            x = block(x)

        cls_out = self.norm(x[:, 0, 0])
        return self.head(cls_out)


def load_vit_weights_into_timesformer(timesformer, vit):
    timesformer.patch_embed.weight.data.copy_(vit.patch_embed.proj.weight.data)
    timesformer.patch_embed.bias.data.copy_(vit.patch_embed.proj.bias.data)

    for ts_block, vit_block in zip(timesformer.blocks, vit.blocks):
        ts_block.spatial_attn.in_proj_weight.data.copy_(vit_block.attn.qkv.weight.data)
        ts_block.spatial_attn.in_proj_bias.data.copy_(vit_block.attn.qkv.bias.data)
        ts_block.spatial_attn.out_proj.weight.data.copy_(vit_block.attn.proj.weight.data)
        ts_block.spatial_attn.out_proj.bias.data.copy_(vit_block.attn.proj.bias.data)

        ts_block.mlp[0].weight.data.copy_(vit_block.mlp.fc1.weight.data)
        ts_block.mlp[0].bias.data.copy_(vit_block.mlp.fc1.bias.data)
        ts_block.mlp[2].weight.data.copy_(vit_block.mlp.fc2.weight.data)
        ts_block.mlp[2].bias.data.copy_(vit_block.mlp.fc2.bias.data)

    print("Loaded ImageNet ViT weights into TimeSformer (spatial only)")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(loader, desc="train"):
        videos = videos.to(device)
        labels = labels.to(device)

        logits = model(videos)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def report_predictions(model, dataset, device, out_path="predictions.png", per_class=3):
    model.eval()
    class_names = dataset.classes
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    results = []
    print("\n=== Predictions by video ===")
    print(f"{'clip':<24} {'true':<24} {'pred':<24} {'ok':<4} {'conf':>6}")
    for i, (videos, labels) in enumerate(loader):
        videos = videos.to(device)
        logits = model(videos)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
        true = int(labels[0])
        clip = os.path.basename(dataset.samples[i][0])
        true_name = class_names[true]
        pred_name = class_names[pred]
        ok = pred == true
        results.append(
            {
                "clip": clip,
                "path": dataset.samples[i][0],
                "true": true_name,
                "pred": pred_name,
                "correct": ok,
                "conf": conf,
                "video": videos[0].cpu(),
            }
        )
        print(
            f"{clip:<24} {true_name:<24} {pred_name:<24} "
            f"{'Y' if ok else 'N':<4} {conf:6.2f}"
        )

    print("\n=== Accuracy by class ===")
    by_class = defaultdict(list)
    for row in results:
        by_class[row["true"]].append(row["correct"])
    for name in class_names:
        vals = by_class[name]
        acc = sum(vals) / len(vals) if vals else 0.0
        print(f"  {name:<24} {sum(vals):2d}/{len(vals):2d}  ({acc:.1%})")
    overall = sum(r["correct"] for r in results) / len(results)
    print(f"  {'overall':<24} {sum(r['correct'] for r in results):2d}/{len(results):2d}  ({overall:.1%})")

    csv_path = os.path.splitext(out_path)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip", "true", "pred", "correct", "conf"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "clip": row["clip"],
                    "true": row["true"],
                    "pred": row["pred"],
                    "correct": row["correct"],
                    "conf": f"{row['conf']:.4f}",
                }
            )
    print(f"\nWrote {csv_path}")

    selected = []
    used = defaultdict(int)
    for row in results:
        if used[row["true"]] < per_class:
            selected.append(row)
            used[row["true"]] += 1
    if not selected:
        return

    n = len(selected)
    cols = per_class
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = [axes] if rows == 1 and cols == 1 else axes.flatten()
    for ax, row in zip(axes, selected):
        mid = row["video"].shape[0] // 2
        frame = row["video"][mid].permute(1, 2, 0).numpy()
        ax.imshow(frame)
        color = "tab:green" if row["correct"] else "tab:red"
        ax.set_title(
            f"{row['clip']}\ntrue: {row['true']}\npred: {row['pred']} ({row['conf']:.2f})",
            color=color,
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TimeSformer on frame folders")
    parser.add_argument(
        "--data-dir",
        default=DATASET_PATH,
        help="Root folder of class/video/frame layout",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Frames per clip (10 ≈ 1 FPS on a 10s Kinetics clip)",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--checkpoint",
        default="timesformer.pt",
        help="Where to save (or load) the trained weights",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and report predictions from --checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = VideoFrameDataset(
        root_dir=args.data_dir,
        num_frames=args.num_frames,
    )
    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"Clips: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    model = TimeSformer(
        num_classes=len(train_dataset.classes),
        num_frames=args.num_frames,
        embed_dim=768,
        depth=12,
        heads=12,
    ).to(device)

    if args.eval_only:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint {args.checkpoint}")
    else:
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit.eval()
        load_vit_weights_into_timesformer(model, vit)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            print(
                f"Epoch {epoch + 1} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train accuracy: {train_acc:.4f}"
            )

        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved {args.checkpoint}")

    report_predictions(model, train_dataset, device)


if __name__ == "__main__":
    main()
