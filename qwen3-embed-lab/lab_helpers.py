"""
Shared helpers for the Qwen3-VL-Embedding learning lab (~/study/qwen3-embed-lab).

IMPORTANT: MODEL_SEQ_LEN must be set BEFORE qwen_vl_utils is imported, because
qwen_vl_utils reads it at import time and uses it as the *default pixel budget*
for file-path video inputs (see fetch_video in qwen_vl_utils/vision_process.py):

    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor**2 * 0.9)

The official wrapper only injects its own `total_pixels` for frame-list inputs,
so for file-path inputs this env var is the budget knob.
"""
import os

# Must happen before any qwen_vl_utils / wrapper import.
os.environ.setdefault("MODEL_SEQ_LEN", "8192")

# Work around a transformers 4.57.x bug: the *network* path of
# list_repo_templates() does not strip the ".jinja" suffix (the local-cache path
# does), which makes AutoProcessor.from_pretrained crash on repos that ship
# additional_chat_templates/ (e.g. Qwen3-VL-Reranker). All lab models are fully
# cached, so we force offline mode and take the correct local path.
# To download NEW models with this env active, unset it or use `hf download` in a
# separate shell (see download_models.sh).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import math
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

LAB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LAB_ROOT / "Qwen3-VL-Embedding"
VIDEO_DIR = LAB_ROOT / "data" / "videos"
FRAME_DIR = LAB_ROOT / "data" / "frames"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402
from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

# ---------------------------------------------------------------------------
# Constants mirrored from the wrapper / qwen_vl_utils (verified in notebooks)
# ---------------------------------------------------------------------------
IMAGE_PATCH = 16          # vision patch size
MERGE = 2                 # 2x2 spatial merge
TOKEN_PIXELS = (IMAGE_PATCH * MERGE) ** 2      # 32*32 = 1024 px per visual token
FRAME_MIN_TOKENS = 128    # per-frame floor (VIDEO_MIN_TOKEN_NUM)
FRAME_MAX_TOKENS = 768    # per-frame ceiling (VIDEO_MAX_TOKEN_NUM)
WRAP_TOTAL_PIXELS = 7_864_320                # wrapper default total_pixels (frame-list mode)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Videos in the lab corpus
# ---------------------------------------------------------------------------
VIDEOS = {
    "bbb_1080": VIDEO_DIR / "bbb_1080.mp4",   # Big Buck Bunny, 1920x1080, 10s, 60fps
    "bbb_720":  VIDEO_DIR / "bbb_720.mp4",    # same content, 1280x720
    "bbb_360":  VIDEO_DIR / "bbb_360.mp4",    # same content, 640x360
    "jellyfish_720": VIDEO_DIR / "jellyfish_720.mp4",  # real footage, 1280x720, 10s
    "flower":   VIDEO_DIR / "flower.mp4",     # real footage, 960x540, 5s (CC0)
    "sintel_10s": VIDEO_DIR / "sintel_10s.mp4",   # 854x480, first 10s
    "sintel_30s": VIDEO_DIR / "sintel_30s.mp4",   # 854x480, first 30s
    "sintel_trailer": VIDEO_DIR / "sintel_trailer.mp4",  # 854x480, 52s
}

# The 4-video corpus used in the retrieval experiments.
CORPUS = ["bbb_1080", "jellyfish_720", "flower", "sintel_10s"]

# Descriptive queries (one detail-oriented + one holistic per video).
# Grounding was verified empirically: argmax over the corpus picks the right video.
VIDEO_QUERIES = {
    "bbb_1080": [
        "A chubby grey rabbit in a green meadow in a 3D animated film.",
        "An animated short film set in a sunny forest meadow with cute animals.",
    ],
    "jellyfish_720": [
        "A translucent jellyfish with trailing tentacles drifting through deep blue water.",
        "Underwater footage of a sea creature glowing as it swims in the ocean.",
    ],
    "flower": [
        "A close-up of a pink flower outdoors with green foliage in the background.",
        "Nature footage of a blooming flower on a windy day in a garden.",
    ],
    "sintel_10s": [
        "A girl with red hair in a snowy mountain landscape in an animated fantasy film.",
        "A fantasy adventure film with mountains, snow and a lone traveller.",
    ],
}
# Queries that should match NOTHING in the corpus (negative controls).
DISTRACTOR_QUERIES = [
    "A rocket launching into the night sky.",
    "People dancing at a crowded concert.",
]

# Canonical query-side instruction for video retrieval experiments.
RETR_INSTRUCTION = "Retrieve the video relevant to the user's query"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_embedder(size: str = "2B", max_length: int = 16384, **kwargs) -> Qwen3VLEmbedder:
    """Load Qwen3-VL-Embedding (2B or 8B) on the GB10 with SDPA attention."""
    return Qwen3VLEmbedder(
        f"Qwen/Qwen3-VL-Embedding-{size}",
        max_length=max_length,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **kwargs,
    )


def load_reranker(size: str = "2B", **kwargs) -> Qwen3VLReranker:
    return Qwen3VLReranker(
        f"Qwen/Qwen3-VL-Reranker-{size}",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Conversation building (mirrors Qwen3VLEmbedder.format_model_input, but with
# total_pixels / min_pixels injectable in file-path mode too, so the pixel
# budget can be controlled per input in *both* input modes).
# ---------------------------------------------------------------------------
def _end_with_period(instr: str) -> str:
    instr = instr.strip()
    if instr and not unicodedata.category(instr[-1]).startswith("P"):
        instr = instr + "."
    return instr


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


def _is_image_path(p: str) -> bool:
    """Mirror of the wrapper's is_image_path: does this path point at an image file?"""
    if p.startswith(("http://", "https://")):
        p = urlparse(p).path
    return os.path.splitext(p.lower())[1] in _IMAGE_EXTS


def build_conversation(embedder: Qwen3VLEmbedder, entry: dict) -> list:
    """Build one chat conversation for an input entry.

    entry keys (all optional): text, image, video, instruction, fps, max_frames,
    total_pixels, min_pixels.
    - `video` as a str  -> file-path mode (fps / max_frames / total_pixels / min_pixels)
    - `video` as a list -> frame-list mode (total_pixels / min_pixels; wrapper's
      constructor max_frames applies at process() time, here we assume the caller
      passes the frames it wants)
    """
    instruction = entry.get("instruction")
    if instruction:
        instruction = _end_with_period(instruction)
    system = instruction or embedder.default_instruction

    content = []
    video = entry.get("video")
    if video is not None:
        # Classify the outer list exactly like the wrapper's is_video_input():
        # - a list whose first element is an IMAGE path / PIL image -> ONE video
        #   given as a frame list (frame-list mode)
        # - a list whose first element is a VIDEO path -> several videos
        if isinstance(video, (str, Path)):
            videos = [video]
        elif isinstance(video, list) and video and isinstance(video[0], (str, Path)) \
                and not _is_image_path(str(video[0])):
            videos = video
        else:
            videos = [video]
        for vid in videos:
            extra = {}
            if entry.get("total_pixels") is not None:
                extra["total_pixels"] = int(entry["total_pixels"])
            if entry.get("min_pixels") is not None:
                extra["min_pixels"] = int(entry["min_pixels"])
            if isinstance(vid, list):
                frames_mode = True
            elif isinstance(vid, (str, Path)):
                frames_mode = _is_image_path(str(vid))  # single image path = 1-frame video
            else:
                frames_mode = True                     # PIL image(s)
            if frames_mode:
                frames = vid if isinstance(vid, list) else [vid]
                if entry.get("max_frames") is not None:
                    frames = frames[: int(entry["max_frames"])]
                frames = [
                    ("file://" + str(f) if isinstance(f, (str, Path)) else f)
                    for f in frames
                ]
                content.append({"type": "video", "video": frames, **extra})
            else:
                p = str(vid)
                body = p if p.startswith(("http://", "https://")) else "file://" + p
                content.append({
                    "type": "video", "video": body,
                    "fps": entry.get("fps") or embedder.fps,
                    "max_frames": entry.get("max_frames") or embedder.max_frames,
                    **extra,
                })
    image = entry.get("image")
    if image is not None:
        images = image if isinstance(image, list) else [image]
        for img in images:
            p = str(img)
            body = p if p.startswith(("http://", "https://")) else "file://" + p
            content.append({
                "type": "image", "image": body,
                "min_pixels": embedder.min_pixels,
                "max_pixels": embedder.max_pixels,
            })
    text = entry.get("text")
    if text is not None:
        texts = text if isinstance(text, list) else [text]
        for t in texts:
            content.append({"type": "text", "text": t})

    if not content:
        content.append({"type": "text", "text": "NULL"})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Embedding (same pipeline as Qwen3VLEmbedder.process, but using
# build_conversation so per-entry budget knobs work)
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed(embedder: Qwen3VLEmbedder, entries: list, timed: bool = False):
    """Embed a list of entries -> (N, D) L2-normalized embeddings on the model device."""
    conversations = [build_conversation(embedder, e) for e in entries]
    t0 = time.perf_counter()
    inputs = embedder._preprocess_inputs(conversations)
    inputs = {k: v.to(embedder.model.device) for k, v in inputs.items()}
    outputs = embedder.forward(inputs)
    emb = embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
    emb = F.normalize(emb, p=2, dim=-1)
    dt = time.perf_counter() - t0
    return (emb, dt) if timed else emb


def embed_official(embedder: Qwen3VLEmbedder, entries: list):
    """The wrapper's own public API (used in notebook 01 to show the official path)."""
    return embedder.process(entries)


# ---------------------------------------------------------------------------
# Instrumentation: what does the model ACTUALLY see for a given entry?
# ---------------------------------------------------------------------------
def inspect_entry(embedder: Qwen3VLEmbedder, entry: dict) -> dict:
    """Run the vision preprocessing only and report frames / resolutions / tokens."""
    conversation = build_conversation(embedder, entry)
    images, video_inputs, video_kwargs = process_vision_info(
        [conversation], image_patch_size=IMAGE_PATCH,
        return_video_metadata=True, return_video_kwargs=True,
    )
    out = {"n_images": 0 if images is None else len(images), "videos": []}
    if video_inputs:
        for vid, meta in video_inputs:
            if torch.is_tensor(vid) and vid.ndim == 4:          # (T, C, H, W)
                T, _, H, W = vid.shape
                frames = vid
            else:                                                # list of frames
                T = len(vid)
                H, W = vid[0].shape[-2], vid[0].shape[-1]
                frames = None
            toks = round(H * W / TOKEN_PIXELS)
            out["videos"].append({
                "nframes": int(T), "height": int(H), "width": int(W),
                "frame_pixels": int(H * W),
                "tokens_per_frame": int(toks),
                "total_visual_tokens": int(toks * T),
                "metadata": meta, "tensor": frames,
            })
    return out


def summarize_inspection(info: dict) -> str:
    parts = []
    for v in info["videos"]:
        parts.append(
            f"{v['nframes']} frames @ {v['width']}x{v['height']} "
            f"({v['frame_pixels']:,} px/frame = {v['tokens_per_frame']} tok/frame "
            f"-> {v['total_visual_tokens']:,} visual tokens)"
        )
    return "; ".join(parts) if parts else "no video content"


def predicted_budget(nframes: int, total_pixels: int | None = None) -> dict:
    """The budget math from qwen_vl_utils.fetch_video, for comparison with reality.

    per-frame max = max( min(FRAME_MAX_TOKENS*1024, total/nframes*2), FRAME_MIN_TOKENS*1024*1.05 )
    """
    if total_pixels is None:
        total_pixels = int(float(os.environ["MODEL_SEQ_LEN"]) * TOKEN_PIXELS * 0.9)
    per_frame = max(
        min(FRAME_MAX_TOKENS * TOKEN_PIXELS, total_pixels / nframes * 2),
        int(FRAME_MIN_TOKENS * TOKEN_PIXELS * 1.05),
    )
    return {
        "total_pixels": total_pixels,
        "per_frame_pixel_cap": int(per_frame),
        "per_frame_token_cap": round(per_frame / TOKEN_PIXELS),
    }


# ---------------------------------------------------------------------------
# Similarity + plotting
# ---------------------------------------------------------------------------
def sim_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Cosine similarity (embeddings are already L2-normalized)."""
    return (a @ b.T).float().cpu().numpy()


def heatmap(mat: np.ndarray, xlabels=None, ylabels=None, title="", fmt=".2f",
            figsize=None, vmin=None, vmax=None, cmap="viridis", ax=None):
    n, m = mat.shape
    figsize = figsize or (max(1.1 * m, 4), max(0.55 * n, 2.2))
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(m)); ax.set_yticks(range(n))
    ax.set_xticklabels(xlabels or range(m), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ylabels or range(n), fontsize=8)
    for i in range(n):
        for j in range(m):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                    color="white" if mat[i, j] < (np.nanmax(mat) if vmax is None else vmax) - 0.15 else "black",
                    fontsize=7)
    if title:
        ax.set_title(title, fontsize=10)
    if own:
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.show()
    return im


def show_frames(video_tensor: torch.Tensor, n: int = 8, ncols: int = 4, title: str = ""):
    """Display the (resized) frames the model actually consumes."""
    T = video_tensor.shape[0]
    idx = np.linspace(0, T - 1, min(n, T)).astype(int)
    nrows = math.ceil(len(idx) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, fi in enumerate(idx):
        frame = video_tensor[fi].permute(1, 2, 0).cpu().to(torch.uint8).numpy()
        axes[k].imshow(frame)
        axes[k].set_title(f"t={fi}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def rank_of(scores: np.ndarray, correct: int) -> int:
    """1-based rank of `correct` index in a descending score array."""
    order = np.argsort(-scores)
    return int(np.where(order == correct)[0][0]) + 1


def margins(scores: np.ndarray, correct: int) -> dict:
    """correct score - best distractor score (separation margin)."""
    s = np.asarray(scores, dtype=float)
    correct_score = s[correct]
    distractors = np.delete(s, correct)
    return {
        "correct_score": float(correct_score),
        "best_distractor": float(distractors.max()),
        "margin": float(correct_score - distractors.max()),
        "rank": rank_of(s, correct),
    }


def drift(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """Cosine distance between two embeddings of the SAME content."""
    return 1.0 - float(F.cosine_similarity(e1.detach().flatten().float(),
                                           e2.detach().flatten().float(), dim=0))


# ---------------------------------------------------------------------------
# Frame extraction (for frame-list mode experiments)
# ---------------------------------------------------------------------------
def extract_frames(video_key: str, n: int, out_name: str | None = None) -> list:
    """Extract n uniformly spaced frames as PNGs using ffmpeg, return sorted paths."""
    import subprocess
    src = str(VIDEOS[video_key])
    out_name = out_name or f"{video_key}_{n}f"
    out_dir = FRAME_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("f_*.png"))
    if len(existing) == n:
        return existing
    for f in out_dir.glob("f_*.png"):
        f.unlink()
    # duration via ffprobe
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "csv=p=0", src,
    ]).decode().strip())
    fps = n / dur
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", src,
        "-vf", f"fps={fps:.6f}", "-frames:v", str(n),
        str(out_dir / "f_%03d.png"),
    ], check=True)
    return sorted(out_dir.glob("f_*.png"))


def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def print_gpu():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({DEVICE})")
    else:
        print("No CUDA GPU visible!")
