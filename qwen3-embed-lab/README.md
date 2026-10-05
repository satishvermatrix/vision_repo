# Qwen3 Embedding Lab (video + text)

A hands-on learning lab for **Qwen3-VL-Embedding / Qwen3-VL-Reranker** - the latest Qwen
embedding & ranking models (Jan 2026) that embed **text, images and video into one shared
vector space**. Runs fully locally on this machine (NVIDIA GB10 / DGX Spark, 128 GB).

## Notebooks

| # | notebook | what you learn |
|---|---|---|
| 01 | `01_text_similarity.ipynb` | how text becomes a vector, cosine similarity intuition, 2B vs 8B |
| 02 | `02_video_text_similarity.ipynb` | how video becomes tokens (frames -> 1024 px/token), cross-modal retrieval, inspecting what the model actually sees |
| 03 | `03_sampling_pixel_budget.ipynb` | **the pixel budget**: fps / max_frames / total_pixels / source resolution / clip duration sweeps - "more frames = smaller frames", margins & drift |
| 04 | `04_instruction_impact.ipynb` | how instructions (system prompts) move embeddings - matched vs mismatched instructions, margins, rank-1 accuracy, drift |
| 05 | `05_reranker_vs_embedding.ipynb` | bi-encoder (recall) vs cross-encoder (precision), hard cases, mixed text+video corpus, cost |
| 06 | `06_event_identification_fight.ipynb` | event identification: chunk footage into 5 s clips (`data/clips/`), embed once, find events by text query - instruction effects on **margin** and detection-threshold windows (uses `~/study/data/fight_2.mp4`) |

## Models used (~27 GB in the HF cache)

- `Qwen/Qwen3-VL-Embedding-2B` (2048-dim) - main workhorse
- `Qwen/Qwen3-VL-Embedding-8B` (4096-dim) - comparisons
- `Qwen/Qwen3-VL-Reranker-2B` - cross-encoder relevance scoring

## Setup (already done on this machine)

```bash
cd ~/study/qwen3-embed-lab
python3 -m venv .venv          # created once
# deps: see requirements.txt (torch 2.9.0+cu130 aarch64 wheels, transformers 4.57.6,
# qwen-vl-utils 0.0.14, av 13.1.0 pinned - see comments in that file)
bash download_models.sh        # (re)downloads the 3 models into ~/.cache/huggingface
```

Videos live in `data/videos/` (sources & licenses in `data/videos/SOURCES.md`);
`lab_helpers.py` holds shared code (model loading, conversation building, budget
instrumentation, similarity/plotting helpers); `tools/` has the notebook builder,
the query-grounding smoke test, and nothing else that the notebooks need at runtime.

Environment gotchas (all handled inside `lab_helpers.py`, documented here because
they are the instructive part):

- `flash_attention_2` is **not** available on GB10 - the lab uses `sdpa`.
- `MODEL_SEQ_LEN=8192` is set **before** `qwen_vl_utils` is imported: it is the pixel
  budget knob for *file-path* video inputs (the wrapper's `total_pixels` only reaches
  qwen_vl_utils for *frame-list* inputs). Read at import time - see notebook 03.
- `HF_HUB_OFFLINE=1` is set to dodge a transformers 4.57.x bug (network template
  listing returns `reranker.jinja` un-stripped, which crashes the reranker's
  `AutoProcessor` load). With it, loads use the local cache - models must be fully
  downloaded first (`bash download_models.sh`).
- `av` is pinned to 13.1.0 because torchvision 0.24's `read_video` uses a kwarg that
  av 19 removed (decord has no aarch64 wheels, so torchvision is the video backend).

## Running

```bash
cd ~/study/qwen3-embed-lab
.venv/bin/jupyter lab notebooks/          # interactive
# or re-execute everything headless:
for nb in notebooks/0*.ipynb; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace "$nb" \
    --ExecutePreprocessor.timeout=3600
done
```

Tips:
- Run notebooks in order the first time; each loads its own model and frees GPU memory
  before loading the next.

## Key mechanics (verified empirically in notebook 03)

- One visual token = 16x16 patch x 2x2 merge = **1024 pixels**.
- Per-frame pixels = `min(786,432, total_pixels * 2 / nframes)`, floor 131,072 px
  (128 tokens), ceiling 786,432 px (768 tokens) per frame.
- Wrapper defaults are self-consistent: `64 frames x 128 tokens = 8192 = max_length`.
- Effective visual-token spend = `2 x total_pixels / 1024`.
- Source resolution above the per-frame allowance is **downscaled away** (1080p vs 720p
  sources give nearly identical embeddings); below it, detail cannot be recovered.
- Instructions are **system messages** and measurably move embeddings (notebook 04).

## References

- Repo: https://github.com/QwenLM/Qwen3-VL-Embedding (this lab uses its `Qwen3VLEmbedder` / `Qwen3VLReranker` wrappers)
- HF: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B (+ 8B, Reranker-2B/8B)
- Tech report: *Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking* (arXiv:2601.04720)
- Text-only family (older, stronger on pure text): https://github.com/QwenLM/Qwen3-Embedding
