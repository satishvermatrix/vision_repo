# videosearch — Natural-language video search with Meta's Perception Encoder

Query a video collection in plain English (an NLQ, e.g. *"a person riding a
horse"*) and get back the most relevant **video chunks** with their source file
and timestamps. Chunks are embedded with Meta's **PE-Core** CLIP encoder
(`PE-Core-L14-336` by default); the query is embedded into the same space and
matched by cosine similarity.

Built for an **NVIDIA GB10 (Grace-Blackwell, sm_121, CUDA 13)** on ARM64, using
`uv` for environment management.

## Quick start (Makefile)

```bash
make setup          # uv sync (incl. pytest) + clone vendor/perception_models
make gpu            # verify CUDA / GB10
make test           # unit tests (no GPU / no PE weights)
make index          # index LIMIT=50 clips into data/index (~2 min)
make search QUERY="a person riding a horse"
make demo QUERY="playing basketball"   # uses data/index_full if present
```

See `make help` for every target and overrideable variables.

## Design

```
video --> chunker (fixed windows, K sampled frames each)
      --> PE-Core image tower (encode frames, mean-pool, L2-normalize)  --> chunk embedding
NLQ   --> PE-Core text tower (encode, L2-normalize)                     --> query embedding
match --> cosine similarity  --> top-k chunks (video, start-end, score)
```

### Why this architecture

| Piece | Role |
|---|---|
| **PE-Core CLIP** | Contrastive vision–language model: frames and text share one embedding space, so text→video retrieval is a dot product. |
| **Chunker** | Splits each video into fixed time windows (default **2s**). Uniformly samples **K** frames (default **8**) per window via OpenCV. Skips tiny trailing slivers (< 0.5s). |
| **Mean-pool** | Per-frame embeddings are averaged then L2-normalized → one vector per chunk. |
| **Index** | `embeddings.npy` (N×D float32) + `chunks.json` (video path, start, end) + `config.json`. |
| **Search** | Embed the NLQ once, cosine-score against all chunk vectors, return top-k. Optional ffmpeg clip export. |

Kinetics clips are ~10s, so a 2s window yields ~5 chunks/clip. The same pipeline works for longer videos without changes.

### Module map

```
src/videosearch/
  pe_model.py   # PE-Core wrapper: embed_images / embed_chunk / embed_texts
  chunker.py    # time windows + frame sampling (OpenCV)
  indexer.py    # walk videos → embed → write index
  search.py     # load index, NLQ → cosine top-k
  cli.py        # `videosearch index` / `videosearch search`
scripts/
  verify_gpu.py # GB10 / CUDA sanity check
tests/
  test_core.py  # unit tests (no GPU)
vendor/perception_models/   # cloned PE source (git-ignored)
data/index*/                # generated indexes (git-ignored)
```

We **vendor** `facebookresearch/perception_models` and import only
`core.vision_encoder`. We deliberately do **not** `pip install -e .` that repo
(its pins are CUDA 12.4 / xformers and break on Blackwell).

### Hardware notes (GB10)

- PyTorch is installed from the **cu130** wheel index (`torch==2.9.1`).
- You may see a warning that sm_121 exceeds the declared max (sm_120); bf16
  forward still works via PTX forward-compat. Bump to a newer cu130 torch if a
  kernel fails.
- First PE load downloads weights into project-local `.hf_home/` (avoids a
  host-wide HF cache permission issue). Export `HF_TOKEN=...` if Hugging Face
  rate-limits you.

## Setup

Requires `uv`, `git`, `ffmpeg`, and a working NVIDIA driver.

```bash
make setup     # preferred
# or manually:
uv sync --extra test
git clone --depth 1 https://github.com/facebookresearch/perception_models.git \
    vendor/perception_models
make gpu
```

## Usage

### Index videos

Default video dir:
`/home/satishv/study/vision/dataset/kinetics-dataset/data` (~1000 Kinetics clips).

```bash
make index                         # LIMIT=50 → data/index
make index LIMIT=200
make index-full                    # all clips → data/index_full (~30+ min)

# Or raw CLI:
uv run videosearch index /path/to/videos \
    --index-dir data/index \
    --chunk-sec 2.0 --frames 8 \
    --model PE-Core-L14-336 \
    --limit 100
```

Writes `embeddings.npy`, `chunks.json`, and `config.json` into the index dir.

### Search

```bash
make search QUERY="a person riding a horse"
make search QUERY="playing basketball" K=10 INDEX_DIR=data/index_full
make search QUERY="cooking food" EXPORT=clips/   # ffmpeg-export matching windows

# Or raw CLI:
uv run videosearch search "swimming in water" -k 5 --index-dir data/index_full
```

Example output:

```
Query: "riding a bicycle"
 1. score=0.2131  [  4.00-  6.00s]  --mI_-gaZLk_000018_000028.mp4
 2. score=0.2129  [  8.00- 10.00s]  --mI_-gaZLk_000018_000028.mp4
 3. score=0.2128  [  6.00-  8.00s]  --mI_-gaZLk_000018_000028.mp4
```

Scores are cosine similarities in \([-1, 1]\). Relative ranking matters more than
the absolute value.

### Tests

```bash
make test       # pytest unit suite (fake PE model; no GPU / no HF download)
make smoke      # real PE-Core forward on GPU + one Kinetics clip
make gpu        # CUDA matmul sanity check only
```

### Makefile variables

| Variable | Default | Used by |
|---|---|---|
| `VIDEO_DIR` | Kinetics `.../data` | `index`, `index-full`, `smoke` |
| `INDEX_DIR` | `data/index` | `index`, `search` |
| `INDEX_FULL` | `data/index_full` | `index-full`, `demo` |
| `LIMIT` | `50` | `index` |
| `QUERY` | `a person riding a horse` | `search`, `demo` |
| `K` | `5` | `search`, `demo` |
| `MODEL` | `PE-Core-L14-336` | `index`, `index-full` |
| `CHUNK_SEC` | `2.0` | `index`, `index-full` |
| `FRAMES` | `8` | `index`, `index-full` |
| `EXPORT` | _(empty)_ | `search` → `--export-clips` |

### Cleanup

```bash
make clean         # __pycache__, .pytest_cache
make clean-index   # data/index*, clips/, logs
make clean-all     # both (keeps .venv, vendor, .hf_home)
```

## Model options

`--model` / `MODEL=` accepts any PE-Core config (larger = better, slower):

`PE-Core-T16-384`, `PE-Core-S16-384`, `PE-Core-B16-224`,
`PE-Core-L14-336` (default), `PE-Core-G14-448`.

## PE → VLM event confirmation

For event demos (fight / aggression / fall / crowd), PE retrieves candidates and
**Qwen3-VL-2B-Instruct** confirms or rejects each window:

```bash
# Needs data/index_events (from: videosearch index data/events --index-dir data/index_events)
uv run videosearch confirm fight -k 5 --index-dir data/index_events \
  --report data/reports/fight_confirm.json \
  --export-clips clips/cascade_fight

make confirm EVENT=crowd K=5 EXPORT=clips/cascade_crowd
```

Output report JSON includes `pe_score`, `vlm_present`, `vlm_confidence`, `rationale`,
and `confirmed` (present and confidence ≥ threshold). Exported clips are prefixed
`ok_` or `fp_`.

## Limitations

- Frame-based PE-Core ignores **audio** and fine temporal motion. For those,
  swap in PE-Video / PE-AV.
- Indexing throughput (~2s/clip here) is dominated by **CPU video decoding**,
  not the GPU. Fewer `--frames` or larger `--chunk-sec` speeds it up.
- Corrupt/truncated mp4s are skipped (OpenCV fails to open them).
- Kinetics has no true **self-harm** class; that event will usually be VLM-rejected.
```
