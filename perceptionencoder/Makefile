# videosearch — Makefile shortcuts
# Usage: make help

.PHONY: help sync vendor setup gpu test test-unit smoke index index-full search demo confirm clean clean-index clean-all

# ---------------------------------------------------------------------------
# Defaults (override on the command line, e.g. make search QUERY="..." K=10)
# ---------------------------------------------------------------------------
VIDEO_DIR  ?= /home/satishv/study/vision/dataset/kinetics-dataset/data
INDEX_DIR  ?= data/index
INDEX_FULL ?= data/index_full
INDEX_EVENTS ?= data/index_events
LIMIT      ?= 50
QUERY      ?= a person riding a horse
EVENT      ?= fight
K          ?= 5
MODEL      ?= PE-Core-L14-336
CHUNK_SEC  ?= 2.0
FRAMES     ?= 8
EXPORT     ?=
UV         := uv run

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
sync: ## Install Python deps with uv (PyTorch cu130 for GB10)
	uv sync --extra test

vendor: ## Clone Meta perception_models into vendor/ (PE-Core source)
	@test -d vendor/perception_models/core/vision_encoder \
		|| git clone --depth 1 https://github.com/facebookresearch/perception_models.git vendor/perception_models
	@echo "vendor/perception_models ready"

setup: sync vendor ## Full first-time setup (deps + vendor PE)

gpu: ## Verify CUDA / GB10 is usable
	$(UV) python scripts/verify_gpu.py

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: ## Run unit tests (no GPU / no PE weights required)
	$(UV) pytest -q

test-unit: test ## Alias for test

smoke: ## GPU smoke: load PE-Core, embed one text + one video chunk
	$(UV) python -c "from videosearch.pe_model import PEModel; from videosearch.chunker import iter_chunks; from pathlib import Path; import numpy as np; m=PEModel(); t=m.embed_texts(['a person riding a horse'])[0]; vids=sorted(Path('$(VIDEO_DIR)').glob('*.mp4')); assert vids, 'no videos'; ch=next(iter_chunks(vids[0], chunk_sec=2.0, frames_per_chunk=4)); v=m.embed_chunk(ch.frames); print('ok dim', v.shape, 'cos', float(t@v))"

# ---------------------------------------------------------------------------
# Index / search
# ---------------------------------------------------------------------------
index: ## Index a subset of videos (LIMIT=$(LIMIT)) into INDEX_DIR
	$(UV) videosearch index "$(VIDEO_DIR)" \
		--index-dir "$(INDEX_DIR)" \
		--model "$(MODEL)" \
		--chunk-sec $(CHUNK_SEC) \
		--frames $(FRAMES) \
		--limit $(LIMIT)

index-full: ## Index ALL videos into INDEX_FULL (~30+ min)
	$(UV) videosearch index "$(VIDEO_DIR)" \
		--index-dir "$(INDEX_FULL)" \
		--model "$(MODEL)" \
		--chunk-sec $(CHUNK_SEC) \
		--frames $(FRAMES)

search: ## Search: make search QUERY="..." [K=5] [INDEX_DIR=...] [EXPORT=clips/]
	$(UV) videosearch search "$(QUERY)" -k $(K) --index-dir "$(INDEX_DIR)" \
		$(if $(EXPORT),--export-clips "$(EXPORT)",)

demo: ## Demo search against the full index (if built)
	$(UV) videosearch search "$(QUERY)" -k $(K) --index-dir "$(INDEX_FULL)"

confirm: ## PE→VLM cascade: make confirm EVENT=fight [K=5] [INDEX_EVENTS=...]
	$(UV) videosearch confirm "$(EVENT)" -k $(K) --index-dir "$(INDEX_EVENTS)" \
		--report "data/reports/$(EVENT)_confirm.json" \
		$(if $(EXPORT),--export-clips "$(EXPORT)",)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove Python caches and pytest junk
	find . -type d -name __pycache__ -not -path './.venv/*' -not -path './vendor/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

clean-index: ## Remove generated indexes and exported clips
	rm -rf data/index data/index_full data/index_events data/reports clips data/*.log

clean-all: clean clean-index ## Clean caches + indexes (keeps .venv, vendor, .hf_home)
