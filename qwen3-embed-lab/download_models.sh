#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd ~/study/qwen3-embed-lab
for m in Qwen/Qwen3-VL-Embedding-2B Qwen/Qwen3-VL-Reranker-2B Qwen/Qwen3-VL-Embedding-8B; do
  echo "=== downloading $m ==="
  hf download "$m" --exclude "*.gguf"
done
echo "ALL_MODELS_DONE"
