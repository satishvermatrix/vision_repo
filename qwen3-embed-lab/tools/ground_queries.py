"""Smoke test + query grounding for the lab.

Verifies: environment works end to end (text + video embed, reranker) and that every
drafted VIDEO_QUERIES entry retrieves its own video with rank 1 and a healthy margin.
Run:  ~/study/qwen3-embed-lab/.venv/bin/python tools/ground_queries.py
"""
import os, sys
LAB_ROOT = "/home/satishv/study/qwen3-embed-lab"
sys.path.insert(0, LAB_ROOT)
from lab_helpers import *
import numpy as np

print_gpu()
embedder = load_embedder("2B")

# --- 1. text embed via the official API -------------------------------------
e = embed_official(embedder, [{"text": "A rocket launching into the night sky."},
                              {"text": "A rocket lifts off at night, flames lighting the launch pad."}])
s = float(e[0] @ e[1])
print(f"[text] dim={tuple(e.shape)} norm={float(e.norm(dim=-1)[0]):.3f} paraphrase-sim={s:.3f}")

# --- 2. video preprocessing at defaults --------------------------------------
for k in CORPUS + ["sintel_trailer", "bbb_360"]:
    info = inspect_entry(embedder, {"video": str(VIDEOS[k])})
    print(f"[inspect] {k:15s} {summarize_inspection(info)}")

# --- 3. ground the queries ----------------------------------------------------
corpus_entries = [{"video": str(VIDEOS[k])} for k in CORPUS]
doc_emb = embed(embedder, corpus_entries)
print("\n[grounding] (rank must be 1; margin comfortably positive)")
for k in CORPUS:
    for q in VIDEO_QUERIES[k]:
        qe = embed(embedder, [{"text": q, "instruction": RETR_INSTRUCTION}])
        sc = (qe[0] @ doc_emb.T).float().cpu().numpy()
        m = margins(sc, CORPUS.index(k))
        print(f"  {k:15s} rank={m['rank']} margin={m['margin']:+.3f} :: {q[:65]}")
print("[grounding] negatives (best score should be low, no clear winner)")
for q in DISTRACTOR_QUERIES:
    qe = embed(embedder, [{"text": q, "instruction": RETR_INSTRUCTION}])
    sc = (qe[0] @ doc_emb.T).float().cpu().numpy()
    print(f"  best={sc.max():.3f} argmax={CORPUS[int(np.argmax(sc))]:15s} :: {q[:65]}")

# --- 4. reranker smoke ---------------------------------------------------------
del embedder; torch.cuda.empty_cache()
reranker = load_reranker("2B")
scores = reranker.process({
    "instruction": "Given a user query, retrieve the video that best matches the description",
    "query": {"text": VIDEO_QUERIES["flower"][0]},
    "documents": [{"video": str(VIDEOS[k])} for k in CORPUS],
    "fps": 1.0, "max_frames": 16,
})
print("\n[reranker] flower query ->", dict(zip(CORPUS, np.round(scores, 3).tolist())))
print("ALL SMOKE CHECKS DONE")
