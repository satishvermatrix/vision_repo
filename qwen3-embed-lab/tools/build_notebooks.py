"""Build the 5 lab notebooks (notebooks/*.ipynb) from cell definitions.

Run:  ~/study/qwen3-embed-lab/.venv/bin/python tools/build_notebooks.py
"""
import nbformat as nbf

NB = nbf.v4


def build(cells, path):
    nb = NB.new_notebook(
        metadata={"kernelspec": {"display_name": "Python 3 (qwen3-embed-lab)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}},
        cells=[NB.new_markdown_cell(s) if t == "md" else NB.new_code_cell(s) for t, s in cells],
    )
    nbf.write(nb, path)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Shared first cells
# ---------------------------------------------------------------------------
SETUP = r'''import os, sys
LAB_ROOT = "/home/satishv/study/qwen3-embed-lab"
sys.path.insert(0, LAB_ROOT)
from lab_helpers import *   # wrappers, VIDEOS, VIDEO_QUERIES, embed(), inspect_entry(), plotting, ...
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
print_gpu()'''


# ===========================================================================
# 01 - Text <-> text similarity basics
# ===========================================================================
nb01 = [
("md", "# 01 - Text embeddings & cosine similarity: the basics\n"
"Qwen3-VL-Embedding maps **text, images and video into one shared vector space**. In this notebook we stay with plain text and learn:\n"
"\n"
"1. How the model turns text into a vector (instruction + text -> chat template -> last-token hidden state, L2-normalized).\n"
"2. What cosine similarity values look like for paraphrases, lexically-similar-but-different sentences, and unrelated text.\n"
"3. Whether the **8B** model separates semantics more sharply than the **2B** model.\n"
"\n"
"**The two Qwen embedding families** (latest as of Oct 2026):\n"
"\n"
"| family | sizes | input | notes |\n"
"|---|---|---|--- |\n"
"| Qwen3-Embedding / Qwen3-Reranker (Jun 2025) | 0.6B / 4B / 8B | text only | MTEB-multilingual #1 at release |\n"
"| **Qwen3-VL-Embedding / Qwen3-VL-Reranker (Jan 2026)** | 2B / 8B | text, image, video, mixed | one shared vector space, MRL 64-4096 dims, 32K ctx |\n"
"\n"
"We use the VL family everywhere because it can embed **video** - but note that on pure-text benchmarks the text-only family is still a bit stronger."),

("code", SETUP),

("md", "## 1. Load the 2B embedder\n"
"On this machine (NVIDIA GB10, 128 GB unified memory) we run in bfloat16 with `sdpa` attention - `flash_attention_2` is not available on this chip.\n"
"\n"
"Every input is formatted as a chat: **the instruction goes into the system message, the content into the user message**, and the embedding is the last hidden state of the final token, then L2-normalized."),

("code", r'''embedder = load_embedder("2B")
n_params = sum(p.numel() for p in embedder.model.parameters())
print(f"{n_params/1e9:.2f}B params | embed dim = {embedder.model.config.text_config.hidden_size} "
      f"| defaults: fps={embedder.fps}, max_frames={embedder.max_frames}, "
      f"total_pixels={embedder.total_pixels:,}")
print("default instruction:", repr(embedder.default_instruction))'''),

("md", "## 2. A sentence set designed to teach\n"
"- **A1/A2** and **B1/B2** are paraphrase pairs (same meaning, different words).\n"
"- **D_lex** shares many *words* with A1 (woman, plays, sun ...) but describes a different scene - the classic lexical-overlap trap.\n"
"- **C1, E1** are from unrelated topics."),

("code", r'''TEXTS = {
    "A1":    "A woman plays with her golden retriever on a sunny beach.",
    "A2":    "A girl and her dog have fun on the sandy shore in bright sunshine.",
    "B1":    "Stock markets rallied after the central bank cut interest rates.",
    "B2":    "Investors cheered as equities surged when policymakers lowered borrowing costs.",
    "C1":    "Photosynthesis allows plants to convert sunlight into chemical energy.",
    "D_lex": "A woman plays a piano piece as the sun sets behind the theater.",
    "E1":    "Gravity is a fundamental force that attracts two bodies toward each other.",
}
labels = list(TEXTS)
entries = [{"text": t} for t in TEXTS.values()]      # no instruction -> default system prompt

# the wrapper's own public API
emb = embed_official(embedder, entries)
print("embeddings:", tuple(emb.shape), "| norms ~1:", emb.norm(dim=-1)[:3].tolist())'''),

("code", r'''S = sim_matrix(emb, emb)
heatmap(S, labels, labels, "Cosine similarity (Qwen3-VL-Embedding-2B)")'''),

("code", r'''print("nearest neighbour for each sentence (excluding itself):")
for i, lab in enumerate(labels):
    sims = S[i].copy(); sims[i] = -1
    j = int(np.argmax(sims))
    print(f"  {lab:>5} -> {labels[j]:>5}   {sims[j]:.3f}")

pairs = [("A1", "A2"), ("B1", "B2")]
def pair_stats(S):
    within = np.mean([S[labels.index(a), labels.index(b)] for a, b in pairs])
    tri = np.triu_indices(len(labels), 1)
    all_pairs = S[tri].mean()
    return within, all_pairs, within - all_pairs

w, a, sep = pair_stats(S)
print(f"\nmean paraphrase-pair similarity: {w:.3f}")
print(f"mean all-pairs similarity:      {a:.3f}")
print(f"separation (within - all):      {sep:.3f}")'''),

("md", "**What to look for**\n"
"- A1-A2 and B1-B2 should be the highest off-diagonal cells: paraphrases land close together even with zero shared keywords (B1/B2).\n"
"- D_lex vs A1 should be *middle*: shared words pull them together, but different meaning keeps them apart. Embeddings are much less fooled by lexical overlap than, say, BM25.\n"
"- Cross-topic cells should be visibly darker (lower)."),

("md", "## 3. Same exercise with the 8B model\n"
"We free the 2B model first, then repeat with 8B. Watch two things: does 8B *sharpen* the separation, and do the two models *agree* on the ordering of all pairs?"),

("code", r'''del embedder; torch.cuda.empty_cache()
embedder8 = load_embedder("8B")
n_params = sum(p.numel() for p in embedder8.model.parameters())
print(f"{n_params/1e9:.2f}B params | embed dim = {embedder8.model.config.text_config.hidden_size}")
emb8 = embed_official(embedder8, entries)
S8 = sim_matrix(emb8, emb8)
heatmap(S8, labels, labels, "Cosine similarity (Qwen3-VL-Embedding-8B)")'''),

("code", r'''from scipy.stats import spearmanr
tri = np.triu_indices(len(labels), 1)
w8, a8, sep8 = pair_stats(S8)
rho, _ = spearmanr(S[tri], S8[tri])

print(f"separation:  2B = {sep:.3f}   8B = {sep8:.3f}")
print(f"Spearman agreement between 2B and 8B over all {len(tri[0])} pairs: {rho:.3f}")
print(f"(paraphrase pairs) 2B: {S[labels.index('A1'),labels.index('A2')]:.3f} | "
      f"8B: {S8[labels.index('A1'),labels.index('A2')]:.3f}")
print(f"(lexical trap A1-D_lex) 2B: {S[labels.index('A1'),labels.index('D_lex')]:.3f} | "
      f"8B: {S8[labels.index('A1'),labels.index('D_lex')]:.3f}")'''),

("md", "**Takeaways**\n"
"- Both models rank paraphrases above the lexical trap - embeddings capture *meaning*, not word overlap.\n"
"- 8B typically widens the gap between paraphrase pairs and everything else (sharper separation), and the two models agree strongly on pair ordering - a good sign that the space is stable across scales.\n"
"Next: **video** enters the same vector space (notebook 02)."),
]


# ===========================================================================
# 02 - Video <-> text similarity
# ===========================================================================
nb02 = [
("md", "# 02 - Video <-> text similarity in the unified space\n"
"Videos never reach the model as pixels-of-a-movie. They are **decoded into frames**, each frame is resized to a pixel budget, chopped into 16x16 patches, and 2x2-merged - so **one visual token = 32x32 = 1024 pixels**. The frame sequence is then just another token sequence the LLM reads.\n"
"\n"
"In this notebook:\n"
"1. We *inspect* exactly what the model sees for a video (frames, resolutions, visual token counts).\n"
"2. We embed 4 videos + 10 text queries and check whether queries find **their** video (cross-modal retrieval).\n"
"3. We compare 2B vs 8B margins, and get a first taste of source-resolution effects.\n"
"\n"
"The video corpus (all CC / public-domain / test footage, see `data/videos/SOURCES.md`):\n"
"- `bbb_1080` - Big Buck Bunny excerpt, 1920x1080, 10 s (also available at 720p / 360p - same content)\n"
"- `jellyfish_720` - real underwater footage, 1280x720, 10 s\n"
"- `flower` - real close-up footage, 960x540, 5 s (CC0)\n"
"- `sintel_10s` - Sintel trailer cut, 854x480, 10 s (we also have 30 s / full 52 s versions)"),

("code", SETUP),

("md", "## 1. What does the model actually see?\n"
"With wrapper defaults (`fps=1`, `max_frames=64`), a 10 s video becomes ~10 frames. The per-frame pixel budget is\n"
"`min(FRAME_MAX_PIXELS, total_pixels * 2 / nframes)` - here the *cap* (786,432 px = 768 tokens/frame) binds first, so each 1080p frame gets downscaled. `inspect_entry()` runs the real preprocessing so we can **verify, not trust**."),

("code", r'''embedder = load_embedder("2B")

info = inspect_entry(embedder, {"video": str(VIDEOS["bbb_1080"])})
print("actual   :", summarize_inspection(info))
print("predicted:", predicted_budget(info["videos"][0]["nframes"]))
print("source   : 1920x1080 =", 1920*1080, "px/frame")
show_frames(info["videos"][0]["tensor"], n=8, ncols=4,
             title="Frames the model actually consumes (defaults)")'''),

("md", "Now a **long** video: the full 52 s Sintel trailer at `fps=1` -> ~52 frames. The per-frame budget `total_pixels*2/52` (~290 K px) is now *below* the source resolution (854x480 = 410 K px), so frames get squeezed. More frames = coarser frames (notebook 03 studies this properly)."),

("code", r'''info_s = inspect_entry(embedder, {"video": str(VIDEOS["sintel_trailer"])})
v = info_s["videos"][0]
print("actual   :", summarize_inspection(info_s))
print("predicted:", predicted_budget(v["nframes"]))
print("source   : 854x480 =", 854*480, "px/frame  ->  squeezed!" if v["frame_pixels"] < 854*480 else "")
show_frames(v["tensor"], n=12, ncols=4, title="Sintel 52 s @ fps=1 - frames are downscaled to fit the budget")'''),

("md", "## 2. Cross-modal retrieval\n"
"We embed the 4 corpus videos (document side, default instruction) and 10 queries (2 descriptions per video + 2 negatives that match nothing, query side uses the canonical retrieval instruction).\n"
"**A query instruction matters** - the model was trained to embed queries with a task description in the system prompt (notebook 04 measures exactly how much it matters)."),

("code", r'''corpus_entries = [{"video": str(VIDEOS[k])} for k in CORPUS]

q_entries, qlabels, qtruth = [], [], []
for k in CORPUS:
    for qi, q in enumerate(VIDEO_QUERIES[k]):
        qlabels.append(f"{k.split('_')[0]}_q{qi+1}")
        qtruth.append(CORPUS.index(k))
        q_entries.append({"text": q, "instruction": RETR_INSTRUCTION})
for qi, q in enumerate(DISTRACTOR_QUERIES):
    qlabels.append(f"neg{qi+1}")
    qtruth.append(None)
    q_entries.append({"text": q, "instruction": RETR_INSTRUCTION})

doc_emb = embed(embedder, corpus_entries)
q_emb   = embed(embedder, q_entries)
S = sim_matrix(q_emb, doc_emb)
heatmap(S, [c.split("_")[0] for c in CORPUS], qlabels, "video<-text cosine similarity (2B)")'''),

("code", r'''rows = []
for i, ql in enumerate(qlabels):
    s = S[i]
    top = CORPUS[int(np.argmax(s))]
    if qtruth[i] is None:
        rows.append(dict(query=ql, top1=top, rank="-", margin=f"{s.max():.3f} (should be LOW)"))
    else:
        m = margins(s, qtruth[i])
        ok = "OK" if m["rank"] == 1 else "MISS"
        rows.append(dict(query=ql, top1=top, rank=m["rank"],
                         margin=f"{m['margin']:+.3f} {ok}"))
pd.DataFrame(rows)'''),

("md", "**What to look for**\n"
"- Each `*_q1/q2` row should peak on its own video (rank 1, positive margin) - text and video really do live in one space.\n"
"- `neg*` rows should have *low* best-match scores and no standout winner - the space is not just clustering everything together.\n"
"- Which query style wins varies by video (detail q1 wins on jellyfish, holistic q2 wins on sintel) - describing concrete visual content is not automatically the stronger signal; both styles retrieve rank 1 here."),

("md", "## 3. 2B vs 8B on the same retrieval task"),

("code", r'''S_2b, margins_2b = S, {}
for i, ql in enumerate(qlabels):
    if qtruth[i] is not None:
        margins_2b[ql] = margins(S[i], qtruth[i])["margin"]

del embedder; torch.cuda.empty_cache()
embedder8 = load_embedder("8B")
doc_emb8 = embed(embedder8, corpus_entries)
q_emb8   = embed(embedder8, q_entries)
S8 = sim_matrix(q_emb8, doc_emb8)
heatmap(S8, [c.split("_")[0] for c in CORPUS], qlabels, "video<-text cosine similarity (8B)")'''),

("code", r'''rows = []
for i, ql in enumerate(qlabels):
    if qtruth[i] is None:
        rows.append(dict(query=ql, margin_2b="-", margin_8b=f"{S8[i].max():.3f} (LOW?)"))
    else:
        m8 = margins(S8[i], qtruth[i])["margin"]
        rows.append(dict(query=ql, margin_2b=f"{margins_2b[ql]:+.3f}", margin_8b=f"{m8:+.3f}"))
rho, _ = spearmanr(S.flatten(), S8.flatten())
print(f"Spearman agreement of the full 2B vs 8B score matrices: {rho:.3f}")
pd.DataFrame(rows)'''),

("md", "## 4. Bonus: does source resolution even matter? (teaser for notebook 03)\n"
"Big Buck Bunny exists in our corpus at 1080p / 720p / 360p - **identical content, different resolution**. We embed all three with identical settings and compare the video embeddings to each other. Prediction: 1080p and 720p give (nearly) the same embedding, because both get downscaled into the same per-frame budget; only 360p (below the budget) stays 'poor'."),

("code", r'''res_keys = ["bbb_1080", "bbb_720", "bbb_360"]
res_emb = embed(embedder8, [{"video": str(VIDEOS[k])} for k in res_keys])
names = ["1080p", "720p", "360p"]
Sres = sim_matrix(res_emb, res_emb)
heatmap(Sres, names, names, "same content, different source resolution (8B)")

q = VIDEO_QUERIES["bbb_1080"][0]          # the detail query
q_emb8 = embed(embedder8, [{"text": q, "instruction": RETR_INSTRUCTION}])
scores = (q_emb8[0] @ res_emb.T).float().cpu().numpy()
for n, s in zip(names, scores):
    print(f"query->video score {n}: {s:.4f}")
print("\nDrift 1080 vs 720:", f"{drift(res_emb[0], res_emb[1]):.4f}",
      "| 1080 vs 360:", f"{drift(res_emb[0], res_emb[2]):.4f}")'''),

("md", "The 1080p and 720p embeddings are nearly identical (drift ~0.02): **once the source exceeds the per-frame budget, extra pixels are normalized away** - storing 1080p masters buys almost nothing here. The 360p embedding drifts further (0.076) - genuinely 'poorer'. And a surprise worth chewing on: the 360p source scores *highest* against the detail query (0.51 vs 0.44) - aggressive downscaling suppressed background clutter while the rabbit stayed salient. **Resolution is not a monotonic quality knob**; it changes what the embedding emphasizes. Notebook 03 turns this into a full study."),
]


# ===========================================================================
# 03 - Sampling & pixel-budget tradeoffs
# ===========================================================================
nb03 = [
("md", "# 03 - Frame rate, frame count & the pixel budget\n"
"**The core tension:** Qwen3-VL sees video as tokens, and one visual token = 1024 pixels (16x16 patch x 2x2 merge). The number of visual tokens a video may consume is capped by a *pixel budget*, so:\n"
"\n"
"> **more frames -> smaller frames** (per-frame allowance = `total_pixels * 2 / nframes`)\n"
"\n"
"The exact code in `qwen_vl_utils.fetch_video`:\n"
"```python\n"
"min_pixels  = ele.get(\"min_pixels\", 128 tokens * 1024)               # per-frame floor\n"
"total_pixels = ele.get(\"total_pixels\", MODEL_SEQ_LEN * 1024 * 0.9)   # budget\n"
"max_pixels  = max(min(768 tokens * 1024, total_pixels / nframes * 2), min_pixels * 1.05)\n"
"```\n"
"Facts worth knowing (all verified empirically below):\n"
"- **1024 px per visual token**; per-frame floor 128 tokens, ceiling 768 tokens.\n"
"- The `* 2` means the *effective* budget is `2 x total_pixels` (in tokens: `2 * total_pixels / 1024`).\n"
"- With the wrapper defaults, `64 frames x 128 tokens = 8192 tokens` - exactly `max_length`. The defaults are internally consistent.\n"
"- **Two input modes, two budget knobs**: for *frame-list* inputs the wrapper injects its `total_pixels` (default 7,864,320). For *file-path* inputs the wrapper only passes `fps`/`max_frames`, and the budget comes from the `MODEL_SEQ_LEN` **environment variable** (we set it to 8192 in `lab_helpers` *before* importing qwen_vl_utils, giving `8192*1024*0.9` = 7.55 M px).\n"
"\n"
"Experiments:\n"
"1. Verify the budget math (prediction vs reality) and *see* the frames the model gets.\n"
"2. **Sweep A** - frame count at fixed content: 4 -> 60 frames.\n"
"3. **Sweep B** - `fps` on a 52 s clip: temporal coverage vs per-frame detail.\n"
"4. **Sweep C** - explicit `total_pixels` budget (frame-list mode).\n"
"5. **Sweep D** - source resolution 1080p/720p/360p (same content).\n"
"6. **Sweep E** - clip duration 10 s/30 s/52 s at fixed `fps`.\n"
"7. 8B spot-check: does the bigger model cope better with squeezed frames?\n"
"\n"
"Throughout we measure: **margin** = correct-video score minus best-distractor score (retrieval quality), **drift** = cosine distance between embeddings of the *same* video under different configs, visual tokens (cost), and wall-clock time."),

("code", SETUP),

("md", "## 0. Load model + fixed experiment apparatus\n"
"Distractor videos (embedded once with defaults) + the two bbb queries (verified to match `bbb_1080`): a **detail** query (single-scene visual content) and a **holistic** query (overall theme)."),

("code", r'''embedder = load_embedder("2B")

DISTRACTORS = ["jellyfish_720", "flower", "sintel_10s"]
distractor_emb = embed(embedder, [{"video": str(VIDEOS[k])} for k in DISTRACTORS])

BBB_DETAIL    = VIDEO_QUERIES["bbb_1080"][0]
BBB_HOLISTIC  = VIDEO_QUERIES["bbb_1080"][1]
SINTEL_DETAIL   = VIDEO_QUERIES["sintel_10s"][0]
SINTEL_HOLISTIC = VIDEO_QUERIES["sintel_10s"][1]

q_emb = embed(embedder, [
    {"text": BBB_DETAIL,    "instruction": RETR_INSTRUCTION},
    {"text": BBB_HOLISTIC,  "instruction": RETR_INSTRUCTION},
    {"text": SINTEL_DETAIL,   "instruction": RETR_INSTRUCTION},
    {"text": SINTEL_HOLISTIC, "instruction": RETR_INSTRUCTION},
])
print("distractors:", DISTRACTORS)
print("detail query  :", BBB_DETAIL)
print("holistic query:", BBB_HOLISTIC)

def evaluate_config(entry, qi):
    """Embed one config of a target video; return per-frame info + detail-query margin vs distractors."""
    info = inspect_entry(embedder, entry)
    v = info["videos"][0]
    emb, dt = embed(embedder, [entry], timed=True)
    stacked = torch.cat([emb, distractor_emb])
    scores = (q_emb[qi:qi+1] @ stacked.T).float().cpu().numpy().flatten()
    m = margins(scores, 0)
    return dict(nframes=v["nframes"], W=v["width"], H=v["height"],
                px_per_frame=v["frame_pixels"], tok_per_frame=v["tokens_per_frame"],
                total_tokens=v["total_visual_tokens"], time_s=round(dt, 2),
                detail_margin=m["margin"], holistic_margin=None, emb=emb[0])

def evaluate_config2(entry, qi_detail, qi_hol):
    """Same, but also score the holistic query (second query index)."""
    r = evaluate_config(entry, qi_detail)
    stacked = torch.cat([r["emb"].unsqueeze(0), distractor_emb])
    s2 = (q_emb[qi_hol:qi_hol+1] @ stacked.T).float().cpu().numpy().flatten()
    r["holistic_margin"] = margins(s2, 0)["margin"]
    return r'''),

("md", "## 1. Verify the budget math\n"
"`fps=6` on the 10 s 1080p clip would give 60 nominal frames; `max_frames` then caps the count. We predict per-frame pixels with `predicted_budget()` and check against what preprocessing actually produced."),

("code", r'''rows = []
for nf in [4, 8, 16, 32, 64]:
    entry = {"video": str(VIDEOS["bbb_1080"]), "fps": 6.0, "max_frames": nf}
    info = inspect_entry(embedder, entry)
    v = info["videos"][0]
    pred = predicted_budget(v["nframes"])
    rows.append(dict(nframes=v["nframes"], actual=f"{v['width']}x{v['height']}",
                     actual_px=v["frame_pixels"],
                     predicted_cap=pred["per_frame_pixel_cap"],
                     tok_per_frame=v["tokens_per_frame"],
                     total_tokens=v["total_visual_tokens"]))
df = pd.DataFrame(rows)
df["matches_prediction"] = (df.actual_px <= df.predicted_cap * 1.05)
df'''),

("code", r'''info4 = inspect_entry(embedder, {"video": str(VIDEOS["bbb_1080"]), "fps": 6.0, "max_frames": 4})
show_frames(info4["videos"][0]["tensor"], n=4, ncols=4, title="4 frames - full per-frame budget (downscaled 1080p)")

info60 = inspect_entry(embedder, {"video": str(VIDEOS["bbb_1080"]), "fps": 6.0, "max_frames": 64})
show_frames(info60["videos"][0]["tensor"], n=16, ncols=4,
            title=f"{info60['videos'][0]['nframes']} frames - per-frame budget squeezed to "
                  f"{info60['videos'][0]['frame_pixels']:,} px")'''),

("md", "Notice the tradeoff in the table above:\n"
"- From 4 -> 16 frames, each frame still gets the **ceiling** (786 K px = 768 tokens): total tokens grow *linearly* with frames (2,880 -> 11,520).\n"
"- At 32 and 60 frames the **budget** becomes the binding constraint: per-frame allowance halves/quarters, and total visual tokens cap out at ~14 K (= `2 x total_pixels / 1024`, minus a little factor-32 rounding loss). The budget just redistributes roughly the same token spend over more, coarser frames.\n"
"- That is exactly the tradeoff your prompt question was about: **more frames -> smaller frames**, until the budget flattens the curve."),

("md", "## 2. Sweep A - frame count (same content, `fps=6`, `max_frames` = 4/8/16/32/64)\n"
"Does retrieval quality care? We measure margins for the **detail** query (needs per-frame resolution) and the **holistic** query (needs coverage of the clip), plus embedding drift vs the wrapper-default config (`fps=1` -> 10 frames)."),

("code", r'''base_emb = embed(embedder, [{"video": str(VIDEOS["bbb_1080"])}])[0]   # default fps=1 -> 10 frames

rows = []
for nf in [4, 8, 16, 32, 64]:
    r = evaluate_config2({"video": str(VIDEOS["bbb_1080"]), "fps": 6.0, "max_frames": nf}, 0, 1)
    r["drift_vs_default"] = drift(r.pop("emb"), base_emb)
    rows.append(r)
dfA = pd.DataFrame(rows)
dfA'''),

("code", r'''fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
axes[0].plot(dfA.nframes, dfA.tok_per_frame, "o-", label="tokens/frame")
axes[0].plot(dfA.nframes, dfA.total_tokens / dfA.nframes, "s--", label="tokens/frame (from totals)")
axes[0].set_xlabel("frames"); axes[0].set_ylabel("tokens / frame"); axes[0].legend(); axes[0].set_title("per-frame resolution shrinks")
axes[1].plot(dfA.nframes, dfA.total_tokens, "o-"); axes[1].set_xlabel("frames")
axes[1].set_ylabel("total visual tokens"); axes[1].set_title("total cost (budget-capped)")
axes[2].plot(dfA.nframes, dfA.detail_margin, "o-", label="detail query margin")
axes[2].plot(dfA.nframes, dfA.holistic_margin, "s-", label="holistic query margin")
axes[2].axhline(0, color="gray", lw=0.5)
axes[2].set_xlabel("frames"); axes[2].set_ylabel("margin"); axes[2].legend(); axes[2].set_title("retrieval quality")
plt.tight_layout(); plt.show()
print(dfA[["nframes", "detail_margin", "holistic_margin", "drift_vs_default", "total_tokens", "time_s"]])'''),

("md", "## 3. Sweep B - `fps` on the long clip (Sintel, 52 s)\n"
"Here the *source* is only 854x480 (410 K px). At low `fps` the per-frame allowance exceeds the source - **no resize, full source detail, but sparse temporal coverage**. At `fps >= 1` the allowance drops *below* the source and frames get squeezed. So this sweep trades temporal coverage against per-frame quality in the opposite direction from Sweep A."),

("code", r'''base_B = embed(embedder, [{"video": str(VIDEOS["sintel_trailer"])}])[0]   # default fps=1 -> 52 frames

rows = []
for fps in [0.25, 0.5, 1.0, 2.0]:
    r = evaluate_config2({"video": str(VIDEOS["sintel_trailer"]), "fps": fps}, 2, 3)
    r["source_px"] = 854*480
    r["resized"] = r["px_per_frame"] < 854*480 - 1024
    r["drift_vs_fps1"] = drift(r.pop("emb"), base_B)
    rows.append(r)
dfB = pd.DataFrame(rows)
dfB[["nframes", "px_per_frame", "resized", "tok_per_frame", "total_tokens",
     "detail_margin", "holistic_margin", "drift_vs_fps1", "time_s"]]'''),

("code", r'''fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(dfB.nframes, dfB.px_per_frame, "o-"); axes[0].axhline(854*480, color="red", ls="--", label="source pixels (854x480)")
axes[0].set_xlabel("frames"); axes[0].set_ylabel("px / frame"); axes[0].legend(); axes[0].set_title("per-frame allowance vs source")
axes[1].plot(dfB.nframes, dfB.holistic_margin, "s-", label="holistic query")
axes[1].plot(dfB.nframes, dfB.detail_margin, "o-", label="detail query")
axes[1].axhline(0, color="gray", lw=0.5); axes[1].set_xlabel("frames"); axes[1].set_ylabel("margin"); axes[1].legend()
axes[1].set_title("retrieval quality vs temporal coverage")
plt.tight_layout(); plt.show()'''),

("md", "Reading Sweep B (check the table/plot above against this):\n"
"- **Sparse sampling** (0.25 fps -> 12 frames): frames keep full 480p source detail (414 K px, no resize), but ~75% of the clip is never sampled - the detail-query margin collapses to ~0 and the embedding drifts farthest (0.087). Unsampled events are simply absent from the vector.\n"
"- **Dense sampling** (2 fps -> 64 capped): coverage is dense but each frame is squeezed to ~225 K px - yet the detail margin is at its *best* (+0.04).\n"
"- Note the direction flip vs Sweep A: for this 52 s clip, **temporal coverage beat per-frame resolution**; for the 10 s excerpt in Sweep A, resolution beat coverage. The sampling sweet spot depends on clip length and query type - which is exactly why you measure instead of guessing."),

("md", "## 4. Sweep C - explicit `total_pixels` budget (frame-list mode)\n"
"We extract 16 uniform frames as PNGs and feed them as a **frame list**, where we control the budget explicitly. Note `build_conversation` injects `total_pixels` in this mode too, so we can vary it per call. Effective per-frame allowance = `total * 2 / 16`: the last two budgets both hit the 786 K px *ceiling* - prediction: identical embeddings."),

("code", r'''frames16 = extract_frames("bbb_1080", 16)
print(len(frames16), "frames extracted, first:", frames16[0].name, "last:", frames16[-1].name)

rows, embs = [], {}
for tp in [1_500_000, 3_000_000, 7_864_320, 15_728_640]:
    entry = {"video": [str(p) for p in frames16], "total_pixels": tp}
    r = evaluate_config2(entry, 0, 1)
    r["budget"] = tp
    embs[tp] = r.pop("emb")
    rows.append(r)
dfC = pd.DataFrame(rows)
dfC[["budget", "px_per_frame", "tok_per_frame", "total_tokens", "detail_margin", "holistic_margin", "time_s"]]'''),

("code", r'''info_lo = inspect_entry(embedder, {"video": [str(p) for p in frames16], "total_pixels": 1_500_000})
show_frames(info_lo["videos"][0]["tensor"], n=2, ncols=2, title="budget 1.5M px -> 16 frames @ ~183 tok each")

info_hi = inspect_entry(embedder, {"video": [str(p) for p in frames16], "total_pixels": 15_728_640})
show_frames(info_hi["videos"][0]["tensor"], n=2, ncols=2, title="budget 15.7M px -> 16 frames @ 768 tok each (ceiling)")

print("drift 7.86M vs 15.7M (both at ceiling):", f"{drift(embs[7_864_320], embs[15_728_640]):.5f}")
print("drift 1.5M vs 15.7M                     :", f"{drift(embs[1_500_000], embs[15_728_640]):.5f}")
print("detail margin by budget:", dfC.set_index("budget")["detail_margin"].round(3).to_dict())'''),

("md", "Confirmed: past the ceiling the budget is **wasted** - 7.86M and 15.7M px give *identical* embeddings (drift 0.00000) and identical margins, both capped at 768 tokens/frame.\n"
"Below the ceiling the story is counter-intuitive: the *starved* 1.5M budget posts the *best* detail margin (0.20 vs 0.06) - heavy downscaling blurred the meadow background while the rabbit stayed salient.\n"
"\n"
"Moral: **resolution is not a monotonic quality knob** - it changes *what the embedding emphasizes*. Do budget your tokens (spend = `2 x total_pixels / 1024`), but treat per-frame resolution as a hyperparameter to **measure**, not to maximize."),

("md", "## 5. Sweep D - source resolution ladder (same content, fixed config)\n"
"Same 10 s of Big Buck Bunny encoded at 1920x1080 / 1280x720 / 640x360. Fixed `fps=1`, `max_frames=8` -> 8 frames, per-frame allowance ~1.89M px -> **ceiling 786 K px binds** for 1080p and 720p. Prediction: 1080p = 720p embedding (both normalized to 786 K px), 360p stays at its source (230 K px) and is coarser than both."),

("code", r'''res = {}
rows = []
for k in ["bbb_1080", "bbb_720", "bbb_360"]:
    entry = {"video": str(VIDEOS[k]), "fps": 1.0, "max_frames": 8}
    r = evaluate_config2(entry, 0, 1)
    src = {"bbb_1080": 1920*1080, "bbb_720": 1280*720, "bbb_360": 640*360}[k]
    r["source_px"] = src
    r["key"] = k
    res[k] = r.pop("emb")
    rows.append(r)
dfD = pd.DataFrame(rows)
dfD[["key", "source_px", "px_per_frame", "tok_per_frame", "total_tokens", "detail_margin", "holistic_margin", "time_s"]]'''),

("code", r'''names = ["1080p", "720p", "360p"]
keys = ["bbb_1080", "bbb_720", "bbb_360"]
Drift = np.array([[drift(res[a], res[b]) for b in keys] for a in keys])
heatmap(Drift, names, names, "embedding drift between resolutions (same content)", fmt=".3f")

info_h = inspect_entry(embedder, {"video": str(VIDEOS["bbb_1080"]), "fps": 1.0, "max_frames": 8})
info_l = inspect_entry(embedder, {"video": str(VIDEOS["bbb_360"]),  "fps": 1.0, "max_frames": 8})
show_frames(info_h["videos"][0]["tensor"], n=2, ncols=2, title="1080p source after budget downscale")
show_frames(info_l["videos"][0]["tensor"], n=2, ncols=2, title="360p source - below budget, no upscale possible")'''),

("md", "The two-sided rule, with an empirical twist:\n"
"- **Source above the allowance** -> downscaled to the same 737 K px: 1080p vs 720p drift is tiny (see heatmap). Storing 1080p masters buys you (almost) nothing at inference time.\n"
"- **Source below the allowance** -> no detail can be recovered: the 360p embedding is clearly the odd one out (largest drift).\n"
"- The twist: 'poorer' is not 'worse for retrieval' - the 360p run again posts the *best* margins (0.21 vs ~0.10), for the same reason as Sweep C: less background clutter. The budget is a ceiling on *detail*, and detail is not the same thing as retrieval quality."),

("md", "## 6. Sweep E - clip duration (10 s / 30 s / 52 s of Sintel at `fps=1`)\n"
"Longer clips at fixed fps mean more frames -> more coverage but a smaller per-frame allowance (52 s hits the budget, 10/30 s do not). Is a 30 s embedding 'closer' to the full-clip embedding than a 10 s embedding is?"),

("code", r'''durs = {"sintel_10s": 10, "sintel_30s": 30, "sintel_trailer": 52}
embsE = {}
rows = []
for k, d in durs.items():
    entry = {"video": str(VIDEOS[k])}                       # default fps=1
    r = evaluate_config2(entry, 2, 3)
    embsE[k] = r.pop("emb")
    rows.append(dict(clip=k, clip_s=d, nframes=r["nframes"], px_per_frame=r["px_per_frame"],
                     tok_per_frame=r["tok_per_frame"], total_tokens=r["total_tokens"],
                     detail_margin=r["detail_margin"], holistic_margin=r["holistic_margin"],
                     time_s=r["time_s"]))
dfE = pd.DataFrame(rows)
full = embsE["sintel_trailer"]
for k in ["sintel_10s", "sintel_30s"]:
    print(f"drift({k} vs full 52s): {drift(embsE[k], full):.4f}")
dfE'''),

("md", "## 7. 8B spot-check\n"
"Does the 8B model tolerate squeezed frames better? We repeat the two extremes of Sweep A (4 vs 60 frames) and the 1080p-vs-360p pair from Sweep D on 8B. (Distractors and queries must be re-embedded - they live in 8B's own space!)"),

("code", r'''del embedder; torch.cuda.empty_cache()
embedder8 = load_embedder("8B")

distractor_emb8 = embed(embedder8, [{"video": str(VIDEOS[k])} for k in DISTRACTORS])
q_emb8 = embed(embedder8, [
    {"text": BBB_DETAIL, "instruction": RETR_INSTRUCTION},
    {"text": BBB_HOLISTIC, "instruction": RETR_INSTRUCTION},
])

def margin8(entry, qi=0):
    emb, _ = embed(embedder8, [entry], timed=True)
    stacked = torch.cat([emb, distractor_emb8])
    s = (q_emb8[qi:qi+1] @ stacked.T).float().cpu().numpy().flatten()
    return margins(s, 0)["margin"], inspect_entry(embedder8, entry)["videos"][0]

rows = []
for label, entry in {
    "A: 4 frames  (fps6, mf4)":   {"video": str(VIDEOS["bbb_1080"]), "fps": 6.0, "max_frames": 4},
    "A: 60 frames (fps6, mf64)":  {"video": str(VIDEOS["bbb_1080"]), "fps": 6.0, "max_frames": 64},
    "D: 1080p src (fps1, mf8)":   {"video": str(VIDEOS["bbb_1080"]), "fps": 1.0, "max_frames": 8},
    "D: 360p src  (fps1, mf8)":   {"video": str(VIDEOS["bbb_360"]),  "fps": 1.0, "max_frames": 8},
}.items():
    m, v = margin8(entry)
    rows.append(dict(config=label, px_per_frame=v["frame_pixels"], total_tokens=v["total_visual_tokens"],
                     detail_margin_8b=m))
pd.DataFrame(rows)'''),

("md", "## Cheat sheet\n"
"| knob | what it controls | gotcha |\n"
"|---|---|---|\n"
"| `fps` | frames per second sampled from a file-path video | budget allowance = `total*2/nframes`; dense fps squeezes frames |\n"
"| `max_frames` | hard cap on frames (uniform `linspace` sampling) | 64 frames x 128-token floor = 8192 tokens = `max_length` |\n"
"| `total_pixels` | pixel budget (frame-list mode; wrapper default 7,864,320) | effective spend = `2 x total_pixels / 1024` tokens; wasted beyond the 786 K px/frame ceiling |\n"
"| `MODEL_SEQ_LEN` env | budget for file-path inputs (`seq_len x 1024 x 0.9`) | read at *import* time of qwen_vl_utils |\n"
"| source resolution | only binds when below the per-frame allowance | above the allowance it is downscaled away - no benefit |\n"
"| clip duration | more frames at fixed fps | long clips hit the budget; consider cutting |\n"
"\n"
"Next: how the **instruction** (system prompt) moves queries and documents in this space (notebook 04)."),
]


# ===========================================================================
# 04 - Instruction impact
# ===========================================================================
nb04 = [
("md", "# 04 - How instructions (system prompts) move embeddings\n"
"Qwen3-VL-Embedding is **instruction-aware**: the instruction travels in the system message of the chat template, in front of your content. Two practical consequences:\n"
"\n"
"1. The *same text* embeds differently under different instructions - the instruction literally re-positions the vector.\n"
"2. Qwen reports **1-5% retrieval-quality differences** from instruction choice, and recommends **task-matched, English** instructions (that is what they trained on).\n"
"\n"
"Today we measure this ourselves:\n"
"- Part 1: exactly what the model consumes (the formatted chat).\n"
"- Part 2: text-query instruction variants (matched vs mismatched vs off-topic) -> margins, rank-1 accuracy, query drift.\n"
"- Part 3: the same for video retrieval.\n"
"- Part 4: instructions on the *document* (video) side.\n"
"\n"
"Note: the wrapper *always* injects a system prompt - if you pass no instruction you get the default `\"Represent the user's input.\"`."),

("code", SETUP),

("code", r'''embedder = load_embedder("2B")

# what does the model actually consume?
conv = embedder.format_model_input(text="What is a black hole?",
                                    instruction="Given a web search query, retrieve relevant passages")
print(embedder.processor.apply_chat_template([conv], tokenize=False, add_generation_prompt=True))
conv2 = embedder.format_model_input(text="What is a black hole?")   # no instruction -> default
print(embedder.processor.apply_chat_template([conv2], tokenize=False, add_generation_prompt=True))'''),

("md", "## Part 2 - text retrieval under different query instructions\n"
"4 queries, 8 short passages (2 relevant per query; one query deliberately has no relevant passage). The **documents are embedded once** (default instruction) - we only vary the *query-side* instruction."),

("code", r'''DOCS = {
    "d1_ev":    "Electric vehicles use large battery packs and electric motors instead of combustion engines. Regenerative braking recharges the battery while driving.",
    "d2_ev":    "Charging an EV at home overnight usually covers a typical daily commute; DC fast chargers can top up a battery in roughly thirty minutes.",
    "d1_tom":   "Tomato plants need consistent moisture. Deep watering once or twice a week encourages strong roots and prevents fruit cracking.",
    "d2_tom":   "Container-grown tomatoes dry out quickly, so check the soil daily in summer and water until it drains freely from the bottom.",
    "d1_coral": "The Great Barrier Reef is the largest coral reef system on Earth, visible from space and home to thousands of marine species.",
    "d2_coral": "Coral bleaching happens when warm water stresses corals, causing them to expel the symbiotic algae that give them color and food.",
}
Q_TEXTS = {
    "q_ev":    "How do electric cars and their chargers work?",
    "q_tom":   "How should I water my tomato plants?",
    "q_coral": "What is happening to the world's coral reefs?",
}
RELEVANT = {"q_ev": ["d1_ev", "d2_ev"], "q_tom": ["d1_tom", "d2_tom"], "q_coral": ["d1_coral", "d2_coral"]}
doc_labels = list(DOCS)
doc_emb = embed(embedder, [{"text": t} for t in DOCS.values()])
print(len(DOCS), "docs,", len(Q_TEXTS), "queries")'''),

("code", r'''VARIANTS = {
    "default":                     None,   # -> "Represent the user's input."
    "websearch (matched)":         "Given a web search query, retrieve relevant passages that answer the query",
    "similarity (matched)":        "Represent the query for semantic similarity matching",
    "sentiment (mismatched)":       "Classify the sentiment of the text as positive or negative",
    "translate (off-topic)":       "Translate the following text into French",
}

rows, q_embs = [], {}
for vname, instr in VARIANTS.items():
    qe = embed(embedder, [{"text": q, "instruction": instr} for q in Q_TEXTS.values()])
    q_embs[vname] = qe
    S = sim_matrix(qe, doc_emb)
    hits, mrgs = 0, []
    for qi, ql in enumerate(Q_TEXTS):
        rel = [doc_labels.index(d) for d in RELEVANT[ql]]
        irr = [i for i in range(len(doc_labels)) if i not in rel]
        mrgs.append(S[qi, rel].max() - S[qi, irr].max())
        hits += int(np.argmax(S[qi]) in rel)
    rows.append(dict(instruction=vname, rank1=f"{hits}/{len(Q_TEXTS)}",
                     mean_margin=np.mean(mrgs).round(3)))
dfV = pd.DataFrame(rows)
dfV'''),

("code", r'''plt.figure(figsize=(8, 3.2))
plt.bar(dfV.instruction, dfV.mean_margin, color=["#4878a8", "#2a9d8f", "#2a9d8f", "#e76f51", "#b0b0b0"])
plt.axhline(0, color="k", lw=0.6); plt.ylabel("mean margin (relevant - best irrelevant)")
plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

# how far does each instruction MOVE the query embedding?
vnames = list(VARIANTS)
D = np.array([[float(F.cosine_similarity(q_embs[a][i].float(), q_embs[b][i].float(), dim=0))
               for b in vnames] for a in vnames for i in range(len(Q_TEXTS))])
D = 1 - D.reshape(len(vnames), len(Q_TEXTS), len(vnames)).mean(axis=1)
heatmap(D, vnames, vnames, "query-embedding drift between instruction variants (1 - cos)", fmt=".2f", cmap="magma")'''),

("md", "Reading the drift matrix: a mismatched instruction does not just *nudge* the query - it can move it a long way (drift similar in magnitude to the distance between unrelated sentences from notebook 01). A moved query can still retrieve fine if all variants move queries *coherently*, but margins usually degrade. The off-topic instruction is the wildest mover."),

("md", "## Part 3 - video retrieval: query-side instruction variants"),

("code", r'''corpus_entries = [{"video": str(VIDEOS[k])} for k in CORPUS]
doc_emb = embed(embedder, corpus_entries)     # document side: default instruction

VIDEO_VARIANTS = {
    "default":                  None,
    "retrieve video (matched)": "Given a user query, retrieve the video that best matches the description",
    "find visual match":        "Find the video that visually matches the user's description",
    "rate quality (mismatched)": "Rate the artistic quality of the video on a scale from one to ten",
}
q_entries_flat, qlabels, qtruth = [], [], []
for k in CORPUS:
    for qi, q in enumerate(VIDEO_QUERIES[k]):
        qlabels.append(f"{k.split('_')[0]}_q{qi+1}"); qtruth.append(CORPUS.index(k))
        q_entries_flat.append({"text": q})

rows = []
for vname, instr in VIDEO_VARIANTS.items():
    qe = embed(embedder, [dict(e, instruction=instr) for e in q_entries_flat])
    S = sim_matrix(qe, doc_emb)
    hits, mrgs = 0, []
    for i, truth in enumerate(qtruth):
        m = margins(S[i], truth); mrgs.append(m["margin"]); hits += m["rank"] == 1
    rows.append(dict(instruction=vname, rank1=f"{hits}/{len(qtruth)}", mean_margin=np.mean(mrgs).round(3)))
pd.DataFrame(rows)'''),

("md", "## Part 4 - instructions on the *document* (video) side\n"
"Now queries keep the matched retrieval instruction and we vary what the **videos** are embedded 'for'."),

("code", r'''q_emb = embed(embedder, [dict(e, instruction=VIDEO_VARIANTS["retrieve video (matched)"]) for e in q_entries_flat])

DOC_VARIANTS = {
    "default":         None,
    "for retrieval":   "Represent the video for retrieval against user queries",
    "rate length (mismatched)": "Estimate how many seconds long this video is",
}
rows = []
for dname, dinstr in DOC_VARIANTS.items():
    de = embed(embedder, [dict(e, instruction=dinstr) for e in corpus_entries])
    S = sim_matrix(q_emb, de)
    hits, mrgs = 0, []
    for i, truth in enumerate(qtruth):
        m = margins(S[i], truth); mrgs.append(m["margin"]); hits += m["rank"] == 1
    # how much did the doc embeddings move?
    dd = np.mean([drift(de[i], doc_emb[i]) for i in range(len(corpus_entries))])
    rows.append(dict(doc_instruction=dname, rank1=f"{hits}/{len(qtruth)}",
                     mean_margin=np.mean(mrgs).round(3), doc_drift=round(float(dd), 3)))
pd.DataFrame(rows)'''),

("md", "## Findings & guidance\n"
"1. **Matched instructions win on margin and rank-1 accuracy** - both for text and video retrieval. The canonical phrasings (`Given a web search query, retrieve ...` / `Given a user query, retrieve the video ...`) are safe defaults.\n"
"2. **Mismatched instructions move queries a lot** (drift matrix) and usually cost margin - the embedding is conditioned on the task, not just the content.\n"
"3. The **document side is less sensitive** than the query side, but a mismatched doc instruction can still cost accuracy - keep both sides task-matched.\n"
"4. Qwen's practical advice: write instructions in **English**, describe the *task*, and be consistent between indexing time and query time. An instruction stored with documents at index time cannot be changed without re-embedding.\n"
"\n"
"Next: the **reranker** - a cross-encoder that reads query and document *together* (notebook 05)."),
]


# ===========================================================================
# 05 - Reranker vs embedding
# ===========================================================================
nb05 = [
("md", "# 05 - Reranker (cross-encoder) vs embedder (bi-encoder)\n"
"Two architectures, one pipeline:\n"
"\n"
"- **Embedding (bi-encoder)**: query and document are encoded *independently* into vectors; relevance = cosine. Cheap, indexable, one-shot - great for **recall** over millions of docs.\n"
"- **Reranker (cross-encoder)**: query and document are concatenated into *one* sequence; the model attends across both and outputs a relevance score - Qwen3-VL-Reranker does this by scoring the probability of the literal token **`yes`** vs **`no`** (a binary head built from the LM head). Expensive (one forward per pair) - great for **precision** on the top-k.\n"
"\n"
"Standard recipe: embed -> recall top-100 -> rerank top-10.\n"
"\n"
"Today: cosine vs reranker scores on our video corpus, hard cases the bi-encoder gets wrong, mixed text+video corpora, instruction sensitivity of the reranker, and the cost story."),

("code", SETUP),

("code", r'''embedder = load_embedder("2B")
reranker = load_reranker("2B")
print(f"reranker default instruction: {reranker.default_instruction!r}\n")

# the actual prompt the reranker consumes for one (query, video) pair:
pair = reranker.format_mm_instruction(
    query_text="A close-up of a flower",
    doc_video=str(VIDEOS["flower"]),
    instruction="Given a user query, retrieve the video that best matches the description",
)
print(reranker.processor.apply_chat_template([pair], tokenize=False, add_generation_prompt=True)[:1200])'''),

("md", "## 1. Score everything: cosine vs reranker\n"
"7 queries x 4 videos. Queries include grounded descriptions, a **hard thematic** query (describes the full Big Buck Bunny *plot*, which the 10 s excerpt does not show), an **ambiguous nature** query, and a negative. Videos use `fps=1, max_frames=16` for the reranker (relevance scoring needs less visual spend than embedding a library)."),

("code", r'''CANON = "Given a user query, retrieve the video that best matches the description"

QUERIES = {
    "bunny":  "A chubby grey rabbit in a green meadow in a 3D animated film.",
    "jelly":  "A translucent jellyfish with trailing tentacles drifting through deep blue water.",
    "flower": "A close-up of a pink flower outdoors with green foliage.",
    "sintel": "A red-haired girl on a snowy mountain in an animated fantasy film.",
    "revenge": "A gentle rabbit takes brutal revenge on three rodents who bully him.",
    "nature":  "Peaceful nature footage with plants and water.",
    "rocket":  "A rocket launch filmed at night.",
}

doc_emb = embed(embedder, [{"video": str(VIDEOS[k])} for k in CORPUS])
q_emb = embed(embedder, [{"text": q, "instruction": RETR_INSTRUCTION} for q in QUERIES.values()])
S = sim_matrix(q_emb, doc_emb)
heatmap(S, [c.split("_")[0] for c in CORPUS], list(QUERIES), "cosine (bi-encoder)")'''),

("code", r'''R = np.zeros((len(QUERIES), len(CORPUS)))
for qi, (qname, qtext) in enumerate(QUERIES.items()):
    R[qi] = reranker.process({
        "instruction": CANON,
        "query": {"text": qtext},
        "documents": [{"video": str(VIDEOS[k])} for k in CORPUS],
        "fps": 1.0, "max_frames": 16,
    })
heatmap(R, [c.split("_")[0] for c in CORPUS], list(QUERIES), "reranker score (cross-encoder)", fmt=".2f")'''),

("code", r'''qlabels = list(QUERIES)
clabels = [c.split("_")[0] for c in CORPUS]
rows = []
for qi, q in enumerate(qlabels):
    cos_top, rr_top = clabels[int(np.argmax(S[qi]))], clabels[int(np.argmax(R[qi]))]
    rows.append(dict(query=q, cosine_top1=cos_top, rerank_top1=rr_top,
                     cos_best=S[qi].max().round(3), rerank_best=R[qi].max().round(3),
                     agree="yes" if cos_top == rr_top else "FLIP"))
pd.DataFrame(rows)'''),

("md", "**What to look for**\n"
"- The two matrices broadly agree on easy grounded queries - the embedding already ranks them correctly.\n"
"- The interesting rows are `revenge` and `nature`:\n"
"  - `revenge`: the 10 s excerpt *shows no revenge*. A cross-encoder that actually 'watches' the video should score it low - watch whether cosine is more easily fooled by the word `rabbit` than the reranker.\n"
"  - `nature`: both flower and jellyfish are plausible; see whether the reranker separates them more decisively.\n"
"- `rocket` should be low across the board for both models."),

("md", "## 2. Mixed-modality corpus: videos vs their text captions\n"
"Because everything lives in one space, the 'documents' can be a **mix of 4 videos and 4 text captions**. Which wins for a content query - the video or its caption?"),

("code", r'''mixed_docs = [{"video": str(VIDEOS[k])} for k in CORPUS] + \
             [{"text": VIDEO_QUERIES[k][0]} for k in CORPUS]
mixed_labels = [f"video:{c.split('_')[0]}" for c in CORPUS] + [f"caption:{c.split('_')[0]}" for c in CORPUS]

md_emb = embed(embedder, mixed_docs)
q = "A chubby grey rabbit in a green meadow in a 3D animated film."
qe = embed(embedder, [{"text": q, "instruction": RETR_INSTRUCTION}])
scores_cos = (qe[0] @ md_emb.T).float().cpu().numpy()
order = np.argsort(-scores_cos)
print("cosine ranking (mixed corpus):")
for i in order:
    print(f"  {scores_cos[i]:.3f}  {mixed_labels[i]}")

scores_rr = reranker.process({"instruction": CANON, "query": {"text": q},
                             "documents": mixed_docs, "fps": 1.0, "max_frames": 16})
order = np.argsort(-np.array(scores_rr))
print("\nreranker ranking (mixed corpus):")
for i in order:
    print(f"  {scores_rr[i]:.3f}  {mixed_labels[i]}")'''),

("md", "## 3. The reranker is also instruction-conditioned\n"
"Same pairs, nonsense instruction -> watch the scores move."),

("code", r'''R_bad = np.zeros((len(QUERIES), len(CORPUS)))
for qi, (qname, qtext) in enumerate(QUERIES.items()):
    R_bad[qi] = reranker.process({
        "instruction": "Judge whether the document is currently trending on social media",
        "query": {"text": qtext},
        "documents": [{"video": str(VIDEOS[k])} for k in CORPUS],
        "fps": 1.0, "max_frames": 16,
    })
heatmap(R - R_bad, [c.split("_")[0] for c in CORPUS], list(QUERIES),
        "score delta: matched - mismatched instruction", fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
print("mean |delta|:", np.abs(R - R_bad).mean().round(3))'''),

("md", "## 4. Cost: why we don't rerank a million documents\n"
"Embedding amortizes: each document is embedded **once** and every query is one dot product. The reranker is a full forward pass **per (query, doc) pair** - quadratic in corpus x queries, and it must re-read the video every time."),

("code", r'''t0 = time.perf_counter(); _ = embed(embedder, [{"video": str(VIDEOS[k])} for k in CORPUS]); t_embed = time.perf_counter() - t0
t0 = time.perf_counter()
_ = reranker.process({"instruction": CANON, "query": {"text": list(QUERIES.values())[0]},
                      "documents": [{"video": str(VIDEOS[k])} for k in CORPUS],
                      "fps": 1.0, "max_frames": 16})
t_rerank = time.perf_counter() - t0

print(f"embed 4 videos once           : {t_embed:.2f}s  ({t_embed/4:.2f}s/doc, reused by every future query)")
print(f"rerank 4 videos for ONE query : {t_rerank:.2f}s  ({t_rerank/4:.2f}s/doc, re-paid for every query)")

# the standard pipeline: recall top-2 by cosine, rerank only those
qi = list(QUERIES).index("nature")
top2 = np.argsort(-S[qi])[:2]
rr = reranker.process({"instruction": CANON, "query": {"text": QUERIES["nature"]},
                       "documents": [{"video": str(VIDEOS[CORPUS[i]])} for i in top2],
                       "fps": 1.0, "max_frames": 16})
print("\npipeline (cosine top-2 -> rerank):",
      [clabels[i] for i in top2][int(np.argmax(rr))], "| full reranker top-1:", clabels[int(np.argmax(R[qi]))])'''),

("md", "## Takeaways\n"
"1. **Bi-encoder for recall, cross-encoder for precision.** The embedding gets you a cheap shortlist from an index; the reranker re-reads query+document jointly and is much harder to fool with superficial overlap.\n"
"2. On hard cases (thematic queries whose events are not visible in the clip, ambiguous queries) the cross-encoder's joint attention usually pays off - but it *reads the video again* for every pair.\n"
"3. The reranker is **also instruction-conditioned** - keep its `<Instruct>` matched to the task, exactly like the embedder.\n"
"4. In a mixed corpus, text captions and videos genuinely compete in the same space - cross-modal retrieval just works.\n"
"\n"
"**Where to go next** (not covered here): MRL dimension truncation (64-4096 dims from one 8B embedding), the 8B reranker, multilingual instructions, and indexing these vectors with a real vector DB."),
]


# ===========================================================================
# 06 - Event identification: chunk a video, embed the clips, find the fight
# ===========================================================================
nb06 = [
    ("md", "# 06 - Event identification: chunk a video, embed the clips, find the fight\n"
    "Notebooks 02-05 asked *which video in this corpus matches the query*. Event identification flips the direction: you hold long footage and a bank of event descriptions, and you want to know **where** - in which seconds - something happens. The standard recipe:\n"
    "\n"
    "> chunk the footage into short clips -> embed each clip once (index time) -> score text queries against all clips (query time) -> alert where a query lights up\n"
    "\n"
    "Today we run that pipeline on **`fight_2.mp4`** (1920x1080, 25 fps, 7.5 s - real fight footage supplied for this lab, `~/study/data/fight_2.mp4`), with the lab's benign corpus chunked the same way as negative control:\n"
    "1. Chunk everything into **5-second clips** with ffmpeg -> a 9-clip 'surveillance index' (2 fight + 7 benign).\n"
    "2. Embed the index and *inspect* what the model actually consumes per clip.\n"
    "3. Identify the fight by **text query** - score matrix and **margin** (fight clip - best benign clip).\n"
    "4. Embed the queries under **different instructions** and watch what happens to the margin.\n"
    "5. Same for instructions on the **clip (index) side**.\n"
    "6. The payoff: **margin = the room you have to place a detection threshold** - we compute the valid threshold window per instruction.\n"
    "\n"
    "(Background from earlier notebooks: instructions are system prompts - nb 04; one visual token = 1024 px and the pixel budget - nb 03.)"),

    ("code", SETUP + "\nimport subprocess\nfrom pathlib import Path"),

    ("md", "## Part 1 - Chunk the footage into 5-second clips\n"
    "Chunking is a plain ffmpeg job with one decision: `-c copy` (stream copy) is instant but cuts land on keyframes - up to seconds off. We **re-encode** (`libx264 -crf 23`) so every window starts exactly on time. Two details that matter for embedding:\n"
    "- the **tail chunk** is whatever is left (7.52 s -> 5.00 s + 2.52 s) - short chunks embed fine, the wrapper enforces a 4-frame floor,\n"
    "- positives and negatives must be **chunked and embedded identically**, so the benign corpus videos go through the exact same function.\n"
    "Derived clips land in `data/clips/` (regenerable; the lab never edits its sources)."),

    ("code", r'''SRC = "/home/satishv/study/data/fight_2.mp4"          # the event footage under test
CLIP_DIR = Path(LAB_ROOT) / "data" / "clips"          # derived data - regenerable

def chunk_video(src, chunk_s=5.0, out_dir=CLIP_DIR):
    """Cut src into chunk_s-second clips (re-encoded for frame-accurate cuts)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "csv=p=0", str(src)]).decode().strip())
    clips, start, i = [], 0.0, 0
    while start < dur - 0.1:                          # -0.1: never emit a zero-length tail
        end = min(start + chunk_s, dur); i += 1
        path = out_dir / f"{Path(src).stem}_c{i:02d}.mp4"
        if not path.exists():                        # cache: chunks are deterministic
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
                            "-t", f"{end - start:.3f}", "-i", str(src),
                            "-c:v", "libx264", "-crf", "23", "-an", str(path)], check=True)
        clips.append(dict(name=path.stem, path=str(path),
                          t0=round(start, 2), dur=round(end - start, 2)))
        start = end
    return clips

fight_clips = chunk_video(SRC)
benign_clips = []
for k in ["bbb_1080", "jellyfish_720", "flower", "sintel_10s"]:
    benign_clips += chunk_video(str(VIDEOS[k]))

INDEX = fight_clips + benign_clips
CLIP_LABELS = [c["name"] for c in INDEX]
FIGHT_IDX = list(range(len(fight_clips)))            # ground truth: clips 0..1 are the fight

print(f"{Path(SRC).name}: {len(fight_clips)} fight chunks + {len(benign_clips)} benign chunks")
for c in INDEX:
    tag = "FIGHT" if c in fight_clips else "benign"
    print(f"  [{tag:5s}] {c['name']:18s} t0={c['t0']:5.2f}s  dur={c['dur']:4.2f}s")'''),

    ("md", "## Part 2 - Embed the clip index & inspect what the model consumes\n"
    "The 9 clips are embedded once, default instruction on the clip side (we vary that in Part 5). Then the ritual from notebook 03 - **verify, don't trust**: what does the model actually consume for one 5 s 1080p chunk?"),

    ("code", r'''embedder = load_embedder("2B")

doc_emb = embed(embedder, [{"video": c["path"]} for c in INDEX])   # clip side: default instruction
print("clip index embeddings:", tuple(doc_emb.shape))

for c in fight_clips:
    info = inspect_entry(embedder, {"video": c["path"]})
    print(f"{c['name']} ({c['dur']:.2f}s): {summarize_inspection(info)}")
    show_frames(info["videos"][0]["tensor"], n=4, ncols=4,
                title=f"{c['name']} - frames the model consumes")'''),

    ("md", "Both fight chunks get **identical preprocessing**: exactly 4 frames at 1152x640 = 737,280 px = **720 tokens/frame -> 2,880 visual tokens per clip**. Why 4: `smart_nframes` computes `total_frames / video_fps * fps`, then floors to a multiple of `FRAME_FACTOR=2` with a floor of `FPS_MIN_FRAMES=4` - so the 5.00 s chunk rounds 5 -> 4, and the 2.52 s tail's ~2 gets lifted to 4. And the 1080p source (2.07 M px) exceeds the per-frame ceiling (768 tokens = 786 K px), so frames are downscaled to ~720 tokens - notebook 03's rule: *source resolution above the allowance is normalized away*.\n"
    "\n"
    "Eyeball the frames: this footage is one continuous altercation, so **both chunks are positives** - detection has to fire on both windows, not on one lucky cut."),

    ("md", "## Part 3 - Identify the event by text query\n"
    "Six queries against the index: three fight phrasings - **grounded visual vocabulary** (`throwing punches`), a **paraphrase** (`hitting and grabbing`), and a **vague** one (`violent altercation`) - plus three distractor events (`sports` is the interesting one: fighting and sports share 'bodies in vigorous physical activity'). Query side uses the canonical matched instruction; clip side stays default.\n"
    "**Margin** here = best fight-clip score minus best benign-clip score - the separation the detector lives on."),

    ("code", r'''QUERIES = {
    "fight (punches)":  "Two people fighting each other, throwing punches.",    # grounded visual detail
    "fight (grabbing)": "A physical fight between people, hitting and grabbing.", # paraphrase
    "fight (vague)":    "A violent altercation between two people.",             # abstract wording
    "sports":           "People playing sports on a field.",                      # distractor: shares 'vigorous activity'
    "cooking":         "Someone cooking in a kitchen.",                          # distractor
    "cars":             "Cars driving on a road.",                                # distractor
}
MATCHED = "Given a user query, retrieve the video that best matches the description"

q_emb = embed(embedder, [{"text": q, "instruction": MATCHED} for q in QUERIES.values()])
S = sim_matrix(q_emb, doc_emb)
heatmap(S, CLIP_LABELS, list(QUERIES), "clip <- query cosine (matched instruction)")

def event_margin(scores, pos_idx):
    """Best positive-clip score minus best negative-clip score: the separation margin."""
    s = np.asarray(scores, dtype=float)
    pos, neg = s[list(pos_idx)], np.delete(s, list(pos_idx))
    return dict(pos=float(pos.max()), neg=float(neg.max()),
                margin=float(pos.max() - neg.max()))

rows = []
for i, qname in enumerate(QUERIES):
    m = event_margin(S[i], FIGHT_IDX)
    rows.append(dict(query=qname, fight_clip=round(m["pos"], 3),
                     best_benign=round(m["neg"], 3), margin=round(m["margin"], 3)))
pd.DataFrame(rows)'''),

    ("code", r'''colors = ["#e76f51" if i in FIGHT_IDX else "#b0b0b0" for i in range(len(INDEX))]
plt.figure(figsize=(9.5, 3.4))
plt.bar(range(len(INDEX)), S[0], color=colors)
plt.axhline(0.45, color="k", ls="--", lw=1)
plt.text(len(INDEX) - 0.6, 0.463, "tau = 0.45", fontsize=8, ha="right")
plt.xticks(range(len(INDEX)), CLIP_LABELS, rotation=30, ha="right", fontsize=8)
plt.ylabel("cosine similarity")
plt.title("which 5s chunk lights up?  ('Two people fighting each other, throwing punches.')")
plt.tight_layout(); plt.show()'''),

    ("md", "**What to look for**\n"
    "- **The event is identified, to the chunk**: the fight query scores ~0.72 on both fight chunks while every benign chunk stays <= 0.41 - the two red bars are the only ones above any sane threshold.\n"
    "- **Phrasing quality = margin**: grounded visual detail (`throwing punches`, ~+0.33) > paraphrase (~+0.24) > abstract wording (`violent altercation`, ~+0.16). The lab's rule from notebook 02 holds for events too: ground your queries empirically before trusting them.\n"
    "- **The `sports` trap**: `sports` scores ~0.49 on the fight chunks (and beats its own best benign clip, margin ~+0.08) - 'vigorous physical activity' is shared vocabulary. A naive `sports` detector with a threshold below 0.49 would flag this footage.\n"
    "- `cooking` / `cars` behave: their best matches are benign chunks and nothing clears 0.45."),

    ("md", "## Part 4 - Different query instructions -> what happens to the margin?\n"
    "Notebook 04 showed instructions *move* embeddings; for detection the question is sharper - does the instruction change the **separation** between event clips and benign clips? Four variants (the wrapper always injects one; `None` means the default `'Represent the user's input.'`):"),

    ("code", r'''Q_VARIANTS = {
    "default":               None,      # -> "Represent the user's input."
    "retrieve (matched)":    MATCHED,
    "event (task-matched)":  "Given a description of an event, retrieve the video clips in which the event takes place",
    "classify (mismatched)": "Rate the artistic quality of the video on a scale from one to ten",
}

q_variant_S = {}
rows = []
for vname, instr in Q_VARIANTS.items():
    qe = embed(embedder, [{"text": q, "instruction": instr} for q in QUERIES.values()])
    Sv = sim_matrix(qe, doc_emb)
    q_variant_S[vname] = Sv
    ms = {qn: event_margin(Sv[i], FIGHT_IDX)["margin"] for i, qn in enumerate(QUERIES)}
    rows.append(dict(instruction=vname, **{qn: round(m, 3) for qn, m in ms.items()},
                     mean_fight_margin=round(float(np.mean([m for qn, m in ms.items()
                                                            if qn.startswith("fight")])), 3)))
pd.DataFrame(rows)'''),

    ("code", r'''fight_qidx = [i for i, qn in enumerate(QUERIES) if qn.startswith("fight")]
fight_qnames = [list(QUERIES)[i] for i in fight_qidx]
x = np.arange(len(fight_qidx)); w = 0.2
plt.figure(figsize=(9, 3.4))
for vi, vname in enumerate(Q_VARIANTS):
    ms = [event_margin(q_variant_S[vname][i], FIGHT_IDX)["margin"] for i in fight_qidx]
    plt.bar(x + (vi - 1.5) * w, ms, w, label=vname)
plt.axhline(0, color="k", lw=0.6)
plt.xticks(x, fight_qnames, fontsize=9)
plt.ylabel("margin (fight clip - best benign)")
plt.legend(fontsize=8)
plt.title("query instruction vs detection margin")
plt.tight_layout(); plt.show()'''),

    ("md", "**Reading the table/chart**\n"
    "- **Healthy instructions are interchangeable**: default, canonical `retrieve`, and the task-phrased `event` instruction all cluster within a couple hundredths of each other (margins ~+0.33 / +0.24 / +0.17). Same as notebook 04 found: on the query side the default is not the problem.\n"
    "- **A mismatched instruction eats the margin**: `classify` cuts the three fight margins to roughly half or less of their healthy values (+0.20 / +0.12 / +0.06) - one wrong sentence in the system prompt. The vague query is hit hardest: its margin ends up one benign lookalike away from a false negative."),

    ("md", "## Part 5 - Instructions on the clip (index) side\n"
    "The deeper asymmetry: query embeddings are made at *query time* - cheap to change. Clip embeddings are made at **index time** - baked into your vector store for millions of chunks. We re-embed the whole index under four clip-side instructions (query stays matched) and watch the fight scores, the margin, and how far the clip embeddings drift:"),

    ("code", r'''q_main = embed(embedder, [{"text": QUERIES["fight (punches)"], "instruction": MATCHED}])

C_VARIANTS = {
    "default":                  None,
    "for event retrieval":      "Represent the video clip for retrieval against event descriptions",
    "surveillance style":       "Represent the surveillance footage clip so that events in it can be found by text queries",
    "rate length (mismatched)": "Estimate how many seconds long this video is",
}
rows = []
for cname, cinstr in C_VARIANTS.items():
    ce = embed(embedder, [{"video": c["path"], "instruction": cinstr} for c in INDEX])
    sc = (q_main[0] @ ce.T).float().cpu().numpy()
    m = event_margin(sc, FIGHT_IDX)
    dr = float(np.mean([drift(ce[i], doc_emb[i]) for i in range(len(INDEX))]))
    rows.append(dict(clip_instruction=cname, fight_c01=round(sc[0], 3), fight_c02=round(sc[1], 3),
                     margin=round(m["margin"], 3), clip_drift=round(dr, 3),
                     both_above_045=bool(sc[0] > 0.45 and sc[1] > 0.45)))
pd.DataFrame(rows)'''),

    ("md", "**Reading the table**\n"
    "- **Re-tasking the index is mostly harmless**: `for event retrieval` and the surveillance phrasing drift the clips only ~0.07-0.09 and the fight chunks still score 0.63-0.69 - detection survives.\n"
    "- **A mismatched index instruction poisons detection**: `rate length` collapses the fight chunks to ~0.44-0.47 - at/near the 0.45 threshold (see `both_above_045`) - while drifting every clip by ~0.37. The damage is invisible at query time; you only notice missed alerts. The instruction is part of the **index schema**: to change it, re-embed everything (Qwen's docs say exactly this)."),

    ("md", "## Part 6 - Margin is the room for your detection threshold\n"
    "Detection needs a number: alert when `score > tau`. For a single query, the *valid* tau interval is exactly its margin's endpoints - `(best benign, best fight clip)` - anything in between fires on the right clips and nothing else. Run several queries off one shared `tau` and it must fit in the **intersection** of their windows. First the fire table at `tau = 0.45`:"),

    ("code", r'''TAU = 0.45
rows = []
for i, qname in enumerate(QUERIES):
    m = event_margin(S[i], FIGHT_IDX)
    fires = [CLIP_LABELS[j] for j in range(len(INDEX)) if S[i, j] > TAU]
    rows.append(dict(query=qname, fight_best=round(m["pos"], 3), benign_best=round(m["neg"], 3),
                     margin=round(m["margin"], 3), fires_at_045=", ".join(fires) or "(none)"))
pd.DataFrame(rows)'''),

    ("code", r'''print("valid shared-tau window over the 3 fight queries (intersection of per-query windows):")
for vname, Sv in q_variant_S.items():
    lo = max(event_margin(Sv[i], FIGHT_IDX)["neg"] for i in fight_qidx)
    hi = min(event_margin(Sv[i], FIGHT_IDX)["pos"] for i in fight_qidx)
    print(f"  {vname:22s} ({lo:.3f}, {hi:.3f})   width = {hi - lo:.3f}")

fig, ax = plt.subplots(figsize=(9.5, 2.8))
for r, (vname, Sv) in enumerate(q_variant_S.items()):
    lo = max(event_margin(Sv[i], FIGHT_IDX)["neg"] for i in fight_qidx)
    hi = min(event_margin(Sv[i], FIGHT_IDX)["pos"] for i in fight_qidx)
    ax.plot([lo, hi], [r, r], lw=8, solid_capstyle="butt", alpha=0.75,
            color="#e76f51" if vname == "classify (mismatched)" else "#2a9d8f")
    ax.text(hi + 0.012, r, f"width {hi - lo:.3f}", va="center", fontsize=8)
ax.axvline(TAU, color="k", ls="--", lw=1)
ax.set_yticks(range(len(q_variant_S))); ax.set_yticklabels(list(q_variant_S), fontsize=8)
ax.set_xlim(0.25, 0.95); ax.set_xlabel("shared threshold tau")
ax.set_title("the valid-tau window per query instruction")
plt.tight_layout(); plt.show()'''),

    ("md", "**Reading the table/chart**\n"
    "- At `tau = 0.45` all three fight phrasings fire **exactly** on the two fight chunks - but `sports` fires on them too (false alarm), and `cooking`/`cars` on nothing. **Absolute thresholds are per-query calibrated** - `sports`' window is only ~0.08 wide.\n"
    "- A shared `tau` must live in the intersection: healthy instructions leave a **~0.15-wide** window; the mismatched `classify` instruction shrinks it to **~0.06** - the vague query's positives sink to ~0.49 while benign noise stays ~0.43. One more sloppy phrasing and the window *closes* (empty intersection -> no single tau works at all).\n"
    "- Two robust alternatives when margins are thin: **rank-based firing** (alert on argmax / top-k, which only needs margin > 0 per query) or the nb-05 recipe - cheap embedding recall + **reranker** precision on the shortlist."),

    ("md", "## Findings & guidance\n"
    "1. **The pipeline works end to end**: chunk -> embed once -> query identified the fight in both 5-second windows (fight ~0.72 vs benign <= 0.41) with zero training. Chunk duration is your time resolution - and the 2.52 s tail chunk embeds fine (the 4-frame floor).\n"
    "2. **Margin is the detection currency**: grounded phrasing (~+0.33) beats paraphrase (~+0.24) beats vague (~+0.16) - and margin is literally the interval of valid thresholds. Thin margins mean fragile detectors.\n"
    "3. **Instructions, both sides**: a mismatched *query* instruction cuts margins roughly in half (+0.35 -> +0.20 on the grounded query; shared-tau window 0.15 -> 0.06). A mismatched *clip* instruction is worse - it **poisons the index at index time** (fight 0.72 -> 0.44, drift ~0.37, missed alert at tau = 0.45) and cannot be repaired at query time.\n"
    "4. **Watch the cross-event traps**: `sports` scored ~0.49 on fight footage. Calibrate per query, fire rank-based, or rerank the shortlist (notebook 05).\n"
    "\n"
    "**Exercise**: `~/study/data/fight_1.mp4` is 17.96 s of 2160x4096 portrait footage - point `SRC` at it and re-run (4 chunks; the 4K portrait source gets downscaled into the same per-frame budget anyway, nb 03's rule). Then try a chunk-length sweep (5 s -> 10 s -> 15 s): longer chunks blur *when* the event happened, shorter chunks cost more index embeddings - the budget-vs-granularity tension of notebook 03, but in time."),
]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
    out.mkdir(exist_ok=True)
    build(nb01, out / "01_text_similarity.ipynb")
    build(nb02, out / "02_video_text_similarity.ipynb")
    build(nb03, out / "03_sampling_pixel_budget.ipynb")
    build(nb04, out / "04_instruction_impact.ipynb")
    build(nb05, out / "05_reranker_vs_embedding.ipynb")
    build(nb06, out / "06_event_identification_fight.ipynb")
