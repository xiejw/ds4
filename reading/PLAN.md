# ds4.c — Reading Plan

A guide for navigating `ds4.c` (16,775 lines) without reading top-down. The file
is a single-translation-unit inference engine for DeepSeek V4 Flash with GGUF
loading, CPU reference kernels, a Metal backend, and a tokenizer. Read the
sections you need; the file is divided by `/* ====... */` banner comments.

## High-level architecture

```
                       +---------------------------+
                       |   Public API (ds4_*)      |   line 14687
                       |   engine open/close,      |
                       |   session sync/eval/save  |
                       +-------------+-------------+
                                     |
            +------------------------+------------------------+
            |                                                 |
   +--------v--------+                              +---------v--------+
   |  Tokenizer/Chat |  line 13412                  |  Session/Engine  |
   |  (BPE, prompt)  |                              |  state machine   |
   +--------+--------+                              +---------+--------+
            |                                                 |
            |                  +---- choose backend ---+      |
            |                  |                       |      |
   +--------v-----+   +--------v---------+   +---------v------v-------+
   | CPU layer    |   | Metal release    |   |  Metal diag/test paths |
   | execution    |   | decode + prefill |   |  (compare vs CPU ref)  |
   |  line 5862   |   |  line 10238      |   |  line 9635             |
   +----+---+-----+   +---------+--------+   +-----------+------------+
        |   |                   |                        |
        |   +-- attention 4438  +-- graph alloc 8196     +-- ref helpers 7733
        |   +-- MoE FFN  4857   +-- decode helpers 8552
        |   +-- HC pre  4028    +-- diag dumps   8090
        |
        |
   +----v-----------------+
   | CPU kernels (q-dot,  |   line 1358
   | rmsnorm, softmax,    |
   | RoPE, swiglu)        |
   +----------+-----------+
              |
   +----------v-----------+   +------------------------+
   | Weight binding /     |   | GGUF parser/loader    |
   | model validation     |   | tensor directory       |
   |  line 1901           |   |  line 804              |
   +----------+-----------+   +-----------+------------+
              |                           |
              +-------- shared --------+--+
                                       |
   +-----------------------------------v---------+
   | Helpers, alloc guards, thread pool, cursor  |   line 374
   | Quant block formats and IQ2 tables          |   line 115
   | Fixed model-shape constants (DS4_*)         |   line 73
   +---------------------------------------------+
```

### Key types (use as anchors when reading)

| Type | Line | Role |
| --- | --- | --- |
| `ds4_model` | 908 | mmap'd GGUF + metadata KV + tensor directory |
| `ds4_weights` / `ds4_layer_weights` | 1887 / 1877 | DS4-specific named pointer table over `ds4_model` |
| `ds4_mtp_weights` | 1899 | Multi-token-prediction (speculative) extra weights |
| `ds4_kv_cache` / `ds4_layer_cache` | 5893 / 5888 | Per-layer KV state for the CPU path |
| `ds4_metal_graph` | 7948 | Metal-side allocated tensor set (decode + prefill) |
| `struct ds4_engine` | 13535 | Top-level: model + vocab + weights + backend flags |
| `struct ds4_session` | 14781 | Per-conversation state: KV graph, logits, checkpoint |
| `ds4_thread_pool` | 610 | Persistent CPU worker pool used by parallel kernels |

### Backend split

The engine has two parallel implementations of the same forward pass:

1. **CPU reference path** — `ds4_cpu_layer_run` family, lines 5862–7732. Slow
   but the source of truth. All Metal output is compared against this in diag
   mode.
2. **Metal release path** — lines 7786–13411. Owns its own per-layer KV
   tensors and dispatches a fixed graph. The decode/prefill helpers live in
   8552–9634; the user-facing `metal_decode` / `metal_prefill` entry points
   are in 10238–13411.

## Recommended reading order (do NOT read top-down)

Pick the lane that matches your goal. Each lane is ordered shortest path first.

### Lane A — "I want to understand the public API surface"

1. `ds4.h` — read first, it lists the public symbols.
2. `ds4_engine_open` — line 15707.
3. `ds4_session_create` — line 15803.
4. `ds4_session_sync` — line 15885 (this is the main forward-pass driver).
5. `ds4_session_eval` — line 16158, plus `ds4_session_eval_speculative_argmax`
   at 16169 (MTP speculative decode).
6. `ds4_session_save_payload` / `ds4_session_load_payload` — 15138 / 15269.

### Lane B — "I want to understand the model architecture"

1. Banner at line 73 — fixed shape constants (head counts, dims, layer count).
2. `ds4_layer_weights` — line 1877 — list of per-layer tensors.
3. CPU layer driver — line 5862 banner; trace one layer:
   - HC pre-transform — line 4028 banner.
   - Attention (QKV proj, RoPE, sink-aware softmax) — line 4438 banner.
   - MoE FFN (router, shared + routed experts, SwiGLU, down proj) — line 4857.
4. `ds4_kv_cache` — line 5893 — read after you've seen attention.

### Lane C — "I want to understand GGUF loading"

1. Banner at line 804 — GGUF parser entry.
2. `ds4_cursor` — line 393 — the safe-read primitive.
3. Banner at line 115 — quant block formats (Q8_0, Q4_0, IQ2_XXS, Q2_K).
4. Banner at line 1901 — fixed weight binding (maps GGUF tensor names to
   `ds4_weights` fields and validates shapes).

### Lane D — "I want to understand the Metal backend"

1. Banner at line 7786 — `ds4_metal_graph` state.
2. Banner at line 8196 — Metal graph allocation (one decode set, one prefill).
3. Banner at line 10238 — release decode + prefill (the real path).
4. Banner at line 8552 — fused decode helpers and the older unfused fallbacks.
5. Banner at line 8090 — diagnostic dumps (no-ops unless env var set).
6. Banner at line 9635 — diagnostic comparisons against CPU reference (skip
   unless debugging a numerical regression).

### Lane E — "I want to understand tokenization / chat format"

1. Banner at line 13412 — tokenizer section.
2. `ds4_tokenize_text` — line 13966.
3. `ds4_encode_chat_prompt` — line 14033 and its `ds4_chat_*` helpers
   (14029–14179).

### Lane F — "I want to understand quantized math kernels"

1. Banner at line 1358 — kernels section header.
2. Read the helpers in order: dequant → matvec → fused norm/swiglu/rope.
3. Cross-reference quant block layouts at line 115 before reading the matvec
   loops.

### Lane G — "I'm chasing one named function/struct"

Use `grep -n '^static .*func_name(' ds4.c` or `grep -n '^typedef struct' ds4.c`
(53 typedefs, 231 static functions). Once you have the line number, look up
which banner it's under to get context, then read.

## Section banner index

Banner lines: 1, 73, 115, 374, 804, 1358, 1901, 4028, 4438, 4857, 5862, 7733,
7786, 8090, 8196, 8552, 9635, 10238, 13412, 14687, 14800.

| Line | Section |
| --- | --- |
| 1 | File header / includes |
| 73 | Fixed DeepSeek V4 Flash shape constants |
| 115 | GGUF quant block formats + IQ2 tables |
| 374 | Shared helpers, alloc guards, thread pool, cursor |
| 804 | GGUF parsing and model mapping |
| 1358 | Scalar conversion and quantized tensor kernels |
| 1901 | Fixed weight binding and model validation |
| 4028 | Hyper-connection transforms |
| 4438 | Attention projections, RoPE, attention output |
| 4857 | Mixture-of-experts FFN |
| 5862 | KV cache, compressors, CPU layer execution |
| 7733 | Metal reference comparison helpers |
| 7786 | Metal release graph state |
| 8090 | Metal diagnostic dump hooks |
| 8196 | Metal release graph allocation |
| 8552 | Metal decode release helpers and reference fallbacks |
| 9635 | Metal diagnostic comparisons |
| 10238 | Metal release decode and prefill |
| 13412 | Tokenizer and chat prompt encoding |
| 14687 | Engine API and process lock |
| 14800 | Session snapshot payloads |

## Companion split files

For sequential reading in five chunks (aligned on top-level `}` boundaries):
`ds4_part1.c` (1–3350), `ds4_part2.c` (3351–6695), `ds4_part3.c` (6696–10097),
`ds4_part4.c` (10098–13410), `ds4_part5.c` (13411–16775).

Mapping to logical sections:

- **part1** — header, shape, quant formats, helpers, GGUF loader, kernels, the
  first half of weight binding.
- **part2** — rest of weight binding, HC transforms, attention, MoE FFN,
  start of KV cache / CPU layer code.
- **part3** — end of CPU layer execution, Metal reference helpers, Metal
  graph state, dump hooks, graph allocation, decode release helpers, diag
  comparisons.
- **part4** — Metal release decode and prefill (largest single section).
- **part5** — tokenizer, chat encoding, engine API, session save/load,
  session eval / speculative decode.
