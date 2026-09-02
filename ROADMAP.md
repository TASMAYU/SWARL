# SWARL — Master Plan

A from-scratch, lightweight Python + PyTorch library for LLM post-training
and RL alignment (RLHF/RLAIF, reasoning-model RL). Built strictly for text
sequences — not general-purpose RL for robotics or games. Architecturally
inspired by [verl](https://verl.readthedocs.io)'s dataflow design, scaled
down deliberately for single-maintainer, single-GPU-first development.

Legend: ✅ done · 🚧 in progress · ⏳ planned · 💰 budget constraint (designed
for, not yet buildable/testable — needs hardware not currently available)

---

## Table of Contents
1. [Design Philosophy](#design-philosophy)
2. [Stage-by-Stage Build Plan](#stage-by-stage-build-plan)
3. [Repository Structure](#repository-structure)
4. [Dataset Handling](#dataset-handling)
5. [Inference Engines: vLLM / TensorRT-LLM](#inference-engines-vllm--tensorrt-llm)
6. [Training Efficiency (No Kernel Programming)](#training-efficiency-no-kernel-programming)
7. [Testing on Remote GPU Hardware](#testing-on-remote-gpu-hardware)
8. [Hardware Compatibility Scope](#hardware-compatibility-scope)
9. [Deployment & Release Process](#deployment--release-process)
10. [Open Source Structure](#open-source-structure)
11. [Honest Scope Note](#honest-scope-note)

---

## Design Philosophy

The build order follows one rule: **isolate the piece with the fewest
moving parts, prove it fully correct, then add exactly one new source of
complexity at a time.**

This is why the library starts with DPO and not PPO or GRPO. DPO needs
only two things every later algorithm needs anyway — a model that computes
log-probabilities (`PolicyModel.sum_logprobs()`), and a dataset loader.
No generation engine, no reward function, no advantage estimator, no
critic network. Starting with an online algorithm would mean debugging
generation, reward scoring, advantage math, and the training loop all
simultaneously with nothing proven to work as a baseline — the classic way
solo RL projects stall out.

Each subsequent stage reuses everything proven in the previous stage and
adds **one** new capability:

```
DPO            → model + data + loss                (no generation)
REINFORCE      → + generation                        (one new thing)
GRPO           → + group-relative advantage           (one new thing)
PPO            → + critic/value network                (one new thing)
FSDP2          → + distributed sharding                 (one new thing, only once algorithms are correct)
```

---

## Stage-by-Stage Build Plan

### Stage 1 — Offline DPO Foundation ✅ (~90%)
The smallest correct slice of the final system.

| File | Purpose | Status |
|---|---|---|
| `core/models.py` | `PolicyModel` (HF model wrapper + log-prob computation), `ReferenceManager` (auto clone/freeze reference copy for KL control) | ✅ |
| `data/preference_dataset.py` | Chosen/rejected pair dataset + collator | ✅ |
| `data/prompt_dataset.py` | Prompt-only dataset, ready for online stages | 🚧 (file created, not yet written) |
| `losses/dpo.py` | DPO loss math | ✅ |
| `trainers/offline.py` | Wires the above into a training loop | ✅ |
| `examples/train_dpo_demo.py` | Runnable script proving it all works | ⏳ |

**Hardening additions (same stage, not a new one):**
- Chat-template formatting via `tokenizer.apply_chat_template` instead of raw string concatenation, since instruct models expect specific formatting.
- `trust_remote_code=True` support for custom model architectures.
- Robust pad/eos token fallback for models that don't define one sensibly.
- `data/loaders.py` — see [Dataset Handling](#dataset-handling) below.
- `tests/` — unit tests for the DPO loss (known values), loss-mask alignment, collator padding; the regression net every later stage reuses.

**Note on HF model support:** already works for any model on the HF Hub —
`PolicyModel` builds directly on `AutoModelForCausalLM.from_pretrained(model_name)`,
so this isn't a separate future stage, it's built in by construction.

### Stage 2 — Online Loop Skeleton (REINFORCE) ⏳
First time generation enters the loop.

| File | Purpose |
|---|---|
| `core/batch.py` | `TextRLBatch` — prompts + generated responses + rewards in one container |
| `core/generation.py` | HF `generate()` backend, behind an interface later backends implement identically |
| `losses/reinforce.py` | Simplest online loss — no clipping, no critic |
| `trainers/online.py` | rollout → reward → loss → update loop |

Reward for this stage is deliberately trivial (length/keyword check) — the
goal is proving the loop runs, not a real reward signal yet.

### Stage 3 — GRPO + DAPO ⏳
Reuses Stage 2's loop unchanged; swaps the loss.

| File | Purpose |
|---|---|
| `losses/grpo.py` | Group Relative Policy Optimization — generate a group of completions per prompt, normalize reward relative to the group mean/std, no critic network. The DeepSeek-R1 method; the flagship algorithm of this library. |
| `losses/dapo.py` | GRPO variant with asymmetric clip bounds + dynamic sampling, fixes GRPO's entropy-collapse failure mode |

### Stage 4 — Real Reward Layer ⏳
Replaces Stage 2's trivial reward with the actual verification engine.

| File | Purpose |
|---|---|
| `rewards/base.py` | Registry selecting a reward function by data source, falls back to a neural reward model (mirrors verl's `RewardManager`) |
| `rewards/verifiers.py` | Regex format-checkers (e.g. `<think>` tag validation), sandboxed code execution + unit-test grading |
| `rewards/models.py` | Load an external HF sequence-classification reward model, batch-score (prompt, response) pairs |

### Stage 5 — Decoupled Generation (vLLM) ⏳
`core/generation.py` gains a vLLM backend behind the same interface as the
HF backend from Stage 2 — trainers don't change at all. Includes the
**weight-sync step**: after every gradient update, the policy's weights
change, but vLLM keeps its own separate loaded copy for fast inference
which goes stale immediately. The generation engine must push the updated
`state_dict()` into vLLM before the next rollout. This is real
infrastructure work, which is why this stage is sequenced *after* Stages
2–4 already work correctly on the slower HF backend — get the algorithm
correct first, then add the speed layer and its complexity on top of
something already known to work.

### Stage 6 — Classic PPO ⏳
`losses/ppo.py` — clipped surrogate objective + GAE. Needs a
`ValueModel`/critic head — the first algorithm in the library requiring
one, hence its own stage rather than bundled with GRPO.

### Stage 6.5 — Training Efficiency Layer ⏳
No CUDA/kernel programming required anywhere in this stage — see
[Training Efficiency](#training-efficiency-no-kernel-programming) for the
full list. Sits here because these single-GPU wins should be exhausted
before reaching for multi-GPU sharding in Stage 7; a lot of projects reach
for FSDP too early when LoRA + gradient checkpointing alone would let a
model fit on one GPU.

### Stage 7 — Scale & Memory (Single-Node) ⏳
| File | Purpose |
|---|---|
| `core/parallel.py` | FSDP2 (`torch.distributed.fsdp.fully_shard`) sharding across GPUs on one machine |
| `core/quantization.py` | 4-bit freezing for `ReferenceManager` (never needs gradients — cheapest memory win, no algorithm changes) |

Only attempted once Stages 1–6 are algorithmically correct on a single
GPU. Testable on a multi-GPU single-node instance if a provider ever
offers one — not blocked by lacking a full cluster.

> **Stages 1–7 are the complete, demoable deliverable. This is the real
> CV-ready library.** Everything below is what turns this into a
> long-running project rather than a bounded one.

### Stage 8 — Newer Optimized Algorithms ⏳
| File | Purpose |
|---|---|
| `losses/gspo.py` | Group Sequence Policy Optimization — sequence-level rather than token-level importance ratios, addresses GRPO instability on long reasoning chains |
| `losses/cispo.py` | Clipped IS-weight Policy Optimization variant |
| `losses/ddpo.py` | Denoising/diffusion-style policy optimization variant |
| `losses/spin.py`, `losses/sppo.py` | Self-play: train the active checkpoint against frozen earlier versions of itself |

### Stage 9 — On-Policy Distillation (OPD) ⏳
Architecturally different from Stages 1–8: needs a **teacher** model
generating alongside the student, and the training and inference engines
must agree on tokenization/logit alignment between two different models,
not just one.

| File | Purpose |
|---|---|
| `trainers/opd.py` | Student learns live from a larger teacher's outputs (on-policy, not a static distillation dataset) |
| `losses/opd_variants.py` | Sequence-level KD, reverse-KL variants |

Sequenced after Stage 5's generation engine is solid — OPD is generation-
engine correctness under harder constraints (two synchronized models), not
new math on top of shaky infra.

### Stage 10 — Multi-Node Cluster Scaling 💰 (budget constraint) ⏳
Everything here needs a multi-GPU cluster to actually test — designed for
now, built and validated later when hardware is available.

- Megatron-LM backend — tensor + pipeline parallelism for large clusters
- DeepSpeed-Ulysses — long-sequence parallelism across nodes
- Training/inference engine alignment at cluster scale (the problem OPD
  previews at single-node scale, now across nodes)
- Fault tolerance — auto-recover training state if a node dies
- FP8 training math, NVFP4 quantization-aware training

### Stage 11 — Deferred ⏳ / 💰
- `hardware/platform.py` — CUDA/ROCm/Ascend NPU auto-detection
- DeepSeek-style MoE attention masking support
- Multi-turn conversational rollout support

---

## Repository Structure

```
swarl/
├── core/
│   ├── models.py          # PolicyModel, ReferenceManager        [Stage 1]
│   ├── batch.py             # TextRLBatch                          [Stage 2]
│   ├── generation.py         # HF -> vLLM rollout engine             [Stage 2/5]
│   ├── parallel.py           # FSDP2                                  [Stage 7]
│   └── quantization.py       # 4-bit, FP8, NVFP4                       [Stage 7+]
├── data/
│   ├── preference_dataset.py                                       [Stage 1]
│   ├── prompt_dataset.py                                            [Stage 1]
│   └── loaders.py            # HF hub / local file / dict adapters   [Stage 1]
├── losses/
│   ├── dpo.py               [Stage 1]
│   ├── reinforce.py           [Stage 2]
│   ├── grpo.py, dapo.py        [Stage 3]
│   ├── ppo.py                    [Stage 6]
│   ├── gspo.py, cispo.py, ddpo.py, spin.py, sppo.py    [Stage 8]
│   └── opd_variants.py                                    [Stage 9]
├── trainers/
│   ├── offline.py           [Stage 1]
│   ├── online.py              [Stage 2/3/6]
│   └── opd.py                    [Stage 9]
├── rewards/
│   ├── base.py, verifiers.py, models.py    [Stage 4]
├── hardware/
│   └── platform.py            [Stage 11, deferred]
├── examples/
│   └── train_dpo_demo.py     [Stage 1]
├── tests/
├── pyproject.toml
├── README.md
├── ROADMAP.md                 (this file)
├── CONTRIBUTING.md
└── LICENSE
```

---

## Dataset Handling

Right now `DPOTrainer.train(records)` only accepts a hand-typed Python
list. A reusable library needs a normalization layer in front of that.

**`data/loaders.py`** provides three entry points, all producing the same
internal record shape (`{"prompt","chosen","rejected"}` or
`{"prompt","meta"}`):

```python
load_from_hf(dataset_name, split, schema_map)   # datasets.load_dataset() + column renaming
load_from_local(path, format="jsonl"|"csv"|"parquet")
load_from_dicts(records)                          # the base case, already supported today
```

The core idea is a **schema adapter**: every HF preference dataset names
its columns differently (`"chosen"` vs `"chosen_response"` vs a full
chat-message list). `schema_map` tells the loader which source column maps
to which internal field, so one function handles every dataset instead of
custom code per dataset:

```python
records = load_from_hf(
    "trl-lib/ultrafeedback_binarized",
    split="train",
    schema_map={"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
)
```

For local files, `.jsonl` (one JSON object per line — the most common
preference-data export format) is the priority; `.csv` and `.parquet` are
straightforward additions via `datasets.load_dataset("json", data_files=path)`.

This isn't its own stage — it folds into Stage 1 alongside the
chat-template hardening.

---

## Inference Engines: vLLM / TensorRT-LLM

**These are independent tools, not code SWARL reimplements.** They're
separate PyTorch packages (`vllm`, `tensorrt-llm`) that `core/generation.py` wraps
behind one interface, so trainer code never needs to know which backend is
active:

```python
engine = GenerationEngine(backend="vllm")   # or "hf", or later "sglang"
responses = engine.generate(prompts)
```

**Where inference is actually needed — broader than just OPD:**

| Stage/algorithm | Needs inference/generation? |
|---|---|
| DPO (Stage 1) | No — scores static, pre-existing text only |
| REINFORCE, GRPO, PPO (Stages 2, 3, 6) | Yes — every training step samples fresh responses from the current policy |
| OPD (Stage 9) | Yes, and harder — needs synchronized inference from *two* models (teacher + student) |

Generation is the defining feature of every *online* algorithm from Stage
2 onward, not an OPD-specific concern. DPO is the sole exception, which is
exactly why Stage 1 could skip the generation engine entirely.

---

## Training Efficiency (No Kernel Programming)

Kernel-level CUDA work is explicitly out of scope. Almost none of the real
speed gains in this space need it — they're either a flag or an existing
library.

| Technique | What it does | Effort |
|---|---|---|
| Mixed precision (bf16) | Already in `PolicyModel` — halves memory, faster matmuls | Done |
| FlashAttention-2 | `attn_implementation="flash_attention_2"` in `from_pretrained` | ~1 line |
| Gradient checkpointing | `model.gradient_checkpointing_enable()` — trades compute for memory | ~1 line |
| LoRA / PEFT | Train small low-rank adapters instead of all weights, via the `peft` library | Wrap `PolicyModel` optionally |
| Sequence packing | Concatenate short examples into one packed sequence instead of padding | Moderate, `data/` layer |
| Gradient accumulation | Simulate a larger batch size across several micro-batches | A few lines in trainer loop |
| `torch.compile` | JIT-compiles the forward pass into fused kernels automatically | `model = torch.compile(model)` |
| Length-grouped batching | Bucket similarly-lengthed sequences together, minimize padding waste | `data/` layer |

---

## Testing on Remote GPU Hardware

For development on a notebook-provided GPU (e.g. a 96GB instance) while
writing code in local VS Code:

1. **VS Code Remote-SSH (preferred, if the provider allows it)** — install
   the Remote-SSH extension, connect to the notebook's SSH endpoint. VS
   Code opens a full session *on* the GPU machine*: same editor, terminal
   and Python interpreter run where the GPU actually is. No sync step.
2. **Git push/pull (works everywhere, including SSH-less notebooks)**:
   ```bash
   # locally
   git add -A && git commit -m "wip" && git push
   ```
   ```python
   # in the notebook
   !git clone https://github.com/<user>/swarl.git   # first time
   %cd swarl && !git pull
   !uv sync && !python examples/train_dpo_demo.py
   ```
3. **VS Code Jupyter extension against a remote kernel** — useful for
   interactive tensor debugging, not for running the actual trainer files
   as programs long-term.

---

## Hardware Compatibility Scope

PyTorch's CUDA layer already abstracts away which specific GPU is in use —
`.to("cuda")` behaves identically on a 24GB consumer card or an 80GB+
datacenter card. What changes across GPUs is capacity (batch size, model
size before OOM), not code paths.

| Scope | Stages covered | Testable how |
|---|---|---|
| Single GPU, any size | 1–6, 6.5 | Directly, on the notebook GPU |
| Single-node, multiple GPUs | 7 (FSDP2) | If the provider ever offers a multi-GPU instance |
| Multi-node cluster | 10 | 🚫 Genuinely untestable without real multi-node infra — designed to the right interface, validated later |

Single-GPU-only for now is the correct and complete choice for Stages
1–6.5, not a limitation being settled for.

---

## Deployment & Release Process

Deployment is not a one-time event at project completion — it's an
ongoing release cycle running alongside continued development.

1. **Publish `0.1.0`** to PyPI (`uv publish`) once Stage 1 is stable.
   `pip install swarl` becomes available to anyone.
2. **Development continues on `main`** exactly as now — new stages don't
   affect anyone who already installed `0.1.0`; their copy is frozen.
3. **Publish `0.2.0`, `0.3.0`, ...`** as later stages land. Users upgrade
   on their own schedule via `pip install --upgrade swarl`.
4. **Semantic versioning**: patch (`0.1.1`) for bug fixes, minor (`0.2.0`)
   for additive features (e.g. a new `GRPOTrainer` alongside `DPOTrainer`),
   major (`1.0.0`) reserved for breaking changes (e.g. renaming
   `DPOTrainer.train()`'s arguments).
5. **Tag GitHub Releases** with changelogs for each version, so the
   project's history is legible without reading raw commits.
6. **Before PyPI is worth setting up**, `pip install git+https://github.com/<user>/swarl.git`
   already works and is the right thing to point early users at while the
   API is still moving (Stages 1–3).
7. **Docs site** (optional, once there's enough surface area): `mkdocs` +
   Read the Docs, turning docstrings into browsable documentation — the
   same stack verl itself uses.

---

## Open Source Structure

- `README.md` — front page: quick start, algorithm status table, project
  structure, link to this roadmap.
- `CONTRIBUTING.md` — one paragraph pointing at the staged roadmap so
  contributors don't jump ahead (e.g. submit a Megatron-LM PR before
  Stage 2 exists).
- **Issue tracker labels**: `good-first-issue`, `help-wanted`,
  `algorithm` (new loss functions), `infra` (generation/parallelism),
  `bug`, `docs`.
- **One issue per unimplemented stub file** (e.g. `losses/reinforce.py`,
  `core/generation.py`), describing the interface it needs to satisfy —
  gives contributors a bounded, well-specified task instead of "help build
  an RL library." This is how verl's and TRL's trackers are structured in
  practice.
- `LICENSE` — MIT.

---

## Honest Scope Note

Stages 1–7 (plus 6.5) form a complete, demoable, single-GPU RL alignment
library — DPO, REINFORCE, GRPO, PPO, a real reward layer, vLLM generation,
and single-node FSDP2 scaling. That is the actual deliverable and the
strong, finishable CV story.

Stages 8–11 are where this becomes a long-running project rather than a
bounded one. Treat them as a living roadmap to revisit and extend over
time, not a checklist to feel behind on.
