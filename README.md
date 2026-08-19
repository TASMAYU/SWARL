# SWARL
RL Post Training and Alignment Library
**S**equential **W**eighted **A**lignment **R**einforcement **L**earning

A lightweight library built from scratch in python and PyTorch for LLM post-training
and RL alignment (RLHF/RLAIF, reasoning-model RL). Built strictly for text sequences and LLM RL Post training 
, not general-purpose RL for robotics or games.

> 🚧 Early stage. Currently implements offline DPO end-to-end; online
> algorithms (REINFORCE, GRPO, PPO) are in active development. See
> [ROADMAP.md](./ROADMAP.md) for the full staged build plan.

## Why SWARL

Most teams reach for [TRL](https://github.com/huggingface/trl) or
[verl](https://github.com/volcengine/verl) for LLM post-training — both
excellent, production-grade libraries. SWARL isn't trying to replace
either. It exists as a from-scratch, deliberately readable implementation
of the same core ideas: DPO, PPO, GRPO, and the reward/verifier machinery
around them, built one algorithm at a time so each piece is understandable
in isolation rather than buried in a large distributed-systems codebase.

If you want to *understand* how DPO or GRPO actually works at the tensor
level, this is a smaller surface to read than verl's full dataflow engine.

## Install

```bash
git clone https://github.com/tasmayu/swarl.git
cd swarl
uv sync
```

(Not yet published to PyPI — see [Roadmap](#roadmap) / installation will
become `pip install swarl` once the API stabilizes past Stage 3.)

## Quick start — offline DPO

```python
from swarl.trainers.offline import DPOTrainer

data = [
    {"prompt": "What is the capital of France?",
     "chosen": "The capital of France is Paris.",
     "rejected": "I don't know."},
    # ... your preference pairs, or load a HF dataset like
    # trl-lib/ultrafeedback_binarized and map to this schema
]

trainer = DPOTrainer(model_name="Qwen/Qwen2.5-0.5B-Instruct", beta=0.1, lr=1e-6)
trainer.train(data, epochs=3, batch_size=2)
```

Any Hugging Face causal LM works via `model_name` — SWARL builds directly
on `transformers.AutoModelForCausalLM`, it doesn't reimplement model
loading.

## What's implemented today

| Algorithm | Status |
|---|---|
| DPO (offline preference tuning) | ✅ working |
| REINFORCE (online) | 🚧 in progress |
| GRPO (group-relative, DeepSeek-R1 style) | ⏳ planned |
| PPO (clipped surrogate + GAE) | ⏳ planned |
| DAPO, GSPO, CISPO, DDPO | ⏳ planned |
| On-policy distillation (OPD) | ⏳ planned |

Full detail, sequencing, and reasoning for the build order in
[ROADMAP.md](./ROADMAP.md).

## Project structure

```
swarl/
├── core/         # PolicyModel, ReferenceManager, generation engine, FSDP/quantization
├── data/         # dataset classes (offline preference pairs, online prompts)
├── losses/       # pure-math loss functions, one file per algorithm
├── trainers/     # training loops that wire core+data+losses together
├── rewards/      # rule-based verifiers, sandboxed code exec, neural reward models
└── hardware/     # device detection (planned)
```

## Contributing

Contributions are welcome, especially on anything marked `⏳ planned` in
the roadmap. Please open an issue before starting a large PR so we can
align on approach — see [CONTRIBUTING.md](./CONTRIBUTING.md).

Good first issues are labeled
[`good-first-issue`](https://github.com/tasmayu/swarl/labels/good-first-issue)
on the issue tracker.

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgments

Architecture and staging heavily inspired by
[verl](https://github.com/volcengine/verl)'s dataflow design, scaled down
for a single-maintainer, single-GPU-first development process.
