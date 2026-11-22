# SWARL examples

Runnable scripts that train **real models on real data**, one per stage.
They show how to use SWARL and prove each algorithm actually learns.

> `examples/` vs `tests/`: tests run on CPU in seconds with tiny random
> models and check that the math and plumbing are correct (`pytest`).
> Examples run on a GPU with real models and check that training works.

| Script | Stage | What it shows | Hardware |
|---|---|---|---|
| `train_dpo_demo.py` | 1 (DPO) | Qwen2.5-0.5B-Instruct on `trl-lib/ultrafeedback_binarized`; held-out reward accuracy rises above 0.5 | 1 GPU, 16–24 GB |
| `train_grpo_gsm8k.py` | 3 (GRPO) | *planned* | |
| `train_ppo_gsm8k.py` | 5 (PPO) | *planned* | |

## Running the DPO demo

```bash
uv sync
uv run python examples/train_dpo_demo.py
# smaller GPU:
uv run python examples/train_dpo_demo.py --batch-size 2 --max-length 384
# bigger run:
uv run python examples/train_dpo_demo.py --train-size 2000 --epochs 2
```

On a notebook (Colab / Kaggle / cloud):
```python
!git clone https://github.com/TASMAYU/SWARL.git
%cd SWARL
!pip install -q torch transformers datasets accelerate
!python examples/train_dpo_demo.py
```

### Expected output

```
[eval before] loss 0.6931 | acc 0.000 | margin 0.0000   (policy == reference, ...)
epoch 0 step 0 | loss 0.6931 | acc 0.00 | margin 0.000
...
[eval after ] loss 0.6xxx | acc 0.6xx | margin 0.0xxx
PASS: held-out reward accuracy > 0.5 with a positive margin -- DPO is learning.
```

The "before" numbers are exactly log 2 / 0 / 0 because the policy starts
identical to the reference. **Stage 1 is done when the "after" accuracy is
clearly above 0.5 on the held-out pairs.**

All options: `python examples/train_dpo_demo.py --help`.
