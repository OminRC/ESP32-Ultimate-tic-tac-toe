# Training on Kaggle (GPU)

Use **`train_kaggle_mp.py`** (not `train_kaggle.py`). The model is tiny, so
a single batched NN forward call finishes almost instantly -- the real
bottleneck is the pure-Python MCTS tree traversal, which is single-threaded
and only ever pins one CPU core no matter how big you make `--parallel`.
`train_kaggle_mp.py` runs self-play across several separate OS processes
(one per CPU core, each doing its own CPU-only inference) while the main
process trains on GPU each iteration and re-saves weights for the workers
to pick up next round. `train_kaggle.py` is kept as the simpler
single-process version if you ever want to read/compare the logic.

## Setup

1. Create a new Kaggle Notebook.
2. Settings (right sidebar) -> Accelerator -> **GPU T4 x2** (or whatever's
   available) -> Internet: on (only needed once, to `pip install torch` if
   it's not already in the image -- it usually is).
3. Upload this `train/` folder's files as a Kaggle **Dataset**, or just
   create each file via a notebook cell (`%%writefile game.py`, etc.) --
   whichever's easier. You need: `game.py`, `mcts.py`, `net_torch.py`,
   `train_kaggle.py`, `export_weights.py`.

## Run

```python
import os; print(os.cpu_count())   # check how many CPU cores you actually have

!python train_kaggle_mp.py \
    --iterations 150 \
    --workers 4 \
    --parallel-per-worker 64 \
    --sims 150 \
    --out /kaggle/working/policy_params.npz
```

- `--workers`: set to your CPU core count (printed above; Kaggle's free tier
  is usually 4). Each worker pins one core running self-play independently.
- `--parallel-per-worker`: games run in lockstep *within* each worker
  process. Total games/iteration = `workers * parallel-per-worker`.
- `--sims`: MCTS simulations per move. Higher = stronger/slower self-play.
  150-200 is a solid upgrade over the local run's 50.
- Watch the session's resource panel while it runs: CPU should now show
  closer to `workers * 100%` instead of pinning a single core, which is
  what "using the CPUs properly" looks like for this workload.
- Checkpoints save every iteration to `--out`, so you can stop anytime
  (Kaggle sessions cap at ~9h/week GPU quota and 12h per session) and still
  keep the latest weights.
- To continue from a previous checkpoint: add `--resume /path/to/previous.npz`.

## After training

Download `/kaggle/working/policy_params.npz` (Kaggle's output file panel),
then either:

- Export to C directly on Kaggle:
  ```python
  !python export_weights.py /kaggle/working/policy_params.npz --out /kaggle/working/model_weights.h
  ```
  and download `model_weights.h`, or
- Copy `policy_params.npz` back into your local `train/` folder and run
  `python export_weights.py policy_params.npz --out ../main/model_weights.h`
  as usual.

Either way, check quality before deploying:
```python
!python eval_vs_random.py /kaggle/working/policy_params.npz --games 200
```
