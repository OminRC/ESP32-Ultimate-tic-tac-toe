# Ultimate Tic-Tac-Toe AI on ESP32-S3

An AlphaZero-lite policy+value network for [Ultimate Tic-Tac-Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe),
trained via self-play MCTS (pure NumPy/PyTorch, no TFLite Micro), exported as
plain C float arrays, and run as greedy single-forward-pass inference on an
**ESP32-S3** (tested on an N8R2 module: 8MB flash / 512KB SRAM). No external
sensors or display modules needed — play against it over the Serial monitor.

```
[self-play + MCTS]  -->  [tiny MLP: 126 -> 96 -> 64 -> {81, 1}]  -->  [C float arrays]  -->  [ESP32-S3 inference]
      (PC / Kaggle GPU)              ~24K params, ~93KB                                        Serial-monitor game
```

## Repository layout

```
train/      Training pipeline (Python) -- self-play MCTS, network, export to C
main/       ESP-IDF project (main.c, game engine, inference, model weights)
arduino/    Arduino IDE version of the same firmware
```

## Quick start

### 1. Train a model

Locally (NumPy, CPU, slow but dependency-free):
```bash
cd train
pip install -r requirements.txt
python train.py --iterations 25 --games-per-iter 25 --sims 50 --out policy_params.npz
python eval_vs_random.py policy_params.npz --games 100
```

Or on [Kaggle](https://www.kaggle.com) with a free GPU (much faster; see
[`train/KAGGLE.md`](train/KAGGLE.md) for the full walkthrough, and
[`train/ultimate_ttt_kaggle.ipynb`](train/ultimate_ttt_kaggle.ipynb) for a
ready-to-import notebook). `train_kaggle_mp.py` runs self-play across
multiple CPU-core worker processes (the model is too small for the GPU
itself to be the bottleneck) and includes early stopping against a
random-move baseline, so you don't have to guess how many iterations you
need.

### 2. Export the trained weights to C

```bash
python export_weights.py policy_params.npz --out ../main/model_weights.h
```

The model is ~93KB as float32 — small enough that no INT8 quantization is
needed; it just lives in flash as a `const` array.

### 3. Build and flash

**ESP-IDF:**
```bash
cd ..
idf.py set-target esp32s3
idf.py -p <PORT> build flash monitor
```

**Arduino IDE:** open [`arduino/ultimate_ttt/ultimate_ttt.ino`](arduino/ultimate_ttt/ultimate_ttt.ino) — see
[`arduino/README.md`](arduino/README.md) for board-manager setup, board
settings (8MB flash, PSRAM enabled), and wiring notes for a bare module
(no onboard USB-UART).

### 4. Play

The board prints as text over the Serial monitor (115200 baud). You are
`O`, the AI is `X`. Enter moves as `sub cell` (both 0-8), e.g.:

```
> 4 4
```

`sub` = which of the 9 mini-boards, `cell` = which of its 9 cells. The
active-sub-board rule is enforced (the cell you play in picks which
mini-board your opponent must play in next — standard Ultimate
Tic-Tac-Toe rules).

## Design notes

- **Why no TFLite Micro**: the network is tiny (126→96→64→{81,1}, ~24K
  params), so hand-rolled float matmuls in [`main/net_infer.c`](main/net_infer.c)
  are simpler than pulling in the whole TFLite Micro component.
- **Why no on-device search**: MCTS at inference time would need many NN
  forward passes per move; a single greedy forward pass is enough for a
  reasonable opponent and keeps the firmware trivial. Porting
  [`train/mcts.py`](train/mcts.py)'s `run_mcts` to C and calling `net_infer`
  inside it would give a stronger on-device AI (memory cost stays tiny,
  just slower per move) if you want to take it further.
- **Why plain NumPy/PyTorch instead of TensorFlow**: keeps the training
  pipeline dependency-light and avoids TF/protobuf version conflicts.
- **Memory budget**: model weights ~93KB (flash, not SRAM), inference
  scratch buffers well under 1KB SRAM, game state struct ~100 bytes —
  comfortably fits within 512KB SRAM.

## License

MIT — see [LICENSE](LICENSE).
