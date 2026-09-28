# Contributing

## Running tests locally

**Python** (game engine, network, MCTS, export script):
```bash
cd train
pip install -r requirements-dev.txt
pytest -v
```

**C** (the same game engine + inference code that runs on-device, compiled
and run natively on your host machine so you don't need real hardware to
catch a logic bug):
```bash
./tests/run_c_tests.sh
```
Requires a host C compiler (`gcc`, already present on most Linux/macOS
setups and in CI; on Windows use WSL or MSYS2/MinGW).

Both suites run automatically on every push/PR via
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), which also does a
compile-only check of the Arduino sketch for the `esp32s3` target.

## Project structure

See the [README](README.md#repository-layout--english) /
[README (فارسی)](README.md#ساختار-مخزن) for the repository layout and
design notes.

## Guidelines

- Keep `main/` (ESP-IDF) and `arduino/ultimate_ttt/` (Arduino) in sync:
  `game.c`/`game.h`/`net_infer.c`/`net_infer.h` should stay byte-for-byte
  identical between the two (only `main.c` vs. `ultimate_ttt.ino` differ,
  since they use different I/O — ESP-IDF's stdio console vs. Arduino's
  `Serial`).
- If you change `train/game.py`'s rules/encoding, mirror the change in
  `main/game.c` (and its Arduino copy) and update both test suites —
  the two implementations are intentionally kept as parallel ports of each
  other, not a single shared codebase, since the device side has no Python
  runtime.
- Whenever you touch `net.py`'s architecture (layer sizes, activation),
  update `net_torch.py` to match — `test_net_torch.py::test_net_torch_checkpoint_is_loadable_by_plain_numpy_net`
  exists specifically to catch the two drifting apart.
- New gameplay/network logic should land with a test in `train/tests/` (or
  `tests/c/` for device-side C changes), not just a manual smoke run.
