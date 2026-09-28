"""Builds a ready-to-upload Kaggle notebook (.ipynb) that writes out all the
training source files via %%writefile cells, then runs self-play training.
Run locally: python make_notebook.py -> writes ultimate_ttt_kaggle.ipynb
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent

SOURCE_FILES = ["game.py", "mcts.py", "net_torch.py", "train_kaggle.py",
                "train_kaggle_mp.py", "export_weights.py", "eval_vs_random.py"]


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main():
    cells = []

    cells.append(md_cell(
        "# Ultimate Tic-Tac-Toe -- self-play training (GPU)\n\n"
        "Writes out the training code, then runs batched self-play + "
        "training. Make sure **Settings -> Accelerator -> GPU** is enabled "
        "before running.\n"
    ))

    cells.append(code_cell("import torch\nprint('cuda available:', torch.cuda.is_available())\n"))

    for fname in SOURCE_FILES:
        content = (HERE / fname).read_text(encoding="utf-8")
        cells.append(code_cell(f"%%writefile {fname}\n" + content))

    cells.append(md_cell(
        "## Train (multi-process self-play + GPU training + early stopping)\n\n"
        "The model is tiny, so the MCTS tree traversal (pure Python, "
        "single-threaded) bottlenecks on one CPU core long before the GPU "
        "does any real work. `train_kaggle_mp.py` runs self-play across "
        "`--workers` separate processes (one per CPU core) while the main "
        "process trains on GPU -- this uses your Kaggle CPU quota, not just "
        "the GPU. Checkpoints save every iteration to "
        "`/kaggle/working/policy_params.npz`, so stopping early (or hitting "
        "the session limit) still leaves you a usable file.\n\n"
        "Every `--eval-every` iterations it also plays a quick batch of "
        "games against a random-move opponent; if the score doesn't improve "
        "for `--patience` checks in a row, training stops automatically and "
        "the best checkpoint is saved separately as "
        "`policy_params_best.npz` -- so you don't have to guess `--iterations` "
        "up front or babysit the notebook.\n"
    ))
    cells.append(code_cell(
        "import os\nprint('CPU cores available:', os.cpu_count())\n"
    ))
    cells.append(code_cell(
        "!python train_kaggle_mp.py \\\n"
        "    --iterations 150 \\\n"
        "    --workers 4 \\\n"
        "    --parallel-per-worker 32 \\\n"
        "    --sims 80 \\\n"
        "    --eval-every 10 \\\n"
        "    --eval-games 40 \\\n"
        "    --patience 5 \\\n"
        "    --out /kaggle/working/policy_params.npz\n"
    ))

    cells.append(md_cell(
        "## Evaluate against a random-move opponent\n\n"
        "Use `policy_params_best.npz` (the early-stopping checkpoint), not "
        "the plain `policy_params.npz` -- it's whichever iteration scored "
        "highest, not necessarily the very last one.\n"
    ))
    cells.append(code_cell(
        "!python eval_vs_random.py /kaggle/working/policy_params_best.npz --games 200\n"
    ))

    cells.append(md_cell(
        "## Export to a C header\n\n"
        "Download `model_weights.h` afterwards (File panel on the right) "
        "and drop it into your ESP-IDF project's `main/` folder.\n"
    ))
    cells.append(code_cell(
        "!python export_weights.py /kaggle/working/policy_params_best.npz "
        "--out /kaggle/working/model_weights.h\n"
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = HERE / "ultimate_ttt_kaggle.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
