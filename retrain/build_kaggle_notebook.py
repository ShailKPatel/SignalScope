"""
Converts a `# %%`-cell Python script into a Jupyter notebook.

Usage:
  python retrain/build_kaggle_notebook.py                       # kaggle_one_shot.py -> kaggle_one_shot.ipynb
  python retrain/build_kaggle_notebook.py SRC.py OUT.ipynb
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def split_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip("\n")
        if kind and body:
            if kind == "markdown":
                body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind:
            buf.append(line)
    flush()
    return cells


def to_notebook(cells):
    nb_cells = []
    for kind, body in cells:
        lines = [l + "\n" for l in body.splitlines()]
        lines[-1] = lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_one_shot.py")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".ipynb"
    with open(src) as f:
        cells = split_cells(f.read())
    with open(out, "w") as f:
        json.dump(to_notebook(cells), f, indent=1)
    print(f"wrote {out} ({len(cells)} cells)")
