"""Build kaggle_notebook.ipynb from arc_verified_solver.py (so the notebook and the repo never drift apart)."""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
solver_src = open(os.path.join(here, 'arc_verified_solver.py'), encoding='utf-8').read()

INTRO = """# True Intelligence — verified program search for ARC-AGI-2

Companion notebook to the writeup *True Intelligence: A Recursive-Reasoning Transformer with Verified Memory for the Abstraction and Reasoning Corpus (ARC)*.
Code: https://github.com/Sivatech24/Project-Intelligence (folder `ARC/BuildVerified/`).

**What this notebook does**
1. Defines a library of 71 grid operations and searches programs of up to two steps.
2. Keeps a program **only if it reproduces every demonstration pair exactly** (the external verifier of the writeup, Section 4.3).
3. Returns two distinct predictions per test input (the competition scores two attempts).
4. Writes `submission.json`. If ground-truth solutions are available (public evaluation set), it also prints the score.

No GPU and no internet are needed when the competition data is attached. If no data is attached and Internet is switched on, it downloads the public ARC-AGI-2 data from GitHub and scores the public evaluation set.
"""

RUNNER = '''import glob, json, os, subprocess, time
import numpy as np

INPUT_ROOT = os.environ.get("ARC_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("ARC_OUTPUT_DIR", "/kaggle/working")
os.makedirs(OUT_DIR, exist_ok=True)

def find(pattern):
    hits = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", pattern), recursive=True))
    return hits[0] if hits else None

# ---- 1. locate data -------------------------------------------------------
test_file = find("*test_challenges*.json")
eval_file = find("*evaluation_challenges*.json")
eval_sol  = find("*evaluation_solutions*.json")
sample    = find("sample_submission.json")
mode, tasks, solutions = None, None, None

if test_file:
    mode, tasks = "competition-test", json.load(open(test_file))
elif eval_file:
    mode, tasks = "public-evaluation", json.load(open(eval_file))
    solutions = json.load(open(eval_sol)) if eval_sol else None
else:
    # no data attached: try the public GitHub repository (needs Internet = On in notebook settings)
    repo = os.path.join(OUT_DIR, "ARC-AGI-2")
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/arcprize/ARC-AGI-2", repo], check=True)
    d = os.path.join(repo, "data", "evaluation")
    tasks, solutions = {}, {}
    for f in sorted(os.listdir(d)):
        t = json.load(open(os.path.join(d, f)))
        tasks[f[:-5]] = {"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]}
        solutions[f[:-5]] = [x["output"] for x in t["test"]]
    mode = "public-evaluation"

print(f"mode={mode}  tasks={len(tasks)}  test inputs={sum(len(t['test']) for t in tasks.values())}")

# ---- 2. solve ---------------------------------------------------------------
t0 = time.time()
sub = build_submission(tasks, time_budget=20.0)
print(f"solved-or-guessed all tasks in {time.time() - t0:.1f}s")

# ---- 3. mirror the official sample file if there is one ----------------------
if sample:
    template = json.load(open(sample))
    for tid, entries in template.items():
        for i, e in enumerate(entries):
            if tid in sub and i < len(sub[tid]):
                e["attempt_1"] = sub[tid][i]["attempt_1"]
                e["attempt_2"] = sub[tid][i]["attempt_2"]
    sub = template

# ---- 4. sanity-check the format ----------------------------------------------
for tid, entries in sub.items():
    assert isinstance(entries, list) and entries, tid
    for e in entries:
        for k in ("attempt_1", "attempt_2"):
            g = e[k]
            assert isinstance(g, list) and g and all(isinstance(r, list) and r for r in g), (tid, k)
            assert all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r), (tid, k)
json.dump(sub, open(os.path.join(OUT_DIR, "submission.json"), "w"))
print("wrote", os.path.join(OUT_DIR, "submission.json"))

# ---- 5. score when ground truth is known --------------------------------------
if solutions:
    solved = total = 0
    full = 0
    for tid, outs in solutions.items():
        ok = []
        for i, gt in enumerate(outs):
            a = sub[tid][i]
            ok.append(np.array_equal(np.array(a["attempt_1"]), np.array(gt)) or np.array_equal(np.array(a["attempt_2"]), np.array(gt)))
        solved += sum(ok); total += len(ok); full += all(ok)
    print(f"PUBLIC EVALUATION: {solved}/{total} test outputs solved, {full}/{len(solutions)} tasks fully solved")
'''

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(True)}

nb = {"cells": [md(INTRO), code(solver_src), code(RUNNER)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(here, 'kaggle_notebook.ipynb'), 'w'), indent=1)
print('notebook written:', len(nb['cells']), 'cells')
