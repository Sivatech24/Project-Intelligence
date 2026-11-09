"""Build kaggle_notebook_recursive.ipynb from trm_verified.py."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'trm_verified.py'), encoding='utf-8').read()

INTRO = """# True Intelligence: recursive transformer + external verifier + write gate (demo-scale evaluation)

Implements the *Target v3* diagram of the writeup and measures it on real ARC tasks (public ARC-AGI-2 **training** split, same-shape tasks up to 10x10).
Code: https://github.com/Sivatech24/Project-Intelligence

## Which diagram boxes this notebook implements

| Diagram box | Here |
|---|---|
| Input assembly: concatenate `[x \\| y \\| z]` on the sequence axis + type embeddings | yes (`RecursiveGridModel.cycle`) |
| Grid tokenizer | yes, each cell is embedded with its 3x3 neighbourhood |
| Recursive core, shared weights, `z <- f(x,y,z)` then `y <- f(y,z)` | yes |
| Deep supervision: `T-1` cycles without gradient, last with gradient, loss at every step, `detach(y,z)` | yes (`train_task`) |
| Halt head trained on the verifier's label | yes (AUROC reported) |
| Contrastive grounding loss on `z` | yes |
| Trajectory logger (every step's draft kept, no early exit) | yes |
| **External verifier**: a candidate must reproduce a held-out demonstration exactly | yes |
| **Selector**: majority among verified candidates (selection, not synthesis) | yes |
| **Write gate**: only verified outputs enter `d_pass`, the rest go to `d_fail` | yes (precision measured) |
| Retrieval `a`, fact store `k`, chunk parser `c`, linear probe, DPO, held-out promotion gate | **not** in this notebook |

## What is measured
Per task, two small models (two seeds) are trained from scratch on the task's demonstrations **except the last one**, which is kept for the verifier. Every supervision step of every seed gives a candidate answer for the test input. Compared on the same test outputs:

* `copy` : return the input grid
* `last_step` : last step of one model, no verifier, no voting
* `vote_all` : majority vote over all candidates, no verifier
* `verified` : majority vote over only the candidates that reproduce the held-out demonstration (the diagram's path)
* write-gate **precision**: how often a buffer entry is actually correct, with and without the gate (ground truth is used only to audit)

Nothing here uses the hidden ARC-AGI-2 test set. This is not a leaderboard score.
"""

DATA = '''import glob, json, os, subprocess
def load_arc_training():
    ch = sorted(glob.glob("/kaggle/input/**/*training_challenges*.json", recursive=True))
    so = sorted(glob.glob("/kaggle/input/**/*training_solutions*.json", recursive=True))
    if ch and so:
        C, S = json.load(open(ch[0])), json.load(open(so[0]))
        return {k: {"train": C[k]["train"], "test": [{"input": t["input"], "output": S[k][i]} for i, t in enumerate(C[k]["test"])]} for k in C}
    repo = "/kaggle/working/ARC-AGI-2" if os.path.isdir("/kaggle/working") else "ARC-AGI-2"
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/arcprize/ARC-AGI-2", repo], check=True)
    return load_tasks(os.path.join(repo, "data", "training"))

ALL = load_arc_training()
ids = [k for k, t in ALL.items() if eligible(t, max_side=10, min_train=3)]
import random; random.Random(0).shuffle(ids)
print(len(ALL), "tasks loaded;", len(ids), "same-shape tasks up to 10x10")
'''

RUN = '''import json, time, multiprocessing as mp
N_TASKS = 60          # about 1 hour on 2 CPU cores; use 15 for a quick look
def _work(tid):
    try:
        return solve_one(tid, ALL[tid], K=4, iters=80, seeds=(0, 1))
    except Exception as e:
        return {"task": tid, "error": str(e)}
t0 = time.time(); results = []
with mp.get_context("fork").Pool(os.cpu_count()) as p:
    for i, r in enumerate(p.imap_unordered(_work, ids[:N_TASKS])):
        results.append(r)
        if (i + 1) % 5 == 0: print(i + 1, "/", N_TASKS, round(time.time() - t0), "s", flush=True)
json.dump(results, open("results_trm.json", "w"))
good = [r for r in results if "error" not in r]
S = summarise(good)
print(json.dumps(S, indent=1))
'''

PLOT = '''import matplotlib.pyplot as plt
names = ["copy", "last_step", "vote_all", "verified"]
n = S["copy"]["of"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(names, [S[k]["pass@2"] / n * 100 for k in names], color=["#999", "#6a8caf", "#6a8caf", "#2a9d5c"])
ax[0].set_ylabel("test outputs solved, % (two attempts)"); ax[0].set_title(f"Solved test outputs (n={n})")
g = S["write_gate"]
ax[1].bar(["no gate", "write gate"], [g["ungated_precision"] * 100, g["gated_precision"] * 100], color=["#999", "#2a9d5c"])
ax[1].set_ylabel("buffer precision, %"); ax[1].set_title(f"Entries that are correct (buffer {g['ungated_buffer_size']} -> {g['gated_buffer_size']})")
plt.tight_layout(); plt.show()
'''

END = """## How to read the result
* `verified` above `vote_all` and `last_step` shows what the external verifier adds on the same candidates.
* The write-gate bars show the contamination argument directly: without the gate the buffer is mostly wrong; with it, entries are far more often correct. That is the reason the memory loop only ever trains on verified data.
* The halt head's AUROC says whether its confidence ranks right answers above wrong ones on the verifier's demonstration.
* Limits: tiny from-scratch models cannot infer rules that the few demonstrations do not show (for example a colour that never appears in them), so absolute accuracy is low; this is a measurement of the verifier and gate, not a claim of solving ARC-AGI-2.
"""

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
nb = {"cells": [md(INTRO), code(src), code(DATA), code(RUN), code(PLOT), md(END)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(here, 'kaggle_notebook_recursive.ipynb'), 'w'), indent=1)
print('ok')
