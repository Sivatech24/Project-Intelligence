"""Score three solvers on a folder of ARC task JSON files (one file per task, with test outputs).

  repo_single : rules from ARC.py, rule chosen from the FIRST train pair only (how find_rule works today)
  repo_all    : the same rules, but a rule must reproduce EVERY train pair
  ext_all     : extended primitive library + two-step programs + every-train-pair check + two attempts

Usage: python eval_local.py <task_dir> <arc_py_path> [limit]
Scoring (competition rule): two attempts per test input; a test output is solved only on an exact match.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arc_verified_solver as S

_ARC = None


def load_repo_rules(path):
    spec = importlib.util.spec_from_file_location('repo_arc', path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod.RULES


def _apply(rule, g):
    try:
        out = np.asarray(rule(np.array(g, int)))
        if out.ndim != 2 or out.size == 0:
            return None
        return out.astype(int)
    except Exception:
        return None


def solve_repo(task, rules, mode):
    X = [np.array(p['input'], int) for p in task['train']]
    Y = [np.array(p['output'], int) for p in task['train']]
    pairs = list(zip(X, Y))[:1] if mode == 'single' else list(zip(X, Y))
    ok = []
    for name, rule in rules:
        good = True
        for x, y in pairs:
            o = _apply(rule, x)
            if o is None or o.shape != y.shape or not np.array_equal(o, y):
                good = False
                break
        if good:
            ok.append(rule)
    res = []
    for t in task['test']:
        tin = np.array(t['input'], int)
        preds = []
        for rule in ok:
            o = _apply(rule, tin)
            if o is not None and not any(np.array_equal(o, p) for p in preds):
                preds.append(o)
            if len(preds) == (1 if mode == 'single' else 2):
                break
        while len(preds) < 2:
            preds.append(preds[0] if (mode == 'single' and preds) else tin)
        res.append([preds[0].tolist(), preds[1].tolist()])
    return res, len(ok)


def _work(args):
    tid, task, which, arc_path = args
    global _ARC
    t0 = time.time()
    if which == 'ext_all':
        att, info = S.solve_task(task, time_budget=20.0)
        n = info['n_consistent_programs']
    else:
        if _ARC is None:
            _ARC = load_repo_rules(arc_path)
        att, n = solve_repo(task, _ARC, 'single' if which == 'repo_single' else 'all')
    solved = []
    for a, t in zip(att, task['test']):
        gt = np.array(t['output'], int)
        solved.append(any(np.array_equal(np.array(g, int), gt) for g in a))
    return tid, solved, n, time.time() - t0


def main():
    task_dir, arc_path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    files = sorted(f for f in os.listdir(task_dir) if f.endswith('.json'))[:limit]
    tasks = {f[:-5]: json.load(open(os.path.join(task_dir, f))) for f in files}
    out = {}
    for which in ('repo_single', 'repo_all', 'ext_all'):
        t0 = time.time()
        jobs = [(tid, t, which, arc_path) for tid, t in tasks.items()]
        with Pool(2) as p:
            rows = p.map(_work, jobs, chunksize=4)
        n_out = sum(len(s) for _, s, _, _ in rows)
        solved_out = sum(sum(s) for _, s, _, _ in rows)
        full = sum(1 for _, s, _, _ in rows if all(s))
        task_score = sum(sum(s) / len(s) for _, s, _, _ in rows) / len(rows)
        with_rule = sum(1 for _, _, n, _ in rows if n > 0)
        out[which] = {'tasks': len(rows), 'test_outputs': n_out, 'outputs_solved': solved_out,
                      'tasks_fully_solved': full, 'task_mean_score': round(task_score, 4),
                      'output_mean_score': round(solved_out / n_out, 4),
                      'tasks_with_any_consistent_program': with_rule,
                      'solved_task_ids': sorted(t for t, s, _, _ in rows if all(s)),
                      'seconds': round(time.time() - t0, 1)}
        r = out[which]
        print(f"{which:12s} outputs solved {r['outputs_solved']:3d}/{n_out}  tasks fully solved {full:3d}/{len(rows)}  "
              f"task-mean {r['task_mean_score']:.4f}  consistent-program found in {with_rule:3d} tasks  ({r['seconds']}s)", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f'results_{os.path.basename(task_dir.rstrip("/"))}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
