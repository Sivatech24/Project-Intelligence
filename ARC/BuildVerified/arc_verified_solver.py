"""Verified program search for ARC tasks.

Idea (Section 4.3, step 6 of the writeup): propose candidate transformations, then
keep a candidate ONLY if it reproduces EVERY demonstration pair exactly. Rank the
survivors (shorter first) and return two distinct predictions, as the competition
scores two attempts per test input.

Pure numpy + scipy. No internet, no GPU, no training.
"""
import time
from itertools import product

import numpy as np
from scipy.ndimage import label

MAX_SIDE = 30


# ------------------------------------------------------------------ helpers
def _crop_bbox(mask, g):
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError('empty')
    return g[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _objects(g, diag=True):
    st = np.ones((3, 3), int) if diag else None
    lab, n = label(g != 0, structure=st)
    return lab, n


def _pick_object(g, largest=True, diag=True):
    lab, n = _objects(g, diag)
    if n == 0:
        raise ValueError('no objects')
    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    idx = (int(np.argmax(sizes)) if largest else int(np.argmin(sizes))) + 1
    return _crop_bbox(lab == idx, g)


def _gravity(g, direction):
    out = np.zeros_like(g)
    h, w = g.shape
    if direction in ('down', 'up'):
        for c in range(w):
            vals = g[:, c][g[:, c] != 0]
            if direction == 'down':
                out[h - len(vals):, c] = vals
            else:
                out[:len(vals), c] = vals
    else:
        for r in range(h):
            vals = g[r][g[r] != 0]
            if direction == 'right':
                out[r, w - len(vals):] = vals
            else:
                out[r, :len(vals)] = vals
    return out


def _fill_holes(g, color):
    lab, n = label(g == 0)  # 4-connected background regions
    out = g.copy()
    h, w = g.shape
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    for i in range(1, n + 1):
        if i not in border:
            out[lab == i] = color
    return out


def _dedupe(g):
    rows = [0] + [i for i in range(1, g.shape[0]) if not np.array_equal(g[i], g[i - 1])]
    g = g[rows]
    cols = [0] + [j for j in range(1, g.shape[1]) if not np.array_equal(g[:, j], g[:, j - 1])]
    return g[:, cols]


def _drop_empty(g):
    rows = np.where((g != 0).any(axis=1))[0]
    cols = np.where((g != 0).any(axis=0))[0]
    if rows.size == 0:
        raise ValueError('empty')
    return g[rows][:, cols]


def _down(g, k):
    h, w = g.shape
    if h % k or w % k:
        raise ValueError('not divisible')
    return g[::k, ::k]


def _overlay(a, b):
    return np.where(a != 0, a, b)


# --------------------------------------------------------------- primitives
def build_primitives():
    P = {}
    P['identity'] = lambda g: g
    P['rot90'] = lambda g: np.rot90(g, 1)
    P['rot180'] = lambda g: np.rot90(g, 2)
    P['rot270'] = lambda g: np.rot90(g, 3)
    P['flip_ud'] = lambda g: g[::-1]
    P['flip_lr'] = lambda g: g[:, ::-1]
    P['transpose'] = lambda g: g.T
    P['anti_transpose'] = lambda g: np.rot90(g, 2).T
    for k in (2, 3, 4, 5):
        P[f'upscale{k}'] = (lambda k: lambda g: np.kron(g, np.ones((k, k), int)))(k)
    for k in (2, 3):
        P[f'downscale{k}'] = (lambda k: lambda g: _down(g, k))(k)
    for a, b in ((1, 2), (2, 1), (2, 2), (3, 3), (1, 3), (3, 1)):
        P[f'tile{a}x{b}'] = (lambda a, b: lambda g: np.tile(g, (a, b)))(a, b)
    P['mirror_h'] = lambda g: np.hstack([g, g[:, ::-1]])
    P['mirror_h_rev'] = lambda g: np.hstack([g[:, ::-1], g])
    P['mirror_v'] = lambda g: np.vstack([g, g[::-1]])
    P['mirror_v_rev'] = lambda g: np.vstack([g[::-1], g])
    P['mirror_quad'] = lambda g: np.vstack([np.hstack([g, g[:, ::-1]]), np.hstack([g[::-1], g[::-1, ::-1]])])
    P['sym_lr'] = lambda g: _overlay(g, g[:, ::-1])
    P['sym_ud'] = lambda g: _overlay(g, g[::-1])
    P['sym_both'] = lambda g: _overlay(_overlay(g, g[:, ::-1]), _overlay(g[::-1], g[::-1, ::-1]))
    P['sym_diag'] = lambda g: _overlay(g, g.T) if g.shape[0] == g.shape[1] else (_ for _ in ()).throw(ValueError('not square'))
    P['crop_nonzero'] = lambda g: _crop_bbox(g != 0, g)
    P['crop_largest_obj'] = lambda g: _pick_object(g, True, True)
    P['crop_smallest_obj'] = lambda g: _pick_object(g, False, True)
    P['crop_largest_obj4'] = lambda g: _pick_object(g, True, False)
    P['crop_smallest_obj4'] = lambda g: _pick_object(g, False, False)
    P['left_half'] = lambda g: g[:, :g.shape[1] // 2]
    P['right_half'] = lambda g: g[:, (g.shape[1] + 1) // 2:]
    P['top_half'] = lambda g: g[:g.shape[0] // 2]
    P['bottom_half'] = lambda g: g[(g.shape[0] + 1) // 2:]
    P['dedupe'] = _dedupe
    P['drop_empty'] = _drop_empty
    for d in ('down', 'up', 'left', 'right'):
        P[f'gravity_{d}'] = (lambda d: lambda g: _gravity(g, d))(d)
    for c in range(1, 10):
        P[f'fill_holes{c}'] = (lambda c: lambda g: _fill_holes(g, c))(c)
        P[f'keep_color{c}'] = (lambda c: lambda g: np.where(g == c, g, 0))(c)
        P[f'remove_color{c}'] = (lambda c: lambda g: np.where(g == c, 0, g))(c)
    return P


# ----------------------------------------------------------------- matching
def _learn_color_map(outs, ys):
    """Return a color->color dict if ys == map(outs) for ALL pairs, else None."""
    m = {}
    for o, y in zip(outs, ys):
        if o.shape != y.shape:
            return None
        for c in np.unique(o):
            vals = np.unique(y[o == c])
            if vals.size != 1:
                return None
            v = int(vals[0])
            if m.setdefault(int(c), v) != v:
                return None
    return m


def _apply_map(g, m):
    out = g.copy()
    for c, v in m.items():
        out[g == c] = v
    return out


def _run(fn, g):
    try:
        out = fn(g)
        out = np.asarray(out)
        if out.ndim != 2 or out.size == 0 or max(out.shape) > MAX_SIDE:
            return None
        return out.astype(int)
    except Exception:
        return None


def solve_task(task, time_budget=20.0, max_candidates=60):
    """task: {'train': [{'input','output'}...], 'test': [{'input'}...]}.
    Returns (attempts_per_test, info) where attempts_per_test is a list of
    [attempt_1, attempt_2] grids (lists of lists), one entry per test input."""
    t0 = time.time()
    X = [np.array(p['input'], int) for p in task['train']]
    Y = [np.array(p['output'], int) for p in task['train']]
    T = [np.array(p['input'], int) for p in task['test']]
    P = build_primitives()
    names = list(P)
    found = []  # (rank_key, program_names, color_map)

    def consider(prog, outs):
        m = _learn_color_map(outs, Y)
        if m is not None:
            nontrivial = sum(1 for c, v in m.items() if c != v)
            found.append(((len(prog), nontrivial), prog, m))

    # constant-output shortcut
    if all(np.array_equal(Y[0], y) for y in Y[1:]):
        found.append(((0, 0), ('constant',), None))

    # depth 1
    level1 = {}
    for n in names:
        outs = [_run(P[n], x) for x in X]
        if any(o is None for o in outs):
            continue
        level1[n] = outs
        consider((n,), outs)
    # depth 2
    for n1, outs1 in level1.items():
        if n1 == 'identity':
            continue
        if time.time() - t0 > time_budget:
            break
        for n2 in names:
            if n2 == 'identity':
                continue
            outs2 = [_run(P[n2], o) for o in outs1]
            if any(o is None for o in outs2):
                continue
            consider((n1, n2), outs2)
        if len(found) >= max_candidates * 20:
            break

    found.sort(key=lambda r: r[0])
    attempts = []
    for tin in T:
        preds = []
        for _, prog, m in found:
            if prog == ('constant',):
                out = Y[0]
            else:
                out = tin
                for n in prog:
                    out = _run(P[n], out)
                    if out is None:
                        break
                if out is None:
                    continue
                if m is not None:
                    out = _apply_map(out, m)
            if not any(np.array_equal(out, p) for p in preds):
                preds.append(out)
            if len(preds) == 2:
                break
        while len(preds) < 2:          # fallback guesses: the input itself
            preds.append(tin)
        attempts.append([preds[0].tolist(), preds[1].tolist()])
    info = {'n_consistent_programs': len(found),
            'best_program': list(found[0][1]) if found else None,
            'seconds': round(time.time() - t0, 2)}
    return attempts, info


# ------------------------------------------------------------ submission IO
def build_submission(tasks, time_budget=20.0, log=print):
    """tasks: {task_id: task}. Returns {task_id: [{'attempt_1':..., 'attempt_2':...}, ...]}."""
    sub = {}
    for i, (tid, task) in enumerate(sorted(tasks.items())):
        try:
            att, info = solve_task(task, time_budget)
        except Exception as e:  # never crash the whole run
            att = [[t['input'], t['input']] for t in task['test']]
            info = {'error': str(e)}
        sub[tid] = [{'attempt_1': a[0], 'attempt_2': a[1]} for a in att]
        if (i + 1) % 20 == 0:
            log(f'  {i + 1}/{len(tasks)} tasks done')
    return sub
