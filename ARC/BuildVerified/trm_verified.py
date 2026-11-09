"""Recursive transformer with an external verifier, selector and write gate (Target v3 diagram), at demo scale.

Per-task test-time training on ARC grids. Everything here is runnable on CPU in minutes.

Boxes of the diagram that are IMPLEMENTED here
  input assembly      concat on the sequence axis: [x | y | z] + type + 2-D position embeddings
  recursive core      one shared transformer; z <- f(x,y,z) n times, then y <- f(y,z) once
  deep supervision    T-1 cycles under no_grad, last with grad, loss at EVERY step, detach (y,z)
  halt head           q_k trained on the external verifier's label
  grounding loss      contrastive: pooled z stays close to its own x
  trajectory logger   every step's draft, q_k and pooled z are kept
  external verifier   exact reproduction of a held-out demonstration (leave-one-out)
  selector            majority among verified candidates (selection, not synthesis)
  write gate          only verified outputs enter d_pass; the rest go to d_fail
NOT implemented here: retrieval `a`, fact store `k`, chunk parser `c`, linear probe, DPO, held-out promotion gate.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 10          # token id of padding cells
NCOL = 11         # colors 0-9 + pad


# ------------------------------------------------------------------ data
def load_tasks(task_dir):
    out = {}
    for f in sorted(os.listdir(task_dir)):
        if f.endswith('.json'):
            out[f[:-5]] = json.load(open(os.path.join(task_dir, f)))
    return out


def eligible(task, max_side=12, min_train=3):
    ps = task['train'] + task['test']
    if len(task['train']) < min_train:
        return False
    for p in ps:
        a, b = np.array(p['input']), np.array(p['output'])
        if a.shape != b.shape or max(a.shape) > max_side:
            return False
    return True


def dih(g, k):
    return np.rot90(g, k) if k < 4 else np.rot90(g[:, ::-1], k - 4)


def dih_inv(g, k):
    return np.rot90(g, -k) if k < 4 else np.rot90(g, -(k - 4))[:, ::-1]


def pad_grid(g, P):
    out = np.full((P, P), PAD, int)
    out[:g.shape[0], :g.shape[1]] = g
    return out


# ------------------------------------------------------------------ model
class RecursiveGridModel(nn.Module):
    def __init__(self, P, d=64, heads=4, layers=2, m=16, n_inner=2, T=2):
        super().__init__()
        self.P, self.m, self.n_inner, self.T, self.d = P, m, n_inner, T, d
        self.tok = nn.Embedding(NCOL, d)
        self.patch = nn.Linear(9 * NCOL, d)                 # grid tokenizer: each cell sees its 3x3 neighbourhood
        self.row, self.col = nn.Embedding(P, d), nn.Embedding(P, d)
        self.typ = nn.Embedding(3, d)                       # x / y / z
        self.y_init = nn.Parameter(torch.zeros(d))
        self.z_init = nn.Parameter(torch.randn(m, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=False)
        self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)   # shared weights, reused every pass
        self.Wz, self.Wy = nn.Linear(d, d), nn.Linear(d, d)
        self.readout = nn.Linear(d, 10)                     # "reverse embedding"
        self.halt = nn.Linear(d, 1)
        self.ground_x, self.ground_z = nn.Linear(d, d), nn.Linear(d, d)
        r = torch.arange(P)
        self.register_buffer('pos', (r[:, None] * P + r[None, :]).reshape(-1))
        self.register_buffer('rr', r[:, None].expand(P, P).reshape(-1).clone())
        self.register_buffer('cc', r[None, :].expand(P, P).reshape(-1).clone())

    def embed_cells(self, g, typ):
        B = g.shape[0]
        G = g.reshape(B, self.P, self.P)
        Gp = F.pad(G, (1, 1, 1, 1), value=PAD)
        patches = torch.stack([Gp[:, i:i + self.P, j:j + self.P] for i in range(3) for j in range(3)], -1)   # B,P,P,9
        oh = F.one_hot(patches, NCOL).float().reshape(B, self.P * self.P, 9 * NCOL)
        return self.tok(g) + self.patch(oh) + self.typ.weight[typ]

    def init_state(self, xg):
        B = xg.shape[0]
        x = self.embed_cells(xg.reshape(B, -1), 0)
        y = self.embed_cells(xg.reshape(B, -1), 1) + self.y_init      # draft starts as a copy of the input grid
        z = (self.z_init + self.typ.weight[2])[None].expand(B, -1, -1).contiguous()
        return x, y, z

    def _core(self, seq, pad_mask):
        return self.core(seq, src_key_padding_mask=pad_mask)

    def cycle(self, x, y, z, valid):
        S = y.shape[1]
        B = x.shape[0]
        zeros = torch.zeros(B, self.m, dtype=torch.bool, device=x.device)
        m_xyz = torch.cat([~valid, ~valid, zeros], 1)
        m_yz = torch.cat([~valid, zeros], 1)
        for _ in range(self.n_inner):                                   # z <- z + Wz core(x,y,z)[z]
            o = self._core(torch.cat([x, y, z], 1), m_xyz)
            z = z + self.Wz(o[:, 2 * S:])
        o = self._core(torch.cat([y, z], 1), m_yz)                      # y <- y + Wy core(y,z)[y]   (x dropped)
        y = y + self.Wy(o[:, :S])
        return y, z

    def supervision_step(self, x, y, z, valid):
        """T-1 cycles without gradient, the last one with gradient."""
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.cycle(x, y, z, valid)
        return self.cycle(x, y, z, valid)

    def heads(self, y, z, valid):
        logits = self.readout(y)
        v = valid.float().unsqueeze(-1)
        ymean = (y.detach() * v).sum(1) / v.sum(1).clamp(min=1)
        q = self.halt(ymean).squeeze(-1)                                # logit of "this draft is exactly right"
        return logits, q, z.mean(1)


# ------------------------------------------------------------ training
def make_batch(pairs, P, bs, rng, device, aug=False):
    X, Y = [], []
    for _ in range(bs):
        x, y = pairs[rng.integers(len(pairs))]
        k = int(rng.integers(8)) if aug else 0
        perm = np.arange(10)
        if aug:
            perm[1:] = rng.permutation(np.arange(1, 10))
        X.append(pad_grid(perm[dih(x, k)], P))
        Y.append(pad_grid(perm[dih(y, k)], P))
    X = torch.tensor(np.stack(X), device=device)
    Y = torch.tensor(np.stack(Y), device=device)
    return X, Y


def train_task(train_pairs, P, K=6, iters=80, bs=16, lr=3e-3, lam=0.05, seed=0, device='cpu', aug=False, **mk):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RecursiveGridModel(P, **mk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = iters * K
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    for it in range(iters):
        X, Y = make_batch(train_pairs, P, bs, rng, device, aug)
        valid = (X != PAD).reshape(bs, -1)
        tgt = Y.reshape(bs, -1).clamp(max=9)
        x, y, z = model.init_state(X)
        for k in range(1, K + 1):                                       # deep supervision: one optimiser step per step k
            x = model.embed_cells(X.reshape(bs, -1), 0)                 # fresh graph for x at every step
            y, z = model.supervision_step(x, y, z, valid)
            logits, q, zp = model.heads(y, z, valid)
            ce = F.cross_entropy(logits.reshape(-1, 10), tgt.reshape(-1), reduction='none').reshape(bs, -1)
            ce = (ce * valid).sum(1) / valid.sum(1)
            correct = ((logits.argmax(-1) == tgt) | ~valid).all(1).float()          # external verifier label
            halt = F.binary_cross_entropy_with_logits(q, correct)
            xp = F.normalize(model.ground_x((x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)), dim=-1)
            zq = F.normalize(model.ground_z(zp), dim=-1)
            sim = zq @ xp.T / 0.2
            ground = F.cross_entropy(sim, torch.arange(bs, device=device))
            w = 0.5 + 0.5 * k / K                                       # later steps weigh more
            loss = w * ce.mean() + 0.1 * halt + lam * ground
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            y, z = y.detach(), z.detach()
    return model


@torch.no_grad()
def trajectory(model, grids, P, K=6, device='cpu', views_used=(0,)):
    """grids: list of 2-D arrays. Returns for each grid, each view v (8), each step k: (prediction, q_k)."""
    model.eval()
    views = [(gi, v) for gi in range(len(grids)) for v in views_used]
    X = torch.tensor(np.stack([pad_grid(dih(grids[gi], v), P) for gi, v in views]), device=device)
    valid = (X != PAD).reshape(len(views), -1)
    x, y, z = model.init_state(X)
    out = {}
    for k in range(1, K + 1):
        y, z = model.supervision_step(x, y, z, valid)
        logits, q, _ = model.heads(y, z, valid)
        pred = logits.argmax(-1).reshape(len(views), P, P).cpu().numpy()
        qs = torch.sigmoid(q).cpu().numpy()
        for i, (gi, v) in enumerate(views):
            h, w = grids[gi].shape
            vh, vw = dih(grids[gi], v).shape
            out[(gi, v, k)] = (dih_inv(pred[i][:vh, :vw], v).copy(), float(qs[i]))
    return out


# ---------------------------------------------------- verifier / selector
def vote(cands):
    """cands: list of grids -> list of (grid, count) sorted by count (stable by first appearance)."""
    keys, groups = [], {}
    for g in cands:
        key = (g.shape, g.tobytes())
        if key not in groups:
            groups[key] = [g, 0]
            keys.append(key)
        groups[key][1] += 1
    return sorted((groups[k] for k in keys), key=lambda t: -t[1])


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def solve_one(tid, task, K=4, iters=80, seeds=(0, 1), device='cpu', **mk):
    t0 = time.time()
    torch.set_num_threads(1)
    train = task['train']
    held = train[-1]                                  # leave-one-out: the verifier's demonstration
    fit = [(np.array(p['input']), np.array(p['output'])) for p in train[:-1]]
    hx, hy = np.array(held['input']), np.array(held['output'])
    tests = [(np.array(p['input']), np.array(p['output'])) for p in task['test']]
    allg = [a for a, _ in fit] + [b for _, b in fit] + [hx, hy] + [a for a, _ in tests]
    P = max(max(g.shape) for g in allg)
    grids = [hx] + [t[0] for t in tests]
    tr = {}                                           # (grid index, seed, step) -> (prediction, q)
    for sd in seeds:
        model = train_task(fit, P, K=K, iters=iters, seed=sd, device=device, **mk)
        for (gi, v, k), val in trajectory(model, grids, P, K=K, device=device).items():
            tr[(gi, sd, k)] = val
    keys = [(sd, k) for sd in seeds for k in range(1, K + 1)]
    same = lambda g, ref: g.shape == ref.shape and np.array_equal(g, ref)
    passed = [key for key in keys if same(tr[(0,) + key][0], hy)]        # external verifier: reproduces the held-out demo
    res = {'task': tid, 'P': P, 'n_fit': len(fit), 'n_verified': len(passed), 'n_candidates': len(keys),
           'halt_q': [tr[(0,) + key][1] for key in keys], 'halt_label': [int(same(tr[(0,) + key][0], hy)) for key in keys],
           'tests': [], 'seconds': 0}
    for ti, (tin, tout) in enumerate(tests):
        gi = ti + 1
        hit = lambda g: same(g, tout)
        every = [tr[(gi,) + key][0] for key in keys]
        last = [tr[(gi, sd, K)][0] for sd in seeds]
        row = {'copy': [hit(tin)],
               'last_step': [hit(last[0])],                                                     # no verifier, no voting
               'vote_all': [hit(vote(every)[i][0]) if i < len(vote(every)) else False for i in range(2)]}   # voting only
        if passed:
            vv = vote([tr[(gi,) + key][0] for key in passed])                                   # verified candidates only
            row['verified'] = [hit(vv[i][0]) if i < len(vv) else False for i in range(2)]
            best = vv[0][0]
        else:
            row['verified'] = list(row['vote_all'])                                             # nothing verified: fall back
            best = vote(every)[0][0]
        row['oracle_any'] = [any(hit(g) for g in every)]
        row['gate'] = {'verified': bool(passed), 'correct': bool(hit(best))}                    # write gate -> d_pass / d_fail
        res['tests'].append(row)
    res['seconds'] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------- summary
def summarise(results):
    def pass_at(name, n):
        s = t = 0
        for r in results:
            for row in r['tests']:
                v = row[name]
                s += any(v[:n]); t += 1
        return s, t
    out = {}
    for name in ('copy', 'last_step', 'vote_all', 'verified'):
        s1, t = pass_at(name, 1)
        s2, _ = pass_at(name, 2)
        out[name] = {'pass@1': s1, 'pass@2': s2, 'of': t}
    out['oracle_any'] = {'any candidate correct': sum(row['oracle_any'][0] for r in results for row in r['tests']), 'of': t}
    gated = [row['gate'] for r in results for row in r['tests'] if row['gate']['verified']]
    ungated = [row['gate'] for r in results for row in r['tests']]
    out['write_gate'] = {
        'ungated_buffer_size': len(ungated), 'ungated_precision': round(sum(g['correct'] for g in ungated) / max(1, len(ungated)), 3),
        'gated_buffer_size': len(gated), 'gated_precision': round(sum(g['correct'] for g in gated) / max(1, len(gated)), 3)}
    q = [x for r in results for x in r['halt_q']]
    l = [x for r in results for x in r['halt_label']]
    out['halt_head_auroc_on_heldout_demo'] = round(auroc(q, l), 3) if sum(l) and sum(l) < len(l) else None
    out['tasks_with_verified_candidate'] = sum(1 for r in results if r['n_verified'] > 0)
    out['tasks'] = len(results)
    return out
