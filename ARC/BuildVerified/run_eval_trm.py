"""Run the verified recursive model on a deterministic subset of ARC tasks. Usage: python run_eval_trm.py <task_dir> <out.json> [n_tasks]"""
import json, random, sys, time
from multiprocessing import Pool
import trm_verified as M

def work(a):
    tid, task = a
    try:
        return M.solve_one(tid, task, K=4, iters=80, seeds=(0, 1))
    except Exception as e:
        return {'task': tid, 'error': str(e)}

if __name__ == '__main__':
    d, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    tasks = M.load_tasks(d)
    ids = [k for k, t in tasks.items() if M.eligible(t, max_side=10, min_train=3)]
    random.Random(0).shuffle(ids)
    ids = ids[:n]
    t0 = time.time(); res = []
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(work, [(k, tasks[k]) for k in ids])):
            res.append(r)
            if (i + 1) % 5 == 0:
                print(i + 1, len(ids), round(time.time() - t0), 's', flush=True)
                json.dump(res, open(out, 'w'))
    json.dump(res, open(out, 'w'))
    good = [r for r in res if 'error' not in r]
    print(json.dumps(M.summarise(good), indent=1))
