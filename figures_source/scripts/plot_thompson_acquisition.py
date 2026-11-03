"""Thompson-sampling acquisition figure: what does the uncertainty bonus buy?

Compares the full DDPP-LB loop (Thompson acquisition) against the same loop with
Thompson disabled, which falls back to greedy top-K by the surrogate's predicted
mean. Everything else in the loop is identical, so the contrast isolates the
acquisition rule.

Both panels avoid thresholds defined from the methods being compared, which
would flatter whichever method happens to score higher overall:

  (a) Distinct molecules found above a per-target bar set by *unconditional
      sampling* -- the 99th percentile of everything the best-of-N baseline ever
      evaluated on that target. The bar is external to both methods.
  (b) Fraction of oracle calls spent on molecules not already evaluated. Greedy
      acquisition keeps re-querying its current favourites, so this measures the
      budget each rule wastes on repeats; it needs no threshold at all.

Usage:  python plot_thompson_tail.py [outdir]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT_BATCH1 = Path('/overflow/trevor/inbox/batch1_results_20260505T1252/outputs/molecular')
TARGETS = ['2VT4', '5SDV', '6CM4', '7BKC', '7C7M', '7YLL']
SEEDS = [0, 1, 2, 3, 42]

METHODS = [
    ('ddpp_orig',    'Thompson acquisition', '#1f78b4'),
    ('ddpp_nothomp', 'Greedy top-$K$',       '#999999'),
]
BASELINE = 'best_of_n'   # unconditional sampling, used only to set the bar
BAR_PCT = 99.0
N_GRID = 200


def load_rows(root, target, method, seed):
    p = root / target / f'{method}_s{seed}' / 'oracle_timeline.jsonl'
    try:
        if not p.exists():
            return None
        fh = open(p)
    except PermissionError:
        return None
    rows = []
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows or None


def run_series(rows):
    """(calls, scores, smiles) sorted by oracle call index."""
    rows = sorted((r for r in rows if r.get('score') is not None),
                  key=lambda r: r['call'])
    return (np.array([r['call'] for r in rows], dtype=float),
            np.array([r['score'] for r in rows], dtype=float),
            [r.get('smiles') for r in rows])


def cumulative_curves(calls, scores, smiles, bar, grid):
    """(distinct molecules above `bar`, unique fraction of calls) read off at grid."""
    seen_hits, seen_all = set(), set()
    hits = np.empty(len(scores))
    uniq = np.empty(len(scores))
    for i, (sc, sm) in enumerate(zip(scores, smiles)):
        key = sm if sm is not None else f'__row{i}'
        seen_all.add(key)
        if sc >= bar:
            seen_hits.add(key)
        hits[i] = len(seen_hits)
        uniq[i] = len(seen_all) / (i + 1)
    idx = np.clip(np.searchsorted(calls, grid, side='right') - 1, 0, len(scores) - 1)
    return hits[idx], uniq[idx]


def collect():
    per_target = {}
    for t in TARGETS:
        base = [run_series(r)[1] for s in SEEDS
                if (r := load_rows(ROOT_BATCH1, t, BASELINE, s))]
        if not base:
            print(f'  {t}: no {BASELINE} baseline, skipped')
            continue
        bar = float(np.percentile(np.concatenate(base), BAR_PCT))

        runs = {}
        for method, _, _ in METHODS:
            seed_runs = [run_series(r) for s in SEEDS
                         if (r := load_rows(ROOT_BATCH1, t, method, s))]
            if seed_runs:
                runs[method] = seed_runs
        if len(runs) != len(METHODS):
            print(f'  {t}: missing method data, skipped')
            continue

        xmax = min(c[-1] for rs in runs.values() for c, _, _ in rs)
        grid = np.linspace(1.0, xmax, N_GRID)
        hits, uniq = {}, {}
        for m, rs in runs.items():
            got = [cumulative_curves(c, sc, sm, bar, grid) for c, sc, sm in rs]
            hits[m] = np.vstack([h for h, _ in got])
            uniq[m] = np.vstack([u for _, u in got])
        per_target[t] = dict(grid=grid, hits=hits, uniq=uniq, bar=bar, xmax=xmax)
        print(f'  {t}: bar(uncond p{BAR_PCT:g})={bar:7.3f}  budget={xmax:.0f} calls')
    return per_target


def _target_mean(per_target, key, method, frac):
    """Interpolate each target onto a common normalized budget, then stack."""
    out = []
    for d in per_target.values():
        g = d['grid']
        out.append(np.interp(frac, (g - g[0]) / (g[-1] - g[0]),
                             d[key][method].mean(axis=0)))
    return np.vstack(out)


def main(outdir):
    print('Collecting runs:')
    per_target = collect()
    if not per_target:
        print('No data found.')
        return

    frac = np.linspace(0.0, 1.0, N_GRID)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    panels = [
        ('hits', '(a) Distinct molecules above the\nunconditional-sampling bar',
         f'Distinct molecules above\nuncond. p{BAR_PCT:g} reward', 'upper left'),
        ('uniq', '(b) Oracle calls spent on\nnot-yet-evaluated molecules',
         'Unique fraction of oracle calls', 'lower left'),
    ]
    for ax, (key, title, ylabel, loc) in zip(axes, panels):
        for method, label, color in METHODS:
            per_t = _target_mean(per_target, key, method, frac)
            mean = per_t.mean(axis=0)
            sem = per_t.std(axis=0, ddof=1) / np.sqrt(per_t.shape[0])
            ax.plot(frac, mean, color=color, lw=2.0, label=label)
            ax.fill_between(frac, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
        ax.set_xlabel('Fraction of oracle budget', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=10, loc=loc)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=10)
    axes[1].set_ylim(0.0, 1.02)

    plt.tight_layout()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = out / f'diagram_thompson_acquisition.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'wrote {path}')
    plt.close()

    print('\nAt the final shared budget (mean over seeds):')
    print(f'{"target":<8}{"method":<22}{"above bar":>11}{"unique %":>10}')
    for t, d in per_target.items():
        for method, label, _ in METHODS:
            print(f'{t:<8}{label:<22}{d["hits"][method][:, -1].mean():>11.1f}'
                  f'{100 * d["uniq"][method][:, -1].mean():>10.1f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/overflow/trevor/inbox/figwork')
