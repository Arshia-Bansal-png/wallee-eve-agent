#!/usr/bin/env python3
"""Generalisation benchmark over random missions:  python benchmark.py --n 40 --out outputs/benchmark"""
import argparse
import json
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wallee.agent import HunterAgent
from wallee.baselines import GradientAscent, Lawnmower, RandomWalk
from wallee.env import random_config
from wallee.runner import run_mission

HARD = False
AGENTS = {
    "Hunter (ours)": lambda s: HunterAgent(seed=s),
    "Hunter, no trust (ablation)": lambda s: HunterAgent(seed=s, use_trust=False),
    "Gradient follower": lambda s: GradientAscent(seed=s),
    "Lawnmower sweep": lambda s: Lawnmower(seed=s),
    "Random walk": lambda s: RandomWalk(seed=s),
}


def one(args):
    name, seed = args
    cfg = random_config(1000 + seed, spoof=True)
    if HARD:
        cfg["resources"]["time"] = 45
        cfg["resources"]["battery"] = 50
    r = run_mission(cfg, AGENTS[name](seed))
    s = r["summary"]
    s["name"] = name
    s["seed"] = seed
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="outputs/benchmark")
    ap.add_argument("--hard", action="store_true", help="tight time/battery budget")
    a = ap.parse_args()
    global HARD
    HARD = a.hard
    os.makedirs(a.out, exist_ok=True)
    jobs = [(n, s) for n in AGENTS for s in range(a.n)]
    with Pool() as p:
        rows = p.map(one, jobs, chunksize=2)
    json.dump(rows, open(f"{a.out}/results.json", "w"), indent=1)
    names = list(AGENTS)
    stats = {}
    for n in names:
        R = [r for r in rows if r["name"] == n]
        succ = np.mean([r["contact"] for r in R])
        loc = np.mean([r["localized"] for r in R])
        ticks = np.mean([r["ticks_used"] for r in R if r["contact"]] or [np.nan])
        err = np.median([r["error_cells"] for r in R])
        stats[n] = {"found_EVE": succ, "localized_within_2": loc, "median_error": err, "mean_ticks_when_found": ticks}
    print(f"{'agent':32s} found  loc<=2  med.err  ticks(found)")
    for n, s in stats.items():
        print(f"{n:32s} {s['found_EVE']:.0%}   {s['localized_within_2']:.0%}    {s['median_error']:.1f}     {s['mean_ticks_when_found']:.0f}")
    # calibration of our agent's confidence
    H = [r for r in rows if r["name"] == "Hunter (ours)" and r["conf"] is not None]
    conf = np.array([r["conf"] for r in H])
    hit = np.array([r["localized"] for r in H], float)
    brier = float(np.mean((conf - hit) ** 2))
    print(f"Hunter calibration: Brier={brier:.3f}, mean conf={conf.mean():.2f}, actual hit rate={hit.mean():.2f}")
    json.dump({"stats": stats, "brier": brier}, open(f"{a.out}/summary.json", "w"), indent=1, default=float)

    col = {"Hunter (ours)": "#D55E00", "Hunter, no trust (ablation)": "#E69F00", "Gradient follower": "#0072B2",
           "Lawnmower sweep": "#56B4E9", "Random walk": "#999999"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(names))
    ax[0].bar(x, [stats[n]["found_EVE"] * 100 for n in names], color=[col[n] for n in names])
    ax[0].set_title("EVE found (visual contact), %")
    ax[1].bar(x, [stats[n]["median_error"] for n in names], color=[col[n] for n in names])
    ax[1].set_title("Median localisation error (cells)")
    ax[2].bar(x, [stats[n]["mean_ticks_when_found"] for n in names], color=[col[n] for n in names])
    ax[2].set_title("Mean ticks to find (when found)")
    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Generalisation benchmark: {a.n} random missions per agent (random maps, decoys, spoof beacon)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{a.out}/benchmark.png", dpi=110)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    bins = np.linspace(0, 1, 6)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf <= hi if hi == 1 else conf < hi)
        if m.sum():
            xs.append(conf[m].mean())
            ys.append(hit[m].mean())
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.plot(xs, ys, "o-", color="#D55E00")
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual hit rate")
    ax.set_title(f"Calibration (Brier {brier:.3f})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{a.out}/calibration.png", dpi=110)


if __name__ == "__main__":
    main()
