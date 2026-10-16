#!/usr/bin/env python3
"""Run one mission:  python run_mission.py configs/mission_alpha.yaml --out outputs/alpha"""
import argparse
import json
import os
import time

from wallee.agent import HunterAgent
from wallee.env import load_config
from wallee.runner import build_report, report_markdown, run_mission, write_trace
from wallee import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--no-gif", action="store_true", help="skip the animated GIF (faster)")
    ap.add_argument("--seed", type=int, default=0, help="agent RNG seed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    t0 = time.time()
    res = run_mission(cfg, HunterAgent(seed=a.seed), collect=True)
    write_trace(res, f"{a.out}/trace.jsonl")
    rep = build_report(res)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=2)
    open(f"{a.out}/report.md", "w").write(report_markdown(rep, res["agent"].strategy_log, cfg))
    viz.dashboard(res, f"{a.out}/dashboard.png")
    open(f"{a.out}/final_map.txt", "w").write(viz.ascii_map(res))
    if not a.no_gif:
        viz.animate(res, f"{a.out}/replay.gif", every=2)
    print(viz.ascii_map(res))
    print(json.dumps(res["summary"], indent=2))
    print("Next step:", rep["recommended_next_step"])
    print(f"Outputs in {a.out}/  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
