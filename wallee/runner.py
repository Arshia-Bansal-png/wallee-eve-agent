"""Mission loop + mission report."""
import json
import math

import numpy as np

from .env import Environment


def run_mission(cfg, agent, collect=False, max_steps=3000):
    env = Environment(cfg)
    info = env.public_info()
    agent.reset(info, info["initial_signal"])
    records, grids = [], []
    last_grid = None
    while not env.done and env.n_steps < max_steps:
        action, meta = agent.decide()
        res = env.step(action)
        agent.integrate(action, res)
        if collect:
            bel = getattr(agent, "belief", None)
            if bel is not None and (last_grid is None or bel.n_updates != last_grid[0]):
                last_grid = (bel.n_updates, bel.grid().astype(np.float32))
            records.append({
                "step": env.n_steps, "time_left": res["time_left"], "pos": list(res["pos"]),
                "battery": round(res["battery"], 1), "action": action,
                "ok": res["ok"], "mode": meta.get("mode"), "reasoning": meta.get("reasoning"),
                "notes": meta.get("notes", []),
                "obs": _jsonable(res["obs"]),
                "belief": None if bel is None else {
                    "entropy": round(bel.entropy(), 3), "map": list(bel.map_cell()),
                    "conf2": round(bel.best_center(2.0)[1], 3),
                    "status": {k: round(float(v), 3) for k, v in zip(["active", "dormant", "damaged"], bel.status)}},
                "trust": {k: round(float(v), 3) for k, v in agent.trust().items()} if hasattr(agent, "trust") else {},
            })
            grids.append(None if last_grid is None else last_grid[1])
    est = agent.final_estimate()
    eve = env.world.eve
    err = math.hypot(est["xy"][0] - eve[0], est["xy"][1] - eve[1])
    summary = {"mission": cfg["mission"]["name"], "agent": agent.name, "outcome": env.outcome, "steps": env.n_steps,
               "ticks_used": int(cfg["resources"]["time"]) - env.time_left, "error_cells": round(err, 2),
               "localized": bool(err <= 2.0), "confirmed": bool(est.get("confirmed")), "conf": est.get("conf"),
               "contact": env.outcome == "contact"}
    return {"env": env, "agent": agent, "records": records, "grids": grids, "summary": summary, "estimate": est}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def write_trace(result, path):
    with open(path, "w") as f:
        for r in result["records"]:
            f.write(json.dumps(r) + "\n")


def build_report(result):
    env, ag, est, s = result["env"], result["agent"], result["estimate"], result["summary"]
    t, w = env.t, env.world
    x, y = est["xy"]
    dcur = t.bfs(env.pos)
    walk = dcur[int(y), int(x)] if t.is_free(int(x), int(y)) else float("nan")
    ch = min(env.world.chargers, key=lambda c: dcur[c[1], c[0]])
    conf = est.get("conf")
    status = est.get("status_best", "unknown")
    if est.get("confirmed"):
        nxt = {
            "active": f"EVE is ACTIVE at ({x},{y}). Move adjacent, send the docking handshake and escort her to charger {ch}.",
            "dormant": f"EVE is DORMANT at ({x},{y}). Bring a charge to her: a solar/charger wake-up on-site is safer than towing.",
            "damaged": f"EVE is DAMAGED at ({x},{y}). Do NOT tow. Stay adjacent, call for repair and shield her from the dust.",
        }[status]
    elif conf is not None and conf >= 0.6:
        nxt = (f"Go to ({x},{y}) (~{int(walk)} steps) and probe to confirm. Recharge at charger {ch} first if battery < "
               f"{int(walk + 6)}. Prepare for status '{status}' ({est['status'][status]:.0%}).")
    else:
        nxt = (f"Confidence is only {0 if conf is None else conf:.0%}. Recharge at charger {ch}, then re-ping from two new "
               f"vantage points and probe the strongest region near ({x},{y}).")
    flagged = [c for c, z in est["trust"].items() if z < 0.3]
    truth = {c["id"]: c["honest"] for c in w.clues}
    caught = [c for c in flagged if not truth[c]]
    false_alarm = [c for c in flagged if truth[c]]
    rep = {"mission": s["mission"], "agent": s["agent"], "outcome": s["outcome"], "estimate_xy": [int(x), int(y)],
           "confidence_within_2_cells": conf, "confirmed_by_visual_contact": est.get("confirmed"),
           "status_posterior": est.get("status"), "inferred_status": status,
           "credible_area_90_cells": est.get("credible_area_90"), "clues_found": len(ag.clues) if hasattr(ag, "clues") else 0,
           "clue_trust": est.get("trust"), "flagged_decoys": flagged, "decoys_caught": caught, "false_alarms": false_alarm,
           "pings": est.get("n_pings"), "probes": est.get("n_probes"), "ticks_used": s["ticks_used"],
           "recommended_next_step": nxt,
           "ground_truth_for_evaluation": {"eve": list(w.eve), "status": w.status, "spoof_beacon": list(w.spoof_at),
                                           "error_cells": s["error_cells"], "decoy_clues": [c["id"] for c in w.clues if not c["honest"]]}}
    return rep


def report_markdown(rep, strategy_log, cfg):
    L = [f"# Mission report: {rep['mission']}", "", f"**Agent:** {rep['agent']}  |  **Outcome:** `{rep['outcome']}`", "",
         "## Verdict", "",
         f"- **Inferred EVE location:** {tuple(rep['estimate_xy'])}",
         f"- **Confidence (within 2 cells):** {rep['confidence_within_2_cells']:.0%}" if rep["confidence_within_2_cells"] is not None else "- Confidence: n/a",
         f"- **Visual contact confirmed:** {rep['confirmed_by_visual_contact']}",
         f"- **Inferred status:** {rep['inferred_status']}  (posterior: " + ", ".join(f"{k} {v:.0%}" for k, v in rep["status_posterior"].items()) + ")",
         f"- **90% credible region:** {rep['credible_area_90_cells']} cells",
         "", "## Recommended next step", "", rep["recommended_next_step"], "", "## Evidence", "",
         f"- {rep['pings']} pings, {rep['probes']} probes, {rep['clues_found']} clues found, {rep['ticks_used']} ticks used.",
         "- Clue trust (1 = believed, 0 = judged a decoy): " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(rep["clue_trust"].items())),
         f"- Flagged as decoys: {rep['flagged_decoys'] or 'none'}",
         "", "## Strategy changes", ""]
    for i, s in enumerate(strategy_log, 1):
        L.append(f"{i}. t-left {s['time_left']}: **{s['from']} -> {s['to']}** because {s['why']}.")
    g = rep["ground_truth_for_evaluation"]
    L += ["", "## Ground truth (for evaluation only; the agent never saw this)", "",
          f"- True EVE: {tuple(g['eve'])} ({g['status']}); localisation error {g['error_cells']} cells",
          f"- Spoof beacon was at {tuple(g['spoof_beacon'])}; true decoy clues: {g['decoy_clues']}",
          f"- Decoys caught: {rep['decoys_caught'] or 'none'}; false alarms: {rep['false_alarms'] or 'none'}", ""]
    return "\n".join(L)
