"""HunterAgent: trust-aware Bayesian search with information-gain-per-cost planning.

Ideas that make it different from a plain hill-climber:
  1. TRUST-WEIGHTED FUSION  - every ping/clue carries a trust score, re-estimated by
     leave-one-out consistency (see belief.py). Decoy clues and spoofed pings get unmasked.
  2. INFO-PER-COST PLANNER  - candidate (cell, action) pairs are scored by expected entropy
     reduction (Monte-Carlo over the current belief) divided by battery+time cost.
  3. LIFELINE CONSTRAINT    - no action is ever taken unless the battery still covers the walk
     back to the nearest charger plus a reserve. WALL-E can never strand himself.
  4. NAMED STRATEGY MODES   - SURVEY / TRIANGULATE / VERIFY / FINAL_APPROACH / RECHARGE /
     RETREAT, with every switch logged together with the reason.
"""
import math

import numpy as np

from .belief import TrustBelief
from .sensors import (PROBE_RADIUS, STATUSES, SensorModel, clue_layer, contact_layer, ping_layers, probe_miss_layer,
                      pulse_layer, simulate_ping, tell_prior)
from .terrain import Terrain


class HunterAgent:
    def __init__(self, seed=0, use_trust=True, n_cand=40, K=6, name=None):
        self.rng = np.random.default_rng(seed)
        self.use_trust, self.n_cand, self.K = use_trust, n_cand, K
        self.name = name or ("Hunter (trust-aware)" if use_trust else "Hunter (no trust / ablation)")

    # ------------------------------------------------------------------ setup
    def reset(self, info, init):
        self.info = info
        self.t = Terrain(info["walls"])
        self.pos = tuple(info["start"])
        self.chargers = [tuple(c) for c in info["chargers"]]
        self.costs = info["costs"]
        self.cap, self.reserve = info["battery_cap"], info["reserve"]
        self.battery, self.time_left = self.cap, info["time_budget"]
        self.sensor = SensorModel(**info["sensor_spec"]).assumed()
        self.life = np.min([self.t.bfs(c) for c in self.chargers], axis=0)
        self.belief = TrustBelief(self.t, use_trust=self.use_trust)
        self.clues, self.ping_pos = {}, []
        self.probed = np.zeros_like(self.t.walls)
        self.n_found = 0
        self.queue, self.dirty = [], True
        self.mode, self.strategy_log, self.notes = "SURVEY", [], []
        self.contact, self.trust_prev = None, {}
        self.n_pings = self.n_probes = 0
        s = info["initial_signal"]
        age = max(0, 0 - s.get("timestamp", 0))
        for kind, L, pr in ping_layers(self.t, self.sensor, self.pos, {"strength": s["strength"], "bearing": s["bearing"]}):
            self.belief.add_soft(f"init_{kind}", "init_" + kind, L, max(0.5, pr - 0.01 * age))
        self.notes.append("Initial signal received; seeding belief with its bearing/strength.")

    # ------------------------------------------------------------- integrate
    def integrate(self, action, res):
        self.pos, self.battery, self.time_left = res["pos"], res["battery"], res["time_left"]
        a, obs, b = action["type"], res["obs"], self.belief
        if a == "ping" and res["ok"]:
            self.n_pings += 1
            self.ping_pos.append(self.pos)
            rd = obs["ping"]
            for kind, L, pr in ping_layers(self.t, self.sensor, self.pos, rd):
                b.add_soft(f"p{self.n_pings}_{kind}", kind, L, pr, {"at": self.pos})
            kind, L, pr = pulse_layer(self.t, self.sensor, self.pos, rd["pulse"])
            b.add_soft(f"p{self.n_pings}_pulse", "pulse", L, pr, {"at": self.pos})
            if rd["pulse"] and b.mass_within(self.pos, self.sensor.hb_radius) > 0.5:
                b.update_status(rd["pulse"])
            self.dirty = True
        if a == "probe" and res["ok"]:
            self.n_probes += 1
            self.probed |= self.t.geometry(self.pos)["dist"] <= PROBE_RADIUS
            if "contact" in obs:
                c = obs["contact"]
                self.contact = c
                b.add_hard(contact_layer(self.t, c["x"], c["y"]))
                b.set_status_observed(c["status"], 0.95)
                self.notes.append(f"VISUAL CONTACT with EVE at ({c['x']},{c['y']}), status={c['status']}.")
            else:
                b.add_hard(probe_miss_layer(self.t, self.pos))
            self.dirty = True
        for c in obs.get("clues", []):
            if c["id"] in self.clues:
                continue
            self.clues[c["id"]] = c
            self.n_found += 1
            pr = tell_prior(c["tell"])
            b.add_soft(c["id"], "clue_" + c["kind"], clue_layer(self.t, c), pr, {"cell": c["cell"]})
            self.notes.append(f"Found clue {c['id']} ({c['kind']}) at {c['cell']}; authenticity tell {c['tell']:.2f} -> prior trust {pr:.2f}.")
            self.dirty = True
        if self.dirty:
            tr = self.trust()
            for cid, z in tr.items():
                old = self.trust_prev.get(cid)
                if old is not None and abs(z - old) > 0.25:
                    verb = "downgraded" if z < old else "upgraded"
                    self.notes.append(f"Trust in {cid} {verb} {old:.2f} -> {z:.2f} (cross-check vs. all other evidence).")
            self.trust_prev = tr

    def trust(self):
        return {l["id"]: l.get("z", l["prior"]) for l in self.belief.layers if l["id"].startswith("c") and l["kind"].startswith("clue")}

    # --------------------------------------------------------------- decision
    def decide(self):
        if self.queue and not self.dirty:
            a, meta = self.queue.pop(0)
            return a, meta
        self.dirty = False
        self.queue = []
        plan = self._plan()
        notes, self.notes = self.notes, []
        self._set_mode(plan["mode"], plan["why"])
        meta = {"mode": self.mode, "reasoning": plan["reasoning"], "notes": notes, "target": plan.get("target"),
                "gain": plan.get("gain"), "score": plan.get("score")}
        acts = plan["actions"]
        self.queue = [(a, {**meta, "notes": []}) for a in acts[1:]]
        return acts[0], meta

    def _set_mode(self, mode, why):
        if mode != self.mode:
            self.strategy_log.append({"from": self.mode, "to": mode, "why": why, "time_left": self.time_left})
            self.mode = mode

    def _plan(self):
        b, t, cm = self.belief, self.t, self.costs["move"]
        H0 = b.entropy()
        center, conf = b.best_center(2.0)
        head = f"H={H0:.2f} nats, best guess {center} holds {conf:.0%} within r=2. "
        at_charger = self.pos in self.chargers
        if self.time_left <= 1:
            return {"mode": "FINAL_APPROACH", "why": "time almost out", "reasoning": head + "Out of time: committing to current best estimate.",
                    "actions": [{"type": "commit"}]}
        if at_charger and self.battery < 0.55 * self.cap:
            n = int(math.ceil((self.cap - self.battery) / self.costs["recharge_rate"]))
            n = max(1, min(n, self.time_left - 1))
            return {"mode": "RECHARGE", "why": "low battery at charger", "reasoning": head + f"Battery {self.battery:.0f}/{self.cap:.0f} at a charger: topping up ({n} ticks).",
                    "actions": [{"type": "recharge"}] * n}
        dcur = t.bfs(self.pos)
        cands = self._candidates(dcur)
        if not cands:
            if at_charger and self.battery >= self.cap - 1:
                return {"mode": "FINAL_APPROACH", "why": "nothing affordable", "reasoning": head + "No affordable useful action left: committing.",
                        "actions": [{"type": "commit"}]}
            if at_charger:
                return {"mode": "RECHARGE", "why": "no affordable action", "reasoning": head + "Nothing affordable on this charge: recharging.",
                        "actions": [{"type": "recharge"}] * max(1, int(math.ceil((self.cap - self.battery) / self.costs["recharge_rate"])))}
            ch = min(self.chargers, key=lambda c: dcur[c[1], c[0]])
            mv = t.path(self.pos, ch) or []
            return {"mode": "RETREAT", "why": "no action keeps the lifeline", "reasoning": head + f"No action keeps battery above lifeline; retreating to charger {ch} ({len(mv)} steps).",
                    "actions": [{"type": "move", "dir": d} for d in mv] or [{"type": "recharge"}], "target": ch}
        # ---- score candidates
        samples = self._sample_sources()
        best = None
        gdens = (self.n_found + 0.5) / (self.probed.sum() + 30.0)
        for q in cands:
            geo = t.geometry(q)
            d = dcur[q[1], q[0]]
            for act in ("ping", "probe"):
                ab = self.costs[act]
                if d * cm + ab + self.life[q[1], q[0]] * cm + self.reserve > self.battery:
                    continue
                tick = 2 if act == "ping" else 1
                if d + tick >= self.time_left:
                    continue
                if act == "ping":
                    gain = self._ping_gain(q, samples, H0)
                    gain *= 0.5 ** sum(1 for p in self.ping_pos if math.hypot(p[0] - q[0], p[1] - q[1]) <= 2.0)
                    m = 0.0
                else:
                    gain, m = self._probe_gain(q, geo, H0, gdens)
                cost = d * cm + ab + 0.5 * tick + 0.5
                score = gain / cost
                if best is None or score > best["score"]:
                    best = {"score": score, "gain": gain, "q": q, "act": act, "cost": cost, "m": m, "d": d}
        if best is None:
            return {"mode": "RETREAT", "why": "no scored action", "reasoning": head + "No scored action; heading to charger.",
                    "actions": [{"type": "move", "dir": d} for d in (t.path(self.pos, min(self.chargers, key=lambda c: dcur[c[1], c[0]])) or [])] or [{"type": "recharge"}]}
        q, act = best["q"], best["act"]
        if act == "ping":
            mode, why = "TRIANGULATE", "ping offers the best information per cost"
        elif best["m"] >= 0.5:
            mode, why = "FINAL_APPROACH", f"{best['m']:.0%} of belief mass inside probe disc"
        elif best["m"] >= 0.2:
            mode, why = "VERIFY", f"{best['m']:.0%} of belief mass inside probe disc: check hypothesis"
        else:
            mode, why = "SURVEY", "hunting for new clues / ruling out regions"
        mv = t.path(self.pos, q) or []
        low = [cid for cid, z in self.trust().items() if z < 0.3]
        txt = head + f"Best info/cost: {act.upper()} at {q} (expected {best['gain']:.2f} nats for cost {best['cost']:.1f}; walk {int(best['d'])}, lifeline back {int(self.life[q[1], q[0]])}). Mode={mode}."
        if low:
            txt += f" Distrusting {', '.join(low)} as likely decoys."
        return {"mode": mode, "why": why, "reasoning": txt, "target": q, "gain": best["gain"], "score": best["score"],
                "actions": [{"type": "move", "dir": d} for d in mv] + [{"type": act}]}

    # ------------------------------------------------------------ planner utils
    def _candidates(self, dcur):
        t, cm = self.t, self.costs["move"]
        ok = np.isfinite(dcur) & t.free & ((dcur * cm + 2 + self.life * cm + self.reserve) <= self.battery) & (dcur + 1 < self.time_left)
        ys, xs = np.nonzero(ok)
        cells = list(zip(xs.tolist(), ys.tolist()))
        if len(cells) <= self.n_cand:
            return cells
        center, _ = self.belief.best_center(2.0)
        near = sorted(cells, key=lambda c: math.hypot(c[0] - center[0], c[1] - center[1]))[:12]
        rest = [c for c in cells if c not in set(near)]
        pick = self.rng.choice(len(rest), self.n_cand - len(near), replace=False)
        out = near + [rest[i] for i in pick]
        if self.pos not in out:
            out.append(self.pos)
        return out

    def _sample_sources(self):
        b = self.belief
        js = self.rng.choice(b.N, self.K, p=b.b)
        out = []
        for j in js:
            f = int(b.idx[j])
            out.append((f % self.t.W, f // self.t.W))
        return out

    def _ping_gain(self, q, samples, H0):
        b, tot = self.belief, 0.0
        for src in samples:
            st = STATUSES[int(self.rng.choice(3, p=b.status))]
            rd = simulate_ping(self.sensor, self.t, q, src, self.rng, st)
            Ls, zs = [], []
            for kind, L, pr in ping_layers(self.t, self.sensor, q, rd) + [pulse_layer(self.t, self.sensor, q, rd["pulse"])]:
                Ls.append(L.ravel()[b.idx])
                zs.append(pr)
            tot += b.hypo_entropy(Ls, zs)
        return max(0.0, H0 - tot / len(samples))

    def _probe_gain(self, q, geo, H0, gdens):
        b = self.belief
        mask = (geo["dist"] <= PROBE_RADIUS).ravel()[b.idx]
        m = float(b.b[mask].sum())
        p_hit = 0.95 * m
        bm = b.b * np.where(mask, 0.05, 1.0)
        bm /= bm.sum()
        gain = H0 - (1 - p_hit) * b.entropy(bm)
        unprobed = (~self.probed.ravel()[b.idx]) & mask
        gain += 0.35 * min(1.5, gdens * unprobed.sum())
        return max(0.0, gain), m

    # ----------------------------------------------------------------- output
    def final_estimate(self):
        b = self.belief
        if self.contact:
            xy, conf = (self.contact["x"], self.contact["y"]), 1.0
        else:
            xy, conf = b.best_center(2.0)
        st = {s: float(p) for s, p in zip(STATUSES, b.status)}
        return {"xy": tuple(xy), "conf": float(conf), "status": st, "status_best": max(st, key=st.get),
                "confirmed": bool(self.contact), "credible_area_90": b.credible_area(0.9), "entropy": b.entropy(),
                "trust": self.trust(), "n_pings": self.n_pings, "n_probes": self.n_probes}
