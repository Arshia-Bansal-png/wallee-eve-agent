"""Mission config loading, world generation and the simulated environment.

The agent only ever sees `Environment.public_info()` (map, chargers, budgets, datasheet,
first signal) and the observations returned by `step()`. Clue positions, EVE's location and
status, the spoof beacon and which clues are decoys are hidden truth.
"""
import copy
import math
from dataclasses import dataclass

import numpy as np

from .sensors import PROBE_RADIUS, STATUSES, SensorModel, simulate_ping
from .terrain import DIRS, Terrain

DEFAULTS = {
    "mission": {"name": "mission", "seed": 1},
    "map": {"width": 24, "height": 24, "wall_density": 0.16, "chargers": 3, "ascii": None},
    "start": [2, 2],
    "initial_signal": {"bearing_deg": None, "strength": None, "timestamp": 0},
    "resources": {"battery": 90, "time": 260, "reserve": 4,
                  "costs": {"move": 1, "ping": 3, "probe": 2, "recharge_rate": 8}},
    "clues": {"honest": 5, "decoy": 3},
    "sensor": {"power": 1.0, "decay": 6.5, "wall_att": 0.6, "floor": 0.04, "sigma_log": 0.24,
               "sigma_bearing": 14.0, "outlier": 0.10, "hb_radius": 5.0, "hb_detect": 0.85},
    "spoof": {"enabled": True, "radius": 6.0, "prob": 0.7},
    "eve": {"position": None, "status": None},
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path_or_dict):
    if isinstance(path_or_dict, dict):
        return deep_merge(DEFAULTS, path_or_dict)
    import json
    text = open(path_or_dict).read()
    if str(path_or_dict).endswith((".yaml", ".yml")):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return deep_merge(DEFAULTS, data)


def random_config(seed, spoof=True):
    r = np.random.default_rng(seed)
    return load_config({
        "mission": {"name": f"random_{seed}", "seed": int(seed)},
        "map": {"width": int(r.integers(20, 27)), "height": int(r.integers(20, 27)),
                "wall_density": float(r.uniform(0.10, 0.20)), "chargers": 3},
        "clues": {"honest": int(r.integers(4, 7)), "decoy": int(r.integers(2, 5))},
        "spoof": {"enabled": bool(spoof)},
    })


@dataclass
class World:
    terrain: Terrain
    start: tuple
    chargers: list
    eve: tuple
    status: str
    spoof_at: tuple
    clues: list


def _blob_walls(rng, W, H, density, start):
    walls = np.zeros((H, W), dtype=bool)
    guard = 0
    while walls.mean() < density and guard < 2000:
        guard += 1
        w, h = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        if rng.random() < 0.5:
            w, h = h, w
        x, y = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        walls[y : y + h, x : x + w] = True
    sx, sy = start
    walls[max(0, sy - 1) : sy + 2, max(0, sx - 1) : sx + 2] = False
    return walls


def build_world(cfg):
    rng = np.random.default_rng(cfg["mission"]["seed"])
    m = cfg["map"]
    start = tuple(cfg["start"])
    if m.get("ascii"):
        rows = m["ascii"]
        walls = np.array([[ch == "#" for ch in row] for row in rows])
        H, W = walls.shape
    else:
        W, H = int(m["width"]), int(m["height"])
        walls = _blob_walls(rng, W, H, float(m["wall_density"]), start)
    walls[start[1], start[0]] = False
    t = Terrain(walls)
    walls = walls | ~np.isfinite(t.bfs(start))          # drop unreachable pockets
    t = Terrain(walls)
    cells = t.free_cells()
    dist0 = t.bfs(start)

    # chargers: start + farthest-point sampling
    chargers = [start]
    while len(chargers) < int(m["chargers"]):
        best = max(cells, key=lambda c: (min(math.hypot(c[0] - a[0], c[1] - a[1]) for a in chargers), rng.random()))
        chargers.append(best)

    # EVE
    if cfg["eve"]["position"] is not None:
        eve = tuple(cfg["eve"]["position"])
    else:
        far = [c for c in cells if dist0[c[1], c[0]] >= 0.55 * dist0[np.isfinite(dist0)].max()]
        eve = far[int(rng.integers(len(far)))]
    status = cfg["eve"]["status"] or STATUSES[int(rng.integers(3))]

    # spoof beacon
    cand = [c for c in cells if math.hypot(c[0] - eve[0], c[1] - eve[1]) >= 8 and math.hypot(c[0] - start[0], c[1] - start[1]) >= 6]
    spoof_at = cand[int(rng.integers(len(cand)))] if cand else cells[0]

    # clues
    def content(kind, cell, target):
        cx, cy = cell
        if kind == "tracks":
            b = (math.degrees(math.atan2(target[0] - cx, -(target[1] - cy))) + rng.normal(0, 12)) % 360
            return {"bearing": float(b)}
        if kind == "scorch":
            return {"radius": float(max(1.0, math.hypot(target[0] - cx, target[1] - cy) + rng.normal(0, 1.5)))}
        return {"x": float(target[0] + rng.normal(0, 2.5)), "y": float(target[1] + rng.normal(0, 2.5))}

    kinds = ["tracks", "scorch", "log"]
    near = [c for c in cells if 2 <= math.hypot(c[0] - eve[0], c[1] - eve[1]) <= 11 and c != start]
    anyc = [c for c in cells if c != start and math.hypot(c[0] - eve[0], c[1] - eve[1]) >= 4]
    clues, used = [], set()
    for i in range(int(cfg["clues"]["honest"]) + int(cfg["clues"]["decoy"])):
        honest = i < int(cfg["clues"]["honest"])
        pool = near if honest else anyc
        cell = pool[int(rng.integers(len(pool)))]
        while cell in used:
            cell = pool[int(rng.integers(len(pool)))]
        used.add(cell)
        kind = kinds[int(rng.integers(3))]
        target = eve if honest else (spoof_at if cfg["spoof"]["enabled"] and rng.random() < 0.7 else cells[int(rng.integers(len(cells)))])
        tell = float(np.clip(rng.normal(0.68 if honest else 0.32, 0.2), 0, 1))
        clues.append({"id": f"c{i}", "kind": kind, "cell": cell, "honest": honest, "tell": tell,
                      "content": content(kind, cell, target)})
    order = rng.permutation(len(clues))
    clues = [clues[i] for i in order]
    for i, c in enumerate(clues):
        c["id"] = f"c{i}"
    return World(t, start, chargers, eve, status, spoof_at, clues)


class Environment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.world = build_world(cfg)
        self.t = self.world.terrain
        self.rng = np.random.default_rng(int(cfg["mission"]["seed"]) + 7919)
        self.sensor = SensorModel(**cfg["sensor"])
        r = cfg["resources"]
        self.cap, self.reserve = float(r["battery"]), float(r["reserve"])
        self.costs = r["costs"]
        self.battery, self.time_left = self.cap, int(r["time"])
        self.pos = self.world.start
        self.done, self.outcome, self.n_steps = False, None, 0
        self.found = set()
        self._init_reading = self._make_initial()

    def _make_initial(self):
        s = self.cfg["initial_signal"]
        if s.get("bearing_deg") is not None and s.get("strength") is not None:
            return {"strength": float(s["strength"]), "bearing": float(s["bearing_deg"]),
                    "timestamp": s.get("timestamp", 0)}
        rng = np.random.default_rng(int(self.cfg["mission"]["seed"]) + 31)
        rd = simulate_ping(self.sensor, self.t, self.world.start, self.world.eve, rng, self.world.status)
        if rd["strength"] is None:                       # too far: give a weak coarse reading
            geo = self.t.geometry(self.world.start)
            rd = {"strength": self.sensor.floor * 1.05, "bearing": float(geo["brg"][self.world.eve[1], self.world.eve[0]])}
        return {"strength": rd["strength"], "bearing": rd["bearing"], "timestamp": s.get("timestamp", 0)}

    # -------------------------------------------------------------- interface
    def public_info(self):
        sp = self.cfg["sensor"]
        return {"width": self.t.W, "height": self.t.H, "walls": self.t.walls.copy(),
                "start": self.world.start, "chargers": list(self.world.chargers),
                "battery_cap": self.cap, "time_budget": int(self.cfg["resources"]["time"]),
                "reserve": self.reserve, "costs": dict(self.costs),
                "sensor_spec": dict(sp), "initial_signal": dict(self._init_reading),
                "name": self.cfg["mission"]["name"]}

    def _pub_clue(self, c):
        return {"id": c["id"], "kind": c["kind"], "cell": c["cell"], "content": c["content"], "tell": c["tell"]}

    def step(self, action):
        if self.done:
            return {"ok": False, "events": ["mission over"], "obs": {}, **self._state()}
        a, c, w = action["type"], self.costs, self.world
        res = {"ok": True, "events": [], "obs": {}}
        batt, ticks = 0.0, 1
        if a == "move":
            dx, dy = DIRS[action["dir"]]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if self.t.is_free(nx, ny) and self.battery >= c["move"]:
                self.pos = (nx, ny)
                batt = c["move"]
                new = [self._pub_clue(k) for k in w.clues if k["cell"] == self.pos and k["id"] not in self.found]
                for k in new:
                    self.found.add(k["id"])
                if new:
                    res["obs"]["clues"] = new
            else:
                res["ok"] = False
                res["events"].append("blocked")
        elif a == "ping" and self.battery >= c["ping"]:
            batt, ticks = c["ping"], 2
            near_spoof = (self.cfg["spoof"]["enabled"] and
                          math.hypot(self.pos[0] - w.spoof_at[0], self.pos[1] - w.spoof_at[1]) <= self.cfg["spoof"]["radius"])
            if near_spoof and self.rng.random() < self.cfg["spoof"]["prob"]:
                res["obs"]["ping"] = simulate_ping(self.sensor, self.t, self.pos, w.spoof_at, self.rng, w.status, spoof=True)
            else:
                res["obs"]["ping"] = simulate_ping(self.sensor, self.t, self.pos, w.eve, self.rng, w.status)
        elif a == "probe" and self.battery >= c["probe"]:
            batt = c["probe"]
            new = []
            for k in w.clues:
                if k["id"] not in self.found and math.hypot(k["cell"][0] - self.pos[0], k["cell"][1] - self.pos[1]) <= PROBE_RADIUS \
                        and self.rng.random() < 0.9:
                    self.found.add(k["id"])
                    new.append(self._pub_clue(k))
            if new:
                res["obs"]["clues"] = new
            if math.hypot(w.eve[0] - self.pos[0], w.eve[1] - self.pos[1]) <= PROBE_RADIUS and self.rng.random() < 0.95:
                res["obs"]["contact"] = {"x": w.eve[0], "y": w.eve[1], "status": w.status}
                self.done, self.outcome = True, "contact"
        elif a == "recharge" and self.pos in w.chargers:
            self.battery = min(self.cap, self.battery + c["recharge_rate"])
        elif a == "commit":
            self.done, self.outcome = True, "commit"
            ticks = 0
        else:
            res["ok"] = False
            res["events"].append("invalid or unaffordable action")
        self.battery -= batt
        self.time_left -= ticks
        self.n_steps += 1
        if not self.done:
            if self.battery <= 0 and self.pos not in w.chargers:
                self.done, self.outcome = True, "stranded"
            elif self.time_left <= 0:
                self.done, self.outcome = True, "timeout"
        res.update(self._state())
        return res

    def _state(self):
        return {"pos": self.pos, "battery": float(self.battery), "time_left": int(self.time_left)}
