"""Classical baselines for the benchmark (all lifeline-safe so the comparison is fair)."""
import math

import numpy as np

from .sensors import PROBE_RADIUS, SensorModel
from .terrain import Terrain


class SafeBase:
    name = "base"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, info, init):
        self.info = info
        self.t = Terrain(info["walls"])
        self.pos = tuple(info["start"])
        self.chargers = [tuple(c) for c in info["chargers"]]
        self.costs, self.cap, self.reserve = info["costs"], info["battery_cap"], info["reserve"]
        self.battery, self.time_left = self.cap, info["time_budget"]
        self.sensor = SensorModel(**info["sensor_spec"])
        self.life = np.min([self.t.bfs(c) for c in self.chargers], axis=0)
        self.contact, self.est = None, None
        self.mode = "BASELINE"
        self.init(info)

    def init(self, info):
        pass

    def integrate(self, action, res):
        self.pos, self.battery, self.time_left = res["pos"], res["battery"], res["time_left"]
        if "contact" in res["obs"]:
            self.contact = res["obs"]["contact"]
        self.on_result(action, res)

    def on_result(self, action, res):
        pass

    def move_to(self, target):
        mv = self.t.path(self.pos, target)
        if mv:
            return {"type": "move", "dir": mv[0]}
        return None

    def safety(self, extra=3):
        """Return a recharge/retreat action if the battery is getting too low, else None."""
        if self.time_left <= 1:
            return {"type": "commit"}
        if self.pos in self.chargers:
            if self.battery < 0.6 * self.cap:
                return {"type": "recharge"}
            return None
        if self.battery <= self.life[self.pos[1], self.pos[0]] * self.costs["move"] + self.reserve + extra:
            dcur = self.t.bfs(self.pos)
            ch = min(self.chargers, key=lambda c: dcur[c[1], c[0]])
            return self.move_to(ch)
        return None

    def decide(self):
        a = self.safety()
        if a:
            return a, {"mode": "SAFETY", "reasoning": "battery/time safety rule"}
        return self.act()

    def final_estimate(self):
        if self.contact:
            return {"xy": (self.contact["x"], self.contact["y"]), "conf": 1.0, "confirmed": True}
        xy = self.est if self.est is not None else self.pos
        return {"xy": tuple(int(v) for v in xy), "conf": None, "confirmed": False}


class RandomWalk(SafeBase):
    name = "Random walk"

    def act(self):
        r = self.rng.random()
        if r < 0.15:
            return {"type": "probe"}, {"mode": "RANDOM", "reasoning": "random probe"}
        d = ["N", "E", "S", "W"][int(self.rng.integers(4))]
        return {"type": "move", "dir": d}, {"mode": "RANDOM", "reasoning": "random step"}

    def final_estimate(self):
        if self.contact:
            return super().final_estimate()
        cells = self.t.free_cells()
        return {"xy": cells[int(self.rng.integers(len(cells)))], "conf": None, "confirmed": False}


class GradientAscent(SafeBase):
    """Classical signal-following: ping, project bearing/strength to a point, walk there, probe."""
    name = "Gradient follower"

    def init(self, info):
        s = info["initial_signal"]
        self.est = self._project(self.pos, s["strength"], s["bearing"])
        self.steps_since_ping = 99
        self.probed_here = set()

    def _project(self, q, s, brg):
        d = max(1.0, -self.sensor.decay * math.log(max(s, 1e-3)))
        x = q[0] + d * math.sin(math.radians(brg))
        y = q[1] - d * math.cos(math.radians(brg))
        return (float(np.clip(x, 0, self.t.W - 1)), float(np.clip(y, 0, self.t.H - 1)))

    def on_result(self, action, res):
        if action["type"] == "ping" and res["obs"].get("ping", {}).get("strength") is not None:
            p = res["obs"]["ping"]
            new = self._project(self.pos, p["strength"], p["bearing"])
            self.est = (0.5 * self.est[0] + 0.5 * new[0], 0.5 * self.est[1] + 0.5 * new[1])
        if action["type"] == "probe":
            self.est = (self.est[0] + self.rng.normal(0, 3), self.est[1] + self.rng.normal(0, 3))
            self.est = (float(np.clip(self.est[0], 0, self.t.W - 1)), float(np.clip(self.est[1], 0, self.t.H - 1)))

    def act(self):
        self.steps_since_ping += 1
        if self.steps_since_ping >= 5:
            self.steps_since_ping = 0
            return {"type": "ping"}, {"mode": "GRADIENT", "reasoning": "periodic ping"}
        tgt = min(self.t.free_cells(), key=lambda c: math.hypot(c[0] - self.est[0], c[1] - self.est[1]))
        if math.hypot(self.pos[0] - tgt[0], self.pos[1] - tgt[1]) <= 1.5:
            return {"type": "probe"}, {"mode": "GRADIENT", "reasoning": "probe at estimate"}
        a = self.move_to(tgt)
        return (a or {"type": "probe"}), {"mode": "GRADIENT", "reasoning": f"walk toward projected source {tgt}"}


class Lawnmower(SafeBase):
    name = "Lawnmower sweep"

    def init(self, info):
        W, H = self.t.W, self.t.H
        pts = []
        for i, y in enumerate(range(2, H, 4)):
            xs = list(range(2, W, 4))
            xs = xs if i % 2 == 0 else xs[::-1]
            for x in xs:
                c = min(self.t.free_cells(), key=lambda c: math.hypot(c[0] - x, c[1] - y))
                pts.append(c)
        self.wps = [p for p in dict.fromkeys(pts)]
        self.i, self.just_probed = 0, False

    def act(self):
        if self.i >= len(self.wps):
            self.i = 0
        tgt = self.wps[self.i]
        if self.pos == tgt:
            if not self.just_probed:
                self.just_probed = True
                return {"type": "probe"}, {"mode": "SWEEP", "reasoning": f"probe at waypoint {tgt}"}
            self.just_probed = False
            self.i += 1
            tgt = self.wps[self.i % len(self.wps)]
        a = self.move_to(tgt)
        return (a or {"type": "probe"}), {"mode": "SWEEP", "reasoning": f"go to waypoint {tgt}"}

    def final_estimate(self):
        if self.contact:
            return super().final_estimate()
        cells = self.t.free_cells()
        return {"xy": cells[int(self.rng.integers(len(cells)))], "conf": None, "confirmed": False}
