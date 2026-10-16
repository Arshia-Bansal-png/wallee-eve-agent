"""Sensor physics (used by BOTH the hidden simulator and the agent's assumed model)
plus the likelihood-layer builders the agent uses to turn observations into evidence.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

try:  # scipy is optional
    from scipy.special import ndtr as _ndtr
except Exception:  # pragma: no cover
    def _ndtr(x):
        return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))

STATUSES = ["active", "dormant", "damaged"]
PULSES = ["steady", "erratic", "faint"]
# P(pulse type | EVE status) -- rows follow STATUSES, columns follow PULSES
PULSE_TABLE = {
    "active": [0.75, 0.15, 0.10],
    "dormant": [0.10, 0.20, 0.70],
    "damaged": [0.10, 0.70, 0.20],
}
PROBE_RADIUS = 2.5


@dataclass
class SensorModel:
    power: float = 1.0
    decay: float = 6.5
    wall_att: float = 0.6
    floor: float = 0.04
    sigma_log: float = 0.3
    sigma_bearing: float = 15.0   # degrees (base)
    outlier: float = 0.1
    hb_radius: float = 5.0
    hb_detect: float = 0.85

    def mu_log(self, geo):
        return math.log(self.power) - geo["dist"] / self.decay + geo["wc"] * math.log(self.wall_att)

    def bearing_sigma(self, geo):
        s = np.radians(self.sigma_bearing) * (1 + geo["dist"] / (2 * self.decay)) * (1 + 0.15 * geo["wc"])
        return np.clip(s, 0.12, 2.5)

    def assumed(self):
        """What the agent believes: same datasheet, but humbler about noise."""
        return replace(self, sigma_log=self.sigma_log * 1.25, sigma_bearing=self.sigma_bearing * 1.2,
                       wall_att=min(0.95, self.wall_att + 0.05))


def simulate_ping(sensor, terrain, q, src, rng, status, spoof=False):
    """Draw one ping reading for a source at `src` seen from `q`."""
    geo = terrain.geometry(q)
    x, y = src
    mu = sensor.mu_log(geo)[y, x]
    s = math.exp(mu + rng.normal(0, sensor.sigma_log))
    if s < sensor.floor:
        return {"strength": None, "bearing": None, "pulse": None}
    sig = sensor.bearing_sigma(geo)[y, x]
    b = (geo["brg"][y, x] + math.degrees(rng.normal(0, sig))) % 360.0
    if rng.random() < sensor.outlier:
        b = float(rng.uniform(0, 360))
    pulse = None
    if (not spoof) and geo["dist"][y, x] <= sensor.hb_radius and rng.random() < sensor.hb_detect:
        pulse = PULSES[int(rng.choice(3, p=PULSE_TABLE[status]))]
    return {"strength": float(s), "bearing": float(b), "pulse": pulse}


# ----------------------------------------------------------------- likelihoods
def _vonmises_layer(obs_deg, brg, sigma_rad):
    d = np.radians(obs_deg - brg)
    kappa = 1.0 / np.square(sigma_rad)
    return np.exp(kappa * (np.cos(d) - 1.0)) / np.i0(kappa) / np.exp(-kappa) / (2 * np.pi) \
        if False else np.exp(kappa * (np.cos(d) - 1.0)) / (np.i0(kappa) * np.exp(-kappa)) / (2 * np.pi)


def ping_layers(terrain, sensor, q, reading):
    """Evidence layers (kind, L(H,W), prior_trust) for one ping reading."""
    geo = terrain.geometry(q)
    mu = sensor.mu_log(geo)
    out = []
    if reading["strength"] is None:
        z = (math.log(sensor.floor) - mu) / sensor.sigma_log
        out.append(("ping_null", _ndtr(z), 0.9))
        return out
    ls = math.log(reading["strength"])
    out.append(("ping_strength", np.exp(-0.5 * np.square((ls - mu) / sensor.sigma_log)), 0.9))
    Lb = _vonmises_layer(reading["bearing"], geo["brg"], sensor.bearing_sigma(geo))
    Lb = np.where(geo["dist"] < 0.5, 1.0 / (2 * np.pi), Lb)
    out.append(("ping_bearing", Lb, 0.85))
    return out


def pulse_layer(terrain, sensor, q, pulse):
    geo = terrain.geometry(q)
    inr = geo["dist"] <= sensor.hb_radius
    if pulse is None:
        return ("pulse", np.where(inr, 1.0 - sensor.hb_detect, 1.0), 0.93)
    return ("pulse", np.where(inr, sensor.hb_detect, 0.02), 0.93)


def clue_layer(terrain, clue):
    """Likelihood over EVE location implied by a clue *if it is honest*."""
    cx, cy = clue["cell"]
    geo = terrain.geometry((cx, cy))
    c = clue["content"]
    if clue["kind"] == "tracks":
        L = _vonmises_layer(c["bearing"], geo["brg"], np.full_like(geo["dist"], math.radians(20.0)))
        return np.where(geo["dist"] < 0.5, 1.0 / (2 * np.pi), L)
    if clue["kind"] == "scorch":
        r = c["radius"]
        sig = 1.8 + 0.12 * r
        return np.exp(-0.5 * np.square((geo["dist"] - r) / sig))
    if clue["kind"] == "log":
        d2 = np.square(terrain.xs - c["x"]) + np.square(terrain.ys - c["y"])
        return np.exp(-0.5 * d2 / 3.2 ** 2)
    raise ValueError(clue["kind"])


def tell_prior(tell, base=0.7):
    """Prior honesty from the noisy 'authenticity tell' (honest ~N(.68,.2), fake ~N(.32,.2))."""
    llr = (-0.5 * ((tell - 0.68) / 0.2) ** 2) - (-0.5 * ((tell - 0.32) / 0.2) ** 2)
    logit = math.log(base / (1 - base)) + llr
    return float(np.clip(1 / (1 + math.exp(-logit)), 0.05, 0.95))


def probe_miss_layer(terrain, q, radius=PROBE_RADIUS):
    geo = terrain.geometry(q)
    return np.where(geo["dist"] <= radius, 0.05, 1.0)


def contact_layer(terrain, x, y):
    d2 = np.square(terrain.xs - x) + np.square(terrain.ys - y)
    return np.exp(-0.5 * d2 / 0.6 ** 2) + 1e-4
