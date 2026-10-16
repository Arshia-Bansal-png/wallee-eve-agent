"""Trust-weighted Bayesian belief over EVE's location and status.

Every soft piece of evidence (a ping reading, a clue...) is a likelihood layer with its
own *trust* z_k = P(layer is honest).  The effective layer is the mixture
    z_k * L_k(p) + (1 - z_k) * mean(L_k)
(a corrupted/spoofed layer says nothing about where EVE is).  Trust is re-estimated by a
leave-one-out EM loop: a layer is believed to the extent it agrees with everything ELSE
we know.  This is how decoy clues and spoofed pings are unmasked.
"""
import numpy as np

from .sensors import PULSES, PULSE_TABLE, STATUSES


class TrustBelief:
    def __init__(self, terrain, use_trust=True, em_iters=4):
        self.t = terrain
        self.use_trust = use_trust
        self.em_iters = em_iters
        self.idx = np.flatnonzero(terrain.free.ravel())
        self.N = len(self.idx)
        self.layers = []                      # soft evidence
        self.hard_log = np.zeros(self.N)      # hard evidence (probe misses, contact)
        self.z = np.zeros(0)
        self.status = np.ones(3) / 3
        self.b = np.full(self.N, 1.0 / self.N)
        self.logb = np.full(self.N, -np.log(self.N))
        self.n_updates = 0

    # ------------------------------------------------------------------ input
    def _prep(self, L2d):
        L = np.asarray(L2d, dtype=float).ravel()[self.idx]
        L = L / (L.max() + 1e-300)
        return np.maximum(L, 1e-9)

    def add_soft(self, lid, kind, L2d, prior, meta=None):
        self.layers.append({"id": lid, "kind": kind, "L": self._prep(L2d), "prior": float(prior), "meta": meta or {}})
        self.rebuild()

    def add_hard(self, L2d):
        self.hard_log += np.log(self._prep(L2d))
        self.rebuild()

    # ---------------------------------------------------------------- inference
    def rebuild(self):
        K = len(self.layers)
        if K == 0:
            logb = self.hard_log.copy()
            self.z = np.zeros(0)
        else:
            Lm = np.stack([l["L"] for l in self.layers])           # K x N
            meanL = Lm.mean(1)
            prior = np.array([l["prior"] for l in self.layers])
            if not self.use_trust:
                prior = np.full(K, 0.9)
            z = prior.copy()
            if self.use_trust:
                for _ in range(self.em_iters):
                    eff = z[:, None] * Lm + (1 - z[:, None]) * meanL[:, None]
                    logeff = np.log(eff + 1e-12)
                    loo = logeff.sum(0)[None, :] - logeff + self.hard_log[None, :]
                    loo -= loo.max(1, keepdims=True)
                    bk = np.exp(loo)
                    bk /= bk.sum(1, keepdims=True)
                    A = (bk * Lm).sum(1)
                    znew = prior * A / (prior * A + (1 - prior) * meanL + 1e-300)
                    z = np.clip(znew, 0.02, 0.98)
            eff = z[:, None] * Lm + (1 - z[:, None]) * meanL[:, None]
            logb = np.log(eff + 1e-12).sum(0) + self.hard_log
            self.z = z
            for l, zk in zip(self.layers, z):
                l["z"] = float(zk)
        logb = logb - logb.max()
        b = np.exp(logb)
        b /= b.sum()
        self.b = b
        self.logb = np.log(b + 1e-300)
        self.n_updates += 1

    def update_status(self, pulse):
        if pulse is None:
            return
        j = PULSES.index(pulse)
        like = np.array([PULSE_TABLE[s][j] for s in STATUSES])
        self.status = self.status * like
        self.status /= self.status.sum()

    def set_status_observed(self, status, acc=0.9):
        like = np.full(3, (1 - acc) / 2)
        like[STATUSES.index(status)] = acc
        self.status = self.status * like
        self.status /= self.status.sum()

    # ----------------------------------------------------------------- queries
    def grid(self):
        g = np.zeros(self.t.H * self.t.W)
        g[self.idx] = self.b
        return g.reshape(self.t.H, self.t.W)

    def entropy(self, b=None):
        b = self.b if b is None else b
        return float(-(b * np.log(b + 1e-300)).sum())

    def map_cell(self):
        j = int(np.argmax(self.b))
        f = int(self.idx[j])
        return (f % self.t.W, f // self.t.W)

    def mass_within(self, center, r):
        g = self.grid()
        d2 = np.square(self.t.xs - center[0]) + np.square(self.t.ys - center[1])
        return float(g[d2 <= r * r].sum())

    def best_center(self, r=2.0):
        """Cell whose radius-r disc holds the most belief mass (a more robust 'estimate' than MAP)."""
        g = self.grid()
        best, bc = -1.0, self.map_cell()
        ys, xs = np.nonzero(g > g.max() * 0.05)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if not self.t.free[y, x]:
                continue
            m = self.mass_within((x, y), r)
            if m > best:
                best, bc = m, (x, y)
        return bc, best

    def credible_area(self, mass=0.9):
        s = np.sort(self.b)[::-1]
        return int(np.searchsorted(np.cumsum(s), mass) + 1)

    # ------------------------------------------------------ planner helpers
    def hypo_entropy(self, L_free_list, priors):
        """Entropy after hypothetically multiplying in soft layers (already in free-index space)."""
        logb = self.logb.copy()
        for L, z in zip(L_free_list, priors):
            L = np.maximum(L / (L.max() + 1e-300), 1e-9)
            logb += np.log(z * L + (1 - z) * L.mean() + 1e-12)
        logb -= logb.max()
        b = np.exp(logb)
        b /= b.sum()
        return self.entropy(b)
