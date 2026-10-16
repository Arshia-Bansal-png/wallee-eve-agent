"""Grid terrain: walls, BFS distances/paths, and cached sensing geometry.

Coordinates are (x, y) with x = column, y = row (y grows downward).
Bearings are compass degrees: 0 = North (up), 90 = East, clockwise.
"""
from collections import deque

import numpy as np

DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
ORDER = ["N", "E", "S", "W"]


class Terrain:
    def __init__(self, walls):
        self.walls = np.asarray(walls, dtype=bool)
        self.H, self.W = self.walls.shape
        self.free = ~self.walls
        self.ys, self.xs = np.mgrid[0 : self.H, 0 : self.W]
        self._bfs = {}
        self._geo = {}

    # ---------------------------------------------------------------- basics
    def is_free(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H and bool(self.free[y, x])

    def free_cells(self):
        ys, xs = np.nonzero(self.free)
        return list(zip(xs.tolist(), ys.tolist()))

    # ------------------------------------------------------------------ BFS
    def bfs(self, src):
        """Walking distance (in steps) from src to every cell (inf = unreachable)."""
        src = (int(src[0]), int(src[1]))
        if src in self._bfs:
            return self._bfs[src]
        dist = np.full((self.H, self.W), np.inf)
        dist[src[1], src[0]] = 0
        dq = deque([src])
        while dq:
            x, y = dq.popleft()
            d = dist[y, x] + 1
            for dx, dy in DIRS.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H and self.free[ny, nx] and dist[ny, nx] == np.inf:
                    dist[ny, nx] = d
                    dq.append((nx, ny))
        self._bfs[src] = dist
        return dist

    def path(self, src, dst):
        """List of move letters from src to dst (shortest walking path)."""
        src = (int(src[0]), int(src[1]))
        dst = (int(dst[0]), int(dst[1]))
        dist = self.bfs(dst)
        if not np.isfinite(dist[src[1], src[0]]):
            return None
        x, y = src
        moves = []
        while (x, y) != dst:
            for k in ORDER:
                dx, dy = DIRS[k]
                nx, ny = x + dx, y + dy
                if self.is_free(nx, ny) and dist[ny, nx] < dist[y, x]:
                    moves.append(k)
                    x, y = nx, ny
                    break
            else:
                return None
        return moves

    # -------------------------------------------------------------- geometry
    def geometry(self, q):
        """Euclid distance, compass bearing and approx walls-crossed from q to every cell."""
        q = (int(q[0]), int(q[1]))
        if q in self._geo:
            return self._geo[q]
        qx, qy = q
        dx = (self.xs - qx).astype(float)
        dy = (self.ys - qy).astype(float)
        dist = np.hypot(dx, dy)
        brg = np.degrees(np.arctan2(dx, -dy)) % 360.0
        n = 24
        t = (np.arange(n) + 0.5) / n
        px = np.clip(np.rint(qx + dx[..., None] * t).astype(int), 0, self.W - 1)
        py = np.clip(np.rint(qy + dy[..., None] * t).astype(int), 0, self.H - 1)
        wc = self.walls[py, px].sum(-1) / n * dist
        geo = {"dist": dist, "brg": brg, "wc": wc}
        self._geo[q] = geo
        return geo
