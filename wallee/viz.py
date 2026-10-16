"""Visualisation: animated replay GIF, static dashboard PNG and an ASCII renderer."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

MODE_COL = {"SURVEY": "#0072B2", "TRIANGULATE": "#56B4E9", "VERIFY": "#CC79A7", "FINAL_APPROACH": "#D55E00",
            "RECHARGE": "#009E73", "RETREAT": "#E69F00", "BASELINE": "#999999", "SAFETY": "#E69F00",
            "GRADIENT": "#999999", "SWEEP": "#999999", "RANDOM": "#999999"}
BELIEF = LinearSegmentedColormap.from_list("b", [(0.05, 0.05, 0.10, 0.0), "#5b2a86", "#d6456b", "#ffd166", "#ffffff"])
WALLE = "#F2B705"


def _draw_map(ax, env, recs, grids, i, final=False):
    t = env.t
    ax.set_facecolor("#14161c")
    ax.imshow(np.where(t.walls, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3, origin="upper", alpha=0.9)
    g = grids[i]
    if g is not None:
        ax.imshow(np.where(t.free, g / (g.max() + 1e-12), np.nan) ** 0.6, cmap=BELIEF, vmin=0, vmax=1, origin="upper")
    for c in env.world.chargers:
        ax.plot(*c, "s", color="#2ecc71", ms=9, mec="white", mew=1)
    path = np.array([r["pos"] for r in recs[: i + 1]])
    ax.plot(path[:, 0], path[:, 1], "-", color=WALLE, lw=1.4, alpha=0.8)
    seen, trust = {}, recs[i]["trust"]
    for r in recs[: i + 1]:
        for c in r["obs"].get("clues", []):
            seen[c["id"]] = c
        a = r["action"]["type"]
        if a == "ping" and r["step"] >= max(1, recs[i]["step"] - 6):
            ax.add_patch(plt.Circle(r["pos"], 1.5 + 1.2 * (recs[i]["step"] - r["step"]), fill=False, ec="#56B4E9", lw=1.2, alpha=0.7))
    cm = plt.get_cmap("RdYlGn")
    for cid, c in seen.items():
        z = trust.get(cid, 0.5)
        ax.plot(*c["cell"], "^", color=cm(z), ms=10, mec="black", mew=0.8)
        ax.text(c["cell"][0] + 0.4, c["cell"][1] - 0.5, cid, color="white", fontsize=6)
    r = recs[i]
    if r["action"]["type"] == "probe":
        ax.add_patch(plt.Circle(r["pos"], 2.5, fill=False, ec="#CC79A7", lw=1.5, ls="--"))
    ax.plot(*r["pos"], "o", color=WALLE, ms=11, mec="black", mew=1.5, zorder=5)
    if final:
        ax.plot(*env.world.eve, "*", color="white", ms=20, mec="#D55E00", mew=1.5, zorder=6)
        ax.plot(*env.world.spoof_at, "X", color="#D55E00", ms=11, mec="white", zorder=6)
    ax.set_xlim(-0.5, t.W - 0.5)
    ax.set_ylim(t.H - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_side(axs, recs, i, env):
    ax_h, ax_b, ax_t = axs
    steps = [r["step"] for r in recs]
    for ax in (ax_h, ax_b):
        ax.clear()
        ax.spines[["top", "right"]].set_visible(False)
    H = [r["belief"]["entropy"] if r["belief"] else np.nan for r in recs]
    ax_h.plot(steps[: i + 1], H[: i + 1], color="#D55E00", lw=2)
    ax_h.set_xlim(0, steps[-1]); ax_h.set_ylim(0, max(np.nanmax(H) * 1.05, 1))
    ax_h.set_title("Uncertainty about EVE (entropy, nats)", fontsize=9, loc="left")
    B = [r["battery"] for r in recs]
    for j in range(i):
        ax_b.axvspan(steps[j], steps[j + 1], color=MODE_COL.get(recs[j]["mode"], "#999"), alpha=0.25, lw=0)
    ax_b.plot(steps[: i + 1], B[: i + 1], color="#222", lw=1.8)
    ax_b.set_xlim(0, steps[-1]); ax_b.set_ylim(0, env.cap * 1.05)
    ax_b.set_title("Battery (shading = strategy mode)", fontsize=9, loc="left")
    ax_t.clear()
    ax_t.spines[["top", "right"]].set_visible(False)
    tr = recs[i]["trust"]
    if tr:
        ids = sorted(tr)
        ax_t.barh(ids, [tr[k] for k in ids], color=[plt.get_cmap("RdYlGn")(tr[k]) for k in ids], edgecolor="black", lw=0.5)
    ax_t.set_xlim(0, 1)
    ax_t.set_title("Trust in each clue (red = decoy suspected)", fontsize=9, loc="left")
    ax_t.tick_params(labelsize=7)


def animate(result, path, fps=8, every=1, dpi=70):
    env, recs, grids = result["env"], result["records"], result["grids"]
    frames = [i for i in range(len(recs)) if i % every == 0 or i == len(recs) - 1]
    frames += [len(recs) - 1] * 12
    fig = plt.figure(figsize=(12.5, 6.6), facecolor="white")
    gs = GridSpec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1], left=0.02, right=0.98, top=0.86, bottom=0.05, wspace=0.12, hspace=0.55)
    axm = fig.add_subplot(gs[:, 0])
    axs = [fig.add_subplot(gs[k, 1]) for k in range(3)]
    title = fig.text(0.02, 0.955, "", fontsize=13, weight="bold")
    sub = fig.text(0.02, 0.89, "", fontsize=8.5, family="monospace", va="bottom")
    n = len(recs)

    def upd(k):
        i = frames[k]
        axm.clear()
        final = k >= len(frames) - 12
        _draw_map(axm, env, recs, grids, i, final)
        _draw_side(axs, recs, i, env)
        r = recs[i]
        title.set_text(f"WALL-E hunts EVE   |   {r['mode']}   |   step {r['step']}   battery {r['battery']:.0f}   time left {r['time_left']}")
        title.set_color(MODE_COL.get(r["mode"], "#333"))
        txt = textwrap.fill(r["reasoning"] or "", 118)
        if r["notes"]:
            txt = textwrap.fill(" * ".join(r["notes"]), 118) + "\n" + txt
        sub.set_text("\n".join(txt.split("\n")[-3:]))
        if final:
            est = result["estimate"]
            axm.set_xlabel(f"FINAL: estimate {est['xy']} | truth {env.world.eve} (star) | spoof beacon X | outcome: {env.outcome}", fontsize=9)

    ani = FuncAnimation(fig, upd, frames=len(frames), interval=1000 / fps)
    ani.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def dashboard(result, path):
    env, recs, grids = result["env"], result["records"], result["grids"]
    fig = plt.figure(figsize=(13, 7), facecolor="white")
    gs = GridSpec(3, 2, width_ratios=[1.5, 1], left=0.03, right=0.98, top=0.93, bottom=0.05, wspace=0.12, hspace=0.55)
    axm = fig.add_subplot(gs[:, 0])
    axs = [fig.add_subplot(gs[k, 1]) for k in range(3)]
    i = len(recs) - 1
    _draw_map(axm, env, recs, grids, i, True)
    _draw_side(axs, recs, i, env)
    s = result["summary"]
    fig.suptitle(f"{s['mission']}: {s['outcome']} | est {result['estimate']['xy']} vs truth {env.world.eve} (error {s['error_cells']} cells) | {s['ticks_used']} ticks", fontsize=12, weight="bold")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def ascii_map(result):
    env, recs = result["env"], result["records"]
    t = env.t
    rows = [["#" if t.walls[y, x] else "." for x in range(t.W)] for y in range(t.H)]
    for r in recs:
        x, y = r["pos"]
        if rows[y][x] == ".":
            rows[y][x] = "+"
    for r in recs:
        for c in r["obs"].get("clues", []):
            rows[c["cell"][1]][c["cell"][0]] = "c"
    for c in env.world.chargers:
        rows[c[1]][c[0]] = "C"
    ex, ey = env.world.eve
    rows[ey][ex] = "E"
    x, y = recs[-1]["pos"]
    rows[y][x] = "W"
    return "\n".join("".join(r) for r in rows) + "\nLegend: W=WALL-E  E=EVE  C=charger  c=clue  +=path  #=wall"
