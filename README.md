# WALL-E and EVE's Signal: a trust-aware Bayesian search agent

An autonomous agent that explores a simulated junk-world, fuses noisy radio pings and clues
(some of them **decoys**, plus a **spoof beacon**), manages a limited battery/time budget and
decides where EVE is and what WALL-E should do next.

## What makes it different
1. **Trust-weighted evidence fusion** (`belief.py`). Every ping and clue carries a trust score. Trust is
   re-estimated by leave-one-out EM: a piece of evidence is believed only as far as it agrees with all
   the *other* evidence. Decoy clues and spoofed pings get exposed (trust ~0.02) without ever being told which are fake.
2. **Information-per-cost planner** (`agent.py`). Each candidate (cell, ping|probe) is scored by Monte-Carlo
   expected entropy reduction / battery+time cost.
3. **Lifeline constraint.** No action is allowed unless the battery still covers the walk to the nearest charger + reserve.
   WALL-E can never strand himself.
4. **Named strategy modes** SURVEY / TRIANGULATE / VERIFY / FINAL_APPROACH / RETREAT / RECHARGE; every switch is logged with a reason.
5. **Status inference.** Heartbeat pulses (steady/erratic/faint) update a posterior over EVE = active/dormant/damaged,
   which drives the recommended next step.
6. **Bonus items done:** visualisation (GIF + dashboard + ASCII), decoy/spoofed signals, noisy sensors,
   generalisation benchmark over random missions, ablation (no trust), calibration check, 3 classical baselines.

## Setup and run
```bash
pip install -r requirements.txt
python run_mission.py configs/mission_alpha.yaml --out outputs/alpha     # easy
python run_mission.py configs/mission_bravo.yaml --out outputs/bravo     # hard: decoys + spoof (makes replay.gif)
python benchmark.py --n 30 --out outputs/benchmark                       # generalisation benchmark
python benchmark.py --n 30 --hard --out outputs/benchmark_hard           # tight budget
```
Outputs per mission: `trace.jsonl` (every action, observation, belief summary, trust and reasoning),
`report.md/json` (final mission report), `dashboard.png`, `replay.gif`, `final_map.txt`.

## Config format (YAML/JSON)
```yaml
mission: {name: x, seed: 51}            # seed drives map, clue placement, EVE
map: {width: 24, height: 24, wall_density: 0.18, chargers: 3}   # or ascii: ["#..", ...]
start: [2, 2]
initial_signal: {bearing_deg: 40, strength: 0.3, timestamp: 0}  # omit to auto-generate from truth
resources: {battery: 90, time: 260, reserve: 4, costs: {move: 1, ping: 3, probe: 2, recharge_rate: 8}}
clues: {honest: 4, decoy: 5}
sensor: {decay: 6.5, wall_att: 0.6, floor: 0.04, sigma_log: 0.24, sigma_bearing: 14, outlier: 0.1}
spoof: {enabled: true, radius: 9, prob: 0.8}
```
The agent sees only the map, chargers, budgets, sensor datasheet and first signal. Clue locations are learned
only by walking over them or probing (radius 2.5).

## Environment design
Grid world of junk heaps (walls). Actions: `move`, `ping` (strength + bearing + heartbeat), `probe` (reveals clues /
makes visual contact), `recharge` (at chargers), `commit`. Pings: log-strength falls with distance and walls, bearing noise grows with distance, 10% glitch outliers,
and a hidden spoof beacon that answers pings with a fake source. Clues: *tracks* (bearing), *scorch* (range ring), *log* (last-seen coordinate); decoys point at
the spoof beacon, and a noisy "authenticity tell" is only a weak hint.

## Files
`wallee/terrain.py` geometry/BFS, `sensors.py` physics + likelihoods, `belief.py` trust EM, `agent.py` planner,
`env.py` world + simulator, `baselines.py`, `runner.py` loop + report, `viz.py` GIF/PNG/ASCII.
