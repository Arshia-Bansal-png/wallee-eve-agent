# Mission report: mission_alpha

**Agent:** Hunter (trust-aware)  |  **Outcome:** `contact`

## Verdict

- **Inferred EVE location:** (12, 21)
- **Confidence (within 2 cells):** 100%
- **Visual contact confirmed:** True
- **Inferred status:** active  (posterior: active 95%, dormant 3%, damaged 3%)
- **90% credible region:** 2 cells

## Recommended next step

EVE is ACTIVE at (12,21). Move adjacent, send the docking handshake and escort her to charger (0, 23).

## Evidence

- 3 pings, 3 probes, 2 clues found, 57 ticks used.
- Clue trust (1 = believed, 0 = judged a decoy): c0=0.98, c7=0.98
- Flagged as decoys: none

## Strategy changes

1. t-left 260: **SURVEY -> TRIANGULATE** because ping offers the best information per cost.
2. t-left 216: **TRIANGULATE -> SURVEY** because hunting for new clues / ruling out regions.
3. t-left 215: **SURVEY -> FINAL_APPROACH** because 66% of belief mass inside probe disc.

## Ground truth (for evaluation only; the agent never saw this)

- True EVE: (12, 21) (active); localisation error 0.0 cells
- Spoof beacon was at (18, 7); true decoy clues: ['c1', 'c4', 'c5']
- Decoys caught: none; false alarms: none
