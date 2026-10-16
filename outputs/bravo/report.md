# Mission report: mission_bravo

**Agent:** Hunter (trust-aware)  |  **Outcome:** `contact`

## Verdict

- **Inferred EVE location:** (22, 14)
- **Confidence (within 2 cells):** 100%
- **Visual contact confirmed:** True
- **Inferred status:** damaged  (posterior: active 3%, dormant 3%, damaged 95%)
- **90% credible region:** 4 cells

## Recommended next step

EVE is DAMAGED at (22,14). Do NOT tow. Stay adjacent, call for repair and shield her from the dust.

## Evidence

- 7 pings, 21 probes, 4 clues found, 229 ticks used.
- Clue trust (1 = believed, 0 = judged a decoy): c4=0.98, c5=0.02, c6=0.03, c7=0.02
- Flagged as decoys: ['c7', 'c6', 'c5']

## Strategy changes

1. t-left 260: **SURVEY -> TRIANGULATE** because ping offers the best information per cost.
2. t-left 235: **TRIANGULATE -> FINAL_APPROACH** because 60% of belief mass inside probe disc.
3. t-left 223: **FINAL_APPROACH -> VERIFY** because 50% of belief mass inside probe disc: check hypothesis.
4. t-left 220: **VERIFY -> FINAL_APPROACH** because 51% of belief mass inside probe disc.
5. t-left 215: **FINAL_APPROACH -> TRIANGULATE** because ping offers the best information per cost.
6. t-left 213: **TRIANGULATE -> FINAL_APPROACH** because 82% of belief mass inside probe disc.
7. t-left 204: **FINAL_APPROACH -> VERIFY** because 37% of belief mass inside probe disc: check hypothesis.
8. t-left 197: **VERIFY -> RETREAT** because no action keeps the lifeline.
9. t-left 184: **RETREAT -> RECHARGE** because low battery at charger.
10. t-left 173: **RECHARGE -> FINAL_APPROACH** because 95% of belief mass inside probe disc.
11. t-left 144: **FINAL_APPROACH -> TRIANGULATE** because ping offers the best information per cost.
12. t-left 142: **TRIANGULATE -> FINAL_APPROACH** because 81% of belief mass inside probe disc.
13. t-left 136: **FINAL_APPROACH -> TRIANGULATE** because ping offers the best information per cost.
14. t-left 131: **TRIANGULATE -> FINAL_APPROACH** because 56% of belief mass inside probe disc.
15. t-left 130: **FINAL_APPROACH -> VERIFY** because 43% of belief mass inside probe disc: check hypothesis.
16. t-left 126: **VERIFY -> FINAL_APPROACH** because 83% of belief mass inside probe disc.
17. t-left 120: **FINAL_APPROACH -> TRIANGULATE** because ping offers the best information per cost.
18. t-left 112: **TRIANGULATE -> RETREAT** because no action keeps the lifeline.
19. t-left 97: **RETREAT -> RECHARGE** because low battery at charger.
20. t-left 86: **RECHARGE -> SURVEY** because hunting for new clues / ruling out regions.
21. t-left 85: **SURVEY -> FINAL_APPROACH** because 99% of belief mass inside probe disc.
22. t-left 58: **FINAL_APPROACH -> VERIFY** because 29% of belief mass inside probe disc: check hypothesis.
23. t-left 56: **VERIFY -> FINAL_APPROACH** because 99% of belief mass inside probe disc.

## Ground truth (for evaluation only; the agent never saw this)

- True EVE: (22, 14) (damaged); localisation error 0.0 cells
- Spoof beacon was at (16, 5); true decoy clues: ['c0', 'c1', 'c5', 'c6', 'c7']
- Decoys caught: ['c7', 'c6', 'c5']; false alarms: none
