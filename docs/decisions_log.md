# Decisions log
One entry per decision: **what**, **why**, **evidence** (figure/table), **alternatives rejected**. This becomes the justification marks in the report.

| Date | Decision | Why | Evidence |
|---|---|---|---|
| 16/10/2026 | Drop lon/lat from models | Impossible values (e.g. lat −645); little direct interpretability | `quality_report` in notebook 01 |
| 16/10/2026 | Use `distance / duration` as average speed | Raw `Speed` column reaches ~930 and does not match distance/duration | `figures/speed_check.png` |
| 16/10/2026 | Split by `vehicle_id` (grouped) | Prevents trips from the same vehicle appearing in both train and test | |
