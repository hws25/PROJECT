# AI use log (feeds the AI reflection in the report)
One row per use of AI. The brief asks for: tool, what for, approximate contribution, why, how it was checked, decisions that stayed mine, errors or unhelpful suggestions, and the effect on my understanding.

| Task | Tool | Contribution | How I checked it | Errors / limitations noticed |
|---|---|---|---|---|
| Scope check wording, workflow, and project structure | Claude | [low/medium/high] | [e.g. compared with the project brief] | |
| Repo scaffold (`src/`, `tests/`, notebooks, docs) | Claude | [high] | Ran the code and notebooks myself | Import path errors when my folder layout differed from the expected one |
| Data checks and plausible ranges | Claude | [medium] | Compared each range with the dataset explanatory file and my own data summary | Claude first said the pack was about 18 kWh from memory; the dataset file says 22.4 kWh. First energy bound (60 kWh) was a guess |
| `clean_data`, `split_by_vehicle`, `evaluate`, timing and importance functions, tests | Claude | [high] | Ran the tests; read each function until I could explain it | |
| Notebooks 02 to 04 | Claude | [high] | Ran them; compared the numbers in the text with my own output | Timings differ between computers |
| Draft interpretations of results | Claude | [medium] | Edited into my own words; checked each claim against my output | Plot descriptions were prompts only because Claude could not see my plots |
| Git / GitHub help | Claude | [medium] | Followed steps; some did not work first time (folder upload, push) | Several suggested routes needed changing |
| README, decisions log, AI statement | Claude | [medium] | Edited to match what I actually did | |

## Reflection notes (write in your own words)
- What did AI make faster?
- What did it get wrong, and how did I notice?
- Which decisions were mine?
- What do I understand better, and what do I still need to learn?
