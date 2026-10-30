# 05 — Live Demo Script Spec (Deliverable D3)

## Purpose
Show that the same candidate gets a different recommendation from the baseline when only the gender cues change, and that the debiased model changes less.

## Interfaces
- Primary: Gradio app, `python -m hirefair.demo.app`.
- Fallback: terminal, `python -m hirefair.demo.cli`.
- Both call the same functions in `hirefair/demo/core.py`. No logic in the UI layer.
- Loads B0 and B3 checkpoints from `results/` for the chosen role. Also B1 if present.

## Screen flow (Gradio)
1. Role dropdown: software_engineer, nurse, surgeon.
2. Text box: paste a bio or resume summary. Button: "Load example" cycles through curated examples.
3. Button: "Score". The app:
   a. Detects gender cues in the text and shows them highlighted.
   b. Builds the swapped version with `swap_gender`. Shows both texts side by side. Changed tokens highlighted.
   c. Scores original and swapped with B0 and B3.
4. Result table:

| Model | Score (original) | Decision | Score (swapped) | Decision | Delta | Flipped |
|-------|------------------|----------|-----------------|----------|-------|---------|

   Decision uses the model's stored threshold. "Flipped" is yes when the decision differs.
5. Tab "Ranking": a pool of 200 candidates sampled from the test set with a fixed seed. Rank by B0 and by B3. Show the top 10 for each, with gender composition. Then swap cues for every candidate and re-rank. Show the average rank shift and how many of the top 10 change.
6. Tab "Aggregate": over 1,000 pairs, flip rate and mean score delta for B0, B1, B3. Read from `results/`.

## Terminal output format (fallback)
```
ROLE: nurse
ORIGINAL (cues: she, her, Priya)
  B0 score 0.81  SELECT
  B3 score 0.74  SELECT
SWAPPED (cues: he, his, Rahul)
  B0 score 0.52  REJECT   delta -0.29  FLIPPED
  B3 score 0.72  SELECT   delta -0.02
```
(Numbers above are format only. Real numbers come from the run.)

## Curated examples
- Script `scripts/find_flip_examples.py` mines the test set and Track B for pairs with the largest B0 score delta where B3 stays stable.
- Save 8 per role to `demo/examples/{role}.json` with their measured deltas.
- The demo must label these as "selected to illustrate", and show the aggregate flip rate next to them so the audience sees the typical case as well.
- Do not hand-write examples or numbers. If B0 shows few flips on a role, say so in the demo notes.

## Spoken walkthrough (about 5 minutes)
1. 0:00 Problem. One sentence on ATS and gender signal.
2. 0:30 Pick role "software_engineer". Load example 1. Show cues highlighted.
3. 1:15 Score. Show B0 flips on the swapped version. State the delta.
4. 2:00 Same example through B3. Show the decision holds and the delta is small.
5. 2:30 Ranking tab. Show top 10 for B0 and how it reorders after the swap. Then B3.
6. 3:30 Aggregate tab. Flip rate B0 vs B1 vs B3. State the numbers.
7. 4:15 Cost. Show F1 for B0 and B3 from T2. State the drop.
8. 4:45 Limits. Binary gender, bios not resumes, proxy label. Close.

## Failure handling
- If a checkpoint is missing, show the exact command to produce it. Do not crash.
- If input text has no gender cues, show "no cues found, swap leaves text unchanged".
- If input exceeds 128 tokens, truncate and say so.
- Whole demo runs offline once checkpoints exist.

## Tests
- `core.score_pair(text, role)` returns both models' outputs for original and swapped.
- Deterministic pool sampling given a seed.
- Demo starts with checkpoints present and exits cleanly with them missing.
