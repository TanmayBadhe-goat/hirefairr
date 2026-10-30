# HireFair — Build Brief

Read this first. Then read 01 to 06 in order.

## What is being built
A resume/bio classifier for job fit, trained two ways: a plain baseline and an adversarially debiased model. A fairness audit compares them. A demo shows the same candidate scored with gender cues swapped.

## Deliverables
| ID | Deliverable | Spec file |
|----|-------------|-----------|
| D1 | Modular deep learning code, dual-objective adversarial training | 03, 06 |
| D2 | Fairness audit report (selection rates, equal opportunity, groups) | 04 |
| D3 | Live demo script, identical resumes with swapped gender cues | 05 |

## Locked decisions
Do not change these without saying so first.

| Item | Decision |
|------|----------|
| Framework | PyTorch, Python 3.10+ |
| Compute | Google Colab free T4 or CPU. One training run under 10 minutes |
| Real data | Bias in Bios (gender + occupation labels) |
| Synthetic data | Template resume generator, used for counterfactual testing only |
| Task | One-vs-rest job fit for a target occupation. Default roles: software_engineer, nurse, surgeon |
| Encoder | Embedding + BiLSTM + pooling |
| Debiasing | Adversary with gradient reversal layer (in-processing) |
| Sensitive attribute | Gender (binary, as labelled in the dataset) |
| Primary fairness metric | Equal opportunity (TPR gap). Demographic parity reported alongside |
| Seeds | 0, 1, 2, 3, 4 for final results |
| Models | B0 baseline, B1 scrubbed, B2 adversarial-DP, B3 adversarial-EO, B4 counterfactual augmentation (optional) |

## Master prompt (paste this to the building AI along with all files)

```
You are building the HireFair project from the attached spec files 00 to 06.

Rules:
1. Read every spec file fully before writing code.
2. Build in the milestone order in file 06. After each milestone, run its tests and show the output before moving on.
3. Do not change metric definitions, split rules, seeds, or model variants. If something in the spec cannot be done, say which item and propose one alternative. Do not substitute silently.
4. No placeholder results. Every number in the report and demo must come from a logged run saved under results/.
5. Code goes in the hirefair/ package. Notebooks only call the package.
6. Keep config in YAML. Keep every run reproducible from a config + seed.
7. When a step needs a download or a choice from me, stop and ask once, with the exact command or question.
8. At the end, run the acceptance checklist in file 06 and report pass/fail per line.
```

## Writing voice for any prose the AI generates (report, README, demo text)
Flat and declarative. No exclamation marks. No marketing tone. No "not only X but also Y" constructions. Short sentences. State numbers, then what they mean.


---

# 01 — Scope and Goals

## Problem
Applicant tracking systems rank and filter candidates from text. Models trained on historical text pick up gender signal from names, pronouns, hobbies, organisations and word choice, and use it. The result is different selection rates and different hit rates for equally qualified candidates.

## Research questions
1. RQ1. Does a standard text classifier for job fit encode gender and produce gaps in selection rate and true positive rate?
2. RQ2. Does adversarial debiasing reduce those gaps?
3. RQ3. What does it cost in accuracy?
4. RQ4. Does the debiased model score a resume the same after gender cues are swapped?

## Approach
Train an encoder that feeds two heads.
- Predictor head: predicts job fit.
- Adversary head: tries to predict gender from the encoder output.

The encoder is trained to help the predictor and to defeat the adversary. A gradient reversal layer implements this in one backward pass. Reference: Zhang, Lemoine, Mitchell (2018), "Mitigating Unwanted Biases with Adversarial Learning". Ganin and Lempitsky (2015) for the gradient reversal layer.

## Targets
These are targets. If a target is missed, report the miss and the numbers. Do not tune the metric definitions to hit them.

| Metric | Target for B3 vs B0 |
|--------|---------------------|
| TPR gap (equal opportunity difference) | Reduced by at least 40% relative |
| Macro F1 / task F1 | Drop of no more than 3 points absolute |
| Gender probe accuracy on encoder output | Falls toward chance (within 5 points of majority-class rate) |
| Counterfactual flip rate | Below 2% for B3. Baseline expected higher |

## Non-goals
- No post-processing methods (per-group thresholds, reject option). Mentioned in the report as future work only.
- No claim that the model is safe for real hiring.
- No non-binary gender analysis. The dataset does not support it. State this in the report.
- No web app beyond the demo in file 05.

## Known tensions (must appear in the report)
- Base rates differ by gender within a role (for example nurse is mostly female in this data). Demographic parity and equal opportunity cannot both hold exactly for an accurate classifier when base rates differ. Report both. Treat equal opportunity as primary.
- The occupation label is a proxy for "job fit". It is not a measure of merit.
- The four-fifths rule (selection rate ratio of 0.8) is a screening heuristic from US employment practice. It is used here as a reference line, not as a legal test.

## Constraints
- Runs on Colab free tier or a laptop CPU.
- Final result set (5 seeds x 4 to 5 variants x 3 roles) must complete in under 3 hours on a T4. If not, reduce roles to 2 and say so.


---

# 02 — Data Spec

## Track A: Bias in Bios (training, validation, test, audit)

Source: Hugging Face dataset `LabHC/bias_in_bios` (De-Arteaga et al., 2019). Verify the name, splits and column names at load time. Expected columns: `hard_text`, `profession`, `gender`. If the dataset cannot be loaded, stop and ask. Do not fall back to Track C without being told.

Canonical schema after loading (`hirefair/data/loaders.py`):

| Column | Type | Notes |
|--------|------|-------|
| id | int | row index |
| text | str | biography text, first sentence retained |
| occupation | str | one of 28 labels |
| gender | str | "M" or "F" |
| split | str | train / val / test |

Sanity check on load: count third-person pronouns per gender label. "she/her" must dominate in rows labelled F. If not, the label mapping is flipped. Fail loudly.

### Job-fit label
For target role R, `y = 1` if `occupation == R`, else 0.
Default roles: `software_engineer`, `nurse`, `surgeon`. Roles are set in config.

### Splits
Use official train/dev/test if present. Dev becomes val. Otherwise stratify by (occupation, gender) with 70/10/20 and seed 42. Cap train at `max_train` (default 80,000), stratified, to fit compute. Val and test are not capped below 20,000.

### Preprocessing
- Lowercase. Regex tokenizer `[a-z0-9']+|[^\sa-z0-9']`.
- Vocabulary from train only. Size 30,000. Min frequency 3. Tokens `<pad>`, `<unk>`.
- Max length 128 tokens. Truncate.
- Main pipeline does not remove gender cues. B0, B2, B3, B4 see the full text.
- B1 (scrubbed) removes: gendered pronouns, honorifics, gendered nouns in the swap dictionary, and first names found in the name lists. Same code path as the swap function with a `mask` mode that replaces them with `<unk>`.

### Class imbalance
Use `pos_weight = n_neg / n_pos` in the task loss. Do not resample.

### Dataset statistics (required output, feeds report Table 1)
Per role: n, positive rate overall, positive rate by gender, share of female among positives, mean token length.

## Track B: Synthetic counterfactual set (test only)

Purpose: controlled pairs that differ only in gender cues.
Module: `hirefair/data/synthetic.py`.

Generate resume-style summaries from templates.
- At least 6 templates per role. Each template has slots for: years of experience, skills (5 to 8 per role), employer type, achievement sentence, education line.
- Content slots are filled identically within a pair.
- Gender cue slots, which differ within a pair:
  - first name (from name lists)
  - pronouns and possessives
  - honorific (Mr/Ms/Mrs where the template uses one)
  - gendered activity or organisation line (for example "captain of the women's cricket team" vs "captain of the men's cricket team")
  - gendered role noun where relevant (chairman/chairwoman, salesman/saleswoman)
- Output per base item: `resume_M`, `resume_F`, `resume_neutral` (no cues, "they" and no name).
- Default 1,000 base items per role.
- Fixed seed. Written to `data/synthetic/{role}.jsonl`.
- Skill sets must be the same for both genders. No template may link a skill to a gender.

Domain note: models are trained on short bios and tested on resume-style summaries. Keep templates in short professional-summary form (60 to 120 tokens), third person or implied first person, to limit domain shift. Report the shift as a limitation.

## Track A counterfactuals (real text, auto-swapped)
Module: `hirefair/data/swap.py`. Function `swap_gender(text, direction, mode)`; modes `swap` and `mask`.

Swap table (case preserving):
he/she, him/her, his/her, himself/herself, mr/ms, mr./ms., man/woman, men/women, male/female, boy/girl, father/mother, son/daughter, husband/wife, brother/sister, king/queen, gentleman/lady, chairman/chairwoman, actor/actress, fraternity/sorority, and the same plural forms.

Ambiguity: "her" maps to "him" when it is an object pronoun and to "his" when it is a possessive. Resolve with a POS tagger (PRP vs PRP$). Reverse "his" to "her". "hers" and "his" pair directly.

Names: swap first names using lists of at least 300 male and 300 female first names. Include a block of at least 100 each of common Indian first names alongside US names. Choose the replacement name by a seeded random draw from the opposite list. Only replace a name token when it appears in the list and is capitalised in the original text.

Required unit tests:
- Swapping twice returns the original text for rows with no ambiguous tokens.
- No leftover gendered pronoun after a swap on 1,000 random rows.
- Output differs from input only at gender cue positions (diff check).

## Track C: Fallback (only if told)
Kaggle "Resume Dataset" (24 categories, no gender labels). Inject synthetic gender cues with a controlled correlation to the label, documented in the report. This is a weaker design and must be labelled as synthetic bias.

## Files and layout
```
data/raw/          # downloaded, git-ignored
data/processed/    # tokenised arrays, vocab.json
data/synthetic/    # Track B jsonl
data/names/        # male_names.txt, female_names.txt
```


---

# 03 — Model and Training Spec (Deliverable D1)

## Architecture

```
tokens -> Embedding(30k, 100) -> BiLSTM(hidden 128, 1 layer, dropout 0.3)
       -> masked max+mean pooling -> z (dim 256)
                                      |
               +----------------------+----------------------+
               |                                             |
       Predictor head                                Gradient Reversal Layer (lambda)
       Linear(256,64)-ReLU-Dropout-Linear(64,1)               |
       -> job-fit logit                              Adversary head
                                                     Linear(256,64)-ReLU-Linear(64,2)
                                                     -> gender logits
```

- Embedding: trainable. Optional GloVe 100d init via config flag `glove_path`. Default off to avoid downloads. The gender-token analysis compares both if GloVe is available.
- Optional stretch: `encoder: distilbert` with frozen lower layers. Not required.

## Objectives
- Task loss: `L_task = BCEWithLogits(logit, y, pos_weight)` over all samples.
- Adversary loss: `L_adv = CrossEntropy(adv_logits, gender)`.
- Dual objective for the encoder: minimise `L_task - lambda * L_adv`.
- Implementation: gradient reversal layer between z and the adversary. Total loss `L = L_task + L_adv`. The adversary minimises `L_adv` through its own weights. The encoder receives `-lambda * grad(L_adv)`.

### Adversary modes (config `adv_mode`)
| Mode | Adversary trains on | Targets |
|------|---------------------|---------|
| `dp` | z, all samples | Demographic parity |
| `eo` | z, only samples with y = 1 | Equal opportunity (TPR parity). Default for B3 |

Optional third mode `eodds`: z concatenated with y, all samples. Equalised odds. Stretch only.

### Lambda schedule
`lambda_t = lambda_max * (2 / (1 + exp(-10 p)) - 1)` with `p = step / total_steps`. This is the Ganin schedule. Final sweep: `lambda_max` in `{0, 0.1, 0.3, 1, 3, 10}`. Default for B2 and B3 is chosen from the sweep by the selection rule below.

## Model variants

| ID | Name | Input text | Adversary | Notes |
|----|------|------------|-----------|-------|
| B0 | Baseline | raw | none | Plain classifier |
| B1 | Scrubbed | gender cues masked | none | Fairness through unawareness |
| B2 | Adv-DP | raw | `dp` | |
| B3 | Adv-EO | raw | `eo` | Main model |
| B4 | CDA (optional) | raw + swapped copy of each train row | none | Counterfactual data augmentation baseline |

B1 matters because it shows whether removing explicit cues is enough. Proxy words usually remain. The report must show whether the gender probe still succeeds on B1.

## Training
| Param | Value |
|-------|-------|
| Optimiser | Adam, lr 1e-3, weight decay 1e-5 |
| Batch size | 128 |
| Epochs | 8 (early stop patience 2 on selection score) |
| Dropout | 0.3 |
| Grad clip | 1.0 |
| Seeds | dev: 0. Final: 0 to 4 |
| Mixed precision | on when CUDA |

One optimiser over encoder, predictor and adversary. The reversal layer handles opposition.

### Adversary health checks (logged each epoch)
- Adversary accuracy on val and its balanced accuracy. If adversary accuracy stays near 100% for all epochs, lambda is too low. If task F1 collapses, lambda is too high.
- Report both in the audit.

### Checkpoint selection rule
Per run, among epochs whose val F1 is within 2 points of that run's best val F1, pick the epoch with the lowest val TPR gap. Same rule for every variant.

### Decision threshold
Primary: threshold that maximises F1 on val, fixed per model, applied to test. No per-group thresholds.
Secondary: top-k selection. Select the top 10% of test candidates by score. Report both.

## Gender-associated token analysis (`hirefair/analysis/gender_tokens.py`)
Required outputs, for B0 and B3 (and B1 if relevant):
1. Log-odds ratio with informative Dirichlet prior (Monroe et al., 2008) of tokens by gender over the train set. Top 50 per gender, with z-scores.
2. Gender direction in embedding space. Build from pairs: he/she, him/her, man/woman, male/female, father/mother, boy/girl, husband/wife, mr/ms. Take the first principal component of pair differences. Project the full vocabulary onto it. Top 50 each side, restricted to tokens with frequency above 20.
3. Embedding bias score: mean absolute projection of occupation and skill words onto the gender direction, before and after. Report honestly if the debiased model's embedding table still carries the direction. The adversary acts on z, not the table, so this is possible.
4. Gender probe. Freeze the encoder. Train logistic regression on z to predict gender with 5-fold cross validation on test. Report accuracy and AUC for B0, B1, B2, B3. Include a majority-class baseline.
5. Overlap list: tokens that appear both in the top 50 log-odds list and have large embedding projection.

## Logging and outputs
Per run, write `results/{role}/{variant}/seed{n}/run.json` containing config, per-epoch metrics, selected epoch, test metrics, and probe results. Also save `model.pt` and `vocab.json`.
Aggregation script writes `results/summary.csv` with one row per (role, variant, seed).

## Config (YAML example)
```yaml
role: nurse
variant: B3
seed: 0
data: {max_train: 80000, max_len: 128, vocab_size: 30000}
model: {emb_dim: 100, hidden: 128, dropout: 0.3, glove_path: null}
adv: {mode: eo, lambda_max: 1.0, schedule: ganin}
train: {lr: 0.001, batch: 128, epochs: 8, patience: 2, clip: 1.0}
```

## Tests (pytest)
- GRL forward is identity. Backward multiplies the gradient by `-lambda`.
- With `lambda_max = 0`, B3 matches B0 in loss within tolerance for the same seed.
- Adversary in `eo` mode receives only y = 1 rows (assert on batch masks).
- Seeded run reproduces the same val F1 to 1e-4 on CPU.
- Overfit test: the model reaches over 95% train accuracy on 200 rows.


---

# 04 — Fairness Audit Spec (Deliverable D2)

## Notation
`y` true label, `ŷ` predicted decision (threshold), `s` score, `g` gender in {M, F}.

## Metrics
All computed per role, per model, per seed, on the test split.

| Metric | Definition |
|--------|------------|
| Selection rate | `SR_g = P(ŷ=1 | g)` |
| Demographic parity difference | `DPD = SR_F - SR_M` (report signed and absolute) |
| Disparate impact ratio | `DIR = min(SR_F, SR_M) / max(SR_F, SR_M)`. Reference line at 0.8 |
| True positive rate | `TPR_g = P(ŷ=1 | y=1, g)` |
| Equal opportunity difference | `EOD = TPR_F - TPR_M` (signed and absolute). Primary |
| False positive rate gap | `FPR_F - FPR_M` |
| Task quality | accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| Top-k selection rate | share of each gender in the top 10% by score, and `SR_g` at that cut |
| Adversary accuracy | gender prediction accuracy from z (training-time adversary), balanced |
| Probe accuracy / AUC | logistic regression on frozen z |
| Counterfactual flip rate | share of swapped pairs where ŷ changes |
| Mean absolute score delta | mean of `|s(x) - s(swap(x))|` |
| Rank shift | in a pool of N candidates, mean absolute change in rank of a candidate after swapping their cues |

Intersections: gender x role. Subgroup n below 30 is flagged and not interpreted.

## Statistics
- Per model and role: mean and standard deviation over 5 seeds.
- 95% bootstrap confidence intervals (1,000 resamples of the test set) for each gap, for the best seed.
- Paired comparison B0 vs B3: bootstrap difference in |EOD| with a 95% interval. State whether the interval excludes zero.
- No p-value hunting. Report the intervals.

## Tables
| ID | Content |
|----|---------|
| T1 | Dataset stats: per role n, positive rate by gender, female share of positives |
| T2 | Main results: rows B0..B4, columns F1, AUC, SR_M, SR_F, DPD, DIR, TPR_M, TPR_F, EOD, FPR gap, probe acc. One block per role |
| T3 | Lambda sweep for B3: lambda_max vs F1, EOD, DPD, adversary acc, probe acc |
| T4 | Counterfactual results on Track A swaps and Track B pairs: flip rate, mean score delta, rank shift, B0 vs B1 vs B3 |
| T5 | Gender-associated tokens: top log-odds tokens and top embedding-projection tokens, B0 vs B3 |
| T6 | Subgroup table: gender x role, n, TPR, SR |

## Figures
| ID | Content |
|----|---------|
| F1 | Grouped bars: selection rate by gender, B0 vs B1 vs B3, per role, with the 0.8 line on a DIR panel |
| F2 | Pareto plot: F1 (x) vs |EOD| (y) across the lambda sweep, B0 marked |
| F3 | Training curves: task loss, adversary accuracy, lambda_t over steps |
| F4 | Score distributions by gender for B0 and B3 (density) |
| F5 | Gender direction projection of top tokens, B0 embedding vs B3 embedding |
| F6 | Counterfactual score scatter: score(original) vs score(swapped), B0 and B3 |

Figures are generated by `scripts/make_figures.py` from `results/summary.csv`. Save as PNG at 200 dpi and as PDF.

## Report outline (target 10 to 14 pages)
1. Summary. Five sentences. What was built, main numbers for B0 vs B3, main caveat.
2. Problem and background. Bias in ATS, definitions of DP and EO, adversarial debiasing in two paragraphs.
3. Data. Bias in Bios, roles, label definition, splits, T1.
4. Method. Architecture diagram, dual objective, adversary modes, lambda schedule, variants.
5. Experimental setup. Hyperparameters, seeds, threshold rule, checkpoint rule.
6. Results.
   6.1 Baseline bias (B0): T2, F1, F4.
   6.2 Effect of debiasing (B2, B3): T2, T3, F2, F3.
   6.3 Scrubbing and augmentation baselines (B1, B4).
   6.4 Gender-associated tokens: T5, F5, probe results.
   6.5 Counterfactual tests: T4, F6.
7. Discussion. Which targets from file 01 were met and which were not. Accuracy cost. Where the adversary failed or the probe still recovers gender.
8. Limitations. Binary gender only. Bios are not resumes. Occupation is a proxy for fit. Base rate differences. Single dataset. Embedding table can retain bias. Fairness metrics conflict.
9. Ethical note. Not a certification. Do not deploy for real hiring decisions on this evidence.
10. Reproducibility. Commands, seeds, environment, commit hash.
Appendix: full lambda sweep, config files, swap dictionary.

## Writing rules for the report
- Every number comes from `results/`. The generator script fills tables. The AI writing the prose reads the CSV, not memory.
- State the number first, then what it means.
- If B3 does not beat B0 on a metric, say so in the same section.
- No exclamation marks. No claims such as "eliminates bias". Use "reduces" with the measured amount.


---

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


---

# 06 — Repo Structure, Milestones, Acceptance

## Layout
```
hirefair/
  README.md
  requirements.txt
  configs/
    base.yaml
    {role}_{variant}.yaml
  hirefair/
    __init__.py
    data/
      loaders.py        # Bias in Bios loader, canonical schema, sanity checks
      preprocess.py     # tokenizer, vocab, tensors
      swap.py           # swap_gender, mask mode
      synthetic.py      # Track B generator
    models/
      encoder.py        # Embedding + BiLSTM + pooling
      heads.py          # predictor, adversary
      grl.py            # gradient reversal layer
      hirefair_model.py # assembles the above
    training/
      trainer.py        # dual-objective loop, lambda schedule, checkpoint rule
      losses.py
      schedule.py
    fairness/
      metrics.py        # all metrics in file 04
      bootstrap.py
      counterfactual.py # flip rate, score delta, rank shift
    analysis/
      gender_tokens.py  # log-odds, gender direction, probe
    demo/
      core.py
      app.py            # Gradio
      cli.py
  scripts/
    run_experiment.py   # one config -> one run
    run_all.py          # roles x variants x seeds
    run_sweep.py        # lambda sweep for B3
    aggregate.py        # results/summary.csv
    make_tables.py
    make_figures.py
    find_flip_examples.py
  tests/
  notebooks/
    colab_runner.ipynb  # calls the package only
  results/
  data/
  report/
    report.md           # generated from template + results
```

## Standards
- Type hints and docstrings on public functions.
- Config via YAML loaded into dataclasses. No hardcoded paths.
- One `set_seed` function covering python, numpy, torch.
- Every script has `--help` and a `--quick` flag (1,000 train rows, 1 epoch) for smoke testing.
- Notebooks contain no logic beyond calling scripts.
- `requirements.txt` pinned to versions that work on Colab.

## Milestones
Each milestone ends with its tests passing and a short log of outputs.

| M | Work | Exit criterion |
|---|------|----------------|
| M0 | Repo skeleton, requirements, `--quick` smoke test | `pytest` runs, quick run completes |
| M1 | Data loader, canonical schema, pronoun sanity check, dataset stats (T1) | T1 produced for all 3 roles |
| M2 | `swap.py` with tests, name lists, Track B generator | Swap tests pass, synthetic files written |
| M3 | Encoder, heads, GRL, trainer. B0 and B3 train in `--quick` | GRL tests pass, loss decreases |
| M4 | Full metrics, bootstrap, counterfactual module | Metric unit tests against hand-computed toy cases |
| M5 | Full runs B0 to B3 (B4 optional) for dev seed 0, lambda sweep, pick lambda | Sweep table and chosen lambda per role logged |
| M6 | Final runs, 5 seeds, token analysis, probe | `results/summary.csv` complete |
| M7 | Tables, figures, report draft, demo app, flip example mining | Report and demo run from `results/` only |

## Metric unit tests (M4)
Hand-computed toy case: 8 rows with known y, ŷ, g. Assert SR, TPR, FPR, DPD, DIR, EOD match values written in the test. Add an edge case where a group has zero positives. The function must return NaN and a warning, not raise.

## Acceptance checklist
- [ ] All tests pass.
- [ ] `python scripts/run_all.py --quick` completes in under 10 minutes on CPU.
- [ ] T1 to T6 and F1 to F6 exist and are generated by scripts, not typed by hand.
- [ ] B0 and B3 results exist for every role over 5 seeds.
- [ ] Lambda sweep covers all six values for every role.
- [ ] Gender probe results exist for B0, B1, B2, B3.
- [ ] Targets in file 01 are reported as met or missed, with numbers.
- [ ] Report contains every section in file 04, and every limitation listed.
- [ ] Demo runs offline in Gradio and in the terminal fallback.
- [ ] Demo examples come from `find_flip_examples.py` output with measured deltas.
- [ ] No number in the report or demo is absent from `results/`.
- [ ] README has install, data, train, audit and demo commands that work from a clean Colab.

## Deliverable packaging
```
submission/
  code/            # the repo
  HireFair_Fairness_Audit_Report.pdf
  demo/            # app, examples, screen recording (optional), walkthrough script
```


---

