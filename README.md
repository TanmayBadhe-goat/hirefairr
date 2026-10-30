# HireFair

A text classifier for job fit, trained two ways, plus the fairness audit that compares them.
One model is a plain baseline. The other uses a gradient-reversal adversary to reduce the
gender gap in true positive rates. A demo scores the same candidate before and after gender
cues are swapped.

Deliverables:

- D1: modular PyTorch code with dual-objective adversarial training (`hirefair/`).
- D2: fairness audit report (`report/report.md`, `submission/HireFair_Fairness_Audit_Report.pdf`).
- D3: live demo, Gradio and terminal (`hirefair/demo/`).

## Install

Python 3.10 or newer. PyTorch CPU is enough. From the project root, on a laptop or a fresh Colab runtime after cloning the repository:

```bash
python -m pip install -r requirements.txt
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

The spaCy model provides the POS tagger used to resolve `her` as an object pronoun or a
possessive. Without it the swapper falls back to a deterministic rule and still runs offline.

## Data

Bias in Bios is downloaded from Hugging Face on first use and cached as
`data/raw/bios.parquet`. Expected size is about 100 MB of download and 165 MB of text.

```bash
python scripts/prepare_data.py --role all --profile configs/fast_cpu.yaml
```

This verifies the label maps (pronoun and role-keyword checks), writes Table T1 source data,
tokenises the raw, masked and swapped inputs, and generates the Track B synthetic pairs in
`data/synthetic/`.

`configs/fast_cpu.yaml` is the compute profile used for the committed results: 12,000 training
rows, 10,000 validation and test rows, 5 epochs. Drop the `--profile` flag to use the full
spec defaults (80,000 rows, 8 epochs). Metric definitions, splits and variants do not change
between profiles.

## Train and audit

One run:

```bash
python scripts/run_experiment.py --role nurse --variant B3 --seed 0 --profile configs/fast_cpu.yaml
```

Lambda sweep, then the grid (this writes the chosen lambda per role and reuses it for B2/B3):

```bash
python scripts/run_sweep.py --role all --profile configs/fast_cpu.yaml
python scripts/run_all.py --role all --variant B0 B1 B2 B3 B4 --seed 0 1 2 3 4 \
    --profile configs/fast_cpu.yaml
```

Aggregate, analyse and generate the report artefacts:

```bash
python scripts/aggregate.py
python scripts/analyze_gender_tokens.py --role all
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/find_flip_examples.py --role all
python scripts/make_report.py --pdf
```

Every script supports `--help`. Add `--quick` for a one-epoch smoke test on 1,000 rows; quick
runs write to `results_quick/` and `data/processed_quick/`, so they cannot overwrite reported
results.

## Demo

Gradio, local and offline once checkpoints exist:

```bash
python -m hirefair.demo.app
```

Terminal fallback:

```bash
python -m hirefair.demo.cli --role nurse
python -m hirefair.demo.cli --role surgeon --text "She led the theatre team for six years."
```

The demo reads B0, B3 and (when present) B1 checkpoints from `results/`. If a checkpoint is
missing it prints the exact command that produces it instead of crashing. Examples come from
`scripts/find_flip_examples.py`; they are labelled as selected to illustrate, and the
aggregate flip rate is shown next to them.

## Layout

```
configs/            base.yaml, compute profiles, one file per role and variant
hirefair/data/      loader, tokenisation, gender-cue swap, synthetic generator
hirefair/models/    encoder, heads, gradient reversal, checkpointing
hirefair/training/  losses, lambda schedule, dual-objective trainer
hirefair/fairness/  metrics, bootstrap, counterfactual summaries
hirefair/analysis/  log-odds tokens, gender direction, probe
hirefair/demo/      shared core, Gradio app, terminal CLI
scripts/            prepare, train, sweep, aggregate, tables, figures, examples, report
tests/              unit tests for every module above
report/             report.md, generated tables and figures
results/            per-run artefacts, summary.csv, sweep.csv, analysis, tables
```

Every run writes `results/{role}/{variant}/seed{n}/run.json`, `model.pt`, `vocab.json`,
`config.yaml` and `predictions.npz`. `results/summary.csv` holds one row per
(role, variant, seed).

## Configuration

YAML is loaded into dataclasses; unknown keys fail loudly. `configs/base.yaml` holds the spec
defaults. `configs/{role}_{variant}.yaml` fixes the variant. Command-line arguments override
individual fields. Nothing is hardcoded except the project root.

Variants: B0 baseline, B1 masked cues, B2 adversarial demographic parity, B3 adversarial equal
opportunity, B4 counterfactual augmentation.

## Tests

```bash
python -m pytest
```

The suite covers the gradient reversal layer, the Ganin schedule, hand-computed fairness
metrics (including the zero-positive edge case), swap involution and cue-only diffs, synthetic
pair construction, counterfactual metrics, bootstrap intervals, config validation, model
reproducibility, the adversary's `eo` row mask, overfitting capacity and demo scoring.

## Limitations and ethics

Binary gender only. Biographies, not resumes. Occupation is a proxy for job fit. Base rates
differ by gender, so demographic parity and equal opportunity cannot both hold exactly. The
embedding table can keep the gender direction even when the encoder output is debiased. The
audit is not a certification and the model must not be used for real hiring decisions on this
evidence. The full list is in Section 8 of the report.

## Troubleshooting

- `ModuleNotFoundError: en_core_web_sm` — install the spaCy model wheel shown above.
- Dataset download fails — check network access; the loader stops rather than falling back to
  another dataset.
- Missing checkpoint in the demo — run the command the error prints.
- Colab — open `notebooks/colab_runner.ipynb`; it only calls the scripts.
