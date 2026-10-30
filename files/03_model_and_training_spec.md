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
