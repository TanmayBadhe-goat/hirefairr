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
