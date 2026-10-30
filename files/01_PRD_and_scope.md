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
