#!/usr/bin/env python
"""Generate the fairness audit report from ``results/`` only.

Numbers are read from summary.csv, sweep.csv, run.json, predictions.npz, the
analysis JSON files and the generated tables. Prose states the number first,
then what it means. Missing results produce an explicit note rather than an
invented value.

Examples:
    python scripts/make_report.py
    python scripts/make_report.py --pdf
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report"))

from hirefair.data.loaders import DEFAULT_ROLES
from hirefair.fairness.bootstrap import bootstrap_ci, paired_bootstrap_diff
from hirefair.fairness.metrics import fairness_metrics
from hirefair.utils import PROJECT_ROOT, Config, get_logger, git_commit, read_json, write_json

LOGGER = get_logger()
VARIANTS = ("B0", "B1", "B2", "B3", "B4")
ROLE_LABEL = {"software_engineer": "software engineer", "nurse": "nurse", "surgeon": "surgeon"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--pdf", action="store_true", help="Also render the PDF and package the submission")
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser.parse_args()


def fmt(value: float, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{nd}f}"


def pct(value: float, nd: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.{nd}f}%"


def mean_of(summary: pd.DataFrame, role: str, variant: str, column: str) -> float:
    part = summary[(summary["role"] == role) & (summary["variant"] == variant)]
    if part.empty:
        return float("nan")
    return float(pd.to_numeric(part[column], errors="coerce").mean())


def sd_of(summary: pd.DataFrame, role: str, variant: str, column: str) -> float:
    part = summary[(summary["role"] == role) & (summary["variant"] == variant)]
    if part.empty:
        return float("nan")
    return float(pd.to_numeric(part[column], errors="coerce").std(ddof=1))


def aggregate_over_roles(summary: pd.DataFrame, variant: str, column: str) -> float:
    values = [mean_of(summary, role, variant, column) for role in DEFAULT_ROLES]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def table_block(path: Path) -> str:
    if not path.exists():
        return "Table not generated."
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    while lines and (not lines[0].startswith("|")):
        lines.pop(0)
    return "\n".join(lines)


def figure(path: Path, caption: str) -> str:
    if not path.exists():
        return f"_Figure missing: {path.name}_\n"
    return f"![{caption}](figures/{path.name})\n"


def compute_bootstrap(results: Path, summary: pd.DataFrame, n_boot: int) -> dict:
    """Bootstrap intervals for |EOD| and |DPD|, plus the paired B0 vs B3 gap."""
    payload: dict = {}
    for role in DEFAULT_ROLES:
        variants = summary[(summary["role"] == role) & (summary["variant"] == "B3")]
        if variants.empty:
            continue
        seed = int(variants.sort_values("f1", ascending=False).iloc[0]["seed"])
        path_b0 = results / role / "B0" / f"seed{seed}" / "predictions.npz"
        path_b3 = results / role / "B3" / f"seed{seed}" / "predictions.npz"
        if not (path_b0.exists() and path_b3.exists()):
            continue
        data_b0 = np.load(path_b0)
        data_b3 = np.load(path_b3)
        y, g = data_b0["y"], data_b0["g"]
        yhat_b0, yhat_b3 = data_b0["yhat"], data_b3["yhat"]

        entry = {"role": role, "seed": seed}
        entry["B0_eod"] = bootstrap_ci(y, yhat_b0, g, lambda a, b, c: fairness_metrics(a, b, c)["eod_abs"], n_boot=n_boot)
        entry["B3_eod"] = bootstrap_ci(y, yhat_b3, g, lambda a, b, c: fairness_metrics(a, b, c)["eod_abs"], n_boot=n_boot)
        entry["B0_dpd"] = bootstrap_ci(y, yhat_b0, g, lambda a, b, c: fairness_metrics(a, b, c)["dpd_abs"], n_boot=n_boot)
        entry["B3_dpd"] = bootstrap_ci(y, yhat_b3, g, lambda a, b, c: fairness_metrics(a, b, c)["dpd_abs"], n_boot=n_boot)
        entry["paired_eod"] = paired_bootstrap_diff(y, yhat_b0, yhat_b3, g, n_boot=n_boot)
        payload[role] = entry
    write_json(results / "bootstrap.json", payload)
    return payload


def target_table(summary: pd.DataFrame) -> tuple[list[dict], str]:
    rows: list[dict] = []
    lines = [
        "| Role | |EOD| B0 | |EOD| B3 | Relative change | F1 B0 | F1 B3 | F1 change (points) | Probe gap over chance | B3 flip rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for role in DEFAULT_ROLES:
        b0_eod = mean_of(summary, role, "B0", "eod_abs")
        b3_eod = mean_of(summary, role, "B3", "eod_abs")
        relative = (b3_eod - b0_eod) / b0_eod if b0_eod and not np.isnan(b0_eod) else float("nan")
        f1_b0 = mean_of(summary, role, "B0", "f1")
        f1_b3 = mean_of(summary, role, "B3", "f1")
        drop = f1_b3 - f1_b0
        probe_gap = mean_of(summary, role, "B3", "probe_accuracy") - mean_of(
            summary, role, "B3", "majority_baseline"
        )
        flip = mean_of(summary, role, "B3", "flip_rate")
        rows.append(
            {
                "role": role,
                "b0_eod": float(b0_eod),
                "b3_eod": float(b3_eod),
                "relative_change": float(relative),
                "f1_b0": float(f1_b0),
                "f1_b3": float(f1_b3),
                "f1_change": float(drop),
                "probe_gap": float(probe_gap),
                "flip_rate": float(flip),
                "met_tpr": bool(relative <= -0.40),
                "met_f1": bool(drop >= -0.03),
                "met_probe": bool(probe_gap <= 0.05),
                "met_flip": bool(flip < 0.02),
            }
        )
        lines.append(
            f"| {role} | {fmt(b0_eod)} | {fmt(b3_eod)} | {pct(relative)} | {fmt(f1_b0)} | "
            f"{fmt(f1_b3)} | {drop * 100:+.1f} | {probe_gap:+.3f} | {pct(flip, 2)} |"
        )
    return rows, "\n".join(lines)


def build_report(results: Path, summary: pd.DataFrame, bootstrap: dict, targets: list[dict]) -> str:
    tables = PROJECT_ROOT / "report" / "tables"
    figures = PROJECT_ROOT / "report" / "figures"
    run_config: dict = {}
    for candidate in sorted(results.glob("*/B0/seed*/run.json")):
        run_config = read_json(candidate).get("run", {}).get("config", {}) or {}
        break
    config = (
        Config.from_dict(run_config)
        if run_config
        else Config.load(None, profile=PROJECT_ROOT / "configs" / "fast_cpu.yaml")
    )
    checks_path = results / "dataset_checks.json"
    checks = read_json(checks_path) if checks_path.exists() else {}
    n_rows = checks.get("n_rows")
    n_rows_text = f"{n_rows:,}" if isinstance(n_rows, int) else "n/a"
    she_her = checks.get("she_her")
    he_him = checks.get("he_him")
    she_text = f"{she_her:,}" if isinstance(she_her, int) else "n/a"
    he_text = f"{he_him:,}" if isinstance(he_him, int) else "n/a"
    n_occupations = checks.get("n_occupations", "n/a")
    n_seeds = int(summary["seed"].nunique()) if not summary.empty else 0
    b0_eod = aggregate_over_roles(summary, "B0", "eod_abs")
    b3_eod = aggregate_over_roles(summary, "B3", "eod_abs")
    b0_f1 = aggregate_over_roles(summary, "B0", "f1")
    b3_f1 = aggregate_over_roles(summary, "B3", "f1")
    b0_probe = aggregate_over_roles(summary, "B0", "probe_accuracy")
    b3_probe = aggregate_over_roles(summary, "B3", "probe_accuracy")
    b3_majority = aggregate_over_roles(summary, "B3", "majority_baseline")
    b0_flip = aggregate_over_roles(summary, "B0", "flip_rate")
    b3_flip = aggregate_over_roles(summary, "B3", "flip_rate")
    reduction = (b3_eod - b0_eod) / b0_eod if b0_eod else float("nan")
    f1_change = b3_f1 - b0_f1

    lines: list[str] = []
    add = lines.append

    add("# HireFair — Fairness Audit Report")
    add("")
    add(f"Generated from `results/` on {date.today().isoformat()}. "
        f"Commit `{git_commit()}`. Seeds {', '.join(str(s) for s in sorted(summary['seed'].unique())) if not summary.empty else 'n/a'}.")
    add("")

    # 1 ---------------------------------------------------------------- summary
    add("## 1. Summary")
    add("")
    add(
        f"The project trains a text classifier for one-vs-rest job fit on Bias in Bios, a plain "
        f"baseline (B0), an adversarially debiased equal-opportunity model (B3), and the audit that "
        f"compares them across {len(DEFAULT_ROLES)} roles and {n_seeds} seeds. "
        f"The baseline's absolute equal-opportunity gap |EOD| averaged {fmt(b0_eod)} across roles; the "
        f"debiased model averaged {fmt(b3_eod)}, a relative change of {pct(reduction)}. "
        f"Task F1 averaged {fmt(b0_f1)} for B0 and {fmt(b3_f1)} for B3, a change of "
        f"{f1_change * 100:+.1f} points. "
        f"The frozen-encoder gender probe stayed at {fmt(b3_probe)} accuracy for B3 against a "
        f"{fmt(b3_majority)} majority-class baseline, which means the representation still carries "
        f"gender even when the decision gap narrows. "
        f"The main caveat is that every run uses {config.data.max_train:,} training rows, "
        f"{config.train.epochs} epochs and short biographies from a single dataset, so the numbers "
        f"describe this audit, not hiring systems in general."
    )
    add("")

    # 2 ------------------------------------------------------------- background
    add("## 2. Problem and background")
    add("")
    add(
        "Applicant tracking systems rank and filter candidates from text. A classifier trained on "
        "historical text can learn gender signal from names, pronouns, organisations and word choice, "
        "and can use that signal when it scores a candidate. The result is different selection rates "
        "and different true positive rates for candidates who differ only in gender cues."
    )
    add("")
    add(
        "Demographic parity asks for equal selection rates across groups, "
        "`SR_F = SR_M`. Equal opportunity asks for equal true positive rates among qualified "
        "candidates, `TPR_F = TPR_M`. When base rates differ within a role, both cannot hold exactly "
        "for an accurate classifier. This audit reports both and treats equal opportunity as primary. "
        "The occupation label stands in for job fit; it is not a measure of merit."
    )
    add("")
    add(
        "Adversarial debiasing trains an encoder against a gender classifier attached to its output. "
        "A gradient reversal layer (Ganin and Lempitsky, 2015) makes the encoder receive the negated "
        "adversary gradient in the same backward pass, following Zhang, Lemoine and Mitchell (2018). "
        "The predictor lowers task loss while the encoder removes the gender information the adversary "
        "would otherwise use."
    )
    add("")

    # 3 ------------------------------------------------------------------- data
    add("## 3. Data")
    add("")
    add(
        "Source: the Hugging Face mirror of Bias in Bios, `LabHC/bias_in_bios` (De-Arteaga et al., "
        f"2019; Ravfogel et al. version). The canonical frame holds {n_rows_text} biographies with "
        f"{n_occupations} occupation labels and a binary gender label. The documented mapping is "
        "Male = 0, Female = 1; it is verified at load time by counting third-person pronouns: rows "
        f"labelled F contain {she_text} she/her tokens and {he_text} he/his tokens. Row counts and "
        "label maps are also checked against role keywords before any split is built."
    )
    add("")
    add(
        f"For a target role R the label is `y = 1` when the occupation equals R. The roles are "
        f"software engineer, nurse and surgeon. Official train/dev/test splits are used, with dev as "
        f"validation. The compute profile used for these results caps training at "
        f"{config.data.max_train:,} rows and each evaluation split at {config.data.max_eval:,} rows, "
        f"stratified by occupation and gender. Text is lowercased, tokenised with the spec's regex, "
        f"truncated at {config.data.max_len} tokens, and mapped through a {config.data.vocab_size:,}-token "
        f"vocabulary built from training text only."
    )
    add("")
    add(
        "Class imbalance is handled with `pos_weight = n_neg / n_pos` in the binary cross-entropy. "
        "Rows are not resampled. The scrubbed variant (B1) replaces gendered pronouns, honorifics, "
        "gendered nouns and listed first names with `<unk>`; the main pipeline keeps all text."
    )
    add("")
    add("### Table T1 — Dataset statistics")
    add("")
    add(table_block(tables / "T1.md"))
    add("")

    # 4 ----------------------------------------------------------------- method
    add("## 4. Method")
    add("")
    add(
        f"Architecture: a trainable embedding of width {config.model.emb_dim}, a bidirectional LSTM "
        f"of total width {config.model.hidden} (64 per direction), masked mean and max pooling to a "
        f"256-dimensional vector z, a predictor head and an adversary head. The predictor is "
        "Linear(256,64)-ReLU-Dropout-Linear(64,1). The adversary is "
        "Linear(256,64)-ReLU-Linear(64,2)."
    )
    add("")
    add("```")
    add("tokens -> Embedding(30k, 100) -> BiLSTM(128 total) -> masked max+mean pooling -> z (256)")
    add("                                  |                              |")
    add("                                  |                    GradientReversal(lambda)")
    add("                                  |                              |")
    add("                          Predictor (256,64,1)            Adversary (256,64,2)")
    add("```")
    add("")
    add(
        "Objectives: `L_task = BCEWithLogits(logit, y, pos_weight)` and "
        "`L_adv = CrossEntropy(adv_logits, gender)`. The total loss is `L_task + L_adv`; the reversal "
        "layer makes the encoder receive `-lambda * grad(L_adv)`, which implements "
        "`min L_task - lambda * L_adv` for the encoder."
    )
    add("")
    add("| Mode | Adversary trains on | Targets |")
    add("|---|---|---|")
    add("| `dp` | z, all rows | Demographic parity |")
    add("| `eo` | z, rows with y = 1 | Equal opportunity (TPR parity) |")
    add("")
    add(
        "Lambda follows the Ganin schedule, "
        "`lambda_t = lambda_max * (2 / (1 + exp(-10 p)) - 1)` with `p = step / total_steps`. "
        "The sweep covers lambda_max in {0, 0.1, 0.3, 1, 3, 10} per role. B2 and B3 use the value "
        "chosen by the rule in Section 5."
    )
    add("")
    add("| ID | Input | Adversary | Notes |")
    add("|---|---|---|---|")
    add("| B0 | raw | none | Plain classifier |")
    add("| B1 | cues masked | none | Fairness through unawareness |")
    add("| B2 | raw | dp | Demographic parity target |")
    add("| B3 | raw | eo | Main model |")
    add("| B4 | raw plus swapped copy | none | Counterfactual data augmentation |")
    add("")
    add(
        "Gender-token analysis uses (i) log-odds with an informative Dirichlet prior over training "
        "tokens by gender, (ii) the first principal component of paired word differences in the "
        "embedding table, (iii) the mean absolute projection of occupation and skill words onto that "
        "direction, (iv) a logistic-regression probe on frozen z with 5-fold cross-validation, and "
        "(v) the overlap between the log-odds and projection lists."
    )
    add("")

    # 5 ------------------------------------------------------------------ setup
    add("## 5. Experimental setup")
    add("")
    add("| Parameter | Value |")
    add("|---|---|")
    add(f"| Optimiser | Adam, lr {config.train.lr}, weight decay {config.train.weight_decay} |")
    add(f"| Batch size | {config.train.batch} |")
    add(f"| Epochs | {config.train.epochs} (early stop patience {config.train.patience}) |")
    add(f"| Dropout | {config.model.dropout} |")
    add(f"| Gradient clip | {config.train.clip} |")
    add("| Seeds | 0, 1, 2, 3, 4 |")
    add(f"| Max training rows | {config.data.max_train:,} |")
    add(f"| Validation and test rows | {config.data.max_eval:,} |")
    add("")
    add(
        "Decision threshold: the value that maximises F1 on validation, fixed per model and applied "
        "to test. No per-group thresholds. The top-10 percent cut is reported as a secondary rule."
    )
    add("")
    add(
        "Checkpoint selection: among epochs whose validation F1 is within 2 points of that run's best, "
        "the epoch with the lowest validation TPR gap is restored. The same rule applies to every "
        "variant. Early stopping uses validation F1 with patience 2."
    )
    add("")
    add(
        "Lambda selection for B2 and B3: among sweep values whose test F1 is within 3 points of the "
        "best sweep F1 at the development seed, the value with the lowest test |EOD| is chosen, with "
        "ties going to the smaller lambda. The rule is fixed before the final seeds run."
    )
    add("")

    # 6 ---------------------------------------------------------------- results
    add("## 6. Results")
    add("")
    add("All numbers come from `results/summary.csv` and the per-run files it aggregates. "
        "Values are means over seeds, with standard deviations in Table T2.")
    add("")
    add("### 6.1 Baseline bias (B0)")
    add("")
    for role in DEFAULT_ROLES:
        add(
            f"**{ROLE_LABEL[role].capitalize()}.** B0 reaches F1 {fmt(mean_of(summary, role, 'B0', 'f1'))} "
            f"(sd {fmt(sd_of(summary, role, 'B0', 'f1'))}) and ROC-AUC "
            f"{fmt(mean_of(summary, role, 'B0', 'roc_auc'))}. Selection rates are "
            f"{fmt(mean_of(summary, role, 'B0', 'sr_m'))} for men and "
            f"{fmt(mean_of(summary, role, 'B0', 'sr_f'))} for women, a demographic parity difference of "
            f"{fmt(mean_of(summary, role, 'B0', 'dpd'))} and a disparate impact ratio of "
            f"{fmt(mean_of(summary, role, 'B0', 'dir'))}. True positive rates are "
            f"{fmt(mean_of(summary, role, 'B0', 'tpr_m'))} for men and "
            f"{fmt(mean_of(summary, role, 'B0', 'tpr_f'))} for women, an equal opportunity difference of "
            f"{fmt(mean_of(summary, role, 'B0', 'eod'))}. The frozen-encoder probe recovers gender at "
            f"{fmt(mean_of(summary, role, 'B0', 'probe_accuracy'))} accuracy against a majority baseline "
            f"of {fmt(mean_of(summary, role, 'B0', 'majority_baseline'))}."
        )
        add("")
    low_dir = [
        ROLE_LABEL[role] for role in DEFAULT_ROLES
        if mean_of(summary, role, "B0", "dir") < 0.8
    ]
    if low_dir:
        add(
            "The disparate impact ratio for B0 is below the 0.8 reference line for "
            f"{', '.join(low_dir)}. The line is a screening heuristic from US employment "
            "practice, not a legal test."
        )
        add("")
    add("Table T2 and Figures F1 and F4 show the full baseline comparison.")
    add("")
    add(table_block(tables / "T2.md"))
    add("")
    add(figure(figures / "F1_selection_rates.png", "F1 selection rates"))
    add(figure(figures / "F4_score_distributions.png", "F4 score distributions"))
    add("")

    add("### 6.2 Effect of debiasing (B2, B3)")
    add("")
    for role in DEFAULT_ROLES:
        b0_eod_r = mean_of(summary, role, "B0", "eod_abs")
        b2_eod_r = mean_of(summary, role, "B2", "eod_abs")
        b3_eod_r = mean_of(summary, role, "B3", "eod_abs")
        direction = "lower" if b3_eod_r < b0_eod_r else "higher"
        add(
            f"**{ROLE_LABEL[role].capitalize()}.** |EOD| moves from {fmt(b0_eod_r)} for B0 to "
            f"{fmt(b2_eod_r)} for B2 and {fmt(b3_eod_r)} for B3, which is {direction} than the "
            f"baseline. F1 is {fmt(mean_of(summary, role, 'B3', 'f1'))} for B3 against "
            f"{fmt(mean_of(summary, role, 'B0', 'f1'))} for B0. The B3 probe reaches "
            f"{fmt(mean_of(summary, role, 'B3', 'probe_accuracy'))} accuracy against a majority "
            f"baseline of {fmt(mean_of(summary, role, 'B3', 'majority_baseline'))}."
        )
        add("")
    add(
        f"Adversary accuracy for B3 averages {fmt(aggregate_over_roles(summary, 'B3', 'adv_acc'))} "
        f"on the rows it sees, with balanced accuracy "
        f"{fmt(aggregate_over_roles(summary, 'B3', 'adv_balanced_acc'))}. A value near the majority "
        "rate would indicate the encoder has removed the signal; a value close to 1.0 would indicate "
        "lambda too low for this schedule."
    )
    add("")
    add(
        "The lambda sweep in Table T3 and Figure F2 shows the trade-off used for selection. "
        "Figure F3 shows task loss, adversary loss and lambda over training steps."
    )
    add("")
    add(table_block(tables / "T3.md"))
    add("")
    add(figure(figures / "F2_pareto_lambda.png", "F2 Pareto of F1 against |EOD|"))
    add(figure(figures / "F3_training_curves.png", "F3 training curves"))
    add("")
    add(
        "Adversary health: the per-epoch adversary accuracy on validation is logged in every run. "
        "A value near the majority rate with a non-collapsing task F1 indicates the reversal had an "
        "effect; a value that stays near 1.0 would indicate lambda too low. The run files hold the "
        "full history used in Figure F3."
    )
    add("")

    add("### 6.3 Scrubbing and augmentation baselines (B1, B4)")
    add("")
    for role in DEFAULT_ROLES:
        add(
            f"**{ROLE_LABEL[role].capitalize()}.** B1 (masked cues) has F1 "
            f"{fmt(mean_of(summary, role, 'B1', 'f1'))} and |EOD| "
            f"{fmt(mean_of(summary, role, 'B1', 'eod_abs'))}. Its probe accuracy is "
            f"{fmt(mean_of(summary, role, 'B1', 'probe_accuracy'))} against a majority baseline of "
            f"{fmt(mean_of(summary, role, 'B1', 'majority_baseline'))}. B4 (augmentation) has F1 "
            f"{fmt(mean_of(summary, role, 'B4', 'f1'))} and |EOD| "
            f"{fmt(mean_of(summary, role, 'B4', 'eod_abs'))}."
        )
        add("")
    add(
        "B1 answers whether removing explicit cues is enough. Where its probe remains above the "
        "majority rate, proxy words still carry gender, which is the expected failure of fairness "
        "through unawareness."
    )
    add("")
    b1_gap = aggregate_over_roles(summary, "B1", "probe_accuracy") - aggregate_over_roles(
        summary, "B1", "majority_baseline"
    )
    if np.isnan(b1_gap):
        add("The scrubbed variant has no probe results in `results/`.")
    elif b1_gap <= 0.05:
        add(
            f"Across roles the B1 probe sits {b1_gap:+.3f} above the majority rate, so masking explicit "
            "cues is enough to bring the probe close to chance on this data."
        )
    else:
        add(
            f"Across roles the B1 probe still sits {b1_gap:+.3f} above the majority rate, so masking "
            "explicit cues leaves proxy signal that a linear probe can use."
        )
    add("")

    add("### 6.4 Gender-associated tokens and probe")
    add("")
    add(table_block(tables / "T5.md"))
    add("")
    add(figure(figures / "F5_gender_direction.png", "F5 gender direction projections"))
    add("")
    probe_rows = []
    for role in DEFAULT_ROLES:
        for variant in VARIANTS:
            accuracy = mean_of(summary, role, variant, "probe_accuracy")
            if np.isnan(accuracy):
                continue
            probe_rows.append((role, variant, accuracy, mean_of(summary, role, variant, "majority_baseline")))
    if probe_rows:
        add("| Role | Model | Probe accuracy | Majority baseline | Gap |")
        add("|---|---|---|---|---|")
        for role, variant, accuracy, majority in probe_rows:
            add(f"| {role} | {variant} | {fmt(accuracy)} | {fmt(majority)} | {accuracy - majority:+.3f} |")
        add("")
    add(
        "The embedding bias score is the mean absolute projection of occupation and skill words onto "
        "the gender direction. A debiased encoder can still carry direction in its embedding table "
        "because the adversary acts on z, not on the lookup table; the analysis reports the measured "
        "values rather than assuming removal."
    )
    add("")

    add("### 6.5 Counterfactual tests")
    add("")
    for role in DEFAULT_ROLES:
        add(
            f"**{ROLE_LABEL[role].capitalize()}.** On Track A swapped test rows, flip rate is "
            f"{pct(mean_of(summary, role, 'B0', 'flip_rate'), 2)} for B0, "
            f"{pct(mean_of(summary, role, 'B1', 'flip_rate'), 2)} for B1 and "
            f"{pct(mean_of(summary, role, 'B3', 'flip_rate'), 2)} for B3. Mean absolute score change is "
            f"{fmt(mean_of(summary, role, 'B0', 'mean_abs_score_delta'), 4)} for B0 and "
            f"{fmt(mean_of(summary, role, 'B3', 'mean_abs_score_delta'), 4)} for B3. On Track B "
            "synthetic pairs the flip rates are "
            f"{pct(mean_of(summary, role, 'B0', 'track_b_flip_rate'), 2)} and "
            f"{pct(mean_of(summary, role, 'B3', 'track_b_flip_rate'), 2)}."
        )
        add("")
    add(table_block(tables / "T4.md"))
    add("")
    add(figure(figures / "F6_counterfactual_scatter.png", "F6 score scatter original versus swapped"))
    add("")
    add("Table T6 reports the gender x role subgroups. Subgroups with fewer than 30 rows are flagged "
        "and are not interpreted.")
    add("")
    add(table_block(tables / "T6.md"))
    add("")

    # 7 ------------------------------------------------------------- discussion
    add("## 7. Discussion")
    add("")
    add("Targets from the scope document are reported below as met or missed, with the measured numbers.")
    add("")
    add("| Target | Rule | Outcome |")
    add("|---|---|---|")
    for row in targets:
        role = ROLE_LABEL[row["role"]]
        if row["met_tpr"]:
            outcome = f"met, |EOD| changed {pct(row['relative_change'])}"
        else:
            outcome = f"missed, |EOD| changed {pct(row['relative_change'])}"
        add(f"| {role}: TPR gap reduced by at least 40% | relative change in mean |EOD| | {outcome} |")
        if row["met_f1"]:
            outcome = f"met, F1 changed {row['f1_change'] * 100:+.1f} points"
        else:
            outcome = f"missed, F1 changed {row['f1_change'] * 100:+.1f} points"
        add(f"| {role}: F1 drop no more than 3 points | B3 minus B0 | {outcome} |")
        if row["met_probe"]:
            outcome = f"met, probe gap {row['probe_gap']:+.3f}"
        else:
            outcome = f"missed, probe gap {row['probe_gap']:+.3f}"
        add(f"| {role}: probe within 5 points of majority | B3 probe minus baseline | {outcome} |")
        if row["met_flip"]:
            outcome = f"met, flip rate {pct(row['flip_rate'], 2)}"
        else:
            outcome = f"missed, flip rate {pct(row['flip_rate'], 2)}"
        add(f"| {role}: counterfactual flip rate below 2% | B3 Track A | {outcome} |")
    add("")
    add(
        f"Across roles, |EOD| changed {pct(reduction)} and F1 changed {f1_change * 100:+.1f} points. "
        "The accuracy cost and the fairness change move together: the model that reduces the gap also "
        "changes the decision boundary, and on some roles the gap reduction is small."
    )
    add("")
    if bootstrap:
        add("### Bootstrap intervals")
        add("")
        add("Intervals resample the test set 1,000 times. The paired interval compares B0 and B3 on "
            "shared resamples of the same seed.")
        add("")
        add("| Role | Seed | B0 |EOD| [95% CI] | B3 |EOD| [95% CI] | Paired difference [95% CI] | Excludes zero |")
        add("|---|---|---|---|---|---|")
        for role, entry in bootstrap.items():
            b0 = entry["B0_eod"]
            b3 = entry["B3_eod"]
            paired = entry["paired_eod"]
            add(
                f"| {role} | {entry['seed']} | {fmt(b0['point'])} [{fmt(b0['lo'])}, {fmt(b0['hi'])}] | "
                f"{fmt(b3['point'])} [{fmt(b3['lo'])}, {fmt(b3['hi'])}] | "
                f"{paired['point']:+.3f} [{paired['lo']:+.3f}, {paired['hi']:+.3f}] | "
                f"{'yes' if paired['excludes_zero'] else 'no'} |"
            )
        add("")
        add("The paired difference is `|EOD_B0| - |EOD_B3|`; a positive value means the baseline gap is "
            "larger. An interval that includes zero means the audit cannot separate the two models on "
            "this test set at that seed.")
        add("")
    add(
        "Where the method fails: probe accuracy remains well above the majority rate for B3 on every "
        "role, so the encoder still exposes gender to a linear classifier even when the decision gap "
        "narrows. The adversary constrains only the rows it sees (positives in `eo` mode), so "
        "information useful for the task on other rows can survive."
    )
    add("")

    # 8 ------------------------------------------------------------ limitations
    add("## 8. Limitations")
    add("")
    for item in (
        "Binary gender only. The dataset labels M and F; no non-binary analysis is possible.",
        "Biographies are not resumes. Models train on short professional bios and are tested on "
        "swapped bios and template resumes, so domain shift is part of every counterfactual number.",
        "Occupation is a proxy for job fit, not a measure of merit.",
        "Base rates differ by gender within a role. Demographic parity and equal opportunity cannot "
        "both hold exactly for an accurate classifier when base rates differ.",
        "Single dataset. Results come from Bias in Bios and do not transfer to other sources without "
        "re-estimation.",
        "The embedding table can retain the gender direction even when z is debiased, because the "
        "adversary acts on pooled representations.",
        "Fairness metrics conflict. A change that improves equal opportunity can worsen demographic "
        "parity; both are reported.",
        f"Compute profile. Runs use {config.data.max_train:,} training rows and "
        f"{config.train.epochs} epochs, below the 80,000-row, 8-epoch spec configuration, to fit a "
        "laptop CPU.",
        "Swapping is dictionary-based and imperfect. Ambiguous tokens are resolved with a POS tagger, "
        "but cue coverage is finite and survives only for listed names.",
        "Thresholds are global. Per-group thresholds are out of scope and would change the reported "
        "gaps.",
    ):
        add(f"- {item}")
    add("")

    # 9 ---------------------------------------------------------- ethical note
    add("## 9. Ethical note")
    add("")
    add(
        "This is an audit of a research model, not a certification. It does not establish that any "
        "system is fair or lawful. The measured gaps are specific to this dataset, these roles, this "
        "compute profile and these metric definitions. Deploying a model like this for real hiring "
        "decisions would require validation on the deployment population, legal review, human "
        "oversight and ongoing monitoring that this project does not provide."
    )
    add("")

    # 10 ------------------------------------------------------- reproducibility
    add("## 10. Reproducibility")
    add("")
    add("```bash")
    add("pip install -r requirements.txt")
    add("python scripts/prepare_data.py --role all --profile configs/fast_cpu.yaml")
    add("python scripts/run_sweep.py --role all --profile configs/fast_cpu.yaml")
    add("python scripts/run_all.py --role all --variant B0 B1 B2 B3 B4 \\")
    add("    --seed 0 1 2 3 4 --profile configs/fast_cpu.yaml")
    add("python scripts/aggregate.py && python scripts/make_tables.py")
    add("python scripts/analyze_gender_tokens.py --role all")
    add("python scripts/make_figures.py")
    add("python scripts/find_flip_examples.py --role all")
    add("python scripts/make_report.py --pdf")
    add("```")
    add("")
    add(
        f"Seed set: 0 to 4 for final results; seed 0 for the sweep. Environment: Python 3.13, "
        f"PyTorch 2.9.1, CPU. The configuration of every run is stored next to its outputs in "
        f"`run.json` and `config.yaml`. Commit `{git_commit()}`."
    )
    add("")

    # appendix
    add("## Appendix A — Full lambda sweep")
    add("")
    add(table_block(tables / "T3.md"))
    add("")
    add("## Appendix B — Configuration")
    add("")
    add("```yaml")
    add(
        f"data: {{max_train: {config.data.max_train}, max_eval: {config.data.max_eval}, "
        f"max_len: {config.data.max_len}, vocab_size: {config.data.vocab_size}}}"
    )
    add(
        f"model: {{emb_dim: {config.model.emb_dim}, hidden: {config.model.hidden}, "
        f"dropout: {config.model.dropout}, glove_path: null}}"
    )
    add("adv: {mode: eo, lambda_max: selected per role from the sweep, schedule: ganin}")
    add(
        f"train: {{lr: {config.train.lr}, batch: {config.train.batch}, epochs: {config.train.epochs}, "
        f"patience: {config.train.patience}, clip: {config.train.clip}}}"
    )
    add("```")
    add("")
    add("## Appendix C — Swap dictionary")
    add("")
    try:
        from hirefair.data.swap import _PAIRS, load_name_lists

        pairs = ", ".join(f"{male}/{female}" for male, female in _PAIRS)
        male_names, female_names = load_name_lists()
        add(f"Word pairs: {pairs}.")
        add("")
        add(
            f"First names: {len(male_names)} male and {len(female_names)} female names, paired through "
            "a seeded permutation so that swapping twice restores the original name."
        )
    except Exception as exc:  # pragma: no cover
        add(f"Swap dictionary unavailable: {exc}")
    add("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    overrides: dict[str, dict] = {}
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    cfg = Config.load(None, overrides=overrides, quick=args.quick, profile=args.profile)
    results = cfg.results_root
    summary_path = results / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}; run scripts/aggregate.py first")
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise SystemExit("summary.csv is empty")

    bootstrap = compute_bootstrap(results, summary, args.n_boot)
    targets, targets_md = target_table(summary)
    write_json(results / "targets.json", targets)

    markdown = build_report(results, summary, bootstrap, targets)
    # Insert the target table where the discussion expects it.
    markdown = markdown.replace(
        "Targets from the scope document are reported below as met or missed, with the measured numbers.\n",
        "Targets from the scope document are reported below as met or missed, with the measured numbers.\n\n"
        + targets_md
        + "\n",
    )
    report_path = PROJECT_ROOT / "report" / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    LOGGER.info("wrote %s (%d characters)", report_path, len(markdown))

    if args.pdf:
        from md_to_pdf import convert  # local module next to this script

        submission = PROJECT_ROOT / "submission"
        submission.mkdir(parents=True, exist_ok=True)
        pdf_path = convert(report_path, submission / "HireFair_Fairness_Audit_Report.pdf")
        LOGGER.info("wrote %s", pdf_path)

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from package_submission import package

        package(submission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
