"""Gender-associated tokens, embedding direction and the frozen-encoder probe."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np

# Pairs from the model spec. Both members must be in the vocabulary to be used.
DIRECTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("he", "she"),
    ("him", "her"),
    ("man", "woman"),
    ("male", "female"),
    ("father", "mother"),
    ("boy", "girl"),
    ("husband", "wife"),
    ("mr", "ms"),
)

OCCUPATION_WORDS = (
    "nurse", "surgeon", "engineer", "teacher", "physician", "professor", "attorney",
    "accountant", "architect", "dentist", "journalist", "psychologist", "poet",
    "painter", "photographer", "paralegal", "comedian", "rapper", "model",
    "software", "programmer", "developer", "doctor", "lawyer", "scientist",
)
SKILL_WORDS = (
    "python", "java", "kubernetes", "sql", "leadership", "communication",
    "management", "research", "coding", "programming", "surgery", "nursing",
    "patient", "care", "teaching", "writing", "analysis", "design",
)


def log_odds_ratio(
    counts_male: Mapping[str, int],
    counts_female: Mapping[str, int],
    alpha0: float = 1000.0,
    min_count: int = 5,
    top_n: int = 50,
) -> dict[str, list[dict[str, float | str]]]:
    """Monroe et al. (2008) log-odds with an informative Dirichlet prior.

    Positive z favours female-associated usage. Tokens below ``min_count`` in
    the combined counts are dropped.
    """
    vocab = set(counts_male) | set(counts_female)
    n_m = float(sum(counts_male.get(t, 0) for t in vocab))
    n_f = float(sum(counts_female.get(t, 0) for t in vocab))
    total = n_m + n_f
    if total == 0:
        raise ValueError("empty token counts")
    rows: list[dict[str, float | str]] = []
    for token in vocab:
        y = float(counts_female.get(token, 0))
        x = float(counts_male.get(token, 0))
        if y + x < min_count:
            continue
        alpha_i = alpha0 * (y + x) / total
        num_f = y + alpha_i
        num_m = x + alpha_i
        den_f = n_f + alpha0 - num_f
        den_m = n_m + alpha0 - num_m
        if num_f <= 0 or num_m <= 0 or den_f <= 0 or den_m <= 0:
            continue
        delta = np.log(num_f / den_f) - np.log(num_m / den_m)
        variance = 1.0 / num_f + 1.0 / num_m + 1.0 / den_f + 1.0 / den_m
        rows.append(
            {
                "token": token,
                "z": float(delta / np.sqrt(variance)),
                "count_f": y,
                "count_m": x,
            }
        )
    rows.sort(key=lambda row: float(row["z"]), reverse=True)
    return {"female": rows[:top_n], "male": rows[-top_n:][::-1]}


def gender_direction(
    embedding: np.ndarray,
    vocab: Mapping[str, int],
    pairs: Iterable[tuple[str, str]] = DIRECTION_PAIRS,
) -> np.ndarray:
    """First principal component of paired word differences.

    The direction points from the male pole to the female pole, so positive
    projections are female-associated.
    """
    diffs: list[np.ndarray] = []
    for male_word, female_word in pairs:
        if male_word in vocab and female_word in vocab:
            diffs.append(
                embedding[vocab[female_word]].astype(np.float64)
                - embedding[vocab[male_word]].astype(np.float64)
            )
    if len(diffs) < 2:
        raise ValueError("need at least two gender pairs present in the vocabulary")
    matrix = np.vstack(diffs)
    _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    direction = vt[0]
    # Orient from male to female using the pair mean difference.
    if float(np.mean(matrix @ direction)) < 0:
        direction = -direction
    norm = np.linalg.norm(direction)
    return direction / norm if norm else direction


def project_vocabulary(
    embedding: np.ndarray,
    direction: np.ndarray,
    vocab: Mapping[str, int],
    counts: Mapping[str, int] | None = None,
    min_freq: int = 20,
    top_n: int = 50,
) -> dict[str, list[dict[str, float | str]]]:
    """Project every eligible token onto the gender direction."""
    rows = []
    for token, index in vocab.items():
        if token.startswith("<"):
            continue
        if counts is not None and counts.get(token, 0) < min_freq:
            continue
        rows.append(
            {
                "token": token,
                "projection": float(np.dot(embedding[index].astype(np.float64), direction)),
                "count": int(counts.get(token, 0)) if counts else 0,
            }
        )
    rows.sort(key=lambda row: float(row["projection"]), reverse=True)
    return {"female": rows[:top_n], "male": rows[-top_n:][::-1]}


def embedding_bias_score(
    embedding: np.ndarray,
    direction: np.ndarray,
    vocab: Mapping[str, int],
    words: Iterable[str] = OCCUPATION_WORDS + SKILL_WORDS,
) -> dict[str, float]:
    """Mean absolute projection of occupation and skill words."""
    projections = [
        abs(float(np.dot(embedding[vocab[word]].astype(np.float64), direction)))
        for word in words
        if word in vocab
    ]
    if not projections:
        return {"mean_abs_projection": float("nan"), "n_words": 0.0}
    return {
        "mean_abs_projection": float(np.mean(projections)),
        "max_abs_projection": float(np.max(projections)),
        "n_words": float(len(projections)),
    }


def overlap_tokens(
    log_odds_female: list[dict[str, float | str]],
    log_odds_male: list[dict[str, float | str]],
    projection_female: list[dict[str, float | str]],
    projection_male: list[dict[str, float | str]],
) -> list[dict[str, str]]:
    """Tokens in both a top-50 log-odds list and a top-50 projection list."""
    odds = {str(row["token"]): "female" for row in log_odds_female}
    odds.update({str(row["token"]): "male" for row in log_odds_male})
    proj = {str(row["token"]): "female" for row in projection_female}
    proj.update({str(row["token"]): "male" for row in projection_male})
    shared = []
    for token in sorted(set(odds) & set(proj)):
        shared.append({"token": token, "log_odds_side": odds[token], "projection_side": proj[token]})
    return shared


def probe_gender(
    z: np.ndarray,
    g: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
) -> dict[str, float]:
    """Logistic-regression probe on frozen encoder output with k-fold CV.

    Returns out-of-fold accuracy and AUC plus the majority-class baseline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z = np.asarray(z, dtype=np.float64)
    g = np.asarray(g).astype(int).ravel()
    majority = float(max(g.mean(), 1 - g.mean())) if len(g) else float("nan")
    if len(g) == 0:
        return {
            "probe_accuracy": float("nan"),
            "probe_auc": float("nan"),
            "majority_baseline": float("nan"),
            "probe_gap_over_chance": float("nan"),
            "n": 0.0,
        }
    counts = np.bincount(g, minlength=2)
    splits = int(min(n_splits, counts.min())) if counts.min() >= 2 else 2
    if counts.min() < 2:
        return {
            "probe_accuracy": float("nan"),
            "probe_auc": float("nan"),
            "majority_baseline": majority,
            "probe_gap_over_chance": float("nan"),
            "n": float(len(g)),
        }
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    predictions = np.zeros(len(g), dtype=float)
    probabilities = np.zeros(len(g), dtype=float)
    # Some BLAS backends (Accelerate on macOS) raise a spurious divide-by-zero
    # RuntimeWarning on all-zero matmuls. Inputs are checked finite above, so
    # floating-point error states are ignored for the fits only.
    with np.errstate(all="ignore"):
        for train_idx, test_idx in cv.split(z, g):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
            )
            model.fit(z[train_idx], g[train_idx])
            probabilities[test_idx] = model.predict_proba(z[test_idx])[:, 1]
            predictions[test_idx] = (probabilities[test_idx] >= 0.5).astype(int)
    accuracy = float((predictions == g).mean())
    try:
        auc = float(roc_auc_score(g, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "probe_accuracy": accuracy,
        "probe_auc": auc,
        "majority_baseline": majority,
        "probe_gap_over_chance": float(accuracy - majority),
        "n": float(len(g)),
        "n_folds": float(splits),
    }


def token_counts(token_lists: Iterable[Iterable[str]]) -> Counter[str]:
    """Count tokens across an iterable of token lists."""
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return counts
