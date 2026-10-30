"""Gender-cue swapping and masking for real biography text.

``swap_gender(text, direction, mode)`` is the single code path used by the
scrubbed variant (``mode="mask"``), the counterfactual tests (``mode="swap"``)
and the demo.

Name replacement is deterministic and involutive: male and female name lists
are equalised in length and paired through a seeded permutation. Applying the
same swap twice returns the original name, which the tests rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from hirefair.utils import PROJECT_ROOT

NAMES_DIR = PROJECT_ROOT / "data" / "names"
MASK_TOKEN = "<unk>"
MAX_NAME_SWAPS_PER_TEXT = 3

# (male, female) pairs. Plural forms are listed explicitly because the
# tokenizer keeps them whole.
_PAIRS: list[tuple[str, str]] = [
    ("he", "she"),
    ("him", "her"),
    ("himself", "herself"),
    ("his", "hers"),
    ("mr", "ms"),
    ("mr.", "ms."),
    ("man", "woman"),
    ("men", "women"),
    ("male", "female"),
    ("males", "females"),
    ("boy", "girl"),
    ("boys", "girls"),
    ("father", "mother"),
    ("fathers", "mothers"),
    ("son", "daughter"),
    ("sons", "daughters"),
    ("husband", "wife"),
    ("husbands", "wives"),
    ("brother", "sister"),
    ("brothers", "sisters"),
    ("king", "queen"),
    ("kings", "queens"),
    ("gentleman", "lady"),
    ("gentlemen", "ladies"),
    ("chairman", "chairwoman"),
    ("chairmen", "chairwomen"),
    ("actor", "actress"),
    ("actors", "actresses"),
    ("fraternity", "sorority"),
    ("fraternities", "sororities"),
    ("salesman", "saleswoman"),
    ("salesmen", "saleswomen"),
    ("businessman", "businesswoman"),
    ("businessmen", "businesswomen"),
    ("policeman", "policewoman"),
    ("policemen", "policewomen"),
    ("fireman", "firewoman"),
    ("firemen", "firewomen"),
    ("spokesman", "spokeswoman"),
    ("spokesmen", "spokeswomen"),
    ("waiter", "waitress"),
    ("waiters", "waitresses"),
    ("nephew", "niece"),
    ("nephews", "nieces"),
    ("uncle", "aunt"),
    ("uncles", "aunts"),
    ("grandfather", "grandmother"),
    ("grandfathers", "grandmothers"),
    ("grandson", "granddaughter"),
    ("grandsons", "granddaughters"),
    ("stepfather", "stepmother"),
    ("stepson", "stepdaughter"),
    ("widower", "widow"),
    ("landlord", "landlady"),
    ("headmaster", "headmistress"),
    ("hero", "heroine"),
    ("prince", "princess"),
    ("duke", "duchess"),
    ("emperor", "empress"),
    ("monk", "nun"),
    ("sir", "madam"),
    ("sportsman", "sportswoman"),
    ("sportsmen", "sportswomen"),
    ("congressman", "congresswoman"),
    ("congressmen", "congresswomen"),
]

MALE_TO_FEMALE: dict[str, str] = {m: f for m, f in _PAIRS}
FEMALE_TO_MALE: dict[str, str] = {f: m for m, f in _PAIRS}

# Tokens that reveal gender regardless of direction, used for cue detection and
# for masking (the B1 variant).
GENDER_CUE_TOKENS: set[str] = set(MALE_TO_FEMALE) | set(FEMALE_TO_MALE) | {"her", "his", "hers"}

_PREPOSITIONS = {
    "to", "for", "with", "by", "of", "from", "at", "on", "in", "about", "around",
    "toward", "towards", "against", "behind", "beside", "between", "into", "over",
    "under", "upon", "without", "within", "after", "before", "near", "than",
}
_VERBS = {
    "awarded", "gave", "give", "gave", "told", "tell", "asked", "ask", "helped",
    "help", "mentored", "mentor", "supported", "support", "saw", "see", "met",
    "meet", "hired", "hire", "promoted", "promote", "taught", "teach", "chose",
    "choose", "selected", "select", "thanked", "thank", "recognized", "recognize",
    "appointed", "appoint", "invited", "invite", "paid", "pay", "joined", "join",
    "trained", "train", "led", "lead", "followed", "follow", "called", "call",
    "made", "make", "found", "find", "left", "leave", "brought", "bring",
}
_OBJECT_AFTER = {
    "is", "was", "were", "are", "has", "had", "have", "will", "would", "can",
    "could", "should", "may", "might", "must", "and", "or", "but", "that", "who",
    "which", "whom", "whose", "when", "while", "because", "if", "as", "so",
}

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z0-9]")


@dataclass(frozen=True)
class Change:
    """One replaced span, with offsets into the output text."""

    start: int
    end: int
    original: str
    replacement: str
    kind: str  # "pronoun", "noun", "honorific", "name"


@dataclass
class SwapResult:
    """Swapped or masked text plus the spans that changed."""

    text: str
    direction: str
    mode: str
    changes: list[Change] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when at least one cue was replaced."""
        return bool(self.changes)


# ------------------------------------------------------------------- names


def _read_names(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        names.append(name)
    return names


def _first_n(items: list[str], n: int) -> list[str]:
    """Take the first ``n`` items. Curated minority blocks come first in the files."""
    return list(items[:n])


@lru_cache(maxsize=4)
def load_name_lists(names_dir: str | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return equal-length male and female first-name tuples.

    Both lists are required to hold at least 300 names. The longer list is
    subsampled evenly so the Indian-name blocks at the end are retained.
    """
    directory = Path(names_dir) if names_dir else NAMES_DIR
    male = _read_names(directory / "male_names.txt")
    female = _read_names(directory / "female_names.txt")
    size = min(len(male), len(female))
    if size < 300:
        raise ValueError(f"need at least 300 names per gender, found {len(male)}/{len(female)}")
    return tuple(_first_n(male, size)), tuple(_first_n(female, size))


@lru_cache(maxsize=8)
def _name_maps(seed: int = 0) -> tuple[dict[str, str], dict[str, str]]:
    """Seeded, involutive male<->female name pairing."""
    male, female = load_name_lists()
    rng = np.random.default_rng(seed)
    size = len(male)
    perm = rng.permutation(size)
    # Keys are lowercased for token lookup; values keep their display casing.
    male_to_female = {male[i].lower(): female[perm[i]] for i in range(size)}
    female_to_male = {female[perm[i]].lower(): male[i] for i in range(size)}
    return male_to_female, female_to_male


# ----------------------------------------------------------------- tagging


@lru_cache(maxsize=1)
def _spacy_nlp():
    """Load the small English POS tagger, or return None if unavailable."""
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])
    except Exception:  # model not installed
        return None


def _tokens_with_tags(text: str) -> list[tuple[str, int, str]]:
    """Return ``(token, char_offset, POS tag)`` triples.

    Uses spaCy when the model is installed, otherwise a regex tokenizer with
    empty tags. The regex fallback is deterministic and offline.
    """
    nlp = _spacy_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [(tok.text, tok.idx, tok.tag_) for tok in doc if not tok.is_space]
    return [(m.group(0), m.start(), "") for m in _TOKEN_RE.finditer(text)]


def _her_is_possessive(tokens: list[tuple[str, int, str]], index: int) -> bool:
    """Decide whether ``her`` at ``index`` is a possessive determiner."""
    tag = tokens[index][2]
    if tag in {"PRP$", "POS"}:
        return True
    if tag == "PRP":
        return False
    lower = [t[0].lower() for t in tokens]
    nxt = lower[index + 1] if index + 1 < len(lower) else ""
    prev = lower[index - 1] if index > 0 else ""
    if prev in _PREPOSITIONS or prev in _VERBS:
        return False
    if nxt in _OBJECT_AFTER or nxt in _PREPOSITIONS:
        return not (prev in _PREPOSITIONS or prev in _VERBS)
    if nxt and (nxt.endswith("ing") or nxt.endswith("ed")):
        return False
    return True


def _his_is_possessive(tokens: list[tuple[str, int, str]], index: int) -> bool:
    """Decide whether ``his`` at ``index`` is a possessive determiner."""
    tag = tokens[index][2]
    if tag in {"PRP$", "POS"}:
        return True
    if tag == "PRP":
        return False
    lower = [t[0].lower() for t in tokens]
    prev = lower[index - 1] if index > 0 else ""
    if prev in _PREPOSITIONS or prev in _VERBS:
        return False
    return True


# ----------------------------------------------------------------- helpers


def _match_case(source: str, target: str) -> str:
    """Apply the capitalisation pattern of ``source`` to ``target``."""
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _lookup_key(token: str) -> tuple[str, str]:
    """Return ``(lookup key, trailing punctuation)`` for a token."""
    stripped = token.rstrip(".")
    suffix = token[len(stripped):]
    return stripped.lower(), suffix


def _infer_direction_tokens(tokens: list[tuple[str, int, str]], seed: int = 0) -> str:
    """Infer the dominant cue direction from precomputed tokens."""
    male_score = 0
    female_score = 0
    male_names, female_names = _name_maps(seed)
    for token, _, _ in tokens:
        key, _ = _lookup_key(token)
        if token[:1].isupper() and key in male_names:
            male_score += 1
        elif token[:1].isupper() and key in female_names:
            female_score += 1
        elif key in MALE_TO_FEMALE:
            male_score += 1
        elif key in FEMALE_TO_MALE:
            female_score += 1
    return "F2M" if female_score > male_score else "M2F"


def infer_direction(text: str) -> str:
    """Infer the dominant cue direction of a text.

    Returns ``"F2M"`` when female cues dominate, else ``"M2F"``.
    """
    return _infer_direction_tokens(_tokens_with_tags(text))


def find_gender_cues(text: str, seed: int = 0) -> list[Change]:
    """Locate every gender cue in ``text`` with offsets into that text.

    Used by the demo to highlight cues and by tests to verify that a swap only
    touches cue positions.
    """
    male_names, female_names = _name_maps(seed)
    cues: list[Change] = []
    for token, offset, _tag in _tokens_with_tags(text):
        key, _ = _lookup_key(token)
        kind = None
        if (key in male_names or key in female_names) and token[:1].isupper():
            kind = "name"
        elif key in {"he", "she", "him", "her", "his", "hers", "himself", "herself"}:
            kind = "pronoun"
        elif key in {"mr", "ms", "sir", "madam"}:
            kind = "honorific"
        elif key in MALE_TO_FEMALE or key in FEMALE_TO_MALE:
            kind = "noun"
        if kind:
            cues.append(Change(offset, offset + len(token), token, token, kind))
    return cues


def _replacement_for(
    tokens: list[tuple[str, int, str]],
    index: int,
    direction: str,
    mode: str,
    male_names: dict[str, str],
    female_names_to_male: dict[str, str],
) -> tuple[str, str] | None:
    """Return ``(replacement, kind)`` for one token, or None to keep it."""
    token = tokens[index][0]
    key, suffix = _lookup_key(token)
    base = token[: len(token) - len(suffix)] if suffix else token

    if key in male_names and token[:1].isupper():
        if mode == "mask":
            return MASK_TOKEN, "name"
        if direction == "M2F":
            return _match_case(token, male_names[key]), "name"
        return None
    if key in female_names_to_male and token[:1].isupper():
        if mode == "mask":
            return MASK_TOKEN, "name"
        if direction == "F2M":
            return _match_case(token, female_names_to_male[key]), "name"
        return None

    if mode == "mask":
        if key in GENDER_CUE_TOKENS or key in MALE_TO_FEMALE or key in FEMALE_TO_MALE:
            kind = "honorific" if key in {"mr", "ms", "sir", "madam"} else (
                "pronoun" if key in {"he", "she", "him", "her", "his", "hers", "himself", "herself"} else "noun"
            )
            return MASK_TOKEN, kind
        return None

    if key == "her" and direction == "F2M":
        possessive = _her_is_possessive(tokens, index)
        return _match_case(base, "his" if possessive else "him"), "pronoun"
    if key == "his" and direction == "M2F":
        possessive = _his_is_possessive(tokens, index)
        return _match_case(base, "her" if possessive else "hers"), "pronoun"
    if key == "hers" and direction == "F2M":
        return _match_case(base, "his"), "pronoun"

    table = FEMALE_TO_MALE if direction == "F2M" else MALE_TO_FEMALE
    if key in table:
        replacement = _match_case(base, table[key]) + suffix
        return replacement, "honorific" if key in {"mr", "ms", "sir", "madam"} else "noun"
    return None


def _apply(
    text: str,
    tokens: list[tuple[str, int, str]],
    direction: str,
    mode: str,
    seed: int,
) -> SwapResult:
    """Apply a resolved direction to precomputed tokens."""
    male_names, female_names = _name_maps(seed)
    pieces: list[str] = []
    changes: list[Change] = []
    cursor = 0
    out_len = 0
    name_swaps = 0

    for index, (token, offset, _tag) in enumerate(tokens):
        key, suffix = _lookup_key(token)
        result = _replacement_for(tokens, index, direction, mode, male_names, female_names)
        if result is None:
            continue
        replacement, kind = result
        if kind == "name" and mode == "swap":
            if name_swaps >= MAX_NAME_SWAPS_PER_TEXT:
                continue
            name_swaps += 1
        between = text[cursor:offset]
        pieces.append(between)
        out_len += len(between)
        pieces.append(replacement)
        changes.append(Change(out_len, out_len + len(replacement), token, replacement, kind))
        out_len += len(replacement)
        cursor = offset + len(token)
    pieces.append(text[cursor:])
    return SwapResult(text="".join(pieces), direction=direction, mode=mode, changes=changes)


def swap_gender(
    text: str,
    direction: str = "auto",
    mode: str = "swap",
    seed: int = 0,
) -> SwapResult:
    """Swap or mask gender cues in ``text``.

    Args:
        text: Input biography or resume text.
        direction: ``"F2M"``, ``"M2F"`` or ``"auto"`` to infer from the text.
        mode: ``"swap"`` replaces cues with their opposite; ``"mask"`` replaces
            every cue with ``<unk>``.
        seed: Seed for the name pairing.

    Returns:
        SwapResult with the new text and the changed spans.
    """
    if mode not in {"swap", "mask"}:
        raise ValueError("mode must be 'swap' or 'mask'")
    if direction not in {"F2M", "M2F", "auto"}:
        raise ValueError("direction must be 'F2M', 'M2F' or 'auto'")
    tokens = _tokens_with_tags(text)
    if direction == "auto":
        direction = _infer_direction_tokens(tokens, seed)
    return _apply(text, tokens, direction, mode, seed)


def swap_many(
    texts: Iterable[str],
    mode: str = "swap",
    seed: int = 0,
    batch_size: int = 256,
) -> list[str]:
    """Swap or mask many texts, batching the tagger when spaCy is available."""
    texts = list(texts)
    if mode not in {"swap", "mask"}:
        raise ValueError("mode must be 'swap' or 'mask'")
    nlp = _spacy_nlp()
    if nlp is None:
        return [swap_gender(text, direction="auto", mode=mode, seed=seed).text for text in texts]
    out: list[str] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        text = doc.text
        tokens = [(tok.text, tok.idx, tok.tag_) for tok in doc if not tok.is_space]
        direction = _infer_direction_tokens(tokens, seed) if mode == "swap" else "F2M"
        out.append(_apply(text, tokens, direction, mode, seed).text)
    return out


def mask_gender(text: str, seed: int = 0) -> SwapResult:
    """Convenience wrapper for the scrubbed (B1) input mode."""
    return swap_gender(text, direction="auto", mode="mask", seed=seed)


def swap_rows(texts: Iterable[str], seed: int = 0) -> list[str]:
    """Swap the cues of many texts with a single name pairing."""
    return [swap_gender(text, direction="auto", mode="swap", seed=seed).text for text in texts]
