"""Model variants B0 to B4.

The variant fixes the input text mode and the adversary setting. Metric
definitions and split rules are never variant-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    """One row of the experiment grid."""

    id: str
    name: str
    text_mode: str  # "raw" or "masked"
    adv_mode: str  # "none", "dp", "eo", "eodds"
    augment: bool
    description: str


VARIANTS: dict[str, Variant] = {
    "B0": Variant(
        id="B0",
        name="Baseline",
        text_mode="raw",
        adv_mode="none",
        augment=False,
        description="Plain classifier, no debiasing.",
    ),
    "B1": Variant(
        id="B1",
        name="Scrubbed",
        text_mode="masked",
        adv_mode="none",
        augment=False,
        description="Gender cues replaced by <unk>; fairness through unawareness.",
    ),
    "B2": Variant(
        id="B2",
        name="Adv-DP",
        text_mode="raw",
        adv_mode="dp",
        augment=False,
        description="Adversary on all rows, targeting demographic parity.",
    ),
    "B3": Variant(
        id="B3",
        name="Adv-EO",
        text_mode="raw",
        adv_mode="eo",
        augment=False,
        description="Adversary on positives only, targeting equal opportunity.",
    ),
    "B4": Variant(
        id="B4",
        name="CDA",
        text_mode="raw",
        adv_mode="none",
        augment=True,
        description="Counterfactual augmentation: each train row plus a swapped copy.",
    ),
}


def resolve_variant(variant: str) -> Variant:
    """Look up a variant id, failing with the valid list."""
    try:
        return VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}") from exc
