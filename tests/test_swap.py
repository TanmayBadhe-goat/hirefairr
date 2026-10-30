import re

import pytest

from hirefair.data.preprocess import tokenize
from hirefair.data.swap import (
    FEMALE_TO_MALE,
    MALE_TO_FEMALE,
    find_gender_cues,
    load_name_lists,
    mask_gender,
    swap_gender,
)

FEMALE_TEXT = (
    "Ms. Priya Sharma is a software engineer. She led her team and presented her "
    "work to the chairwoman. She is close to her sister and her mother."
)
MALE_TEXT = (
    "Mr. Rahul Sharma is a software engineer. He led his team and presented his "
    "work to the chairman. He is close to his brother and his father."
)

CUE_TOKENS = set(MALE_TO_FEMALE) | set(FEMALE_TO_MALE) | {"her", "his", "hers"}
NAME_TOKENS = {name.lower() for name in load_name_lists()[0]} | {
    name.lower() for name in load_name_lists()[1]
}


def test_name_lists_meet_minimums():
    male, female = load_name_lists()
    assert len(male) == len(female) >= 300
    assert "Rahul" in male and "Priya" in female


def test_swap_twice_returns_original():
    first = swap_gender(FEMALE_TEXT)
    assert first.direction == "F2M"
    second = swap_gender(first.text)
    assert second.text == FEMALE_TEXT


def test_swap_male_direction():
    result = swap_gender(MALE_TEXT)
    assert result.direction == "M2F"
    assert "She" in result.text
    assert "chairwoman" in result.text
    assert "sister" in result.text
    assert swap_gender(result.text).text == MALE_TEXT


def test_no_leftover_female_pronoun_after_f2m():
    result = swap_gender(FEMALE_TEXT)
    assert not re.search(r"\b(she|her|hers|herself)\b", result.text, re.IGNORECASE)


def test_no_leftover_male_pronoun_after_m2f():
    result = swap_gender(MALE_TEXT)
    assert not re.search(r"\b(he|him|his|himself)\b", result.text, re.IGNORECASE)


def test_diff_only_at_cue_positions():
    swapped = swap_gender(FEMALE_TEXT).text
    original_tokens = tokenize(FEMALE_TEXT)
    swapped_tokens = tokenize(swapped)
    assert len(original_tokens) == len(swapped_tokens)
    for left, right in zip(original_tokens, swapped_tokens):
        if left == right:
            continue
        assert left in CUE_TOKENS or right in CUE_TOKENS or left in NAME_TOKENS or right in NAME_TOKENS, (
            left,
            right,
        )


def test_possessive_and_object_her():
    possessive = swap_gender("Her research was funded.")
    assert possessive.text.startswith("His research")
    objective = swap_gender("The committee thanked her.")
    assert objective.text.rstrip(".").endswith("thanked him")


def test_mask_mode_replaces_cues():
    result = mask_gender(FEMALE_TEXT)
    assert "<unk>" in result.text
    assert "she" not in result.text.lower()
    assert "priya" not in result.text.lower()


def test_swap_leaves_neutral_text_unchanged():
    text = "The candidate has ten years of experience and holds a BSc."
    result = swap_gender(text)
    assert result.text == text
    assert not result.changed


def test_find_gender_cues_counts_names_and_pronouns():
    cues = find_gender_cues(FEMALE_TEXT)
    tokens = {cue.original for cue in cues}
    assert "She" in tokens
    assert "Priya" in tokens
    assert "chairwoman" in tokens


def test_name_only_replaced_when_capitalised():
    result = swap_gender("priya is a nurse.", direction="F2M")
    assert "priya" in result.text


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        swap_gender(FEMALE_TEXT, mode="replace")
