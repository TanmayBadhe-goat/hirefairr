from collections import Counter

from hirefair.data.preprocess import tokenize
from hirefair.data.synthetic import ROLE_CONTENT, TEMPLATES, generate_role, validate_pairs

ROLES = ("software_engineer", "nurse", "surgeon")


def test_at_least_six_templates_per_role():
    assert len(TEMPLATES) >= 6
    for role in ROLES:
        assert role in ROLE_CONTENT


def test_all_templates_are_used():
    records = generate_role("nurse", count=12, seed=0)
    used = Counter(record["template"] for record in records)
    assert set(used) == set(range(len(TEMPLATES)))


def test_pairs_share_content_slots():
    records = generate_role("software_engineer", count=20, seed=1)
    for record in records:
        assert record["content"]["skills"] in record["resume_M"]
        assert record["content"]["skills"] in record["resume_F"]
        assert record["content"]["skills"] in record["resume_neutral"]
        assert str(record["content"]["years"]) in record["resume_M"]
        assert str(record["content"]["years"]) in record["resume_F"]


def test_lengths_in_range():
    for role in ROLES:
        records = generate_role(role, count=60, seed=2)
        for record in records:
            for key in ("resume_M", "resume_F", "resume_neutral"):
                length = len(tokenize(record[key]))
                assert 40 <= length <= 140, (role, key, length)


def test_gender_pronouns_do_not_leak_across_renders():
    records = generate_role("surgeon", count=40, seed=3)
    for record in records:
        male = tokenize(record["resume_M"])
        female = tokenize(record["resume_F"])
        assert "she" not in male and "her" not in male
        assert "he" not in female and "his" not in female


def test_validation_returns_stats():
    stats = validate_pairs(generate_role("nurse", count=30, seed=4))
    assert stats["n"] == 30
    assert stats["mean_tokens"] > 0


def test_deterministic_with_seed():
    first = generate_role("nurse", count=5, seed=7)
    second = generate_role("nurse", count=5, seed=7)
    assert [r["resume_F"] for r in first] == [r["resume_F"] for r in second]
