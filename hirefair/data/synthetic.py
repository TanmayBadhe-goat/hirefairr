"""Track B: synthetic counterfactual resume summaries.

Each base item is rendered three ways from one set of content slots:
``resume_M`` (male cues), ``resume_F`` (female cues) and ``resume_neutral``
(no name, singular they). Content slots never depend on gender, so a pair
differs only in cue positions. Test-only data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hirefair.data.preprocess import tokenize
from hirefair.data.swap import load_name_lists
from hirefair.utils import PROJECT_ROOT, get_logger

LOGGER = get_logger()


@dataclass(frozen=True)
class RoleContent:
    """Role-specific slot values. Neutral with respect to gender."""

    role_label: str
    skills: tuple[str, ...]
    employers: tuple[str, ...]
    achievements: tuple[str, ...]
    educations: tuple[str, ...]
    teams: tuple[str, ...]
    councils: tuple[str, ...]


ROLE_CONTENT: dict[str, RoleContent] = {
    "software_engineer": RoleContent(
        role_label="software engineer",
        skills=(
            "Python", "Java", "Go", "Kubernetes", "PostgreSQL", "React",
            "distributed systems", "CI/CD pipelines", "AWS", "TypeScript",
            "API design", "unit testing",
        ),
        employers=(
            "a mid-sized payments company", "a healthcare analytics startup",
            "a public-sector digital services team", "a logistics platform",
            "an enterprise software vendor", "a fintech scale-up",
        ),
        achievements=(
            "led the migration of a monolith to containerised services",
            "cut checkout latency by 38 percent over two quarters",
            "built the internal deployment pipeline used by twelve teams",
            "shipped a search service handling four million queries a day",
            "reduced cloud spend by 22 percent through query tuning",
            "mentored four junior engineers through their first on-call rotation",
        ),
        educations=(
            "a BSc in Computer Science", "an MSc in Software Engineering",
            "a BEng in Information Technology", "a BSc in Mathematics",
            "an MSc in Data Science", "a diploma in Computer Engineering",
        ),
        teams=("cricket", "debating", "rowing", "volleyball", "quiz", "chess"),
        councils=("engineering", "technology", "platform", "architecture", "open source", "developer"),
    ),
    "nurse": RoleContent(
        role_label="nurse",
        skills=(
            "patient triage", "IV therapy", "Epic EMR", "wound care",
            "medication administration", "care planning", "BLS certification",
            "post-operative monitoring", "infection control", "patient education",
            "cardiac monitoring", "shift handover",
        ),
        employers=(
            "a regional teaching hospital", "a community health centre",
            "a private surgical clinic", "a paediatric ward",
            "an emergency department", "a rehabilitation facility",
        ),
        achievements=(
            "reduced patient wait times by 18 percent across the unit",
            "led the ward handover redesign for three shifts",
            "trained twelve new nurses on the EMR rollout",
            "maintained a zero-pressure-ulcer record for two years",
            "coordinated discharge planning for over 400 patients",
            "introduced a medication double-check that cut errors by a third",
        ),
        educations=(
            "a BSc in Nursing", "an Associate Degree in Nursing",
            "a diploma in General Nursing and Midwifery", "a BSc in Health Sciences",
            "an MSc in Nursing Practice", "a certificate in Critical Care Nursing",
        ),
        teams=("cricket", "debating", "netball", "volleyball", "badminton", "athletics"),
        councils=("nursing", "clinical", "patient safety", "quality improvement", "community health", "education"),
    ),
    "surgeon": RoleContent(
        role_label="surgeon",
        skills=(
            "laparoscopic surgery", "robotic-assisted surgery", "trauma care",
            "surgical oncology", "pre-operative assessment", "team leadership",
            "clinical research", "minimally invasive techniques", "post-operative care",
            "quality assurance", "teaching", "audit",
        ),
        employers=(
            "a tertiary referral hospital", "a university teaching hospital",
            "a specialist cancer centre", "a regional trauma centre",
            "a private surgical practice", "a children's hospital",
        ),
        achievements=(
            "performed over 600 elective procedures with a 1.2 percent complication rate",
            "established a day-surgery pathway that freed 300 bed days a year",
            "led a surgical audit that changed antibiotic prophylaxis policy",
            "published twelve peer-reviewed papers on minimally invasive technique",
            "reduced average theatre turnaround time by 25 percent",
            "supervised eight residents through their operative logbook",
        ),
        educations=(
            "an MBBS and an MS in General Surgery", "an MD in Surgery",
            "an MBBS and a fellowship in surgical oncology",
            "an MD and a fellowship in trauma surgery",
            "an MBBS and an MCh in gastrointestinal surgery",
            "an MD and a fellowship in minimally invasive surgery",
        ),
        teams=("cricket", "debating", "rowing", "tennis", "hockey", "swimming"),
        councils=("surgical", "clinical governance", "medical education", "quality", "research", "hospital"),
    ),
}

# Verb slots are part of the cue set so the neutral rendering (singular they)
# stays grammatical while the content slots stay identical across a pair.
SURNAMES: tuple[str, ...] = (
    "Sharma", "Patel", "Singh", "Kumar", "Reddy", "Iyer", "Nair", "Gupta", "Mehta",
    "Shah", "Brown", "Smith", "Johnson", "Williams", "Miller", "Davis", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Lewis", "Walker", "Hall",
    "Young", "King", "Wright", "Lopez", "Hill", "Scott", "Green",
)

TEMPLATES: tuple[str, ...] = (
    "{intro} {intro_v} a {role_label} with {years} years of experience at {employer}. "
    "{Subj} {achievement}. {Subj} {works_v} with {skills}. {Subj} graduated with "
    "{education}. {Subj} {was_v} captain of the {team} team.",

    "{intro} {intro_works_v} as a {role_label} at {employer}, reporting to the {senior}. "
    "Over {years} years {subj_l} {achievement}. Core skills include {skills}. "
    "{Subj} completed {education} and {volunteers_v} with the {team} mentoring programme.",

    "{intro} {intro_v} a {role_label} with {years} years in practice. {Subj} {achievement}. "
    "{Subj} {works_v} confidently with {skills}. Education: {education}. Outside work "
    "{subj_l} {coaches_v} the {team} team.",

    "Profile: {intro}, {role_label} at {employer}. {Subj} {achievement} over {years} "
    "years. {Poss} toolkit covers {skills}. {Subj} {holds_v} {education}. {Subj} "
    "{serves_v} on the {council} advisory board.",

    "{intro} {has_v} worked as a {role_label} for {years} years, most recently at "
    "{employer}. {Subj} {achievement}. {Subj} {lists_v} {skills} among {poss} strengths. "
    "{Subj} studied for {education} and led the {team} volunteer group.",

    "Summary. {intro}, {role_label}, {years} years at {employer}. {Subj} {achievement}. "
    "Key skills: {skills}. {Subj} earned {education}. {Subj} {represents_v} the "
    "{council} staff council.",
)


def _cue_variants(
    honorific_m: str,
    honorific_f: str,
    name_m: str,
    name_f: str,
    team_base: str,
    surname: str,
) -> dict[str, dict[str, str]]:
    """Return pronoun and cue slots for the male, female and neutral renderings."""
    return {
        "M": {
            "intro": f"{honorific_m} {name_m} {surname}",
            "Subj": "He",
            "subj_l": "he",
            "obj": "him",
            "poss": "his",
            "Poss": "His",
            "team": f"men's {team_base}",
            "senior": "chairman",
            "intro_v": "is",
            "intro_works_v": "works",
            "works_v": "works",
            "was_v": "was",
            "holds_v": "holds",
            "serves_v": "serves",
            "lists_v": "lists",
            "has_v": "has",
            "represents_v": "represents",
            "coaches_v": "coaches",
            "volunteers_v": "volunteers",
        },
        "F": {
            "intro": f"{honorific_f} {name_f} {surname}",
            "Subj": "She",
            "subj_l": "she",
            "obj": "her",
            "poss": "her",
            "Poss": "Her",
            "team": f"women's {team_base}",
            "senior": "chairwoman",
            "intro_v": "is",
            "intro_works_v": "works",
            "works_v": "works",
            "was_v": "was",
            "holds_v": "holds",
            "serves_v": "serves",
            "lists_v": "lists",
            "has_v": "has",
            "represents_v": "represents",
            "coaches_v": "coaches",
            "volunteers_v": "volunteers",
        },
        "N": {
            "intro": "The candidate",
            "Subj": "They",
            "subj_l": "they",
            "obj": "them",
            "poss": "their",
            "Poss": "Their",
            "team": f"staff {team_base}",
            "senior": "chair",
            "intro_v": "is",
            "intro_works_v": "works",
            "works_v": "work",
            "was_v": "were",
            "holds_v": "hold",
            "serves_v": "serve",
            "lists_v": "list",
            "has_v": "have",
            "represents_v": "represent",
            "coaches_v": "coach",
            "volunteers_v": "volunteer",
        },
    }


def _render(template: str, content: dict[str, Any], cues: dict[str, str]) -> str:
    values = {**content, **cues}
    # The closing line is identical in every rendering and contains no cues.
    text = template.format(**values) + " References available on request."
    return " ".join(text.split())


def generate_role(role: str, count: int = 1000, seed: int = 0) -> list[dict[str, Any]]:
    """Generate ``count`` counterfactual triples for one role."""
    if role not in ROLE_CONTENT:
        raise ValueError(f"unknown synthetic role {role!r}; expected {sorted(ROLE_CONTENT)}")
    spec = ROLE_CONTENT[role]
    rng = np.random.default_rng(seed)
    male_names, female_names = load_name_lists()
    honorifics = (("Mr.", "Ms."), ("Mr.", "Mrs."), ("Mr.", "Ms."))

    records: list[dict[str, Any]] = []
    for i in range(count):
        template = TEMPLATES[i % len(TEMPLATES)]
        male_name = str(male_names[rng.integers(len(male_names))])
        female_name = str(female_names[rng.integers(len(female_names))])
        if male_name == female_name:
            female_name = str(female_names[(female_names.index(female_name) + 1) % len(female_names)])
        honorific_m, honorific_f = honorifics[int(rng.integers(len(honorifics)))]

        n_skills = int(rng.integers(5, 9))
        skills = rng.choice(spec.skills, size=n_skills, replace=False).tolist()
        skills_phrase = ", ".join(skills[:-1]) + " and " + skills[-1]
        content = {
            "role_label": spec.role_label,
            "years": int(rng.integers(2, 21)),
            "employer": str(rng.choice(spec.employers)),
            "achievement": str(rng.choice(spec.achievements)),
            "education": str(rng.choice(spec.educations)),
            "skills": skills_phrase,
            "council": str(rng.choice(spec.councils)),
        }
        team_base = str(rng.choice(spec.teams))
        surname = str(rng.choice(SURNAMES))
        cues = _cue_variants(honorific_m, honorific_f, male_name, female_name, team_base, surname)

        record = {
            "id": f"{role}-{i:04d}",
            "role": role,
            "template": i % len(TEMPLATES),
            "content": {**content, "team_base": team_base},
            "resume_M": _render(template, content, cues["M"]),
            "resume_F": _render(template, content, cues["F"]),
            "resume_neutral": _render(template, content, cues["N"]),
        }
        records.append(record)
    return records


def validate_pairs(records: list[dict[str, Any]]) -> dict[str, float]:
    """Check that paired resumes differ only in cue positions.

    Returns summary statistics. Raises when content leaked across genders.
    """
    if not records:
        raise ValueError("no records")
    lengths = []
    for record in records:
        tokens_m = tokenize(record["resume_M"])
        tokens_f = tokenize(record["resume_F"])
        lengths.extend([len(tokens_m), len(tokens_f)])
        if not record["resume_M"] or not record["resume_F"]:
            raise ValueError("empty synthetic resume")
        if record["content"]["skills"] not in record["resume_M"]:
            raise ValueError("skills missing from the male rendering")
        if record["content"]["skills"] not in record["resume_F"]:
            raise ValueError("skills missing from the female rendering")
    mean = float(np.mean(lengths))
    if not 40 <= mean <= 140:
        raise ValueError(f"synthetic resume length out of range: mean={mean:.1f} tokens")
    return {"n": len(records), "mean_tokens": mean, "min_tokens": int(np.min(lengths)), "max_tokens": int(np.max(lengths))}


def write_synthetic(
    out_dir: str | Path,
    roles: tuple[str, ...] = ("software_engineer", "nurse", "surgeon"),
    count: int = 1000,
    seed: int = 0,
) -> dict[str, Path]:
    """Write one JSONL file per role under ``data/synthetic``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role in roles:
        records = generate_role(role, count=count, seed=seed)
        stats = validate_pairs(records)
        path = out_dir / f"{role}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info("synthetic %s: %d pairs, %.1f tokens mean", role, stats["n"], stats["mean_tokens"])
        paths[role] = path
    return paths


def default_out_dir() -> Path:
    """Project-relative output directory."""
    return PROJECT_ROOT / "data" / "synthetic"
