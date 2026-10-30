# 02 — Data Spec

## Track A: Bias in Bios (training, validation, test, audit)

Source: Hugging Face dataset `LabHC/bias_in_bios` (De-Arteaga et al., 2019). Verify the name, splits and column names at load time. Expected columns: `hard_text`, `profession`, `gender`. If the dataset cannot be loaded, stop and ask. Do not fall back to Track C without being told.

Canonical schema after loading (`hirefair/data/loaders.py`):

| Column | Type | Notes |
|--------|------|-------|
| id | int | row index |
| text | str | biography text, first sentence retained |
| occupation | str | one of 28 labels |
| gender | str | "M" or "F" |
| split | str | train / val / test |

Sanity check on load: count third-person pronouns per gender label. "she/her" must dominate in rows labelled F. If not, the label mapping is flipped. Fail loudly.

### Job-fit label
For target role R, `y = 1` if `occupation == R`, else 0.
Default roles: `software_engineer`, `nurse`, `surgeon`. Roles are set in config.

### Splits
Use official train/dev/test if present. Dev becomes val. Otherwise stratify by (occupation, gender) with 70/10/20 and seed 42. Cap train at `max_train` (default 80,000), stratified, to fit compute. Val and test are not capped below 20,000.

### Preprocessing
- Lowercase. Regex tokenizer `[a-z0-9']+|[^\sa-z0-9']`.
- Vocabulary from train only. Size 30,000. Min frequency 3. Tokens `<pad>`, `<unk>`.
- Max length 128 tokens. Truncate.
- Main pipeline does not remove gender cues. B0, B2, B3, B4 see the full text.
- B1 (scrubbed) removes: gendered pronouns, honorifics, gendered nouns in the swap dictionary, and first names found in the name lists. Same code path as the swap function with a `mask` mode that replaces them with `<unk>`.

### Class imbalance
Use `pos_weight = n_neg / n_pos` in the task loss. Do not resample.

### Dataset statistics (required output, feeds report Table 1)
Per role: n, positive rate overall, positive rate by gender, share of female among positives, mean token length.

## Track B: Synthetic counterfactual set (test only)

Purpose: controlled pairs that differ only in gender cues.
Module: `hirefair/data/synthetic.py`.

Generate resume-style summaries from templates.
- At least 6 templates per role. Each template has slots for: years of experience, skills (5 to 8 per role), employer type, achievement sentence, education line.
- Content slots are filled identically within a pair.
- Gender cue slots, which differ within a pair:
  - first name (from name lists)
  - pronouns and possessives
  - honorific (Mr/Ms/Mrs where the template uses one)
  - gendered activity or organisation line (for example "captain of the women's cricket team" vs "captain of the men's cricket team")
  - gendered role noun where relevant (chairman/chairwoman, salesman/saleswoman)
- Output per base item: `resume_M`, `resume_F`, `resume_neutral` (no cues, "they" and no name).
- Default 1,000 base items per role.
- Fixed seed. Written to `data/synthetic/{role}.jsonl`.
- Skill sets must be the same for both genders. No template may link a skill to a gender.

Domain note: models are trained on short bios and tested on resume-style summaries. Keep templates in short professional-summary form (60 to 120 tokens), third person or implied first person, to limit domain shift. Report the shift as a limitation.

## Track A counterfactuals (real text, auto-swapped)
Module: `hirefair/data/swap.py`. Function `swap_gender(text, direction, mode)`; modes `swap` and `mask`.

Swap table (case preserving):
he/she, him/her, his/her, himself/herself, mr/ms, mr./ms., man/woman, men/women, male/female, boy/girl, father/mother, son/daughter, husband/wife, brother/sister, king/queen, gentleman/lady, chairman/chairwoman, actor/actress, fraternity/sorority, and the same plural forms.

Ambiguity: "her" maps to "him" when it is an object pronoun and to "his" when it is a possessive. Resolve with a POS tagger (PRP vs PRP$). Reverse "his" to "her". "hers" and "his" pair directly.

Names: swap first names using lists of at least 300 male and 300 female first names. Include a block of at least 100 each of common Indian first names alongside US names. Choose the replacement name by a seeded random draw from the opposite list. Only replace a name token when it appears in the list and is capitalised in the original text.

Required unit tests:
- Swapping twice returns the original text for rows with no ambiguous tokens.
- No leftover gendered pronoun after a swap on 1,000 random rows.
- Output differs from input only at gender cue positions (diff check).

## Track C: Fallback (only if told)
Kaggle "Resume Dataset" (24 categories, no gender labels). Inject synthetic gender cues with a controlled correlation to the label, documented in the report. This is a weaker design and must be labelled as synthetic bias.

## Files and layout
```
data/raw/          # downloaded, git-ignored
data/processed/    # tokenised arrays, vocab.json
data/synthetic/    # Track B jsonl
data/names/        # male_names.txt, female_names.txt
```
