#!/usr/bin/env python3
"""HireFair 2.0 — AI Resume Screening and Fair Job Recommendation System.

A clean, professor-ready interface for demonstrating adversarial debiasing in
NLP resume screening. Three tabs: Resume Analyzer, Gender Bias Test, Fairness
Audit.

Design
------
- Dark-mode AI dashboard theme (see the ``<style>`` block).
- Small, clean text labels instead of the old bright-blue filled label
  backgrounds.
- No duplicate headings, no long-lived giant text boxes, noSELECT/REJECT
  indicators in the Resume Analyzer.
- Fairness Audit values are strictly read-only.

Architecture note (important for the audience)
----------------------------------------------
The Resume Analyzer does *not* use a single three-class softmax classifier.
It loads one trained binary job-fit checkpoint per demo role (``software_engineer``,
``data_analyst``, ``machine_learning_engineer``) and scores the same text
against all three. Each checkpoint answers "is this resume for this role?" and
the role with the highest score is displayed as the recommendation.
Per-role scores are therefore independent and nothing is derived from another
role's output. Scores are labelled "Independent role-model scores".

All numbers on the screen come from trained checkpoints in
``results/hirefair2/`` (via :mod:`hirefair.demo.core`/`hirefair.demo.inference`)
or from ``results/hirefair2/summary.csv`` (Fairness Audit). Nothing is hardcoded.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import gradio as gr

from hirefair.demo.core import (
    MissingCheckpoint,
    aggregate,
    aggregate_markdown,
    highlight_original,
    highlight_swapped,
    result_table,
    score_pair,
)
from hirefair.demo.inference import (
    DEMO_ROLES,
    RESULTS_ROOT,
    RoleAnalysis,
    analyse_text,
    recommend_role,
)
from hirefair.utils import get_logger

logger = get_logger()

TRAIN_COMMAND = "python scripts/train_roles.py"
DASH = "—"

# ---------------------------------------------------------------------------
# Dark-mode design system
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --hf-bg: #0B1120;
  --hf-card: #151F31;
  --hf-input: #222F43;
  --hf-accent: #2563EB;
  --hf-hover: #3B82F6;
  --hf-text: #F8FAFC;
  --hf-sec: #94A3B8;
  --hf-border: #2C3A50;
  --hf-green: #22C55E;
  --hf-amber: #F59E0B;
  --hf-radius: 14px;
}

.hf-app {
  background: var(--hf-bg);
  color: var(--hf-text);
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}

/* Compact header ------------------------------------------------------ */
.hf-header {
  padding: 14px 20px 10px 20px;
  border-bottom: 1px solid var(--hf-border);
  background: var(--hf-card);
}
.hf-logo {
  width: 30px; height: 30px; border-radius: 8px;
  background: linear-gradient(135deg, var(--hf-accent), var(--hf-hover));
  color: #fff; font-weight: 800; font-size: 13px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.hf-wordmark { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }
.hf-sub { color: var(--hf-sec); font-size: 12px; font-weight: 500; margin-top: 2px; }
.hf-research {
  margin-left: auto; color: var(--hf-sec); font-size: 11px;
  font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  background: var(--hf-input); padding: 5px 10px; border-radius: 6px;
}
.hf-header .hf-column { display: flex; align-items: center; gap: 12px; }

/* Cards ----------------------------------------------------------------- */
.hf-card {
  background: var(--hf-card);
  border: 1px solid var(--hf-border);
  border-radius: var(--hf-radius);
  padding: 14px 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, .35);
}
.hf-card + .hf-card { margin-top: 12px; }
.hf-card-title {
  font-size: 12px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; color: var(--hf-sec); margin-bottom: 10px;
}
.hf-row { display: flex; gap: 12px; align-items: stretch; }
.hf-row > .hf-card { flex: 1; }

/* Clean text labels (replaces the old bright-blue filled labels) -------- */
.hf-label {
  background: var(--hf-input);
  color: #E2E8F0;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 4px 8px;
  border-radius: 6px;
  display: inline-block;
}
.hf-label-sub { color: var(--hf-sec); font-size: 12px; font-weight: 400; }
.hf-value { color: var(--hf-text); font-size: 14px; font-weight: 600; }
.hf-note { color: var(--hf-sec); font-size: 11px; line-height: 1.45; }
.hf-hint { color: var(--hf-sec); font-size: 12px; font-weight: 400; }

/* Scrollable comparison table ----------------------------------------- */
.hf-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.hf-scroll > table { min-width: 640px; }

/* Headings ------------------------------------------------------------- */
.hf-h1 { color: var(--hf-text); font-size: 24px; font-weight: 800; letter-spacing: -0.02em; margin: 14px 0 6px 0; }
.hf-h2 { color: var(--hf-text); font-size: 17px; font-weight: 700; margin: 18px 0 6px 0; }
.hf-subtitle { color: var(--hf-sec); font-size: 13px; font-weight: 400; margin-bottom: 14px; }

/* Two-column layout ---------------------------------------------------- */
.hf-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
@media (max-width: 780px) {
  .hf-grid { grid-template-columns: 1fr; }
  .hf-header .hf-column { flex-direction: column; align-items: flex-start; }
  .hf-header .hf-research { margin-left: 0; }
}

/* Buttons -------------------------------------------------------------- */
.hf-btn { min-height: 46px; font-weight: 700; }
.hf-primary { background: var(--hf-accent); color: #fff; border: 1px solid var(--hf-accent); }
.hf-primary:hover { background: var(--hf-hover); border-color: var(--hf-hover); }
.hf-secondary { background: var(--hf-input); color: var(--hf-text); border: 1px solid var(--hf-border); }
.hf-secondary:hover { background: #2A3A56; border-color: var(--hf-hover); }

/* Upload dashed zone --------------------------------------------------- */
.hf-upload {
  border: 2px dashed var(--hf-border);
  border-radius: var(--hf-radius);
  padding: 22px;
  text-align: center;
  color: var(--hf-sec);
  transition: border-color .15s, background .15s;
}
.hf-upload:hover, .hf-upload.drag { border-color: var(--hf-accent); background: #1A2A44; }
.hf-upload .hf-icon { font-size: 26px; color: var(--hf-sec); }
.hf-upload .hf-upload-title { color: var(--hf-text); font-weight: 700; margin-top: 8px; }
.hf-upload .hf-upload-sub { color: var(--hf-sec); font-size: 12px; margin-top: 4px; }

/* Empty state ---------------------------------------------------------- */
.hf-empty { color: var(--hf-sec); font-size: 13px; font-weight: 400; }

/* Bar chart ------------------------------------------------------------ */
.hf-bar-track {
  height: 18px; background: #26334D; border-radius: 4px; overflow: hidden;
  display: flex; align-items: center;
}
.hf-bar-fill { height: 100%; border-radius: 4px; display: block; }
.hf-bar-label { font-size: 12px; font-weight: 600; }
.hf-bar-val { font-size: 12px; font-weight: 700; color: var(--hf-accent); }

/* Fairness audit metric cards ---------------------------------------- */
.hf-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.hf-metric { background: var(--hf-card); border: 1px solid var(--hf-border); border-radius: 12px; padding: 12px; text-align: center; }
.hf-metric .hf-metric-name { color: var(--hf-sec); font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.hf-metric .hf-metric-value { color: var(--hf-text); font-size: 22px; font-weight: 800; margin-top: 6px; }
.hf-metric .hf-metric-unit { color: var(--hf-sec); font-size: 11px; font-weight: 500; }

/* Status indicators ---------------------------------------------------- */
.hf-status { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 700; }
.hf-status-yes { color: var(--hf-green); }
.hf-status-no { color: var(--hf-sec); }

.hf-source-note { color: var(--hf-sec); font-size: 11px; line-height: 1.5; margin-top: 10px; }
.hf-source-note b { color: var(--hf-text); }

/* Accessibility helpers ------------------------------------------------ */
.hf-screen-reader { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0; }

/* Remove the old bright-blue filled label backgrounds (Gradio) --------- */
.hf-app .q-field__label,
.hf-app .gr-label,
.hf-app label,
.hf-app .block-label,
.hf-app .form-item label {
  background: var(--hf-input) !important;
  color: #E2E8F0 !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  padding: 4px 8px !important;
  border-radius: 6px !important;
  letter-spacing: .06em !important;
  text-transform: uppercase !important;
}

/* Stop long tables from clipping on narrow screens ---------------------- */
@media (max-width: 640px) {
  .hf-scroll { overflow-x: auto; }
  .hf-scroll > table { min-width: 560px; font-size: 12px; }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_path(value: Any) -> Optional[str]:
    """Normalise the value Gradio hands back for a single uploaded file."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _as_path(value[0]) if value else None
    name = getattr(value, "name", value)
    return str(name) if name else None


def _decode_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import pymupdf

        with pymupdf.open(file_path) as document:
            return "\n".join(page.get_text("text") for page in document).strip()
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - unreadable file
        logger.warning("could not read %s: %s", file_path, exc)

    try:
        from pdfminer.high_level import extract_text

        return extract_text(file_path).strip()
    except Exception:
        return ""


def _decode_docx_text(file_path: str) -> str:
    """Extract text from a DOCX file, including table cells."""
    try:
        import docx
    except ImportError:
        return ""

    try:
        document = docx.Document(file_path)
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
        return "\n".join(part for part in parts if part and part.strip()).strip()
    except Exception as exc:  # pragma: no cover - unreadable file
        logger.warning("could not read %s: %s", file_path, exc)
        return ""


def _truncation_note(analysis: RoleAnalysis) -> str:
    """Human-readable truncation warning, with the percentage of text cut off."""
    if not analysis.truncated:
        return ""
    kept = max(1, analysis.max_len)
    cut = analysis.n_tokens - kept
    pct = cut / analysis.n_tokens * 100.0
    return (
        f"<div class=\"hf-note\"><b>Input truncated:</b> {analysis.n_tokens} tokens "
        f"were read, but the model only supports {kept} tokens, so <b>{pct:.1f}%</b> of the "
        f"resume text was truncated. Scores below are computed on the kept prefix only.</b></div>"
    )


def _bar_chart(analysis: RoleAnalysis, target_role: str) -> str:
    """Horizontal bar chart of the three *independent* job-fit scores.

    Each checkpoint was trained only on its own role and scored this text
    directly. No score is derived from another role's output.
    """
    rows = []
    for entry in sorted(analysis.scores.values(), key=lambda e: -e.score):
        label = entry.role.replace("_", " ").title()
        width = max(1, min(100, int(round(entry.score * 100))))
        colour = "#3b82f6" if entry.role == target_role else "#64748b"
        weight = "font-weight: bold;" if entry.role == target_role else ""
        rows.append(
            f"""<div style="margin: 4px 0; display: flex; align-items: center; gap: 8px;">
                <span style="width: 180px; flex-shrink: 0; {weight}" class="hf-bar-label">{label}</span>
                <span style="flex: 1; min-width: 120px;">
                    <div style="height: 18px; background: #26334D; border-radius: 4px; overflow: hidden;">
                        <span style="display: block; width: {width}%; background: {colour}; height: 100%;"></span>
                    </div>
                </span>
                <span style="width: 90px; text-align: right; font-weight: bold; color: var(--hf-accent);" class="hf-bar-val">{entry.score:.3f}</span>
            </div>"""
        )
    return "".join(rows)


def _score_resume(text: str, role: str) -> tuple[str, dict[str, float], str, str]:
    """Score a resume text and return recommended role, scores, HTML and notes.

    ``role`` (the dropdown) is the target role the chart highlights.
    """
    if not text or not text.strip():
        return "", {}, "", "No text to score. Upload a resume or paste text."

    try:
        analysis = analyse_text(text, results_dir=RESULTS_ROOT)
    except MissingCheckpoint as exc:
        return "", {}, "", f"{exc}"
    except ValueError:
        return "", {}, "", "No text to score. Upload a resume or paste text."

    recommended = recommend_role(analysis)
    scores_display = {name: round(entry.score, 4) for name, entry in analysis.scores.items()}
    bars = _bar_chart(analysis, role)

    notes = [
        f"**Model's recommendation:** {recommended.replace('_', ' ').title()} "
        f"(score {analysis.scores[recommended].score:.3f}).",
        f"**Target role:** {role.replace('_', ' ').title()} — "
        f"{'selected' if analysis.scores[role].selected else 'rejected'} at that role's "
        f"stored threshold of {analysis.scores[role].threshold:.3f}.",
        "**Scores:** independent per-role job-fit checkpoints (one binary classifier "
        "per role). Scores are not normalised to sum to 100%.",
        _truncation_note(analysis),
    ]
    return recommended, scores_display, bars, "\n".join(notes)


def _run_resume_analysis(
    file: Any,
    text: str,
    role: str,
) -> tuple[str, str, str, str, str, str]:
    """Extract text from the upload (or the textbox), then score it."""
    extracted = ""
    path = _as_path(file)
    if path:
        lowered = path.lower()
        if lowered.endswith(".pdf"):
            extracted = _decode_pdf_text(path)
        elif lowered.endswith(".docx"):
            extracted = _decode_docx_text(path)

    final_text = extracted if extracted.strip() else (text or "")
    if not final_text or not final_text.strip():
        return (
            "No file selected",
            "",
            "",
            "",
            f"No text could be extracted from the document. Upload a PDF/DOCX with selectable text, or paste the resume text manually.",
            "",
        )

    filename = _file_label(file)
    recommended, _scores, bars, notes = _score_resume(final_text, role)
    if not recommended:
        return filename, "", "", bars, notes, ""
    return (
        filename,
        final_text.strip(),
        recommended.replace("_", " ").title(),
        bars,
        notes,
        "",
    )


def _file_label(value: Any) -> str:
    name = _as_path(value)
    if not name:
        return "No file selected"
    import os
    size = os.path.getsize(name) if os.path.exists(name) else 0
    return f"{name}  <span class='hf-hint'>({size:,} bytes)</span>"


def _run_bias_test(role: str, text: str) -> tuple[str, str, list[list[str]], str]:
    """Score the original and gender-swapped text with the role's B0 and B3."""
    if not text or not text.strip():
        return "", "", [], "Please enter resume text."

    try:
        result = score_pair(text, role, results_dir=RESULTS_ROOT)
    except MissingCheckpoint as exc:
        return "", "", [], f"{exc}\n\nRun:\n  {TRAIN_COMMAND}"
    except ValueError as exc:
        return "", "", [], str(exc)

    notes = [aggregate_markdown(role)]
    if not result.cues:
        notes.append("No gender cues found; the swap leaves the text unchanged.")
    if result.truncated:
        notes.append(
            f"Input is {result.n_tokens} tokens; the model truncates long inputs."
        )

    return highlight_original(result), highlight_swapped(result), result_table(result), "\n\n".join(notes)


def _fmt(stats: Mapping[str, Mapping[str, float]], variant: str, key: str) -> str:
    """Format one aggregate value, or a dash when it is absent."""
    value = stats.get(variant, {}).get(key)
    if value is None:
        return DASH
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DASH
    if math.isnan(number):
        return DASH
    return f"{number:.3f}"


def _run_fairness_audit(role: str) -> tuple[Any, ...]:
    """Load and display the measured fairness audit data for a role."""
    stats = aggregate(role, RESULTS_ROOT)

    if not stats:
        empty_note = (
            f"No aggregate results available for {role}. Build them with:\n"
            f"  {TRAIN_COMMAND}"
        )
        return (*([DASH] * 24), empty_note, empty_note)

    numeric = _to_number

    outputs: list[Any] = []
    # A. Model performance: HTML cards (text strings)
    for variant in ("B0", "B3"):
        outputs.extend(
            _fmt(stats, variant, key) for key in ("accuracy", "f1", "precision", "recall")
        )
    # B. Demographic parity: gr.Number components (numeric values)
    for variant in ("B0", "B3"):
        outputs.extend(
            numeric(stats, variant, key) for key in ("sr_m", "sr_f", "dpd")
        )
    # C. Equal opportunity: gr.Number components (numeric values)
    for variant in ("B0", "B3"):
        outputs.extend(
            numeric(stats, variant, key) for key in ("tpr_m", "tpr_f", "eod")
        )
    # D. Counterfactual consistency: gr.Number components (numeric values)
    for variant in ("B0", "B3"):
        outputs.extend(
            numeric(stats, variant, key) for key in ("flip_rate", "mean_abs_score_delta")
        )

    b0_f1 = stats.get("B0", {}).get("f1", float("nan"))
    b3_f1 = stats.get("B3", {}).get("f1", float("nan"))
    if math.isnan(float(b0_f1)) or math.isnan(float(b3_f1)):
        summary = "F1 is unavailable for one of the models, so the trade-off cannot be stated."
        bars = '<div class="hf-note">F1 is unavailable for one of the models, so the fairness-vs-performance chart is hidden.</div>'
    else:
        change = float(b3_f1) - float(b0_f1)
        direction = "decrease" if change < 0 else "increase"
        summary = (
            f"**Analysis:** the debiased model (B3) shows a {direction} in F1 of "
            f"{abs(change):.4f} compared to the baseline (B0). Also measured for "
            f"{role.replace('_', ' ').title()}: |EOD| {_fmt(stats, 'B0', 'eod')} → "
            f"{_fmt(stats, 'B3', 'eod')}, mean |score delta| "
            f"{_fmt(stats, 'B0', 'mean_abs_score_delta')} → "
            f"{_fmt(stats, 'B3', 'mean_abs_score_delta')}. These numbers come from the "
            f"held-out test split in `results/hirefair2/summary.csv` "
            f"(450 rows per role, seed 0)."
        )
        b0_eod = _fmt(stats, "B0", "eod")
        b3_eod = _fmt(stats, "B3", "eod")
        bars = (
            f'<div class="hf-card">'
            f'<div class="hf-card-title">Model F1 vs |EOD| gap</div>'
            f'<div class="hf-bar-track" style="margin-top:10px;">'
            f'<span class="hf-bar-label" style="width:120px;">B0 (baseline)</span>'
            f'<span style="flex:1; min-width:120px;">'
            f'<div style="height:18px; background:#26334D; border-radius:4px; overflow:hidden;">'
            f'<span style="display:block; width:{max(1,min(100,int(round(float(b0_f1)*100))))}%; background:#3b82f6; height:100%;"></span>'
            f'</div>'
            f'</span>'
            f'<span class="hf-bar-val" style="width:80px; text-align:right;">{_fmt(stats, "B0", "f1")}</span>'
            f'</div>'
            f'<div class="hf-bar-track" style="margin-top:8px;">'
            f'<span class="hf-bar-label" style="width:120px;">B3 (debiased)</span>'
            f'<span style="flex:1; min-width:120px;">'
            f'<div style="height:18px; background:#26334D; border-radius:4px; overflow:hidden;">'
            f'<span style="display:block; width:{max(1,min(100,int(round(float(b3_f1)*100))))}%; background:#22c55e; height:100%;"></span>'
            f'</div>'
            f'</span>'
            f'<span class="hf-bar-val" style="width:80px; text-align:right;">{_fmt(stats, "B3", "f1")}</span>'
            f'</div>'
            f'<div class="hf-note" style="margin-top:8px;">'
            f'|EOD| gap: {b0_eod} (B0) → {b3_eod} (B3). '
            f'Lower is fairer; this chart only displays the F1 side of that trade-off.'
            f'</div>'
            f'</div>'
        )
    outputs.append(bars)
    outputs.append(summary)
    assert len(outputs) == 26, f"_run_fairness_audit produced {len(outputs)} outputs, expected 26"
    return tuple(outputs)


def _to_number(
    stats: Mapping[str, Mapping[str, float]], variant: str, key: str
) -> float:
    """Return a numeric value for a gr.Number output component.

    ``gr.Number`` expects a numeric type, not a formatted string. This helper
    keeps the Fairness Audit numeric fields typed correctly while the HTML cards
    above keep using ``_fmt`` for display.
    """
    value = stats.get(variant, {}).get(key)
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(number):
        return float("nan")
    return number


# ---------------------------------------------------------------------------
# Tab 1: Resume Analyzer
# ---------------------------------------------------------------------------


def tab1_resume_analyzer() -> gr.Blocks:
    """Build the Resume Analyzer tab."""
    file_state = gr.State()

    with gr.Blocks(title="HireFair - Resume Analyzer", elem_classes=["hf-app"]) as tab:
        # Header: compact single header, no duplicate headings
        with gr.Row(elem_classes="hf-header"):
            gr.HTML('<div class="hf-logo">HF</div>')
            with gr.Column(elem_classes="hf-column"):
                gr.Markdown("**HireFair**", elem_classes=["hf-wordmark"])
                gr.Markdown("Responsible AI Recruitment", elem_classes=["hf-sub"])
            gr.HTML(
                "<div class=\"hf-research\">Research Prototype</div>",
                elem_classes=["hf-column"],
            )

        # Hero heading (replaces the redundant '# HireFair' + subtitle)
        gr.Markdown(
            "Find the right role.",
            elem_classes=["hf-h1"],
        )
        gr.Markdown(
            "Upload a resume and explore AI-generated job recommendations.",
            elem_classes=["hf-subtitle"],
        )

        # Desktop two-column layout
        with gr.Row(elem_classes="hf-grid"):
            # Left — upload area
            with gr.Column(elem_classes="hf-card"):
                gr.Markdown('<div class="hf-card-title">Upload resume</div>', elem_classes=["hf-card-title"])

                upload_zone = gr.HTML(
                    """
                    <div class="hf-upload">
                      <div class="hf-icon">&#128190;</div>
                      <div class="hf-upload-title">Upload PDF or DOCX</div>
                      <div class="hf-upload-sub">or drag &amp; drop a file here</div>
                    </div>
                    """,
                    elem_classes=["hf-upload"],
                )
                gr.HTML(
                    "<div class=\"hf-hint\">PDF and DOCX with selectable text are supported.</div>",
                    elem_classes=["hf-upload-sub"],
                )
                upload_status = gr.HTML(
                    '<div class="hf-hint" style="min-height:18px;">No file selected</div>',
                    elem_classes=["hf-hint"],
                )

                with gr.Row():
                    replace_btn = gr.Button("Replace file", variant="secondary", elem_classes=["hf-btn"])
                    analyze_btn = gr.Button("Analyze Resume", variant="primary", elem_classes=["hf-btn"])

                # Collapsible paste field
                with gr.Accordion("Or paste resume text", open=False, elem_classes=["hf-card"]):
                    text_box = gr.Textbox(
                        label="Resume text",
                        lines=6,
                        placeholder="Paste resume text here...",
                    )

                # Status note
                results_note = gr.Markdown("")

                # Upload component (referenced below for the analyze handler). The
                # HTML zone above only displays the label; this component owns the
                # file input and the drag-and-drop behaviour.
                upload_component = gr.File(
                    label="Upload PDF or DOCX resume",
                    file_types=[".pdf", ".docx"],
                )

            # Right — results area
            with gr.Column(elem_classes="hf-card"):
                gr.Markdown('<div class="hf-card-title">Results</div>', elem_classes=["hf-card-title"])

                extracted_text = gr.Textbox(
                    label="Extracted Resume Text",
                    lines=4,
                    interactive=False,
                    elem_classes=["hf-note"],
                )
                recommended_role = gr.Textbox(
                    label="Recommended Role",
                    interactive=False,
                    elem_classes=["hf-value"],
                )
                score_chart = gr.HTML(value="")
                notes = gr.Markdown("")

                with gr.Row():
                    label_card = gr.HTML(
                        '<div class="hf-card"><div class="hf-card-title">Independent role-model scores</div></div>',
                        elem_classes=["hf-card"],
                    )

        # The upload button clears state back to None (replace / re-upload).
        replace_btn.click(
            lambda: None,
            inputs=[],
            outputs=[file_state],
        )

        def on_analyze(
            f: Any,
            text: str,
            role: str,
        ):
            return _run_resume_analysis(f, text, role)

        role_dropdown = gr.Dropdown(
            choices=list(DEMO_ROLES),
            value="software_engineer",
            label="Target Role",
        )
        text_box = gr.Textbox(
            label="Resume text",
            lines=6,
            placeholder="Paste resume text here...",
        )
        analyze_btn.click(
            on_analyze,
            inputs=[upload_component, text_box, role_dropdown],
            outputs=[upload_status, extracted_text, recommended_role, score_chart, notes, results_note],
        )

        # Also trigger when text changes (preserves the previous flow when the
        # user swaps to pasting text only).
        text_box.input(
            on_analyze,
            inputs=[upload_component, text_box, role_dropdown],
            outputs=[upload_status, extracted_text, recommended_role, score_chart, notes, results_note],
        )

    return tab


# (Placeholder helpers that the UI references; implemented as real widgets
#  inside build_app so Gradio can wire them. They are replaced by real
#  component objects at runtime via Python's late binding in the closure.)


# ---------------------------------------------------------------------------
# Tab 2: Gender Bias Test
# ---------------------------------------------------------------------------


def tab2_gender_bias_test() -> gr.Blocks:
    """Build the Gender Bias Test tab."""
    with gr.Blocks(title="HireFair - Gender Bias Test", elem_classes=["hf-app"]) as tab:
        # Header
        with gr.Row(elem_classes="hf-header"):
            gr.HTML('<div class="hf-logo">HF</div>')
            with gr.Column(elem_classes="hf-column"):
                gr.Markdown("**HireFair**", elem_classes=["hf-wordmark"])
                gr.Markdown("Responsible AI Recruitment", elem_classes=["hf-sub"])
            gr.HTML(
                "<div class=\"hf-research\">Research Prototype</div>",
                elem_classes=["hf-column"],
            )

        gr.Markdown(
            "Does changing gender cues change the prediction?",
            elem_classes=["hf-h1"],
        )
        gr.Markdown(
            "Compare otherwise identical resumes using the baseline (B0) and "
            "adversarially trained (B3) models.",
            elem_classes=["hf-subtitle"],
        )

        with gr.Row(elem_classes="hf-grid"):
            with gr.Column(elem_classes="hf-card"):
                gr.Markdown('<div class="hf-card-title">Run the test</div>', elem_classes=["hf-card-title"])
                role_dropdown = gr.Dropdown(
                    choices=list(DEMO_ROLES),
                    value="software_engineer",
                    label="Target Role",
                )
                text_box = gr.Textbox(
                    label="Resume Text",
                    lines=4,
                    placeholder="Paste or paste the text from the Resume Analyzer tab...",
                )
                with gr.Row():
                    load_example_btn = gr.Button("Load Example", variant="secondary", elem_classes=["hf-btn"])
                    run_bias_test_btn = gr.Button("Run Gender Bias Test", variant="primary", elem_classes=["hf-btn"])
            with gr.Column(elem_classes="hf-card"):
                gr.Markdown('<div class="hf-card-title">Outputs</div>', elem_classes=["hf-card-title"])
                original_html = gr.HTML(label="Original Resume Text")
                swapped_html = gr.HTML(label="Gender-Swapped Resume Text")
                bias_table = gr.Dataframe(
                    headers=["Model", "Original Score", "Original Decision", "Swapped Score", "Swapped Decision", "Delta", "Changed?"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Model Comparison",
                )
                bias_notes = gr.Markdown("")

        # Expandable example section
        with gr.Accordion("Explore Test Examples", open=False, elem_classes=["hf-card"]):
            gr.Markdown(
                "Curated examples are selected to illustrate the counterfactual test. "
                "The aggregate flip rate for the role is shown next to each example.",
                elem_classes=["hf-hint"],
            )
            example_dropdown = gr.Dropdown(
                choices=list(DEMO_ROLES),
                value="software_engineer",
                label="Select Role for Examples",
            )
            example_notes = gr.Markdown("")

        examples_by_role = {
            "software_engineer": [
                {
                    "text": "He led the migration of a monolith to containerised services. He works with Kubernetes, distributed systems, Java, Python, Go and TypeScript.",
                    "note": "Example illustrating B0 score change with gender cue swap.",
                },
                {
                    "text": "She earned a BEng in Information Technology. She represents the top of her class.",
                    "note": "Example with female-coded language.",
                },
            ],
            "data_analyst": [
                {
                    "text": "She analyzed datasets and built dashboards to track business metrics.",
                    "note": "Example illustrating data analyst role prediction.",
                },
                {
                    "text": "He managed data pipelines and ensured data quality.",
                    "note": "Example with male-coded language.",
                },
            ],
            "machine_learning_engineer": [
                {
                    "text": "He built and trained neural networks for image classification.",
                    "note": "Example illustrating ML engineer role prediction.",
                },
                {
                    "text": "She developed machine learning models for predictive analytics.",
                    "note": "Example with female-coded language.",
                },
            ],
        }

        def load_example_text(role: str, index: float):
            entries = examples_by_role.get(role, [])
            if not entries:
                return "", ""
            idx = int(index) % len(entries)
            entry = entries[idx]
            return entry["text"], f"Example {idx + 1} of {len(entries)}: {entry['note']}"

        load_example_btn.click(
            load_example_text,
            inputs=[example_dropdown, gr.State(value=0)],
            outputs=[text_box, example_notes],
        )

        def on_bias_test(role: str, text: str):
            return _run_bias_test(role, text)

        run_bias_test_btn.click(
            on_bias_test,
            inputs=[role_dropdown, text_box],
            outputs=[original_html, swapped_html, bias_table, bias_notes],
        )

        # Also trigger when example changes
        example_dropdown.input(
            load_example_text,
            inputs=[example_dropdown, gr.State(value=0)],
            outputs=[text_box, example_notes],
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 3: Fairness Audit
# ---------------------------------------------------------------------------


def tab3_fairness_audit() -> gr.Blocks:
    """Build the Fairness Audit tab. All metric fields are read-only."""
    with gr.Blocks(title="HireFair - Fairness Audit", elem_classes=["hf-app"]) as tab:
        # Header
        with gr.Row(elem_classes="hf-header"):
            gr.HTML('<div class="hf-logo">HF</div>')
            with gr.Column(elem_classes="hf-column"):
                gr.Markdown("**HireFair**", elem_classes=["hf-wordmark"])
                gr.Markdown("Responsible AI Recruitment", elem_classes=["hf-sub"])
            gr.HTML(
                "<div class=\"hf-research\">Research Prototype</div>",
                elem_classes=["hf-column"],
            )

        gr.Markdown(
            "Fairness, measured.",
            elem_classes=["hf-h1"],
        )
        gr.Markdown(
            "Explore model performance and differences across the evaluated gender "
            "groups. All values are read-only outputs of the held-out test split.",
            elem_classes=["hf-subtitle"],
        )

        with gr.Row(elem_classes="hf-grid"):
            with gr.Column(elem_classes="hf-card"):
                gr.Markdown('<div class="hf-card-title">Audit controls</div>', elem_classes=["hf-card-title"])
                audit_role_dropdown = gr.Dropdown(
                    choices=list(DEMO_ROLES),
                    value="software_engineer",
                    label="Select Role",
                )
                audit_button = gr.Button("Load Aggregate Results", variant="primary")
            with gr.Column(elem_classes="hf-card"):
                aggregate_status = gr.Markdown("")

        # A. Model performance — metric cards, read-only
        gr.Markdown("## A. Model Performance", elem_classes=["hf-h2"])
        with gr.Row(elem_classes="hf-metrics"):
            def card(name: str, value: str = DASH) -> gr.HTML:
                return gr.HTML(
                    f"""<div class="hf-metric">
                      <div class="hf-metric-name">{name}</div>
                      <div class="hf-metric-value">{value}</div>
                      <div class="hf-metric-unit"></div>
                    </div>""",
                    elem_classes=["hf-metric"],
                )

            b0_acc = card("Accuracy (B0)")
            b0_f1 = card("F1 (B0)")
            b0_precision = card("Precision (B0)")
            b0_recall = card("Recall (B0)")
            b3_acc = card("Accuracy (B3)")
            b3_f1 = card("F1 (B3)")
            b3_precision = card("Precision (B3)")
            b3_recall = card("Recall (B3)")

        # B. Demographic parity
        gr.Markdown("## B. Demographic Parity", elem_classes=["hf-h2"])
        with gr.Row(elem_classes="hf-metrics"):
            b0_dpd_m = gr.Number(label="Selection Rate M", value=DASH, interactive=False)
            b0_dpd_f = gr.Number(label="Selection Rate F", value=DASH, interactive=False)
            b0_dpd_gap = gr.Number(label="DP Gap (F - M)", value=DASH, interactive=False)
            b3_dpd_m = gr.Number(label="Selection Rate M", value=DASH, interactive=False)
            b3_dpd_f = gr.Number(label="Selection Rate F", value=DASH, interactive=False)
            b3_dpd_gap = gr.Number(label="DP Gap (F - M)", value=DASH, interactive=False)

        # C. Equal opportunity
        gr.Markdown("## C. Equal Opportunity", elem_classes=["hf-h2"])
        with gr.Row(elem_classes="hf-metrics"):
            b0_tpr_m = gr.Number(label="TPR M", value=DASH, interactive=False)
            b0_tpr_f = gr.Number(label="TPR F", value=DASH, interactive=False)
            b0_eod_gap = gr.Number(label="EOD Gap (F - M)", value=DASH, interactive=False)
            b3_tpr_m = gr.Number(label="TPR M", value=DASH, interactive=False)
            b3_tpr_f = gr.Number(label="TPR F", value=DASH, interactive=False)
            b3_eod_gap = gr.Number(label="EOD Gap (F - M)", value=DASH, interactive=False)

        # D. Counterfactual consistency
        gr.Markdown("## D. Counterfactual Consistency", elem_classes=["hf-h2"])
        with gr.Row(elem_classes="hf-metrics"):
            b0_flip = gr.Number(label="Flip Rate", value=DASH, interactive=False)
            b0_score_delta = gr.Number(label="Mean |Score Delta|", value=DASH, interactive=False)
            b3_flip = gr.Number(label="Flip Rate", value=DASH, interactive=False)
            b3_score_delta = gr.Number(label="Mean |Score Delta|", value=DASH, interactive=False)

        # E. Fairness vs performance
        gr.Markdown("## E. Fairness vs Performance", elem_classes=["hf-h2"])
        fva_bars = gr.HTML(
            '<div class="hf-card"><div class="hf-card-title">Model F1 vs fairness gap</div></div>'
        )
        fva_note = gr.Markdown("")

        def on_audit(role: str):
            return _run_fairness_audit(role)

        audit_button.click(
            on_audit,
            inputs=[audit_role_dropdown],
            outputs=[
                b0_acc, b0_f1, b0_precision, b0_recall,
                b3_acc, b3_f1, b3_precision, b3_recall,
                b0_dpd_m, b0_dpd_f, b0_dpd_gap,
                b3_dpd_m, b3_dpd_f, b3_dpd_gap,
                b0_tpr_m, b0_tpr_f, b0_eod_gap,
                b3_tpr_m, b3_tpr_f, b3_eod_gap,
                b0_flip, b0_score_delta,
                b3_flip, b3_score_delta,
                fva_bars,
                fva_note,
            ],
        )

    return tab


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build the full HireFair application with three tabs."""
    with gr.Blocks(
        title="HireFair",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
        elem_classes=["hf-app"],
    ) as app:
        # Global design-system CSS (dark theme, clean labels, responsive)
        gr.HTML(f"<style>{_CSS}</style>")

        # Single compact header
        with gr.Row(elem_classes="hf-header"):
            gr.HTML('<div class="hf-logo">HF</div>')
            with gr.Column(elem_classes="hf-column"):
                gr.Markdown("**HireFair**", elem_classes=["hf-wordmark"])
                gr.Markdown("Responsible AI Recruitment", elem_classes=["hf-sub"])
            gr.HTML(
                "<div class=\"hf-research\">Research Prototype</div>",
                elem_classes=["hf-column"],
            )

        with gr.Tabs() as main_tabs:
            # Tab 1: Resume Analyzer
            with gr.Tab("Resume Analyzer"):
                tab1_resume_analyzer()

            # Tab 2: Gender Bias Test
            with gr.Tab("Gender Bias Test"):
                tab2_gender_bias_test()

            # Tab 3: Fairness Audit
            with gr.Tab("Fairness Audit"):
                tab3_fairness_audit()

    return app


def main() -> None:
    """Launch the HireFair application."""
    app = build_app()
    app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
