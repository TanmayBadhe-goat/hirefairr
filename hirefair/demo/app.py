"""Gradio demo. All logic lives in :mod:`hirefair.demo.core`.

Run:
    python -m hirefair.demo.app
"""

from __future__ import annotations

import gradio as gr

from hirefair.demo import core


def _example_payload(role: str):
    try:
        return core.examples(role), None
    except core.MissingCheckpoint as exc:
        return None, str(exc)


def load_example(role: str, index: int):
    """Cycle through curated examples."""
    payload, error = _example_payload(role)
    if payload is None:
        return "", index, 0, error or ""
    entries = payload["examples"]
    if not entries:
        return "", index, 0, "No curated examples for this role."
    index = int(index) % len(entries)
    entry = entries[index]
    note = (
        f"{payload.get('note', '')}\n\n"
        f"Example {index + 1} of {len(entries)}: selected to illustrate."
    )
    return entry["text"], index, len(entries), note


def score(role: str, text: str):
    """Score the original and swapped text."""
    if not text or not text.strip():
        return "", "", [], "Paste a bio or load an example."
    try:
        result = core.score_pair(text, role)
    except core.MissingCheckpoint as exc:
        return "", "", [], str(exc)
    notes = [core.aggregate_markdown(role)]
    if not result.cues:
        notes.append("No gender cues found; the swap leaves the text unchanged.")
    if result.truncated:
        notes.append(
            f"Input is {result.n_tokens} tokens; the model truncates at 128."
        )
    return core.highlight_original(result), core.highlight_swapped(result), core.result_table(result), "\n\n".join(notes)


def ranking_tab(role: str, pool_size: int, seed: int):
    """Build the ranking comparison."""
    try:
        payload = core.ranking(role, pool_size=int(pool_size), seed=int(seed))
    except core.MissingCheckpoint as exc:
        return "", str(exc)
    lines = [f"Pool: {payload['pool_size']} test candidates, seed {payload['seed']}.", ""]
    for variant, stats in payload["models"].items():
        lines.append(f"## {variant}")
        lines.append(
            f"Top-10 gender composition before swap: {stats['top_gender_before']['M']} M / "
            f"{stats['top_gender_before']['F']} F; after swap: "
            f"{stats['top_gender_after']['M']} M / {stats['top_gender_after']['F']} F."
        )
        lines.append(
            f"Top-10 changes: {stats['top_changes']}. Mean absolute rank shift: "
            f"{stats['mean_rank_shift']:.2f}. Pool flip rate: {stats['pool_flip_rate']:.2%}."
        )
        before = ", ".join(f"{row['id']} ({row['gender']})" for row in stats["top_before"])
        after = ", ".join(f"{row['id']} ({row['gender']})" for row in stats["top_after"])
        lines.append(f"Top 10 before: {before}")
        lines.append(f"Top 10 after: {after}")
        lines.append("")
    return "\n".join(lines), ""


def build_app() -> gr.Blocks:
    """Assemble the Gradio interface."""
    with gr.Blocks(title="HireFair demo") as demo:
        gr.Markdown(
            "# HireFair\n"
            "The same candidate, scored before and after swapping gender cues. "
            "Examples are selected to illustrate; the aggregate flip rate for the role "
            "is shown next to them so the typical case is visible too."
        )
        example_index = gr.State(0)
        example_total = gr.State(0)

        with gr.Tab("Score"):
            with gr.Row():
                role = gr.Dropdown(choices=list(core.ROLES), value="nurse", label="Role")
                load_button = gr.Button("Load example")
                score_button = gr.Button("Score", variant="primary")
            text = gr.Textbox(label="Bio or resume summary", lines=6)
            example_note = gr.Markdown("")
            with gr.Row():
                original_html = gr.HTML(label="Original")
                swapped_html = gr.HTML(label="Swapped")
            table = gr.Dataframe(
                headers=["Model", "Score (original)", "Decision", "Score (swapped)", "Decision", "Delta", "Flipped"],
                interactive=False,
                label="Model comparison",
            )
            notes = gr.Markdown("")

            load_button.click(
                load_example,
                inputs=[role, example_index],
                outputs=[text, example_index, example_total, example_note],
            )
            score_button.click(
                score,
                inputs=[role, text],
                outputs=[original_html, swapped_html, table, notes],
            )

        with gr.Tab("Ranking"):
            with gr.Row():
                rank_role = gr.Dropdown(choices=list(core.ROLES), value="nurse", label="Role")
                pool_size = gr.Slider(50, 500, value=200, step=50, label="Pool size")
                seed = gr.Number(value=42, precision=0, label="Seed")
                rank_button = gr.Button("Build ranking", variant="primary")
            ranking_markdown = gr.Markdown("")
            ranking_error = gr.Markdown("")
            rank_button.click(
                ranking_tab,
                inputs=[rank_role, pool_size, seed],
                outputs=[ranking_markdown, ranking_error],
            )

        with gr.Tab("Aggregate"):
            with gr.Row():
                aggregate_role = gr.Dropdown(choices=list(core.ROLES), value="nurse", label="Role")
                aggregate_button = gr.Button("Load aggregate")
            aggregate_text = gr.Markdown("")
            aggregate_button.click(
                lambda role_value: core.aggregate_markdown(role_value),
                inputs=[aggregate_role],
                outputs=[aggregate_text],
            )
    return demo


def main() -> None:
    """Launch the local Gradio app."""
    build_app().launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
