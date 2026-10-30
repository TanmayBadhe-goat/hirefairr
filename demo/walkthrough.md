# Demo walkthrough (about 5 minutes)

Run the app from the repository root:

```bash
python -m hirefair.demo.app
```

Terminal fallback:

```bash
python -m hirefair.demo.cli --role nurse
```

The app and the terminal call the same functions in `hirefair/demo/core.py`. Every
number shown is read from `results/` or produced by the loaded checkpoint.

## Spoken script

1. 0:00 Problem. An applicant tracking system scores text. Historical text carries
   gender signal in names, pronouns, organisations and word choice.
2. 0:30 Pick role `software_engineer`. Press "Load example" once. Point at the
   highlighted cues.
3. 1:15 Press "Score". Show the B0 row: the swapped text changes the score by the
   measured delta. If the decision flips, say so with the number.
4. 2:00 Point at the B3 row for the same example. The decision holds and the delta
   is smaller. Read both deltas from the table.
5. 2:30 Open the "Ranking" tab. Build the pool. Show the top 10 for B0 before and
   after the swap, then the B3 panel. Read the mean rank shift and the number of
   top-10 changes.
6. 3:30 Open "Aggregate". Read the flip rate for B0, B1 and B3. These are means
   over seeds, not the selected examples.
7. 4:15 Cost. Show F1 for B0 and B3 from Table T2 in the report. State the drop.
8. 4:45 Limits. Binary gender, biographies rather than resumes, occupation as a
   proxy for fit. Close on the ethical note: this is not a certification.

## Handling questions

- Selected examples are labelled "selected to illustrate". The aggregate flip
  rate sits next to them so the typical case is visible.
- If a checkpoint is missing, the app prints the exact command that produces it.
- If the text has no cues, the app says the swap leaves the text unchanged.
- If the input exceeds 128 tokens, it is truncated and the app says so.
