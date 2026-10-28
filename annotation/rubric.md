# Human evaluation rubric

Each row of the sheet is one **pair**: an original premise/hypothesis and a modified version
(`mod_type`: label_change, paraphrase or irrelevant_addition). For each side you see the
model's **Label**, its **Cited evidence** (quoted spans, or "(none)") and its **Explanation**.
You do not know which model or prompt produced it. Rate on your own; do not discuss items
with the other rater until both sheets are done.

Write answers exactly as given in the scale column (lower case). Leave a cell empty only if
you really cannot judge it, and say why in `comments`.

## Per output (fill in for `orig_` and `var_`)

| Column | Question | Scale |
|---|---|---|
| Q1 | If the explanation were true, would it support the **predicted** label? (Ignore whether the label is right.) | `yes` / `partly` / `no` |
| Q2 | Is the explanation correct about the two sentences? Judge the reasoning, not the writing. | `correct` / `partly` / `incorrect` |
| Q3 | Does it state something the sentences do not support (an invented fact or assumption)? | `yes` / `no` |
| Q4 | Is the cited evidence relevant to the decision? `n.a.` if nothing was cited. | `all` / `some` / `none` / `n.a.` |

## Per pair

| Column | Question | Scale |
|---|---|---|
| P1 | Did the explanation change appropriately from original to variant? For a label change it should address the changed words; for paraphrase / irrelevant addition it should keep the same reasoning. | `appropriate` / `stale` (kept the old reasoning when it should have changed) / `changed_when_it_shouldnt` / `n.a.` |
| P2 | Irrelevant addition only: does the variant treat the added text as decisive? Leave empty for other types. | `yes` / `no` |

## Failure tags (`failure_tags`, separate several with `;`)

- `correct_pred_invalid_expl`: correct label, but the explanation is wrong or doesn't support it
- `wrong_pred_fluent_expl`: wrong label with a confident, plausible-sounding explanation
- `pred_updated_expl_inconsistent`: the label changed but the explanation doesn't match it
- `unsupported_assumption`: an assumption turns neutral into entailment or contradiction
- `irrelevant_sensitive`: reacts to irrelevant details
- `missed_evidence_change`: ignores a meaningful change in the evidence
- `other`: describe in `comments`

## Examples of borderline calls

- Explanation says "a man is not a woman" for a contradiction where the premise says *person*: Q2 `incorrect`, Q3 `yes`.
- Neutral answer whose explanation says the hypothesis "contradicts" because the premise doesn't mention it: Q1 `no` (the text argues for contradiction), tag `unsupported_assumption` if the label was entailment/contradiction instead.
- Cited span exists but is irrelevant (e.g. "in the background") and the explanation relies on another part: Q4 `some` or `none`.

## Procedure

1. Calibration: both raters do `calibration.csv` (dev pairs), then discuss disagreements and clarify this file. Calibration items are not part of the results.
2. Rate your own sheet (`sheet_r1.csv` or `sheet_r2.csv`) independently.
3. `python -m cfx.human_eval agreement` reports kappa per question and writes `adjudicated.csv`, where disagreements are blank and both answers are shown next to them.
4. A third team member fills the blanks and the final `failure_tags`. Then run `python -m cfx.human_eval results`.
