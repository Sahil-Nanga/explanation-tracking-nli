# Do LLM Explanations Track the Evidence?

Course project (Project 26, NLP, University of Oulu). We test whether the explanations an LLM gives for its
NLI predictions actually follow the evidence. Starting from e-SNLI, we make small controlled edits to each
example (a label-changing edit, a paraphrase and an irrelevant addition) and check whether the label and the
explanation change together when they should, and stay put when they shouldn't.

Models: Qwen3-235B-A22B-Instruct-2507 (M1) and Qwen3-30B-A3B-Instruct-2507 (M2), served through Nebius Token Factory.

Prompting conditions:

| Condition | Prompt |
|---|---|
| `c0` | label only |
| `c1` | label first, then explanation |
| `c2` | quote evidence first, then label |
| `c3` | structured JSON with quoted evidence spans |
| `c3n` | structured JSON without quotes |

On top of these we run a verify-and-revise step, a generic self-check baseline, an explanation-to-label probe,
a noise-floor rerun and cited/uncited span interventions. The write-up is in `reports/report.md`.

## Repository layout

```
configs/            models.json (endpoints, model ids), experiment.json (decoding, limits)
prompts/            prompt templates, frozen before the main runs (see CHANGELOG.md)
src/cfx/            the pipeline
jobs/               run_nebius.sh, plus Slurm scripts for CSC Roihu
annotate/           local web app for the manual annotation, and rater kits
annotation/         human evaluation rubric
mod_editor/         small editor used to write and check the modifications
data/annotations/   modifications, interventions, human ratings, phenomenon tags
data/processed/     dev and eval items built from e-SNLI
outputs/runs/       one folder per run (predictions.jsonl, manifest.json)
reports/            tables, figures, summary.md, report.md
tests/              unit and end-to-end tests with a fake OpenAI-compatible server
```

## Setup

Python 3.10+.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

The pipeline itself only needs the standard library; numpy, pandas and matplotlib are for data preparation and analysis.

The processed data is already in `data/processed/`. To rebuild it from scratch, clone e-SNLI next to this repo and run:

```bash
git clone https://github.com/OanaMariaCamburu/e-SNLI.git
venv/bin/python prepare_esnli.py --esnli_dir e-SNLI/dataset
PYTHONPATH=src venv/bin/python -m cfx.data
```

## Running the models

Set your Nebius key in the environment (don't put it in any file in the repo):

```bash
export NEBIUS_API_KEY=...
```

Smoke test, 10 dev items with M2:

```bash
bash jobs/run_nebius.sh smoke
```

Everything (dev, main, noise and revise for both models, then the probe with M1):

```bash
bash jobs/run_nebius.sh all
```

Responses are cached in `outputs/cache/`, so if a run stops you can just start it again. If you hit rate limits,
lower `concurrency` in `configs/models.json`. The full experiment is around 20k calls.

A single run:

```bash
PYTHONPATH=src venv/bin/python -m cfx.run --model m2 --condition c3 --items dev --limit 5 --run-id try_m2_c3
```

`--items` takes `dev`, `eval`, `interventions` or a path to a JSONL file. `--salt` gives a separate cache
namespace and `--refresh` ignores the cache.

Phases (`python -m cfx.phase --model <m> --phases a,b,c`):

| Phase | Runs | Run ids |
|---|---|---|
| `smoke` | c3 on 10 dev items | `smoke_<m>_c3` |
| `dev` | all conditions on dev | `dev_<m>_<c>` |
| `main` | all conditions on the 600 eval items | `main_<m>_<c>` |
| `noise` | c3 again in a separate cache namespace | `noise_<m>_c3` |
| `revise` | verifier and generic self-check on c2 and c3 | `revise_main_<m>_<c>_<method>` |
| `probe` | the model labels each explanation without seeing the label | `probe_<run>` |
| `interventions` | c2 and c3 on the cited/uncited edits | `int_<m>_<c>` |

Any OpenAI-compatible endpoint works: add an entry to `configs/models.json`, or set `CFX_BASE_URL`
(e.g. `http://localhost:11434/v1` for Ollama).

### Roihu (CSC)

The `roihu_m1` / `roihu_m2` entries run the models with vLLM on Roihu instead. Set your project in
`jobs/env.sh`, then on a login node:

```bash
bash jobs/download_models.sh
jobs/submit.sh roihu_m2 smoke --test
bash jobs/submit_all.sh
```

`download_models.sh` pins the model revisions in `configs/models.json`. Copy the results back with
`CFX_REMOTE=<user>@roihu-gpu.csc.fi:<outputs dir> bash jobs/fetch_results.sh`.

## Analysis

```bash
PYTHONPATH=src venv/bin/python -m cfx.analysis
```

Reads every run in `outputs/runs/` and writes `reports/tables/` (CSV and LaTeX), `reports/figures/` and
`reports/summary.md`: accuracy and macro-F1 with cluster-bootstrap CIs, counterfactual pair categories,
span validity and highlight F1, paired comparisons (bootstrap, McNemar, Holm), revision flips, probe
consistency, the intervention 2x2 table, the noise floor and token usage.

## Manual annotation

```bash
venv/bin/python annotate/app.py
```

This opens a local page (http://127.0.0.1:8771) with a view per role: Rater 1, Rater 2, adjudicator and a
joint view. Answers are saved straight into the CSVs in `data/annotations/`, with a backup taken the first
time each file is opened. Set `CFX_ANNOTATIONS=<folder>` to work on a copy instead.

For raters working on their own machine:

```bash
venv/bin/python annotate/make_kit.py r2 --send-to <name>
venv/bin/python annotate/merge_kit.py <returned zip> --dry-run
```

A kit contains only that rater's sheets, with the other rater's answers removed.

### Interventions

```bash
PYTHONPATH=src venv/bin/python -m cfx.faithfulness select --base-run main_m1_c3 --n 40
```

Fill in the edited sentences and both raters' labels in `data/annotations/interventions.csv` (or in the
web app), agree on `*_gold_final`, then:

```bash
PYTHONPATH=src venv/bin/python -m cfx.faithfulness build
bash jobs/run_nebius.sh interventions
```

### Human evaluation

```bash
PYTHONPATH=src venv/bin/python -m cfx.human_eval sample
PYTHONPATH=src venv/bin/python -m cfx.human_eval agreement
PYTHONPATH=src venv/bin/python -m cfx.human_eval results
```

`sample` writes the blinded sheets to `data/annotations/human_eval/`. `agreement` computes Cohen's kappa and
writes `adjudicated.csv` with the disagreements left blank; fill those in before running `results`.
The rubric is `annotation/rubric.md`.

## Tests

```bash
PYTHONPATH=src venv/bin/python -m unittest discover -s tests -v
```

The end-to-end tests run every phase against `tests/fake_server.py`, so no API key or GPU is needed.
To try the pipeline by hand:

```bash
venv/bin/python tests/fake_server.py --uds outputs/fake.sock &
CFX_UDS=$PWD/outputs/fake.sock PYTHONPATH=src venv/bin/python -m cfx.phase --model fake --phases smoke
```

Delete the `*_fake_*` runs afterwards so they don't end up in the analysis.

## Environment variables

| Variable | Use |
|---|---|
| `NEBIUS_API_KEY` | API key for the hosted models |
| `CFX_OUTPUTS` | where runs and the cache go (default `outputs/`) |
| `CFX_REPORTS` | where the analysis writes (default `reports/`) |
| `CFX_ANNOTATIONS` | annotation folder (default `data/annotations/`) |
| `CFX_BASE_URL`, `CFX_UDS` | endpoint for models without a fixed `base_url` |

## Notes

The hosted models can't be pinned to a weight revision and may be served quantized, so exact numbers can
drift between providers. 103 of the 300 modifications were first drafted with an LLM and then checked and
corrected by hand; this is recorded in the `provenance` column of `modifications_all.csv`.

e-SNLI is by Camburu et al. (2018) and is not included in this repository.
