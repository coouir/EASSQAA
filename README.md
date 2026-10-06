# Error Analysis of SAR Ship Question Answering Agents (EASSQAA)

EASSQAA is the acronym of the paper title, and the repository name follows it. The Python package and command keep the short name `sarqa`.

This repository is an experiment repository that analyzes **where errors come from** in LLM agents that detect ships in SAR (synthetic aperture radar) images and answer questions by calling the detection results through tools. Each wrong answer is split into an "input error" (the boxes given by the detector were wrong) and an "agent error" (the agent mishandled the boxes it received), and each agent error is classified by the stage where the agent first deviated (question interpretation, tool selection/calls, result reading, planning/branching, calculation, answer formatting, execution failure). The package and command name is `sarqa`.

Code and data for an error analysis of tool-using LLM agents that answer questions about SAR ship-detection results.

## Study design

| Item | Description |
|---|---|
| Data | HRSID (ship class only), 5,604 images of 800×800. They were cut with overlap from 136 original scenes, so the images were re-split by scene group into `det_train` / `det_val` / `dev` / `test` (88 / 13 / 10 / 23 groups) |
| Detector | Faster R-CNN (torchvision `fasterrcnn_resnet50_fpn`, starting from COCO pretraining, 2 classes). The score threshold is the F1-maximizing value on `det_val` (0.96) |
| 5 tools | `detect_ships`, `spatial_query`, `get_metadata`, `image_stats`, `calc` (`docs/tools.md`, `src/sarqa/tools/`) |
| Questions | 23 templates, types L1–L5, 360 test questions (`docs/dev_questions.md` lists the dev questions). The gold solution and the tools use the same functions |
| Agents | Qwen3-8B (`qwen3:8b`, Ollama), text-only input. Two methods: **stepwise planning** (one tool call per step) and **batch planning** (the whole plan is written first and then executed) |
| Conditions | 11 (`configs/conditions.yaml`): label boxes, detected boxes, injection of miss / false-positive / localization errors into label boxes, correction of one error type in detected boxes, re-run (variability), and batch planning with label and detected boxes. 11 conditions × 360 questions = 3,960 runs |
| Classification | Wrong answers are split into input only / agent only / both, and the first-deviation stage of each agent error is recorded (SPEC §11, `src/sarqa/classify/`) |

## Key results

- Accuracy: 79.2% with label boxes (stepwise planning) → 63.9% with detected boxes. Input errors are involved in 60% of the wrong answers
- With label boxes, the accuracy difference between stepwise and batch planning is 6.1 percentage points
- Agreement between the automatic error classification and the AI judgments was low (κ=0.137): the stage distribution is an exploratory result

## Environment

- Python 3.10 or later (`pyproject.toml`); development and runs use the conda env `easqaa` (Python 3.10.21).
- Detector training and inference need a GPU (development machine: RTX 3080 10GB, torch 2.7.1+cu118). Install torch and torchvision with `.[detector]` (`docs/env_notes.md`).
- Running the agents (`sarqa run`) needs an Ollama server. The settings are in `llm` of `configs/default.yaml`: model `qwen3:8b`, address `http://localhost:11434`, `think: false`, `temperature: 0.2`, `num_ctx: 16384`. Ollama on the development machine is 0.34.0 (`docs/env_notes.md`).
- Tests that need data are marked `@pytest.mark.data`, and tests that need a GPU or Ollama are marked `@pytest.mark.gpu`. CI runs only the tests without these two marks.

## Installation and tests

```bash
conda create -n easqaa python=3.10
conda activate easqaa
pip install -e ".[dev]"        # add ".[detector]" for torch/torchvision
pytest -m "not data and not gpu"
```

## Data

HRSID is **not included** in this repository. Download it from its original source under the HRSID license and place it in `data/hrsid/` (file structure: `docs/data_notes.md`). `data/` and `outputs/` are in `.gitignore`.

HRSID: S. Wei, X. Zeng, Q. Qu, M. Wang, H. Su, J. Shi, 'HRSID: A High-Resolution SAR Images Dataset for Ship Detection and Instance Segmentation,' IEEE Access, vol. 8, pp. 120234–120254, 2020.

The questions, run records, classification results, and aggregate tables released with the paper are in [`release/`](release/).

The files in `splits/` (`scenes.json`, `splits.json`, `splits_report.json`) contain only image file names, scene group numbers, split and inshore/offshore tags, and the number of boxes per image. They contain neither the images themselves nor box coordinates.

## Reproduction

The order and the commands are exactly as written in `PROTOCOL.md`, `docs/freeze_checklist.md`, `docs/env_notes.md`, `configs/default.yaml`, and `scripts/`.

1. Build the data splits (`docs/data_notes.md` §10–11): `sarqa data scenes`, `sarqa data splits`. The resulting `splits/*.json` are in the repository; compare their SHA-256 with `PROTOCOL.md` §1 and `sha256sum splits/*.json`.
2. Train the detector: `bash scripts/train_detector.sh` (trains on `det_train`, records `det_val` AP50). Then build the inference cache with `sarqa detector infer` and set the score threshold from the `det_val` F1.
3. Measure the error-injection rates: `sarqa inject calibrate` (based on dev, `configs/injection.yaml`).
4. Generate the test questions and the injected/corrected boxes (once only): `sarqa questions generate --split test --allow-test`, `sarqa inject build --split test --allow-test`.
5. Freeze check: fill the hash rows of PROTOCOL.md with `sarqa freeze hashes --split test --write` and verify with `sarqa freeze check --split test`.
6. Main run: `sarqa run --conditions all --split test --out outputs/runs/` (needs Ollama; if interrupted, rerun the same command to resume).
7. Classification: `sarqa classify --runs outputs/runs/ --split test`.
8. Analysis: `python -m sarqa.analysis.main_run --runs outputs/runs --questions data/questions/test.json --out outputs/analysis` (usage: the header of `src/sarqa/analysis/main_run.py`).

The detector weights, the detection cache, the test question file, and the run records are not in the repository. The SHA-256 of the weights and data files is in `PROTOCOL.md` §1.

## Freeze point: `freeze-v1`

The `freeze-v1` tag is the point at which the experimental design, code, prompts, and classification rules were fixed **before the results were seen**. The test questions were generated only once, and the SHA-256 of the data files and the design decisions are recorded in `PROTOCOL.md`. The main run started from the commit of this tag, and at startup the runner checks that the working tree is clean, that no frozen file has changed since the tag, and that the hashes match. After that, `src/sarqa/` (except `analysis/`), `configs/`, `pyproject.toml`, and `PROTOCOL.md` are not modified. Deviations from the specification and changes after the freeze are recorded in `docs/deviations.md`.

## Validation and limitations

The first-deviation stage assigned by the automatic error classification was validated against the judgments of an independent AI assessor (second validation, 100 runs: 30 agreements, Cohen's kappa 0.137). The plan, method, results, and disagreement analysis are in [docs/validation_round2.md](docs/validation_round2.md). The detector score threshold was set on `det_val`, and test was only reported (`PROTOCOL.md` §4). The number of injected false positives was matched to the expected number of misses, so it does not mimic the false-positive rate of a real detector (`docs/deviations.md`).

## Citation

This repository contains the code and data of the paper 'Error Analysis of SAR Ship Question Answering Agents', submitted to the undergraduate paper competition of the 2026 Fall Conference of The Korean Institute of Broadcast and Media Engineers.

## License

The code and documents in this repository are under the [MIT License](LICENSE). The HRSID data, however, is under a separate license and is not included in this repository.
