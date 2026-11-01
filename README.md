# Harpreet Singh AI Ensemble

**Maintainer:** Harpreet Singh  
**Type:** Research / Kaggle inference pipeline  
**Models:** Qwen3, OLMo2, SmolLM3  
**Methods:** two-stage LoRA, grouped Gaussian readouts, ensemble classification

## About

This repository reorganizes the supplied Kaggle notebook, *Two-Stage LoRA, Grouped Gaussian Readouts, OLMo2 and SmolLM3*, into individually inspectable Python modules and configuration files. It **preserves the original source code** instead of claiming it was newly written by the maintainer.

**Important:** This is a source-code archive and GitHub-ready structure, **not a verified standalone training/inference package**. Full execution requires the Kaggle environment, attached external model weights, datasets, and exact input paths expected by the original scripts. The GitHub CI only validates Python syntax and source integrity; it does not run GPU inference or establish the reported Kaggle score.

## Structure

- `run_candidate.py` — original candidate orchestration
- `run_baseline.py`, `run_baseline_with_groups.py` — baseline stages
- `gaussian_core.py`, `gaussian_integration.txt` — grouped Gaussian readout
- `compute_olmo.py`, `olmo_component.py`, `olmo_owned_path.py` — OLMo2 component
- `compute_smol.py`, `smol_component.py`, `smol_owned_path.py` — SmolLM3 component
- `finish_prediction.py` — prediction aggregation
- `candidate-config.json`, `olmo-spec.json`, `smol-spec.json` — original configuration
- `notebooks/original_kaggle_notebook.ipynb` — original notebook with cell outputs removed
- `THIRD_PARTY_NOTICES.md`, `LICENSE_APACHE_2_0.txt` — notices/license text shipped in the notebook
- `tools/verify_sources.py` — checks the extracted files against the notebook's embedded hashes

## Setup (Kaggle recommended)

1. Create a Kaggle notebook with a GPU and attach the datasets/model assets required by the original notebook.
2. Upload or clone this repository into the Kaggle environment.
3. Inspect `candidate-config.json` and the `/kaggle/input` references in the scripts; make sure the referenced files are attached.
4. From the repository root, run `python tools/verify_sources.py`. Hash warnings are expected for the two files listed under Known issues.
5. Only after checking the model assets and dependencies, run `python run_candidate.py` from the Kaggle working environment. This command is **not** expected to work on a bare GitHub Actions runner or CPU-only laptop.

For a local environment, install `pip install -r requirements.txt` first, but this list is a starting point rather than a fully locked reproducible environment.

## Known issues

- **Two declared hashes do not match their files.** The original notebook declares SHA-256 values for `baseline_source.py` and `candidate-config.json` that differ from the files' actual contents. Both files match the notebook's embedded text exactly, so the declared values themselves appear to be wrong. The author should confirm which version is correct. `tools/verify_sources.py` reports these as warnings; run it with `--strict` to fail on them.
- **A second copy of the notebook is at the repository root** (`nishant_0592_original.ipynb`). It differs from `notebooks/original_kaggle_notebook.ipynb`. Keep only one, after checking which one is the intended original.
- **Scripts depend on Kaggle paths** such as `/kaggle/input`, and no model weights or datasets are included.

## Attribution and license

Original notebook credits the following Kaggle works:

- [kirilltym — Two-Stage LoRA Gaussian Readout](https://www.kaggle.com/code/kirilltym/two-stage-lora-gaussian-readout?scriptVersionId=360050107)
- [Nishant Dahal — Two-Stage LoRA on the Public Pool](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool?scriptVersionId=359887828)
- [prvsiyan — The Prompt Is Not the Author](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=359792979)

Read the included `THIRD_PARTY_NOTICES.md` and `LICENSE_APACHE_2_0.txt` before redistributing. **The presence of an Apache license file in the notebook does not automatically establish that every bundled component and external model/data asset is Apache-licensed.** Verify the upstream rights before publishing this repository publicly. No new license is asserted for the combined work.

## Maintainer

Harpreet Singh — repository packaging and maintenance. Original code authorship and external sources remain credited above.
