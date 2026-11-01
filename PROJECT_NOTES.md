# Project notes

- Source: user-supplied `.ipynb` notebook.
- 27 embedded files extracted and checked against notebook contents. Note: the notebook itself contains mismatched declared SHA-256 values for `baseline_source.py` and `candidate-config.json`; these original discrepancies are preserved and must be resolved before running its original extraction cell.
- Original notebook preserved under `notebooks/` with outputs removed.
- No model weights, external datasets, Kaggle inputs, or credentials are bundled.
- GitHub CI intentionally does not execute heavyweight or externally dependent scripts.
- Some source scripts reference hardcoded `/kaggle/input` paths and may require Kaggle-specific package versions.
- Check third-party rights before public release.
