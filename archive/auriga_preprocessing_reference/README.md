# Auriga preprocessing reference

These scripts are preserved as provenance for the supplied Halo12 stellar
catalogue. They depend on site-specific Auriga paths and external analysis
packages, so they are not active project entry points.

- `extract_halo12_stars.py` records the original snapshot extraction,
  centering, alignment, and diagnostic workflow. Its binding-energy
  classification has been corrected to use `Potential + 0.5 * v^2`.
- `Elz_bin.py` is the original Zhu-style circularity reference supplied with
  the data and is retained verbatim.

Use `python -m experiments.prepare_auriga` for the maintained, tested project
pipeline.
