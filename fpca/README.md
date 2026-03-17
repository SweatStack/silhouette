# FPCA Population Model

This directory contains everything needed to refit the FPCA population model from scratch.

## Contents

- `fit.py` - Python script that fits the population model and writes it to `src/silhouette/_fpca_model.npz`.
- `Reviewed_working_Study_FPCA.R` - Original R script from Puchowicz & Skiba (2025), for reference and cross-validation. Not committed to git (no license). See below for how to obtain it.
- `GCdatcrop3.csv` - Training data: 2445 athletes, 90 durations each. Extracted from the GoldenCheetah Open Data project.
- `GCdat2time.csv` - The 90-point time grid (1s to 7200s).

## Original source

The original R script and data are available at:
https://drive.google.com/drive/folders/1D750xiaSualEaINF6LztOQs-zyqdDgiJ

The R script does not have a license, so it is not committed to git. Download `Reviewed_working_Study_FPCA.R` from the link above and place it in this directory for reference.

## Refitting

```bash
python fpca/fit.py
```

This will overwrite `src/silhouette/_fpca_model.npz`. Run the test suite afterwards to verify.

## Note

This directory is not included in published wheels. It exists only in the source repository for reproducibility.

## Reference

Puchowicz, M. J., & Skiba, P. F. (2025). Functional Data Analysis of the Power-Duration Relationship in Cyclists. International Journal of Sports Physiology and Performance, 1(aop), 1-10.
