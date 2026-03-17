# Development

## Setup

```bash
uv sync
```

## Tests

```bash
make test
```

## Refitting the FPCA population model

The `FPCARegressor` ships with a pre-fitted population model (`src/silhouette/_fpca_model.npz`). This model was trained on 2000 cyclists from the GoldenCheetah Open Data project.

Everything needed to reproduce or update this model lives in `fpca/`:

```
fpca/
    fit.py             # Fitting script
    GCdatcrop3.csv     # Training data
    GCdat2time.csv     # Time grid
    README.md          # Details
```

To refit:

```bash
python fpca/fit.py
```

This writes the model artifact to `src/silhouette/_fpca_model.npz`. Run `make test` afterwards to verify.

The original R script (`Reviewed_working_Study_FPCA.R`) and data are available at https://drive.google.com/drive/folders/1D750xiaSualEaINF6LztOQs-zyqdDgiJ. The R script is not committed to git as it does not have a license, but can be placed in `fpca/` locally for reference.

The `fpca/` directory is not included in published wheels. It exists only in the source repository.
