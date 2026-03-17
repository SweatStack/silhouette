# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.4.0] - 2026-03-17

### Added

- `fitting` parameter on `TwoParameterRegressor` with `"nonlinear"` (default) and `"work_duration"` options. The work-duration method linearizes the model to W = W' + CP·t and fits via OLS, minimizing error in work space rather than power space.
- Interactive playground at [silhouette.sweatstack.no](https://silhouette.sweatstack.no) for fitting models in the browser.
- `FPCARegressor` for data-driven power-duration modelling (Puchowicz & Skiba, 2025).
- `percentiles()` and `z_scores()` on `FPCARegressor` for population comparison.
- Pre-fitted population model shipped with the library.
- `fpca/` directory with training data and refitting scripts.
- `PowerDurationDisplay` for plotting data with model curves (sklearn Display pattern).
- `ModeOfVarianceDisplay` for FPCA mode of variance plots.
- `matplotlib` as optional dependency (`silhouette[plotting]`).
- `DEVELOPMENT.md` with contributor setup and FPCA refitting instructions.

## [0.3.0] - 2026-03-17

### Added

- Public `curve_inverse()` class method on all models for time-to-exhaustion with known parameters.
- `predict_inverse()` for time-to-exhaustion on fitted models.

## [0.2.0] - 2026-03-17

### Added

- `TwoParameterRegressor` for the classic critical power model (Monod & Scherrer, 1965).
- `ThreeParameterRegressor` adding maximum instantaneous power (Morton, 1996).
- `OmniDurationRegressor` for full-range power-duration modelling (Puchowicz et al., 2020).
- Public `curve()` static method on all models for evaluating with known parameters.
- Configurable optimization bounds and initial parameters.


## [0.1.0] - 2026-03-17

### Added
- Initializes project.
