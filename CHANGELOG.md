# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.6.0] - 2026-03-18

### Added

- `VDOTSpeedRegressor` implementing the Daniels-Gilbert (1979) VDOT model for running. Single-parameter model predicting equivalent race performances across distances (3 min – 2 hours).
- `VDOTPowerRegressor` (experimental) adapting the VDOT model to cycling power using 11.7 mL O2/W conversion. Requires `body_mass` parameter.
- Experimental model badges in the playground: models marked as experimental show a warning icon that opens a dialog explaining the model has not been validated for this sport/metric.

## [0.5.0] - 2026-03-18

### Added

- `ExpPowerRegressor` and `ExpSpeedRegressor` (speed experimental) implementing the exponential power-duration model (Hopkins et al., 1989): P(t) = (P_max - CP) · exp(-t/τ) + CP.
- `MinimalPowerPowerRegressor` and `MinimalPowerSpeedRegressor` (both experimental) implementing the Lambert W minimal power model (Mulligan et al., 2018).
- `MinimalPowerDisplay` for normalized minimal power model plots with reference endurance band, replicating Figure 2 from Mulligan et al. (2018). Works for both power and speed regressors.
- Speed (running) variants of all parametric models: `TwoParamCriticalSpeedRegressor`, `ThreeParamCriticalSpeedRegressor`, `OmniDomainSpeedRegressor` (experimental). Same formulas as power models with speed-appropriate parameter names (CS, D'), bounds, and defaults.
- `duration_range` parameter on all parametric models to restrict which data points are used for fitting. Exposes `duration_mask_` fitted attribute.
- Recommended duration range warnings: `TwoParamCriticalPowerRegressor` warns when data falls outside 2-15 minutes, `ThreeParamCriticalPowerRegressor` warns for data above 15 minutes.

### Changed

- Renamed all regressors to include the metrics: `TwoParamCriticalPowerRegressor`, `ThreeParamCriticalPowerRegressor`, `OmniDomainPowerRegressor`, `FPCAPowerRegressor`.

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
