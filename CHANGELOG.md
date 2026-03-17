# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-17

### Added

- `TwoParameterRegressor` for the classic critical power model (Monod & Scherrer, 1965).
- `ThreeParameterRegressor` adding maximum instantaneous power (Morton, 1996).
- `OmniDurationRegressor` for full-range power-duration modelling (Puchowicz et al., 2020).
- Public `curve()` static method on all models for evaluating with known parameters.
- `predict_inverse()` for time-to-exhaustion calculations.
- Configurable optimization bounds and initial parameters.


## [0.1.0] - 2026-03-17

### Added
- Initializes project.
