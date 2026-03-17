import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting. "
        "Install it with: pip install silhouette[plotting]"
    ) from e


_TICK_CANDIDATES = [
    (1, "1s"), (2, "2s"), (3, "3s"), (5, "5s"),
    (10, "10s"), (15, "15s"), (20, "20s"), (30, "30s"),
    (60, "1min"), (120, "2min"), (180, "3min"), (300, "5min"),
    (600, "10min"), (900, "15min"), (1200, "20min"), (1800, "30min"),
    (3600, "1h"), (7200, "2h"),
]


def _set_duration_ticks(ax, min_t, max_t, target_count=8):
    """Set human-readable tick labels at natural durations."""
    candidates = [(t, label) for t, label in _TICK_CANDIDATES if min_t <= t <= max_t]

    if not candidates:
        return

    if len(candidates) <= target_count:
        selected = candidates
    else:
        log_range = np.log10(max_t) - np.log10(min_t)
        min_spacing = log_range / target_count

        selected = [candidates[0]]
        for t, label in candidates[1:]:
            if np.log10(t) - np.log10(selected[-1][0]) >= min_spacing:
                selected.append((t, label))

        # Always include the last candidate
        if selected[-1] is not candidates[-1]:
            selected.append(candidates[-1])

    ticks, labels = zip(*selected)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.minorticks_off()


def _setup_power_duration_axes(ax, min_t=None, max_t=None):
    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("duration")
    ax.set_ylabel("power [W]")
    if min_t is not None and max_t is not None:
        _set_duration_ticks(ax, min_t, max_t)


def _create_axes(ax):
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _model_curve(estimator, duration_range):
    """Generate a smooth model curve over a duration range."""
    t = np.geomspace(duration_range[0], duration_range[1], 200)
    power = estimator.predict(t.reshape(-1, 1))
    return t, power


class PowerDurationDisplay:
    """Visualization of power-duration data with model curves.

    Use the classmethods ``from_estimator`` or ``from_estimators`` to create
    a display. Do not instantiate this class directly.

    Attributes
    ----------
    ax_ : matplotlib Axes
        The axes containing the plot.
    figure_ : matplotlib Figure
        The figure containing the plot.
    scatter_ : matplotlib PathCollection or None
        The data scatter artist.
    lines_ : list of matplotlib Line2D
        The model curve artists.
    """

    def __init__(self, *, ax, scatter, lines):
        self.ax_ = ax
        self.figure_ = ax.figure
        self.scatter_ = scatter
        self.lines_ = lines

    @classmethod
    def from_estimator(cls, estimator, X=None, y=None, *, name=None, ax=None):
        """Plot a single fitted model, optionally with data.

        Parameters
        ----------
        estimator : fitted regressor
            A fitted silhouette regressor.
        X : array-like of shape (n_samples, 1), optional
            Durations in seconds.
        y : array-like of shape (n_samples,), optional
            Power in watts.
        name : str, optional
            Label for the model curve. Defaults to the class name.
        ax : matplotlib Axes, optional
            Axes to plot on. Created if None.

        Returns
        -------
        PowerDurationDisplay
        """
        return cls.from_estimators(
            [estimator], X, y, names=[name] if name else None, ax=ax,
        )

    @classmethod
    def from_estimators(cls, estimators, X=None, y=None, *, names=None, ax=None):
        """Plot multiple fitted models, optionally with data.

        Parameters
        ----------
        estimators : list of fitted regressors
            Fitted silhouette regressors.
        X : array-like of shape (n_samples, 1), optional
            Durations in seconds.
        y : array-like of shape (n_samples,), optional
            Power in watts.
        names : list of str, optional
            Labels for each model. Defaults to class names.
        ax : matplotlib Axes, optional
            Axes to plot on. Created if None.

        Returns
        -------
        PowerDurationDisplay
        """
        from sklearn.utils.validation import check_is_fitted

        for est in estimators:
            check_is_fitted(est)

        ax = _create_axes(ax)

        # Data scatter
        scatter = None
        if X is not None and y is not None:
            durations = np.asarray(X).ravel()
            power = np.asarray(y).ravel()
            scatter = ax.scatter(
                durations, power, color="black", s=20, zorder=2, label="data",
            )
            duration_range = (durations.min(), durations.max())
        else:
            duration_range = (1, 7200)

        # Model curves
        if names is None:
            names = [est.__class__.__name__ for est in estimators]

        lines = []
        for estimator, name in zip(estimators, names):
            t, power = _model_curve(estimator, duration_range)
            line, = ax.plot(t, power, linewidth=2, zorder=3, label=name)
            lines.append(line)

        _setup_power_duration_axes(ax, *duration_range)
        ax.legend()

        return cls(ax=ax, scatter=scatter, lines=lines)


class ModeOfVarianceDisplay:
    """Visualization of FPCA mode of variance.

    Shows how each FPC modifies the mean power-duration curve.

    Use ``from_model`` or ``from_estimator`` to create a display.

    Attributes
    ----------
    axes_ : ndarray of matplotlib Axes, or single Axes
        The axes containing the plots.
    figure_ : matplotlib Figure
        The figure containing the plots.
    mean_line_ : matplotlib Line2D or list of Line2D
        The mean function line artist(s).
    athlete_line_ : matplotlib Line2D, list of Line2D, or None
        The athlete's line artist(s), if an estimator was passed.
    """

    def __init__(self, *, axes, mean_line, athlete_line):
        self.axes_ = axes
        self.figure_ = axes[0].figure if isinstance(axes, np.ndarray) else axes.figure
        self.mean_line_ = mean_line
        self.athlete_line_ = athlete_line

    @classmethod
    def from_model(cls, *, component=None, n_lines=50, n_sd=2, axes=None):
        """Plot mode of variance from the bundled population model.

        Parameters
        ----------
        component : int (1, 2, or 3), optional
            Plot a single component. If None, plots all three.
        n_lines : int, default=50
            Number of gradient lines per component.
        n_sd : float, default=2
            Score range in standard deviations.
        axes : matplotlib Axes or array of Axes, optional
            Axes to plot on. Created if None.

        Returns
        -------
        ModeOfVarianceDisplay
        """
        return cls._plot(
            estimator=None,
            component=component,
            n_lines=n_lines,
            n_sd=n_sd,
            axes=axes,
        )

    @classmethod
    def from_estimator(cls, estimator, *, component=None, n_lines=50, n_sd=2, axes=None):
        """Plot mode of variance with a fitted athlete's position highlighted.

        Parameters
        ----------
        estimator : FPCARegressor
            A fitted FPCARegressor.
        component : int (1, 2, or 3), optional
            Plot a single component. If None, plots all three.
        n_lines : int, default=50
            Number of gradient lines per component.
        n_sd : float, default=2
            Score range in standard deviations.
        axes : matplotlib Axes or array of Axes, optional
            Axes to plot on. Created if None.

        Returns
        -------
        ModeOfVarianceDisplay
        """
        return cls._plot(
            estimator=estimator,
            component=component,
            n_lines=n_lines,
            n_sd=n_sd,
            axes=axes,
        )

    @classmethod
    def _plot(cls, *, estimator, component, n_lines, n_sd, axes):
        from silhouette.fpca import _get_model

        model = estimator.population_model if estimator is not None else _get_model()

        if component is not None:
            components = [component - 1]  # 1-indexed to 0-indexed
        else:
            components = list(range(model.n_components))

        single = len(components) == 1

        # Create axes
        if axes is None:
            if single:
                fig, axes_arr = plt.subplots(1, 1, figsize=(8, 5))
                axes_arr = np.array([axes_arr])
            else:
                fig, axes_arr = plt.subplots(
                    1, len(components), figsize=(6 * len(components), 5),
                )
        else:
            axes_arr = np.atleast_1d(axes)

        time_grid = model.time_grid
        mean_power = np.exp(model.mean_function)
        log_time = model.log_time_grid

        fpc_labels = ["FPC1: Overall power", "FPC2: Sprint/endurance bias", "FPC3: Mid-duration"]
        scores = None
        if estimator is not None:
            scores = [estimator.fpc1_, estimator.fpc2_, estimator.fpc3_]

        mean_lines = []
        athlete_lines = []

        for ax, comp_idx in zip(axes_arr, components):
            sd = np.sqrt(model.eigenvalues[comp_idx])
            score_range = np.linspace(-n_sd * sd, n_sd * sd, n_lines)

            # Gradient curves with diverging colors
            for score_val in score_range:
                modified = np.exp(
                    model.mean_function + model.eigenfunctions[:, comp_idx] * score_val
                )

                fraction = score_val / (n_sd * sd)  # -1 to 1
                if fraction < 0:
                    color = plt.cm.Reds(0.2 + 0.5 * abs(fraction))
                else:
                    color = plt.cm.Blues(0.2 + 0.5 * fraction)

                ax.semilogx(
                    time_grid, modified,
                    color=color, alpha=0.4, linewidth=0.5, zorder=1,
                )

            # Mean function
            mean_line, = ax.semilogx(
                time_grid, mean_power, "k-", linewidth=2, zorder=10, label="Mean",
            )
            mean_lines.append(mean_line)

            # Athlete's position
            if scores is not None:
                athlete_curve = np.exp(
                    model.mean_function
                    + model.eigenfunctions[:, comp_idx] * scores[comp_idx]
                )
                athlete_line, = ax.semilogx(
                    time_grid, athlete_curve,
                    color="#C8102E", linewidth=2.5, zorder=15,
                    label=f"Athlete ({scores[comp_idx]:+.2f})",
                )
                athlete_lines.append(athlete_line)

            ax.set_ylabel("power [W]")
            ax.set_xlabel("duration")
            _set_duration_ticks(ax, time_grid[0], time_grid[-1])
            ax.set_title(fpc_labels[comp_idx] if comp_idx < len(fpc_labels) else f"FPC{comp_idx + 1}")
            ax.legend(loc="upper right")

        if not single:
            plt.tight_layout()

        result_axes = axes_arr[0] if single else axes_arr
        result_mean = mean_lines[0] if single else mean_lines
        result_athlete = (
            (athlete_lines[0] if single else athlete_lines) if athlete_lines else None
        )

        return cls(axes=result_axes, mean_line=result_mean, athlete_line=result_athlete)
