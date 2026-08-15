"""A from-scratch, NumPy-only implementation of the difference-curve knee locator.

This module implements the method credited in the project's Acknowledgements
section (README.md) so that ``elbow_helper`` depends on **numpy only**: no
``scipy`` at runtime.

Two scipy calls in a typical implementation are replaced here:

* ``scipy.interpolate.interp1d(x, y)(x)`` evaluated at the *same* ``x`` is the
  identity, so with ``interp_method="interp1d"`` the fitted line is just ``y``.
* ``scipy.signal.argrelextrema(a, comparator, order, mode="clip")`` is a short
  NumPy helper (:func:`_argrelextrema`) that compares each sample against its
  clip-indexed neighbours, bit-for-bit equivalent for 1-D input.

The traversal logic in :meth:`KneeLocator.find_knee`, the ``transform_y``
orientation table, the sensitivity threshold ``Tmx`` and the online-correction
behaviour follow the reference implementation closely, so results match it
for the supported inputs.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

VALID_CURVE = ("convex", "concave")
VALID_DIRECTION = ("increasing", "decreasing")


def _argrelextrema(data: np.ndarray, comparator, order: int = 1) -> np.ndarray:
    """Indices of relative extrema, matching ``scipy.signal.argrelextrema``.

    Replicates SciPy's ``mode="clip"`` boundary handling: a sample is compared
    against the ``order`` neighbours on each side, with out-of-range neighbour
    indices clipped to the array bounds (so an endpoint is compared against
    itself on its missing side).

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional signal.
    comparator : callable
        Element-wise comparison, ``numpy.greater_equal`` for maxima or
        ``numpy.less_equal`` for minima.
    order : int, optional
        Number of neighbours on each side to compare against. Default ``1``.

    Returns
    -------
    numpy.ndarray
        Integer indices where ``data`` is a relative extremum.
    """
    n = data.shape[0]
    locs = np.arange(n)
    results = np.ones(n, dtype=bool)
    for shift in range(1, order + 1):
        plus = data[np.clip(locs + shift, 0, n - 1)]
        minus = data[np.clip(locs - shift, 0, n - 1)]
        results &= comparator(data, plus)
        results &= comparator(data, minus)
        if not results.any():
            break
    return np.nonzero(results)[0]


class KneeLocator:
    """Locate the point of maximum curvature (knee/elbow) of a curve.

    A NumPy-only implementation of the difference-curve locator, exposing the
    public surface used by this package: ``knee``, ``norm_knee``, ``all_knees``,
    ``all_norm_knees``, ``x_difference`` / ``y_difference`` and the extrema
    indices.

    Parameters
    ----------
    x, y : array-like
        Input coordinates, equal length. ``x`` must be strictly increasing.
    S : float, optional
        Sensitivity; larger values are more conservative. Default ``1.0``.
    curve : str, optional
        ``"concave"`` to detect knees, ``"convex"`` to detect elbows.
    direction : str, optional
        ``"increasing"`` or ``"decreasing"``.
    interp_method : str, optional
        ``"interp1d"`` (identity fit) or ``"polynomial"`` (``numpy.polyfit``).
    online : bool, optional
        If ``True``, keep correcting the knee while traversing; if ``False``,
        return the first knee found. Default ``False``.
    polynomial_degree : int, optional
        Degree used when ``interp_method="polynomial"``. Default ``7``.
    """

    def __init__(
        self,
        x: Iterable[float],
        y: Iterable[float],
        S: float = 1.0,
        curve: str = "concave",
        direction: str = "increasing",
        interp_method: str = "interp1d",
        online: bool = False,
        polynomial_degree: int = 7,
    ):
        # Step 0: raw input
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.x.size == 0:
            # Every step below (normalizing to [0, 1], averaging the
            # difference-curve step size) divides or reduces over the array,
            # so an empty input surfaces as a confusing "zero-size array to
            # reduction operation minimum which has no identity" deep inside
            # __normalize rather than a clear error at the actual boundary.
            raise ValueError(
                "KneeLocator needs at least one (x, y) point, got an empty array"
            )
        self.curve = curve
        self.direction = direction
        self.N = len(self.x)
        self.S = S
        self.all_knees: set = set()
        self.all_norm_knees: set = set()
        self.all_knees_y: list = []
        self.all_norm_knees_y: list = []
        # Extension over the base algorithm: one record per distinct knee,
        # pairing its data-space location, normalized location, and the index
        # of the generating peak on the difference curve (used for prominence).
        self.all_knee_records: list = []
        self.online = online
        self.polynomial_degree = polynomial_degree

        if curve not in VALID_CURVE or direction not in VALID_DIRECTION:
            raise ValueError(
                "Please check that the curve and direction arguments are valid."
            )

        # Step 1: fit a smooth line.
        # interp1d(x, y) evaluated at x is the identity, so we skip scipy here.
        if interp_method == "interp1d":
            self.Ds_y = self.y.copy()
        elif interp_method == "polynomial":
            coefs = np.polyfit(self.x, self.y, self.polynomial_degree)
            self.Ds_y = np.poly1d(coefs)(self.x)
        else:
            raise ValueError(
                f"{interp_method} is an invalid interp_method parameter, "
                "use either 'interp1d' or 'polynomial'"
            )

        # Step 2: normalize to the unit square.
        self.x_normalized = self.__normalize(self.x)
        self.y_normalized = self.__normalize(self.Ds_y)

        # Step 3: orient to concave-increasing, then subtract the diagonal.
        self.y_normalized = self.transform_y(
            self.y_normalized, self.direction, self.curve
        )
        self.y_difference = self.y_normalized - self.x_normalized
        self.x_difference = self.x_normalized.copy()

        # Step 4: local maxima / minima of the difference curve.
        self.maxima_indices = _argrelextrema(self.y_difference, np.greater_equal)
        self.x_difference_maxima = self.x_difference[self.maxima_indices]
        self.y_difference_maxima = self.y_difference[self.maxima_indices]

        self.minima_indices = _argrelextrema(self.y_difference, np.less_equal)
        self.x_difference_minima = self.x_difference[self.minima_indices]
        self.y_difference_minima = self.y_difference[self.minima_indices]

        # Step 5: sensitivity thresholds.
        self.Tmx = self.y_difference_maxima - (
            self.S * np.abs(np.diff(self.x_normalized).mean())
        )

        # Step 6: find the knee.
        self.knee, self.norm_knee = self.find_knee()

        # Step 7: attach y values if a knee exists.
        self.knee_y = self.norm_knee_y = None
        if self.knee is not None:
            self.knee_y = self.y[self.x == self.knee][0]
            self.norm_knee_y = self.y_normalized[self.x_normalized == self.norm_knee][0]

    @staticmethod
    def __normalize(a: np.ndarray) -> np.ndarray:
        """Scale an array to ``[0, 1]``.

        Parameters
        ----------
        a : numpy.ndarray
            One-dimensional array to rescale.

        Returns
        -------
        numpy.ndarray
            ``a`` linearly rescaled so its minimum is ``0`` and maximum ``1``.
        """
        return (a - a.min()) / (a.max() - a.min())

    @staticmethod
    def transform_y(y: np.ndarray, direction: str, curve: str) -> np.ndarray:
        """Orient ``y`` to a concave, increasing frame (elbows become knees).

        Parameters
        ----------
        y : numpy.ndarray
            Normalized ``y`` values.
        direction : str
            ``"increasing"`` or ``"decreasing"``.
        curve : str
            ``"concave"`` or ``"convex"``.

        Returns
        -------
        numpy.ndarray
            ``y``, flipped and/or mirrored so the concave-increasing bump
            logic in :meth:`find_knee` applies unchanged.
        """
        if direction == "decreasing":
            if curve == "concave":
                y = np.flip(y)
            elif curve == "convex":
                y = y.max() - y
        elif direction == "increasing" and curve == "convex":
            y = np.flip(y.max() - y)
        return y

    def find_knee(self) -> Tuple[Optional[float], Optional[float]]:
        """Traverse the difference curve and return ``(knee, norm_knee)``.

        Returns
        -------
        tuple of (float or None), (float or None)
            ``(knee, norm_knee)`` in data-space and normalized-space
            coordinates respectively, or ``(None, None)`` if no candidate
            clears its sensitivity threshold.
        """
        if not self.maxima_indices.size:
            return None, None

        maxima_threshold_index = 0
        minima_threshold_index = 0
        threshold = 0.0
        threshold_index = 0
        knee = None
        norm_knee = None
        detection_active = True

        for i, _ in enumerate(self.x_difference):
            if i < self.maxima_indices[0]:
                continue

            j = i + 1
            if i == (len(self.x_difference) - 1):
                break

            if (self.maxima_indices == i).any():
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = i
                maxima_threshold_index += 1
                detection_active = True
            if (self.minima_indices == i).any():
                threshold = 0.0
                minima_threshold_index += 1
                detection_active = False

            if detection_active and self.y_difference[j] < threshold:
                if self.curve == "convex":
                    if self.direction == "decreasing":
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]
                elif self.curve == "concave":
                    if self.direction == "decreasing":
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]

                y_at_knee = self.y[self.x == knee][0]
                y_norm_at_knee = self.y_normalized[self.x_normalized == norm_knee][0]
                if knee not in self.all_knees:
                    self.all_knees_y.append(y_at_knee)
                    self.all_norm_knees_y.append(y_norm_at_knee)
                    self.all_knee_records.append(
                        {
                            "knee": knee,
                            "norm_knee": norm_knee,
                            "threshold_index": threshold_index,
                        }
                    )

                self.all_knees.add(knee)
                self.all_norm_knees.add(norm_knee)

                if self.online is False:
                    return knee, norm_knee

        if not self.all_knees:
            return None, None
        return knee, norm_knee

    # Elbow aliases, for callers who think in "elbows".
    @property
    def elbow(self):
        """Alias for :attr:`knee`, for callers who think in "elbows"."""
        return self.knee

    @property
    def norm_elbow(self):
        """Alias for :attr:`norm_knee`, for callers who think in "elbows"."""
        return self.norm_knee

    @property
    def all_elbows(self):
        """Alias for :attr:`all_knees`, for callers who think in "elbows"."""
        return self.all_knees

    @property
    def all_norm_elbows(self):
        """Alias for :attr:`all_norm_knees`, for callers who think in "elbows"."""
        return self.all_norm_knees
