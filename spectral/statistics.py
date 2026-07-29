"""
statistics.py

Statistical analysis of unfolded spectra.

This module computes universal statistics used in
Random Matrix Theory.

Currently implemented

    • Nearest-neighbour spacings

Future additions

    • Pair correlation function
    • Number variance
    • Dyson-Mehta Δ₃ statistic
    • Spectral rigidity
    • Spectral form factor
"""

from __future__ import annotations

import numpy as np

from .ensemble import SpectrumEnsemble
from .spectrum import Spectrum


class Statistics:
    """
    Collection of statistical analysis routines.
    """

    # ---------------------------------------------------------
    # Nearest-neighbour spacings
    # ---------------------------------------------------------

    @staticmethod
    def compute_spacings(
        spectrum: Spectrum,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute nearest-neighbour spacings for ONE spectrum.

        Parameters
        ----------
        spectrum
            Unfolded spectrum.

        normalize
            Divide by the mean spacing.

        Returns
        -------
        np.ndarray
            Spacing array.
        """

        if spectrum.unfolded is None:
            raise RuntimeError(
                "Spectrum must be unfolded first."
            )

        spacings = np.diff(spectrum.unfolded)

        if normalize:

            mean_spacing = np.mean(spacings)

            if mean_spacing <= 0:
                raise RuntimeError(
                    "Mean spacing is non-positive."
                )

            spacings = spacings / mean_spacing

        spectrum.spacings = spacings

        return spacings

    # ---------------------------------------------------------

    @staticmethod
    def compute_ensemble_spacings(
        ensemble: SpectrumEnsemble,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute spacings for every spectrum in the ensemble.

        Returns
        -------
        np.ndarray
            Concatenated spacings.
        """

        all_spacings = []

        for spectrum in ensemble:

            spacings = Statistics.compute_spacings(
                spectrum,
                normalize=normalize,
            )

            all_spacings.append(spacings)

        return np.concatenate(all_spacings)

    # ---------------------------------------------------------

    @staticmethod
    def spacing_histogram(
        spacings: np.ndarray,
        *,
        bins: int = 50,
        density: bool = True,
        range: tuple[float, float] = (0.0, 4.0),
    ):
        """
        Compute histogram data without plotting.

        Returns
        -------
        bin_centres
        histogram
        """

        hist, edges = np.histogram(
            spacings,
            bins=bins,
            density=density,
            range=range,
        )

        centres = 0.5 * (edges[:-1] + edges[1:])

        return centres, hist

    # ---------------------------------------------------------
    # Theoretical distributions
    # ---------------------------------------------------------

    @staticmethod
    def wigner_surmise_goe(
        s: np.ndarray,
    ) -> np.ndarray:
        """
        Wigner surmise for GOE.

        P(s) =
        (π/2)s exp(-πs²/4)
        """

        return (
            np.pi
            / 2
            * s
            * np.exp(
                -np.pi * s**2 / 4
            )
        )

    # ---------------------------------------------------------

    @staticmethod
    def wigner_surmise_gue(
        s: np.ndarray,
    ) -> np.ndarray:
        """
        Wigner surmise for GUE.

        P(s) =
        (32/π²)s² exp(-4s²/π)
        """

        return (
            32
            / np.pi**2
            * s**2
            * np.exp(
                -4 * s**2 / np.pi
            )
        )

    # ---------------------------------------------------------

    @staticmethod
    def poisson_distribution(
        s: np.ndarray,
    ) -> np.ndarray:
        """
        Poisson spacing distribution.

        Integrable systems.
        """

        return np.exp(-s)

    # ---------------------------------------------------------
    # Convenience summary
    # ---------------------------------------------------------

    @staticmethod
    def summary(
        spacings: np.ndarray,
    ) -> dict:
        """
        Compute basic statistics of a spacing set.
        """

        return {

            "count": len(spacings),

            "mean": float(
                np.mean(spacings)
            ),

            "variance": float(
                np.var(spacings)
            ),

            "std": float(
                np.std(spacings)
            ),

            "minimum": float(
                np.min(spacings)
            ),

            "maximum": float(
                np.max(spacings)
            ),

        }

    # ---------------------------------------------------------

    @staticmethod
    def print_summary(
        spacings: np.ndarray,
    ) -> None:
        """
        Nicely print spacing statistics.
        """

        summary = Statistics.summary(
            spacings
        )

        print()

        print("=" * 40)
        print("Spacing Statistics")
        print("=" * 40)

        for key, value in summary.items():

            if isinstance(value, float):

                print(
                    f"{key:12s}: {value:.6f}"
                )

            else:

                print(
                    f"{key:12s}: {value}"
                )

        print("=" * 40)