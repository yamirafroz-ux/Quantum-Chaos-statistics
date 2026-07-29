"""
unfolding.py

Algorithms for unfolding spectra.

Unfolding transforms a raw spectrum

    λ₁, λ₂, λ₃, ...

into an unfolded spectrum

    x₁, x₂, x₃, ...

whose average local level density is unity.

Implemented
-----------
    • RiemannUnfolder
    • SemicircleUnfolder
    • NumericalUnfolder
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .spectrum import Spectrum
from .ensemble import SpectrumEnsemble


# ==========================================================
# Base Class
# ==========================================================

class Unfolder(ABC):
    """
    Base class for unfolding algorithms.
    """

    @abstractmethod
    def unfold(
        self,
        spectrum: Spectrum,
    ) -> Spectrum:
        """
        Unfold a single spectrum.
        """
        pass

    def unfold_ensemble(
        self,
        ensemble: SpectrumEnsemble,
    ) -> SpectrumEnsemble:
        """
        Apply unfolding independently to every spectrum
        in an ensemble.
        """

        for spectrum in ensemble:
            self.unfold(spectrum)

        return ensemble


# ==========================================================
# Riemann-von Mangoldt Unfolding
# ==========================================================

class RiemannUnfolder(Unfolder):
    """
    Unfold the imaginary parts of the Riemann zeta zeros
    using the smooth Riemann-von Mangoldt counting function.

    N(T) ≈

        T/(2π) log(T/(2π))
        - T/(2π)
        + 7/8
    """

    @staticmethod
    def counting_function(
        gamma: np.ndarray,
    ) -> np.ndarray:

        gamma = np.asarray(gamma, dtype=float)

        return (
            gamma / (2 * np.pi)
            * np.log(gamma / (2 * np.pi))
            - gamma / (2 * np.pi)
            + 7 / 8
        )

    def unfold(
        self,
        spectrum: Spectrum,
    ) -> Spectrum:

        spectrum.unfolded = self.counting_function(
            spectrum.eigenvalues
        )

        spectrum.metadata["unfolding"] = (
            "Riemann-von Mangoldt"
        )

        return spectrum


# ==========================================================
# Wigner Semicircle Unfolding
# ==========================================================

class SemicircleUnfolder(Unfolder):
    """
    Unfold spectra generated from GUE/GOE matrices using
    the integrated Wigner semicircle law.
    """

    @staticmethod
    def semicircle_cdf(
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Integrated semicircle law.

        Parameters
        ----------
        x : np.ndarray
            Values scaled into [-1,1].

        Returns
        -------
        np.ndarray
        """

        x = np.clip(x, -1.0, 1.0)

        return (
            0.5
            +
            (
                x * np.sqrt(1.0 - x**2)
                + np.arcsin(x)
            ) / np.pi
        )

    def unfold(
        self,
        spectrum: Spectrum,
    ) -> Spectrum:

        if "matrix_size" not in spectrum.metadata:
            raise RuntimeError(
                "Spectrum metadata does not contain "
                "'matrix_size'."
            )

        N = spectrum.metadata["matrix_size"]

        # Raw eigenvalues live approximately on
        # [-2√N, 2√N]

        scaled = spectrum.eigenvalues / (
            2.0 * np.sqrt(N)
        )

        cumulative = self.semicircle_cdf(
            scaled
        )

        # N(E)=N·F(E)

        spectrum.unfolded = N * cumulative

        spectrum.metadata["unfolding"] = (
            "Semicircle"
        )

        return spectrum


# ==========================================================
# Polynomial Unfolding
# ==========================================================

class NumericalUnfolder(Unfolder):
    """
    Generic polynomial unfolding.

    Fits the staircase counting function with
    a polynomial.
    """

    def __init__(
        self,
        degree: int = 7,
    ) -> None:

        if degree < 1:
            raise ValueError(
                "Polynomial degree must be positive."
            )

        self.degree = degree

    def unfold(
        self,
        spectrum: Spectrum,
    ) -> Spectrum:

        eig = spectrum.eigenvalues

        staircase = np.arange(
            1,
            len(eig) + 1,
        )

        coeff = np.polyfit(
            eig,
            staircase,
            self.degree,
        )

        poly = np.poly1d(coeff)

        spectrum.unfolded = poly(eig)

        spectrum.metadata["unfolding"] = (
            f"Polynomial (degree={self.degree})"
        )

        return spectrum


# ==========================================================
# Utility
# ==========================================================

def available_unfolders() -> list[str]:
    """
    Return available unfolding algorithms.
    """

    return [
        "RiemannUnfolder",
        "SemicircleUnfolder",
        "NumericalUnfolder",
    ]