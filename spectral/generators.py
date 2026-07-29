"""
generators.py

Generators for random matrix ensembles.

Currently implemented

    • GUEGenerator

Future additions

    • GOEGenerator
    • GSEGenerator
    • PoissonGenerator
"""

from __future__ import annotations

import numpy as np

from .ensemble import SpectrumEnsemble
from .spectrum import Spectrum


# ==========================================================
# GUE Generator
# ==========================================================


class GUEGenerator:
    """
    Generate an ensemble of GUE spectra.

    Parameters
    ----------
    matrix_size
        Dimension N of each Hermitian matrix.

    samples
        Number of matrices to generate.

    seed
        Optional random seed.
    """

    def __init__(
        self,
        matrix_size: int = 300,
        samples: int = 500,
        seed: int | None = None,
    ) -> None:

        if matrix_size < 2:
            raise ValueError(
                "matrix_size must be at least 2."
            )

        if samples < 1:
            raise ValueError(
                "samples must be at least 1."
            )

        self.matrix_size = matrix_size
        self.samples = samples
        self.seed = seed

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------
    # Matrix generation
    # ------------------------------------------------------

    def _generate_matrix(self) -> np.ndarray:
        """
        Generate one GUE matrix.

        The entries are complex Gaussian random variables.
        """

        real = self.rng.normal(
            0.0,
            1.0,
            (self.matrix_size, self.matrix_size),
        )

        imag = self.rng.normal(
            0.0,
            1.0,
            (self.matrix_size, self.matrix_size),
        )

        H = real + 1j * imag

        # Hermitian projection

        H = (H + H.conjugate().T) / 2

        return H

    # ------------------------------------------------------

    @staticmethod
    def _eigenvalues(
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute eigenvalues of a Hermitian matrix.
        """

        return np.linalg.eigvalsh(matrix)

    # ------------------------------------------------------
    # Public interface
    # ------------------------------------------------------

    def generate(self) -> SpectrumEnsemble:
        """
        Generate an ensemble of GUE spectra.

        Returns
        -------
        SpectrumEnsemble
        """

        spectra: list[Spectrum] = []

        for i in range(self.samples):

            H = self._generate_matrix()

            eigenvalues = self._eigenvalues(H)

            metadata = {

                "ensemble": "GUE",

                "matrix_size": self.matrix_size,

                "matrix_index": i,

                "seed": self.seed,

            }

            spectra.append(

                Spectrum.from_array(

                    eigenvalues,

                    name=f"GUE Matrix {i + 1}",

                    metadata=metadata,

                )

            )

        ensemble_metadata = {

            "ensemble": "GUE",

            "matrix_size": self.matrix_size,

            "samples": self.samples,

            "seed": self.seed,

        }

        return SpectrumEnsemble(

            spectra,

            name=f"GUE Ensemble ({self.samples} matrices)",

            metadata=ensemble_metadata,

        )

    # ------------------------------------------------------

    def __repr__(self) -> str:

        return (

            "GUEGenerator("

            f"matrix_size={self.matrix_size}, "

            f"samples={self.samples}, "

            f"seed={self.seed})"

        )