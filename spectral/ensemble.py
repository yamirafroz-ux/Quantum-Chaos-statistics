"""
ensemble.py

Defines the SpectrumEnsemble class.

A SpectrumEnsemble is a collection of Spectrum objects.

Example
-------

500 GUE matrices

↓

500 Spectrum objects

↓

SpectrumEnsemble

The ensemble provides convenient methods
for collecting spacings and unfolded spectra.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .spectrum import Spectrum


class SpectrumEnsemble:
    """
    Collection of Spectrum objects.
    """

    def __init__(
        self,
        spectra: Iterable[Spectrum],
        *,
        name: str = "Spectrum Ensemble",
        metadata: dict | None = None,
    ) -> None:

        self.name = name

        self.metadata = metadata or {}

        self.spectra = list(spectra)

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of spectra.
        """

        return len(self.spectra)

    @property
    def total_levels(self) -> int:
        """
        Total number of eigenvalues.
        """

        return sum(
            spectrum.size
            for spectrum in self.spectra
        )

    # ---------------------------------------------------------
    # Access
    # ---------------------------------------------------------

    def __len__(self):

        return len(self.spectra)

    def __iter__(self):

        return iter(self.spectra)

    def __getitem__(self, index):

        return self.spectra[index]

    # ---------------------------------------------------------
    # Data collection
    # ---------------------------------------------------------

    def unfolded_levels(self) -> np.ndarray:
        """
        Concatenate all unfolded levels.
        """

        arrays = []

        for spectrum in self.spectra:

            if spectrum.unfolded is None:
                raise RuntimeError(
                    "Spectrum has not been unfolded."
                )

            arrays.append(spectrum.unfolded)

        return np.concatenate(arrays)

    def spacings(self) -> np.ndarray:
        """
        Concatenate all nearest-neighbour spacings.

        IMPORTANT

        Spacings are computed INSIDE each spectrum.

        They are never computed between
        different spectra.
        """

        arrays = []

        for spectrum in self.spectra:

            if spectrum.spacings is None:
                raise RuntimeError(
                    "Spectrum has no spacings."
                )

            arrays.append(spectrum.spacings)

        return np.concatenate(arrays)

    # ---------------------------------------------------------

    def append(
        self,
        spectrum: Spectrum,
    ) -> None:

        self.spectra.append(spectrum)

    # ---------------------------------------------------------

    def __repr__(self):

        return (
            f"SpectrumEnsemble("
            f"name='{self.name}', "
            f"spectra={self.size}, "
            f"levels={self.total_levels})"
        )