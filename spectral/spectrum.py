"""
spectrum.py

Defines the Spectrum class.

A Spectrum represents ONE ordered sequence of eigenvalues.

Examples
--------
• One GUE Hamiltonian
• One GOE Hamiltonian
• One sequence of Riemann zeros

The class stores

    raw eigenvalues
    unfolded eigenvalues
    nearest-neighbour spacings

It intentionally knows nothing about
how the spectrum was generated.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Spectrum:
    """
    Represents a single ordered spectrum.
    """

    def __init__(
        self,
        eigenvalues: np.ndarray,
        *,
        name: str = "Spectrum",
        metadata: Optional[dict] = None,
    ) -> None:

        self.name = name

        self.metadata = metadata or {}

        self.eigenvalues = np.asarray(
            eigenvalues,
            dtype=float,
        )

        self.eigenvalues.sort()

        self.unfolded: Optional[np.ndarray] = None

        self.spacings: Optional[np.ndarray] = None

    # ---------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        *,
        name: str = "Spectrum",
        metadata: Optional[dict] = None,
    ) -> "Spectrum":

        return cls(
            values,
            name=name,
            metadata=metadata,
        )

    @classmethod
    def from_file(
        cls,
        filename: str,
        *,
        name: str = "Spectrum",
        metadata: Optional[dict] = None,
    ) -> "Spectrum":

        values = np.loadtxt(filename)

        return cls(
            values,
            name=name,
            metadata=metadata,
        )

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def size(self) -> int:
        return self.eigenvalues.size

    @property
    def is_unfolded(self) -> bool:
        return self.unfolded is not None

    @property
    def has_spacings(self) -> bool:
        return self.spacings is not None

    # ---------------------------------------------------------
    # Methods
    # ---------------------------------------------------------

    def reset(self) -> None:
        """
        Remove computed quantities while keeping
        the raw eigenvalues.
        """

        self.unfolded = None
        self.spacings = None

    def copy(self) -> "Spectrum":

        new = Spectrum(
            self.eigenvalues.copy(),
            name=self.name,
            metadata=self.metadata.copy(),
        )

        if self.unfolded is not None:
            new.unfolded = self.unfolded.copy()

        if self.spacings is not None:
            new.spacings = self.spacings.copy()

        return new

    # ---------------------------------------------------------

    def __len__(self):

        return self.size

    def __repr__(self):

        return (
            f"Spectrum("
            f"name='{self.name}', "
            f"size={self.size}, "
            f"unfolded={self.is_unfolded}, "
            f"spacings={self.has_spacings})"
        )