"""
plotting.py

Visualization utilities for spectral analysis.

This module provides publication-quality plots for
spectra and Random Matrix Theory statistics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .spectrum import Spectrum


class Plotter:
    """
    Plotting utilities for Spectrum objects.
    """

    # ---------------------------------------------------------

    @staticmethod
    def spacing_histogram(
        spectrum: Spectrum,
        bins: int = 50,
        density: bool = True,
        ax=None,
        label: str | None = None,
        alpha: float = 0.7,
    ):
        """
        Plot a nearest-neighbour spacing histogram.
        """

        if spectrum.spacings is None:
            raise ValueError(
                "Compute spacings before plotting."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(7,5))

        ax.hist(
            spectrum.spacings,
            bins=bins,
            density=density,
            alpha=alpha,
            label=label or spectrum.name,
        )

        ax.set_xlabel("Spacing s")

        ax.set_ylabel("Probability Density")

        ax.set_title("Nearest-Neighbor Spacing Distribution")

        if label is not None:
            ax.legend()

        return ax

    # ---------------------------------------------------------

    @staticmethod
    def compare_spacing_histograms(
        spectrum1: Spectrum,
        spectrum2: Spectrum,
        bins: int = 50,
    ):
        """
        Overlay two spacing histograms.
        """

        fig, ax = plt.subplots(figsize=(8,6))

        Plotter.spacing_histogram(
            spectrum1,
            bins=bins,
            ax=ax,
            label=spectrum1.name,
            alpha=0.6,
        )

        Plotter.spacing_histogram(
            spectrum2,
            bins=bins,
            ax=ax,
            label=spectrum2.name,
            alpha=0.6,
        )

        ax.legend()

        return fig, ax

    # ---------------------------------------------------------

    @staticmethod
    def spectral_density(
        spectrum: Spectrum,
        bins: int = 80,
    ):
        """
        Plot the raw eigenvalue density.
        """

        fig, ax = plt.subplots(figsize=(7,5))

        ax.hist(
            spectrum.eigenvalues,
            bins=bins,
            density=True,
        )

        ax.set_xlabel("Eigenvalue")

        ax.set_ylabel("Density")

        ax.set_title(spectrum.name)

        return fig, ax

    # ---------------------------------------------------------

    @staticmethod
    def save(
        figure,
        filename: str,
        dpi: int = 300,
    ):
        """
        Save a matplotlib figure.
        """

        figure.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )