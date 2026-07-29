"""
plotting.py

Visualization utilities for Random Matrix Theory.

This module intentionally contains NO physics.

It simply plots data supplied by the caller.

Currently implemented

    • Spacing histogram
    • Theoretical distributions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .statistics import Statistics


class Plotter:
    """
    Utility plotting class.
    """

    # ---------------------------------------------------------
    # Nearest-neighbour spacing distribution
    # ---------------------------------------------------------

    @staticmethod
    def spacing_distribution(
        riemann_spacings: np.ndarray,
        gue_spacings: np.ndarray,
        *,
        bins: int = 60,
        show_theory: bool = True,
        title: str = "Nearest-Neighbour Spacing Distribution",
    ):
        """
        Plot spacing distributions for the Riemann zeros
        and the GUE ensemble.
        """

        fig, ax = plt.subplots(figsize=(8, 5))

        # --------------------------------------------------
        # Histograms
        # --------------------------------------------------

        ax.hist(
            riemann_spacings,
            bins=bins,
            density=True,
            alpha=0.60,
            label="Riemann Zeros",
        )

        ax.hist(
            gue_spacings,
            bins=bins,
            density=True,
            alpha=0.60,
            label="GUE",
        )

        # --------------------------------------------------
        # Theoretical curve
        # --------------------------------------------------

        if show_theory:

            s = np.linspace(0.0, 4.0, 500)

            ax.plot(
                s,
                Statistics.wigner_surmise_gue(s),
                linewidth=2.5,
                label="GUE Wigner Surmise",
            )

        # --------------------------------------------------

        ax.set_xlim(0, 4)

        ax.set_xlabel(
            "Normalized spacing $s$",
            fontsize=12,
        )

        ax.set_ylabel(
            "Probability Density",
            fontsize=12,
        )

        ax.set_title(
            title,
            fontsize=14,
        )

        ax.legend()

        ax.grid(
            alpha=0.25,
        )

        fig.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Single histogram
    # ---------------------------------------------------------

    @staticmethod
    def histogram(
        data: np.ndarray,
        *,
        bins: int = 50,
        density: bool = True,
        xlabel: str = "",
        ylabel: str = "Density",
        title: str = "",
    ):
        """
        Generic histogram.
        """

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            data,
            bins=bins,
            density=density,
            alpha=0.75,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.grid(alpha=0.30)

        fig.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Compare with theory
    # ---------------------------------------------------------

    @staticmethod
    def compare_with_theory(
        spacings: np.ndarray,
        *,
        ensemble: str = "GUE",
        bins: int = 60,
    ):
        """
        Compare empirical spacings against
        GOE/GUE/Poisson predictions.
        """

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            spacings,
            bins=bins,
            density=True,
            alpha=0.65,
            label="Simulation",
        )

        s = np.linspace(0, 4, 500)

        ensemble = ensemble.upper()

        if ensemble == "GOE":

            theory = Statistics.wigner_surmise_goe(s)

        elif ensemble == "GUE":

            theory = Statistics.wigner_surmise_gue(s)

        elif ensemble == "POISSON":

            theory = Statistics.poisson_distribution(s)

        else:

            raise ValueError(
                f"Unknown ensemble '{ensemble}'."
            )

        ax.plot(
            s,
            theory,
            linewidth=2.5,
            label=ensemble,
        )

        ax.set_xlim(0, 4)

        ax.set_xlabel("Normalized spacing")
        ax.set_ylabel("Probability Density")

        ax.legend()

        ax.grid(alpha=0.25)

        fig.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------

    @staticmethod
    def save(
        figure,
        filename: str,
        *,
        dpi: int = 300,
    ) -> None:
        """
        Save a figure.

        Creates directories automatically.
        """

        path = Path(filename)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    @staticmethod
    def show():

        plt.show()

    # ---------------------------------------------------------

    @staticmethod
    def close():

        plt.close("all")