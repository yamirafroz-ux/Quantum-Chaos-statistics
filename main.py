"""
main.py

Compare the nearest-neighbour spacing distribution
of the Riemann zeta zeros against the Gaussian
Unitary Ensemble (GUE).

Author
------
Afroz Yamir
"""

from spectral.generators import GUEGenerator
from spectral.plotting import Plotter
from spectral.spectrum import Spectrum
from spectral.statistics import Statistics
from spectral.unfolding import (
    RiemannUnfolder,
    SemicircleUnfolder,
)


def main():

    print("=" * 60)
    print("Quantum Chaos Spectral Statistics")
    print("=" * 60)

    # -----------------------------------------------------
    # Load Riemann zeros
    # -----------------------------------------------------

    print("\nLoading Riemann zeros...")

    riemann = Spectrum.from_file(
        "data/zeros1.txt",
        name="Riemann Zeros",
    )

    print(riemann)

    # -----------------------------------------------------
    # Unfold Riemann zeros
    # -----------------------------------------------------

    print("\nUnfolding Riemann spectrum...")

    RiemannUnfolder().unfold(riemann)

    riemann_spacings = Statistics.compute_spacings(
        riemann
    )

    Statistics.print_summary(
        riemann_spacings
    )

    # -----------------------------------------------------
    # Generate GUE ensemble
    # -----------------------------------------------------

    print("\nGenerating GUE ensemble...")

    generator = GUEGenerator(
        matrix_size=300,
        samples=500,
        seed=42,
    )

    gue = generator.generate()

    print(gue)

    # -----------------------------------------------------
    # Unfold every Hamiltonian
    # -----------------------------------------------------

    print("\nUnfolding GUE ensemble...")

    SemicircleUnfolder().unfold_ensemble(
        gue
    )

    # -----------------------------------------------------
    # Compute spacing statistics
    # -----------------------------------------------------

    gue_spacings = Statistics.compute_ensemble_spacings(
        gue
    )

    Statistics.print_summary(
        gue_spacings
    )

    # -----------------------------------------------------
    # Plot comparison
    # -----------------------------------------------------

    print("\nGenerating figure...")

    fig, ax = Plotter.spacing_distribution(
        riemann_spacings,
        gue_spacings,
        bins=60,
    )

    Plotter.save(
        fig,
        "plots/spacing_distribution.png",
    )

    Plotter.show()

    print("\nFigure saved to")
    print("plots/spacing_distribution.png")

    print("\nDone.")


if __name__ == "__main__":
    main()