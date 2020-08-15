# gpvolve

[![PyPI version](https://badge.fury.io/py/gpvolve.svg)](https://badge.fury.io/py/gpvolve)
[![](https://readthedocs.org/projects/gpvolve/badge/?version=latest)](https://gpvolve.readthedocs.io/en/latest/?badge=latest)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/harmslab/gpvolve/blob/master/examples/Introduction%20to%20gpvolve.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)]([![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/harmslab/gpvolve/master?filepath=examples%2FIntroduction%20to%20gpvolve.ipynb))

**A python package for extracting evolutionary trajectories from genotype-phenotype-maps**

A Python API for the simulation and analysis of evolution in genotype-phenotype space.
You can use this library to:

   1. Build a markov state model from a genotype-phenotype-map.
   2. Find clusters of genotypes that represent metastable states of the system, using PCCA+.
   3. Compute fluxes and pathways between pairs of genotypes and/or clusters of interest, using Transition Path Theory.
   4. Visualize the outputs of all of the above.

## Basic Example

Build a Markov model from a genotype-phenotype map.
```python

# Import base class, Transition Path Theory class and functions for building Markov Model.
from gpvolve import GenotypePhenotypeMSM, TransitionPathTheory, linear_skew, mccandlish, find_max

# Import visualization tool.
from gpvolve.visualization import plot_network

# Import GenotypePhenotypeMap class for handling genotype-phenotype data.
from gpmap import GenotypePhenotypeMap

# Helper functions.
from scipy.sparse import dok_matrix

# Genotype-phenotype map data.
wildtype = "AAA"
genotypes = ["AAA", "AAT", "ATA", "TAA", "ATT", "TAT", "TTA", "TTT"]
phenotypes = [0.8, 0.81, 0.88, 0.89, 0.82, 0.82, 0.95, 1.0]

# Instantiate Markov model class.
gpm = GenotypePhenotypeMap(wildtype=wildtype,
                           genotypes=genotypes,
                           phenotypes=phenotypes)


# Instantiate a evolutionary Markov State Model from the genotype-phenotype map.
gpmsm = GenotypePhenotypeMSM(gpm)
```
Apply an evolutionary model to describe transitions between genotypes.
```python
# Map fitnesses to phenotypes.
gpmsm.apply_selection(fitness_function=linear_skew, selection_gradient=1)

# Build Markov State Model based on 'mccandlish' fixation probability function.
gpmsm.build_transition_matrix(fixation_model=mccandlish, population_size=100)

# Find global fitness peak.
fitness_peak = find_max(gpmsm=gpmsm, attribute='fitness')
```

Calculate and plot the trajectory flux between the wildtype and triple mutant.
```python

# Compute fluxes from wildtype to fitness peak.
fluxes = TransitionPathTheory(gpmsm, source=[0], target=[fitness_peak])

# Normalize flux.
norm_fluxes = fluxes.net_flux/fluxes.total_flux

# Plot the network and the fluxes
fig, ax = plot_network(gpmsm,
                       flux=dok_matrix(norm_fluxes),
                       edge_labels=True,
                       colorbar=True)

```
<img src="docs/img/basic_example.png" width="700">


## Install

To install from PyPI:
```
pip install gpvolve
```

To install a development version:
```
git clone https://github.com/harmslab/gpvolve
cd gpvolve
pip install  -e .
```


