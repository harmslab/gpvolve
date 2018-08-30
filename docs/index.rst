``gpvolve``
===========

A Python API for the simulation and analysis of evolution in genotype-phenotype space.
You can use this library to:

   1. Build a markov state model from a genotype-phenotype-map.
   2. Find clusters of genotypes that represent metastable states of the system.
   3. Compute fluxes and pathways between pairs of genotypes and/or clusters of interest.
   4. Visualize the outputs of all of the above.

The core-utilities of this library are built on top of the pyemma and msmtools packages.
For a deeper understanding of these tools, we recommend to read the docs and scientific
references of the respective libraries ([1]_, [2]_, [3]_).

A rationale for treating fitness landscapes as markov systems can be found in [4]_.

Currently, this package works only as an API. There is no command-line
interface. Instead, we encourage you use this package inside `Jupyter notebooks`_ .

References
----------

.. [1] https://github.com/markovmodel/PyEMMA
.. [2] https://github.com/markovmodel/msmtools
.. [3] M K Scherer, B Trendelkamp-Schroer, F. Paul, G. Pérez-Hernández, M. Hoffmann, N. Plattner, C. Wehmeyer, J.-H. Prinz and F. Noé: PyEMMA 2: A Software Package for Estimation, Validation, and Analysis of Markov Models, J. Chem. Theory Comput. 11, 5525-5542 (2015)
.. [4] G Sella, A. E. Hirsh: The application of statistical physics to evolutionary biology, Proceedings of the National Academy of Sciences Jul 2005, 102 (27) 9541-9546; DOI: 10.1073/pnas.0501865102


Basic Example
-------------

Calculate and plot the fluxes between wildtype and triple mutant on an example genotype-phenotype map.

.. code-block:: python

   # Import the base class and a visualization tool.
   from gpvolve.markovmodel import EvoMarkovStateModel
   from gpvolve.visualization import draw_network

   # Genotype-phenotype map data.
   wildtype = "AAA"
   genotypes = ["AAA", "AAT", "ATA", "TAA", "ATT", "TAT", "TTA", "TTT"]
   phenotypes = [0.8, 0.81, 0.88, 0.89, 0.82, 0.82, 0.95, 1.0]

   # Create genotype-phenotype map object.
   gpm = GenotypePhenotypeMap(wildtype=wildtype,
                              genotypes=genotypes,
                              phenotypes=phenotypes)

   # Define source and target (Can be done at a later point as well)
   source = ["AAA"]
   target = ["TTT"]

   # Create a evolutionary markov state model from the genotype-phenotype map.
   M = EvoMarkovStateModel(gpm=gpm,
                           selection_gradient=1,
                           population_size=100,
                           two_step_probability=0,
                           source=source,
                           target=target)

   # Compute reactive fluxes between source and target.
   M.tpt()
   fluxes = M.net_flux
   total_flux = M.total_flux

   # Normalize flux.
   norm_fluxes = fluxes/total_flux

   # Plot the network and the fluxes
   fig, ax = draw_network(M=M,
                          clusters=None,
                          flux=norm_fluxes,
                          colorbar=True)

.. image:: img/basic_example.png

Documentation
-------------
.. toctree::
   :maxdepth: 2

   readme
   installation
   usage
   modules
   contributing
   history
   api

