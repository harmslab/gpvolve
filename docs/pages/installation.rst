
Installation and Dependencies
=============================

Installation
------------

To get the lastest published version, install using pip:

.. code-block:: bash

    pip install gpvolve


To install from source, clone this repo and install using pip:

User can clone the github repository and install it locally.

.. code-block:: bash

    git clone https://github.com/harmslab/gpvolve
    cd gpvolve
    pip install -e .


Dependencies
------------

The following dependencies are required for the ``gpvolve`` package.

* gpmap_: Module for constructing powerful genotype-phenotype map python data-structures.
* gpgraph_: Module for graphic representation of genotype-phenotype maps built on top of networkx.
* networkx_: Python package for construction and analysis of networks and graphs.
* msmtools_: Python package containing tools for construction and analysis of markov models, including Transition Path Theory and PCCA+.
* numpy_: Python's array manipulation package.
* cython_: Programming language written in Python with C-like performance.
* matplotlib_: Python plotting library.

.. _gpmap: https://github.com/harmslab/gpmap
.. _gpgraph:: https://github.com/harmslab/gpgraph
.. _networkx: https://github.com/networkx
.. _msmtools: https://github.com/markovmodel/msmtools
.. _numpy: http://www.numpy.org/
.. _cython: https://github.com/cython/cython
.. _matplotlib: http://matplotlib.org/
