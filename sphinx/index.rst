.. V-Mapper documentation master file, created by
   sphinx-quickstart on Fri Oct  8 15:45:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

V-Mapper: 
====================================
A topological data analysis (TDA) method for high-dimensional data, Mapper, represents a topology of data as a simplicial complex or graph abstraction based on the nerve of clusters [Singh et al., 2007]. V-Mapper (velocity Mapper) is an extension of Mapper for high-dimensional data with position and velocity. 
V-Mapper describes a topological structure and flows on it simultaneously as a weighted directed graph (V-Mapper graph) by embedding given velocity data in the edges of the Mapper graph.

.. image:: ../../images/VMapper_Top.jpg

Installation
====================================
V-Mapper package supports PyPI install. 

.. code-block:: bash

 $ pip install vmapper



You can also install the development version of V-Mapper from GitHub:

.. code-block:: bash

	$ pip install git+https://github.com/yusuke-imoto-lab/V-Mapper.git


To use V-Mapper, import ``vmapper``. 

.. code-block:: python

 import vmapper


.. toctree::
	:maxdepth: 2
	:caption: Contents:
	
	examples/index
	reference/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
