.. fggs documentation master file, created by
   sphinx-quickstart on Fri Oct 29 18:00:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fggs's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Factor graph grammars
---------------------

.. autoclass:: fggs.FGG
   :members:
      
.. autoclass:: fggs.FactorGraph
   :members:
      
.. autoclass:: fggs.HRG
   :members:
      
.. autoclass:: fggs.HRGRule
   :members:

.. autoclass:: fggs.Node
   :members:
      
.. autoclass:: fggs.NodeLabel
   :members:
      
.. autoclass:: fggs.Edge
   :members:
      
.. autoclass:: fggs.EdgeLabel
   :members:
   
.. autoclass:: fggs.Domain
   :members:

.. autoclass:: fggs.FiniteDomain
   :members:

.. autoclass:: fggs.Factor
   :members:

.. autoclass:: fggs.CategoricalFactor
   :members:

.. automodule:: fggs.formats
   :members:

Operations on FGGs
------------------

.. autofunction:: fggs.start_graph
                  
.. autofunction:: fggs.replace_edges
                  
.. autofunction:: fggs.conjoin_hrgs
                  
.. autofunction:: fggs.factorize

Inference
---------
      
.. autofunction:: fggs.sum_product

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
