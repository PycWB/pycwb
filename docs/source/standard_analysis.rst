.. _standard_analysis:

===================
Production Analysis
===================

This section covers how to set up and run production pycWB burst searches
using configuration templates and computing clusters.

.. mermaid::

   flowchart TD
     A[pycWB environment] --> C[Standard analysis setup]
     B[Config template<br/>and required files] --> C

     C --> D[Production runs]

     subgraph prod[Production]
       D1[Background]
       D2[Training set]
       D3[Simulations]
     end

     D --> D1
     D --> D2
     D --> D3

     D1 --> E[Postproduction]
     D2 --> E
     D3 --> E

     subgraph post[Postproduction]
       E1[Model training]
       E2[Background statistics]
       E3[Sensitivity study]
       E4[Open box / zero-lag review]

       E1 --> E2
       E2 --> E3
       E2 --> E4
     end

     E --> E1

.. toctree::
   :maxdepth: 2

   config_repository
   run_on_clusters
