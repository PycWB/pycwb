.. _credits:

Use of pycWB in Scientific Publications
=======================================

If you use pycWB in a scientific publication, cite the pycWB paper and the
underlying cWB method paper. Depending on the analysis, also cite the cWB
software overview, the cWB-2G performance paper, and any release-specific
software DOI or version archive that applies to the exact software you used.

Please also report the pycWB version, tag, or git hash used for the analysis.


pycWB Paper
-----------

.. code-block:: bibtex

   @article{pycwb,
     title = {PycWB: A user-friendly, Modular, and python-based framework for gravitational wave unmodelled search},
     journal = {SoftwareX},
     volume = {26},
     pages = {101639},
     year = {2024},
     issn = {2352-7110},
     doi = {10.1016/j.softx.2024.101639},
     url = {https://www.sciencedirect.com/science/article/pii/S2352711024000104},
     author = {Xu, Yumeng and Tiwari, Shubhanshu and Drago, Marco},
     keywords = {Gravitational waves, Burst search}
   }

The article is available from
`SoftwareX <https://www.sciencedirect.com/science/article/pii/S2352711024000104>`_.


cWB Method Paper
----------------

pycWB implements the coherent WaveBurst search method, so publications should
also cite the core cWB method paper:

.. code-block:: bibtex

   @article{cwb_method,
     title = {Method for detection and reconstruction of gravitational wave transients with networks of advanced detectors},
     author = {Klimenko, S. and Vedovato, G. and Drago, M. and Salemi, F. and Tiwari, V. and Prodi, G. A. and Lazzaro, C. and Ackley, K. and Tiwari, S. and Da Silva, C. F. and Mitselmakher, G.},
     journal = {Phys. Rev. D},
     volume = {93},
     issue = {4},
     pages = {042004},
     year = {2016},
     doi = {10.1103/PhysRevD.93.042004},
     url = {https://link.aps.org/doi/10.1103/PhysRevD.93.042004}
   }


cWB Software Overview
---------------------

For work that discusses the cWB software framework, public cWB releases, or
the software lineage that pycWB builds on, cite the cWB SoftwareX overview:

.. code-block:: bibtex

   @article{cwb_softwarex,
     title = {coherent WaveBurst, a pipeline for unmodeled gravitational-wave data analysis},
     author = {Salemi, F. and others},
     journal = {SoftwareX},
     volume = {14},
     pages = {100678},
     year = {2021},
     doi = {10.1016/j.softx.2021.100678},
     url = {https://doi.org/10.1016/j.softx.2021.100678}
   }


cWB-2G Paper
------------

For analyses that rely on or compare against cWB-2G search performance,
ranking, or production-search strategy, cite the cWB-2G paper:

.. code-block:: bibtex

   @article{cwb_2g,
     title = {Optimizing searches for gravitational wave bursts using coherent WaveBurst-2G},
     author = {Martini, A. and others},
     journal = {Classical and Quantum Gravity},
     volume = {43},
     pages = {055016},
     year = {2026},
     doi = {10.1088/1361-6382/ae4717},
     url = {https://doi.org/10.1088/1361-6382/ae4717}
   }


Software Releases and Derived Work
----------------------------------

If your result depends on a specific pycWB or cWB release, cite the
release-specific DOI or archived software record when one is available. Avoid
using generic "latest release" wording in papers and internal notes; record
the exact version, tag, or git hash instead.

If you develop software from pycWB or cWB, cite the relevant project URL and
state which code version your work is derived from.
