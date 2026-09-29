.. _about:

About PycWB
===========

What is pycWB?
--------------

pycWB is a Python package for **coherent gravitational-wave burst searches**.
It implements the same cWB/cWB-2G algorithmic chain used by the ROOT/C++ cWB
pipeline: WDM time-frequency analysis, coherent pixel selection, clustering and
superclustering, coherent likelihood evaluation, waveform reconstruction, and
postproduction ranking.

pycWB implements the cWB/cWB-2G algorithms for coherent burst searches. It
analyzes strain data from the LIGO-Virgo-KAGRA detector network, transforms it
into a wavelet time-frequency representation, and searches for short
gravitational-wave transients with minimal assumptions about the signal
waveform by identifying coherent excess-power structures across the detector
network.

Unlike template-based searches that look for specific waveforms, pycWB
identifies **any statistically significant coherence** between detectors,
making it sensitive to both known and unknown source types.

.. image:: _static/diagrams/pipeline_overview.svg
   :alt: pycWB pipeline overview


Search animation
----------------

.. raw:: html

   <span id="id1"></span>

Watch the :ref:`60-second search lifecycle animation <search_lifecycle_animation>`
to follow a simulated signal from detector projection through reconstruction
and time-slide background estimation. The :doc:`pipeline_lifecycle` page
explains each stage and the animation's scope.


Project links
-------------

.. raw:: html

   <p>
     <a href="https://docs.pycwb.org">
       <img src="https://readthedocs.org/projects/pycwb/badge/?version=latest" alt="Documentation">
     </a>
     <a href="https://git.ligo.org/yumeng.xu/pycwb/-/pipelines">
       <img src="https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg" alt="Build Status">
     </a>
     <a href="https://git.ligo.org/yumeng.xu/pycwb/-/releases">
       <img src="https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg" alt="Releases">
     </a>
     <a href="https://badge.fury.io/py/pycWB">
       <img src="https://badge.fury.io/py/pycWB.svg" alt="PyPI version">
     </a>
     <a href="https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE">
       <img src="https://img.shields.io/badge/license-GPLv3-blue" alt="License">
     </a>
   </p>


For citation guidance and BibTeX entries, see :doc:`credit`.

.. toctree::
   :maxdepth: 1

   cwb_heritage
   public_gwtc_references
   release_policy
   support

For the earlier audience-based navigation, see :doc:`choose_your_path`.
