.. _getting_started:

Getting started
===============

Start with installation, complete the synthetic first search, then learn what
its catalog and plots mean. The example uses generated noise and a simulated
signal; it does not require detector data or a collaboration account.

.. toctree::
   :maxdepth: 1

   install
   start_here
   understanding_results

Already installed?
------------------

.. code-block:: bash

   # From a source checkout, with PycWB installed
   pycwb run examples/demo/user_parameters.yaml --work-dir my_first_search
   pycwb progress --work-dir my_first_search

See :ref:`start_here` for a guided first run, or :ref:`installing_pycwb`
for detailed installation options.


For help with installation or a failed run, use :doc:`troubleshooting`
or :doc:`support`.

Continue with :doc:`tutorials` for worked examples or :doc:`run_analyses` for
workflows using your own inputs.
