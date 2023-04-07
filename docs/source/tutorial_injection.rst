.. _tutorial_injection:

Step by step injection search
==============================

If you want to know more about burst search, you can run this tutorial by injection a signal and find it step by step


First, create a project folder and copy the `user_parameters_injection.yaml` file from the examples folder.


.. code-block:: bash

    mkdir my_search
    cp [path_to_source_code]/examples/injection/user_parameters_injection.yaml user_parameters.yaml


Next, load the environment and the configuration file:

.. code-block:: python

    import os

    import pyburst
    from pyburst.config import Config
    from pyburst.utils import logger_init

    if not os.environ.get('HOME_WAT_FILTERS'):
        pyburst_path = os.path.dirname(os.path.abspath(pyburst.__file__))
        os.environ['HOME_WAT_FILTERS'] = f"{os.path.abspath(pyburst_path)}/vendor"

    logger_init()

    config = Config('./user_parameters_injection.yaml')


Now, create an injection using the parameters specified in the configuration file:

.. code-block:: python

    from pyburst.modules.read_data import generate_injection

    data = generate_injection(config)

A pyCBC time series is generated for each detector defined in the configuration file.

Use the data conditioning module to remove lines and whiten the data. The strains are the whitened data,
and the nRMS is the noise RMS of the data:

.. code-block:: python

    from pyburst.modules.data_conditioning import data_conditioning
    from pyburst.modules.plot import plot_spectrogram

    strains, nRMS = data_conditioning(config, data)




Create a network using the whitened data and the noise RMS, then generate a set of wavelet decomposition modules
for each resolution:

.. code-block:: python

    # initialize network
    from pyburst.modules.wavelet import create_wdm_set
    from pyburst.types import WDMXTalkCatalog, Network

    job_id = 0

    wdm_MRA = WDMXTalkCatalog(config.MRAcatalog)
    wdm_list = create_wdm_set(config, wdm_MRA)
    network = Network(config, strains, nRMS, wdm_MRA)

Find the coherent pixel clusters and generate the sparse table to reduce the computational cost in the following steps:

.. code-block:: python

    from pyburst.modules.coherence import coherence, sparse_table_from_fragment_clusters

    # calculate coherence
    fragment_clusters = coherence(config, strains, wdm_list, nRMS)

    # generate sparse table
    sparse_table_list = sparse_table_from_fragment_clusters(config, network.get_max_delay(),
                                                            strains, wdm_list, fragment_clusters)

Then merge the clusters to superclusters

.. code-block:: python

    from pyburst.modules.super_cluster import supercluster

    pwc_list = supercluster(config, network, wdm_list, fragment_clusters, sparse_table_list)


Finally, calculate the likelihood for each supercluster:

.. code-block:: python

    from pyburst.modules.likelihood import likelihood

    events, clusters = likelihood(job_id, config, network, pwc_list)

You can use the following code to plot the events on the spectrogram:

.. code-block:: python

    from pyburst.modules.plot import plot_event_on_spectrogram

    for i, tf_map in enumerate(strains):
        plt = plot_event_on_spectrogram(tf_map, events)
        plt.show()
