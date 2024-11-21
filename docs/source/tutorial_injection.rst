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

    import pycwb
    from pycwb.config import Config
    from pycwb.modules.logger import logger_init

    logger_init()

    config = Config('./user_parameters_injection.yaml')
    config.ifo, config.injection


Now, create an injection job using the parameters specified in the configuration file:

.. code-block:: python

    from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg
    from pycwb.modules.job_segment import create_job_segment_from_injection

    job_segments = create_job_segment_from_injection(config.ifo, config.simulation, config.injection)

    data = generate_noise_for_job_seg(job_segments[0], config.inRate)
    data = generate_injection(config, job_segments[0], data)

A pyCBC time series is generated for each detector defined in the configuration file.

Use the data conditioning module to remove lines and whiten the data. The strains are the whitened data,
and the nRMS is the noise RMS of the data:

.. code-block:: python

    from pycwb.modules.data_conditioning import data_conditioning
    from pycwb.modules.plot import plot_spectrogram

    strains, nRMS = data_conditioning(config, data)


Find the coherent pixel clusters and generate the sparse table to reduce the computational cost in the following steps:

.. code-block:: python

    from pycwb.modules.coherence import coherence

    # calculate coherence
    fragment_clusters = coherence(config, strains, nRMS)

Create a network using the whitened data and the noise RMS, then merge the clusters to superclusters

.. code-block:: python

    from pycwb.modules.super_cluster import supercluster
    from pycwb.types.network import Network

    network = Network(config, strains, nRMS)

    pwc_list = supercluster(config, network, fragment_clusters, strains)

Finally, calculate the likelihood for each supercluster:

.. code-block:: python

    from pycwb.modules.likelihood import likelihood

    events, clusters, skymap_statistics = likelihood(config, network, pwc_list)

You can use the following code to plot the events on the spectrogram:

.. code-block:: python

    %matplotlib inline
    from pycwb.modules.plot import plot_event_on_spectrogram

    for i, tf_map in enumerate(strains):
        plt = plot_event_on_spectrogram(tf_map, events)
        plt.show()

the likelihood map and null map reconstructed from the clusters
will also be plotted with

.. code-block:: python

    %matplotlib inline
    from gwpy.spectrogram import Spectrogram

    for cluster in clusters:
        merged_map, start, dt, df = cluster.get_sparse_map("likelihood")

        plt = Spectrogram(merged_map, t0=start, dt=dt, f0=0, df=df).plot()
        plt.colorbar()

    for cluster in clusters:
        merged_map, start, dt, df = cluster.get_sparse_map("null")

        plt = Spectrogram(merged_map, t0=start, dt=dt, f0=0, df=df).plot()
        plt.colorbar()
