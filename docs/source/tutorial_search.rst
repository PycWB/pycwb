.. _tutorial_search:

Go deeper into pycWB.search
==============================

Although you can use each module in pycWB freely, the module :py:func:`pycwb.search` provides
a easier way to manage the job and run the search with the input from a yaml file if you just want to
run the search with default modules.

We can go deeper into the search function to see how it works. The search function is composed of
three parts: job generation and data analysis

Job Control
-----------------

The job control part is done by the functions in :py:func:`pycwb.workflow`.

Initialize logger with log_file and log_level, if log_file is None, log will be printed to stdout.

.. code-block:: python

   from pycwb.modules.logger import logger_init
   logger_init(log_file, log_level)


read user parameters from user_parameters.yaml to a :py:class:`.Config` object.

.. code-block:: python

   from pycwb.config import Config
   config = Config('./user_parameters.yaml')

then, it will create directories for output files.

.. code-block:: python

    if not os.path.exists(config.outputDir):
        os.makedirs(config.outputDir)
    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)

generate the job segments from the config settings and create a :py:class:`.WaveSegment` object for each segment.
If the injections are set in the config, the injections will be added to the job segments according to your injection type.

.. code-block:: python

   from pycwb.modules.job_segment import create_job_segment_from_config

   job_segments = create_job_segment_from_config(config)

A catalog file will be created in the output folder. The catalog file is a json file containing all the events found
in the search. The catalog file will be updated after each job segment is analyzed. A web viewer will also be created in the output folder. It is a web app that can be used to view the events in the
output folder. The web viewer is created by copying the web_viewer folder in pycWB to the output folder.

.. code-block:: python

    from pycwb.modules.catalog import create_catalog, add_events_to_catalog
    from pycwb.modules.web_viewer.create import create_web_viewer

    # create catalog
    create_catalog(f"{config.outputDir}/catalog.json", config, job_segments)

    # copy all files in web_viewer to output folder
    create_web_viewer(config.outputDir)

each job segment will be analyzed with :py:func:`pycwb.search.analyze_job_segment`.
To avoid memory leak in c code, the function is called in a subprocess.

.. code-block:: python

   from pycwb.search import analyze_job_segment

   for job_segment in job_segments:
       process = multiprocessing.Process(target=analyze_job_segment, args=(config, job_seg))
       process.start()
       process.join()

For macOS users, by default, you might encounter a safety check error when running the code.
To aviod this, you should not use subprocess to run the code. Instead, you can run the code directly in the main process.

.. code-block:: python

   from pycwb.search import analyze_job_segment

   for job_segment in job_segments:
       analyze_job_segment(config, job_segment)

Data Analysis
-----------------

The data analysis part is done by :py:func:`pycwb.search.analyze_job_segment`.
It analyzes the input job segment with config settings.


First, it will read the data from the job segment with :py:func:`pycwb.modules.read_data.read_from_job_segment`
and/or :py:func:`pycwb.modules.read_data.generate_injection` if the job segment contains injections. The data will be
stored in a pycbc TimeSeries object.

.. code-block:: python

    from pycwb.modules.read_data import read_from_job_segment, generate_injection

    data = None
    if job_seg.frames:
        data = read_from_job_segment(config, job_seg)
    if job_seg.injections:
        data = generate_injection(config, job_seg, data)


Then, it will do data conditioning with :py:func:`pycwb.modules.data_conditioning.data_conditioning`.
A list of conditioned :py:class:`.TimeFrequencySeries` objects and a list of nRMS :py:class:`.TimeFrequencySeries`
will be returned.

.. code-block:: python

    from pycwb.modules.data_conditioning import data_conditioning

    # data conditioning
    tf_maps, nRMS_list = data_conditioning(config, data)


Next, it will select the pixels and do the clustering with :py:func:`pycwb.modules.coherence.coherence`
and :py:func:`pycwb.modules.super_cluster.supercluster`. The output is a list of :py:class:`.Cluster` objects.

.. code-block:: python

    from pycwb.modules.coherence import coherence
    from pycwb.modules.super_cluster import supercluster

    fragment_clusters = coherence(config, tf_maps, nRMS_list)

    pwc_list = supercluster(config, network, fragment_clusters, tf_maps)

Finally, it will do the likelihood analysis with :py:func:`pycwb.modules.likelihood.likelihood`.
The output is a list of :py:class:`.Event` objects containing the statistics of each event from the likelihood analysis.
and a list of :py:class:`.Cluster` objects which contains the more detailed statistics of each pixels.
The clusters and events will be saved in the output folder. The catalog file will be updated with the new events.

.. code-block:: python

    from pycwb.modules.likelihood import likelihood, save_likelihood_data
    from pycwb.modules.catalog import add_events_to_catalog

    events, clusters = likelihood(config, network, pwc_list)

    for i, event in enumerate(events):
        save_likelihood_data(job_id, i+1, config.outputDir, event, clusters[i])
        # save event to catalog
        add_events_to_catalog(f"{config.outputDir}/catalog.json", event.summary(job_id, i+1))


The events will be marked on the spectrogram and the likelihood map and null map reconstructed from the clusters
will also be plotted.

.. code-block:: python

    from pycwb.modules.plot.cluster_statistics import plot_statistics
    from pycwb.modules.plot import plot_event_on_spectrogram


    for i, tf_map in enumerate(tf_maps):
        plot_event_on_spectrogram(tf_map, events, filename=f'{config.outputDir}/events_{job_id}_all_{i}.png')

    # plot the likelihood map
    for i, cluster in enumerate(clusters):
        if cluster.cluster_status != -1:
            continue
        plot_statistics(cluster, 'likelihood', filename=f'{config.outputDir}/likelihood_map_{job_id}_{i+1}.png')
        plot_statistics(cluster, 'null', filename=f'{config.outputDir}/null_map_{job_id}_{i+1}.png')

