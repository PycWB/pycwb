.. _tutorial_search:

Go deeper into PyBurst.search
==============================

Initialize logger with log_file and log_level, if log_file is None, log to stdout.

.. code-block:: python

   from pyburst.utils import logger_init
   logger_init(log_file, log_level)

set env HOME_WAT_FILTERS to the path of xdmXTalk. PyBurst contains a sample xdmXTalk file in vendor folder.

.. code-block:: python

   import os, pyburst

   pyburst_path = os.path.dirname(os.path.abspath(pyburst.__file__))
   os.environ['HOME_WAT_FILTERS'] = f"{os.path.abspath(pyburst_path)}/vendor"

read user parameters from user_parameters.yaml to a :py:class:`~pyburst.config.Config` object.

.. code-block:: python

   from pyburst.config import Config
   user_parameters = Config('./user_parameters.yaml')

then, create directories for output files.

.. code-block:: python

    if not os.path.exists(config.outputDir):
        os.makedirs(config.outputDir)
    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)

find the job segments from the config settings and create a :py:class:`~pyburst.types.JobSegment` object for each segment.

.. code-block:: python

   from pyburst.modules.read_data import read_from_job_segment

   job_segments = select_job_segment(config.dq_files, config.ifo, config.frFiles,
                                     config.segLen, config.segMLS, config.segEdge, config.segOverlap,
                                     config.rateANA, config.l_high)

analyze the job segments with :py:func:`pyburst.search.analyze_job_segment`.
To avoid memory leak, the function is called in a subprocess.

.. code-block:: python

   from pyburst.search import analyze_job_segment

   for job_segment in job_segments:
       process = multiprocessing.Process(target=analyze_job_segment, args=(config, job_seg))
       process.start()
       process.join()

For macOS users, by default, you might encounter a safety check error when running the code.
To aviod this, you should not use subprocess to run the code. Instead, you can run the code directly in the main process.

.. code-block:: python

   from pyburst.search import analyze_job_segment

   for job_segment in job_segments:
       analyze_job_segment(config, job_segment)
