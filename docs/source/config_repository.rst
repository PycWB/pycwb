.. _config_repository:

.. _pycwb_config:

Setup Config Templates
======================

``pycwb-config`` is the central configuration repository used to prepare
standard pycWB burst searches. It stores search parameter templates, chunk
definitions, data-quality segment files, data-source settings, machine
profiles, and helper scripts for downloading public GWOSC inputs.

Use this guide when you want to create a working directory from a named search
configuration and run it with ``pycwb config-setup``.


Repository Layout
-----------------

A typical checkout has this structure:

.. code-block:: text

   pycwb-config/
   |-- config/
   |   |-- settings.yaml
   |   |-- machine/
   |   |   |-- default.yaml
   |   |   |-- cit.yaml
   |   |   |-- picasso.yaml
   |   |   `-- ...
   |   |-- BurstLF_chunk.lst
   |   |-- BurstHF_chunk.lst
   |   |-- BurstLD_chunk.lst
   |   |-- BurstLF/
   |   |   `-- LH/
   |   |       |-- BKG/
   |   |       |   `-- user_parameters.yaml
   |   |       `-- SIM/
   |   |           `-- ...
   |   |-- DQ/
   |   |   `-- C00/
   |   |       `-- BurstLF/
   |   |           |-- metadata.yaml
   |   |           |-- H1_cat0.txt
   |   |           |-- H1_cat1.txt
   |   |           `-- H1_cat2.txt
   |   `-- frames/
   |       |-- H1.frames
   |       `-- H1_frames/
   `-- scripts/
       |-- download_gwosc_dq
       |-- download_gwosc_frames
       `-- update_framelist

The important inputs are:

``settings.yaml``
   Defines interferometer aliases, network short codes such as ``LH`` and
   ``HLV``, and data-source profiles such as ``gwosc``, ``igwn-osg``,
   ``cit-local``, and ``local``.

``machine/*.yaml``
   Defines execution defaults for a site or cluster, including the submission
   backend, container or conda environment, memory, disk, accounting group,
   SLURM options, and the default data source.

``*_chunk.lst``
   Defines the GPS start and stop times for each observation/chunk pair used
   by a search.

``{Search}/{Network}/{JobType}/user_parameters.yaml``
   Stores the pycWB user-parameter template for a specific search
   configuration. ``pycwb config-setup`` copies this file into the working
   directory and fills template variables such as GPS times, IFOs, DQ files,
   frame settings, and batch settings.

``DQ/{DQ}/{Search}/``
   Stores data-quality segment files and ``metadata.yaml``. These files are
   copied into the run ``input`` directory during setup.

``frames/``
   Stores local frame path lists for ``--datatype local``. Each
   ``{ifo}.frames`` file contains one frame path per line.


Quick Start
-----------

Clone the configuration repository and enter it:

.. code-block:: bash

   git clone git@git.ligo.org:yumeng.xu/pycwb-config.git pycwb-config
   cd pycwb-config

The GWOSC download helpers require the ``gwosc`` Python package:

.. code-block:: bash

   python -m pip install gwosc

Download data-quality segment files:

.. code-block:: bash

   # All searches, all chunks, default IFOs: H1 L1 V1
   scripts/download_gwosc_dq

   # One search and one observing run
   scripts/download_gwosc_dq --search BurstLF --obs O4

   # Preview without writing files
   scripts/download_gwosc_dq --search BurstLF --obs O4 --dry-run

Download GWOSC frame files when you want to run from local frame lists:

.. code-block:: bash

   # All searches, default IFOs, 16 kHz data
   scripts/download_gwosc_frames

   # One search and selected chunks
   scripts/download_gwosc_frames --search BurstLF --obs O4 --chunks 1-5

   # H1/L1 4 kHz data
   scripts/download_gwosc_frames --ifos H1 L1 --sample-rate 4096

   # Disable DQ-based trimming of the download window
   scripts/download_gwosc_frames --search BurstLF --obs O4 --dq ''

Then create a pycWB working directory:

.. code-block:: bash

   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine default \
       --datatype gwosc

Add ``--submit`` and the appropriate cluster options when the working
directory should be submitted immediately.


Work Directory Names
--------------------

``pycwb config-setup`` uses the working directory name to select the
configuration, chunk, DQ category, network, job type, and optional run label:

.. code-block:: text

   {OBS}_K{CHUNK_ID}_{DQ}_{Search}_{Network}_{JobType}[_{label}]

Examples:

.. code-block:: text

   O4_K02_C00_BurstLF_LH_BKG_standard
   O4_K05_C00_BurstHF_LHV_SIM_STDINJs_Set1_run1

The fixed fields mean:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Field
     - Example
     - Meaning
   * - ``OBS``
     - ``O4``
     - Observation run. Used with the chunk ID to select GPS times from
       ``{Search}_chunk.lst``.
   * - ``K{CHUNK_ID}``
     - ``K02``
     - Chunk identifier in the search chunk list.
   * - ``DQ``
     - ``C00``
     - Data-quality category under ``config/DQ/``.
   * - ``Search``
     - ``BurstLF``
     - Search directory and chunk-list prefix.
   * - ``Network``
     - ``LH`` or ``HLV``
     - Network short code resolved through ``settings.yaml``.
   * - ``JobType``
     - ``BKG`` or ``SIM``
     - Configuration subtree containing ``user_parameters.yaml``.
   * - ``label``
     - ``standard``
     - Optional run label. Everything after the matched configuration path is
       treated as the label.

Directory names inside the configuration path may contain underscores. The
parser finds the longest matching directory that contains
``user_parameters.yaml`` and treats the remaining name components as the run
label.


Data-Quality Files
------------------

DQ files are stored under ``config/DQ/{dq_category}/{search}/``:

.. code-block:: text

   config/DQ/C00/BurstLF/
   |-- metadata.yaml
   |-- H1_cat0.txt
   |-- H1_cat1.txt
   |-- H1_cat2.txt
   |-- L1_cat0.txt
   `-- ...

Each ``*_cat*.txt`` file contains one segment per line:

.. code-block:: text

   GPS_START GPS_END

The ``metadata.yaml`` file tells pycWB how to interpret each DQ file:

.. code-block:: yaml

   H1_cat0:
     type: CWB_CAT0
     inverted: false
     column4: false
   H1_cat1:
     type: CWB_CAT1
     inverted: false
     column4: false

For GWOSC-generated files, the flags are non-inverted because GWOSC ``*_DATA``
flags already represent valid science time.


Download Helper Options
-----------------------

``scripts/download_gwosc_dq`` writes DQ segments to
``config/DQ/{dq}/{search}/`` and rebuilds the selected output files.

.. list-table:: ``download_gwosc_dq`` options
   :header-rows: 1
   :widths: 26 18 56

   * - Option
     - Default
     - Description
   * - ``--config CONFIG_DIR``
     - ``./config``
     - Config directory containing ``*_chunk.lst`` files.
   * - ``--search SEARCH``
     - ``all``
     - Search name such as ``BurstLF``, or ``all``.
   * - ``--obs OBS``
     - all
     - Observation-run filter, for example ``O4``.
   * - ``--chunks SPEC``
     - ``all``
     - Chunk IDs, for example ``2,5-9,16a``.
   * - ``--dq DQ_CATEGORY``
     - ``C00``
     - DQ subdirectory label.
   * - ``--ifos IFO [...]``
     - ``H1 L1 V1``
     - IFOs to process. Space-separated and comma-separated forms are both
       accepted.
   * - ``--dry-run``
     - off
     - Preview without writing files.
   * - ``-v, --verbose``
     - off
     - Enable DEBUG logging.

``scripts/download_gwosc_frames`` downloads GWOSC ``.gwf`` files and writes
``config/frames/{ifo}.frames`` path lists.

.. list-table:: ``download_gwosc_frames`` options
   :header-rows: 1
   :widths: 26 18 56

   * - Option
     - Default
     - Description
   * - ``--config CONFIG_DIR``
     - ``./config``
     - Config directory containing ``*_chunk.lst`` files.
   * - ``--search SEARCH``
     - ``all``
     - Search name such as ``BurstLF``, or ``all``.
   * - ``--obs OBS``
     - all
     - Observation-run filter, for example ``O4``.
   * - ``--chunks SPEC``
     - ``all``
     - Chunk IDs, for example ``1,3-9,16a``.
   * - ``--ifos IFO [...]``
     - ``H1 L1 V1``
     - IFOs to process. Space-separated and comma-separated forms are both
       accepted.
   * - ``--sample-rate HZ``
     - ``16384``
     - GWOSC frame sample rate.
   * - ``--dq DQ_CAT``
     - ``C00``
     - DQ category used to trim each download window to the first and last
       overlapping CAT0 segment. Pass ``''`` to disable trimming.
   * - ``--dry-run``
     - off
     - Preview without downloading.
   * - ``-v, --verbose``
     - off
     - Enable DEBUG logging.

When frame files have been moved or added manually, regenerate the path-list
files:

.. code-block:: bash

   scripts/update_framelist
   scripts/update_framelist --ifos H1 L1
   scripts/update_framelist --dry-run


Running ``pycwb config-setup``
------------------------------

The command creates the working directory, copies and renders
``user_parameters.yaml``, copies DQ files into ``input/``, copies local frame
lists when needed, and runs the normal pycWB batch setup.

Common examples:

.. code-block:: bash

   # Set up a working directory without submitting
   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine default \
       --datatype gwosc

   # Set up and submit to HTCondor
   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine cit \
       --cluster condor --submit \
       --accounting-group ligo.dev.o4.burst.cwb \
       --container-image /cvmfs/container.gwdc.org/pycwb_latest.sif

   # Set up a SLURM run using a machine profile
   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine picasso-small \
       --cluster slurm --submit

   # Preview setup actions
   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine default \
       --dry-run

   # Count jobs without submitting
   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine default \
       --list-n-jobs

The ``--machine`` option loads ``config/machine/{machine}.yaml``. If
``config/settings.yaml`` contains a top-level ``machine`` key, that profile is
used by default; otherwise pass ``--machine`` explicitly.

.. list-table:: ``pycwb config-setup`` options
   :header-rows: 1
   :widths: 28 18 54

   * - Option
     - Default
     - Description
   * - ``workdir``
     - required
     - Working directory name. The basename is parsed as the project name.
   * - ``--config-base-path, -c``
     - ``./prototypes/config``
     - Base path to the configuration repository.
   * - ``--machine``
     - from ``settings.yaml``
     - Machine profile under ``machine/{name}.yaml``.
   * - ``--datatype, -d``
     - from machine profile
     - Data source profile from ``settings.yaml``. Common values are
       ``gwosc``, ``igwn-osg``, ``cit-local``, and ``local``.
   * - ``--cluster``
     - from rendered config
     - Submission backend: ``condor`` or ``slurm``.
   * - ``--submit, -s``
     - off
     - Submit after setup.
   * - ``--dry-run``
     - off
     - Preview only; do not create files.
   * - ``--n-proc, -n``
     - from rendered config
     - Number of CPUs per job.
   * - ``--memory, -m``
     - from rendered config
     - Memory per job, for example ``4GB``.
   * - ``--disk, -k``
     - from rendered config
     - Disk per job, for example ``8GB``.
   * - ``--accounting-group, -g``
     - from rendered config
     - Condor accounting group.
   * - ``--container-image, --image``
     - from rendered config
     - Container image URI.
   * - ``--force-overwrite, -f``
     - off
     - Overwrite existing batch output.
   * - ``--list-n-jobs``
     - off
     - Print the number of jobs without submitting.
   * - ``--walltime``
     - from rendered config
     - SLURM wall-clock limit, for example ``72:00:00``.
   * - ``--slurm-constraint``
     - from rendered config
     - SLURM node feature constraint.
   * - ``--slurm-partition``
     - from rendered config
     - SLURM partition.
   * - ``--n-retries``
     - ``5``
     - Application-level retries for SLURM jobs.


End-to-End Example
------------------

This example uses public GWOSC inputs and a Condor-style default machine
profile:

.. code-block:: bash

   git clone git@git.ligo.org:yumeng.xu/pycwb-config.git pycwb-config
   cd pycwb-config

   scripts/download_gwosc_dq --search BurstLF --obs O4
   scripts/download_gwosc_frames --search BurstLF --obs O4

   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine default \
       --datatype gwosc \
       --cluster condor \
       --accounting-group ligo.dev.o4.burst.cwb

Add ``--submit`` after checking the generated working directory and batch
files.
