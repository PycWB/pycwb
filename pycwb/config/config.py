"""
Config dataclass for pycWB analysis parameters.

Parameters are loaded from a YAML file and validated against a JSON schema.  The
default schema is defined in ``pycwb.constants.user_parameters_schema``.  A YAML
file may include an optional ``pycwb_schema`` block to extend or replace that
default schema – see :class:`Config` and :func:`pycwb.utils.yaml_helper.resolve_schema`
for details.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import os.path
import logging
import pathlib

from ..modules.xtalk.xtalk_data import check_and_download_xtalk_data
from ..modules.xtalk.monster import read_catalog_metadata
from ..types.data_quality_file import DQFile
from ..utils.network import max_delay
from ..utils.yaml_helper import load_yaml
from ..constants import user_parameters_schema

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Dataclass that stores all user parameters for a pycWB analysis run.

    Instances are normally created empty and then populated via one of the two
    loading helpers:

    * :meth:`load_from_yaml` – load from a YAML file, validate against a JSON
      schema, and compute all derived fields.  The YAML file may contain an
      optional ``pycwb_schema`` block at the top level to customise which
      parameters are accepted (see below).
    * :meth:`load_from_dict` – load directly from a plain Python dict (no
      validation or derived-field computation).

    Example
    -------
    .. code-block:: python

        config = Config()
        config.load_from_yaml("user_parameters.yaml")
        print(config.ifo, config.fLow, config.rateANA)

    Custom schema (``pycwb_schema`` block in the YAML file)
    --------------------------------------------------------
    The ``pycwb_schema`` top-level key lets you extend or replace the built-in
    parameter schema without touching any Python code.

    **Extend** – add or override individual fields while keeping all defaults:

    .. code-block:: yaml

        pycwb_schema:
          mode: extend          # optional, 'extend' is the default
          properties:
            my_tag:
              type: string
              default: "unset"

    **Extend via external file** (file contains only the extra ``properties`` dict):

    .. code-block:: yaml

        pycwb_schema:
          mode: extend
          schema_file: ./my_extra_fields.yaml

    **Replace** – supply a complete JSON schema, discarding the built-in one:

    .. code-block:: yaml

        pycwb_schema:
          mode: replace
          schema_file: ./my_full_schema.yaml

    Relative ``schema_file`` paths are resolved relative to the directory that
    contains the config YAML file.

    Derived fields (computed by :meth:`add_derived_key`)
    -----------------------------------------------------
    After loading, the following attributes are set automatically:

    * ``gamma`` – alias of ``cfg_gamma``
    * ``search`` – alias of ``cfg_search``
    * ``rateANA`` – analysis sample rate (``inRate`` or ``fResample`` shifted by ``levelR``)
    * ``nRES`` – number of resolution levels (``l_high - l_low + 1``)
    * ``MRAcatalog`` – full path to the WDM cross-talk catalog
    * ``TDRate`` – time-delay filter sample rate
    * ``nIFO`` / ``nDQF`` – number of IFOs / DQ flag entries
    * ``max_delay`` – maximum light-travel delay between IFOs
    * ``WDM_level`` – list of WDM resolution levels
    * ``dq_files`` – ``DQF`` rows converted to :class:`~pycwb.types.data_quality_file.DQFile` objects
    """
    dq_files: List[DQFile] = field(default_factory=list)
    
    # Loadable/expected parameters from YAML
    outputDir: Optional[str] = None
    logDir: Optional[str] = None
    nproc: Optional[int] = None
    cfg_gamma: Optional[float] = None
    gamma: Optional[float] = field(init=False)
    fResample: Optional[int] = None
    rateANA: Optional[float] = field(init=False)
    levelR: Optional[int] = None
    inRate: Optional[int] = None
    nRES: Optional[int] = field(init=False)
    l_low: Optional[int] = None
    l_high: Optional[int] = None
    fLow: Optional[float] = None
    fHigh: Optional[float] = None
    whiteWindow: Optional[float] = None
    filter_dir: Optional[str] = None
    wdmXTalk: Optional[str] = None
    MRAcatalog: Optional[str] = field(init=False)
    TDRate: Optional[float] = field(init=False)
    lagStep: Optional[float] = None
    lagBuffer: Optional[str] = None
    lagMode: Optional[str] = None
    max_delay: Optional[float] = field(init=False)
    injection: Dict = field(default_factory=dict)
    parallel_injection_trail: bool = False
    analyze_injection_only: bool = False
    injection_padding: float = 1.0
    WDM_beta_order: Optional[int] = None
    WDM_precision: Optional[int] = None
    WDM_level: List[int] = field(default_factory=list)
    cfg_search: Optional[Any] = None
    ifo: List[str] = field(default_factory=list)
    DQF: List[List[Any]] = field(default_factory=list)
    upTDF: Optional[int] = None
    segEdge: Optional[float] = None
    segMLS: Optional[float] = None

    def load_from_yaml(self, file_name, schema=None):
        """
        Load user parameters from a YAML file, validate them, and compute all
        derived fields.

        The YAML file is first scanned for an optional top-level
        ``pycwb_schema`` block.  If present, that block controls which JSON
        schema is used for validation – either extending the default schema
        with extra properties (``mode: extend``) or replacing it wholesale
        (``mode: replace``).  The block is then stripped before validation so
        it does not appear in the resulting parameter set.  See
        :func:`~pycwb.utils.yaml_helper.resolve_schema` for the full format.

        After loading and validating the YAML values, :meth:`add_derived_key`
        is called to compute dependent attributes, followed by
        :meth:`check_xtalk_file`, :meth:`check_MRA_catalog`, and
        :meth:`check_lagStep`.

        Parameters
        ----------
        file_name : str
            Path to the YAML configuration file.
        schema : dict, optional
            JSON schema to validate against.  Defaults to
            ``pycwb.constants.user_parameters_schema``.  A ``pycwb_schema``
            block inside the file can further modify or replace this schema.

        Raises
        ------
        jsonschema.ValidationError
            If the YAML parameters fail schema validation.
        FileNotFoundError
            If the WDM cross-talk catalog cannot be found or downloaded.
        ValueError
            If the MRA catalog layers do not match the configured resolution
            range, or if ``lagStep`` / ``segEdge`` / ``segMLS`` are
            incompatible with WDM pixel parity.
        """
        if schema is None:
            schema = user_parameters_schema
        # load_yaml resolves any pycwb_schema metadata in the YAML file first
        # and merges/replaces the default schema before validation (see
        # yaml_helper.resolve_schema for the supported modes and format).
        params = load_yaml(file_name, schema)

        for key in params:
            setattr(self, key, params[key])

        self.add_derived_key()
        self.check_xtalk_file(self.MRAcatalog)
        self.check_MRA_catalog()
        self.check_lagStep()
        self.check_analyze_injection_only()

    def load_from_dict(self, params: Dict[str, Any]):
        """
        Load user parameters from a plain Python dictionary.

        This is intended for restoring a :class:`Config` from a previously
        serialised (e.g. JSON-dumped) parameter dict.  No schema validation
        or derived-field computation is performed – the dict values are applied
        directly as attributes.

        Parameters
        ----------
        params : dict
            Mapping of parameter names to values.  Unknown keys are accepted
            and set as attributes without validation.
        """
        for key in params:
            setattr(self, key, params[key])

    def add_derived_key(self):
        """
        Compute and set all derived attributes from the loaded base parameters.

        Called automatically by :meth:`load_from_yaml` after the YAML values
        have been applied.  The following attributes are set:

        * ``gamma`` – alias of ``cfg_gamma``
        * ``search`` – alias of ``cfg_search``
        * ``rateANA`` – analysis sample rate: ``fResample >> levelR`` when
          ``fResample > 0``, otherwise ``inRate >> levelR``
        * ``nRES`` – number of resolution levels: ``l_high - l_low + 1``
        * ``filter_dir`` – resolved from the ``HOME_WAT_FILTERS`` environment
          variable when not set explicitly in the YAML file
        * ``MRAcatalog`` – full path ``filter_dir / wdmXTalk``
        * ``TDRate`` – time-delay filter sample rate: ``rateANA * upTDF``
        * ``nIFO`` – number of interferometers
        * ``nDQF`` – number of DQ flag entries
        * ``dq_files`` – ``DQF`` rows converted to
          :class:`~pycwb.types.data_quality_file.DQFile` objects
        * ``max_delay`` – maximum light-travel delay between IFOs
        * ``WDM_level`` – ordered list of WDM resolution levels from
          ``l_low`` to ``l_high``
        """

        self.gamma = self.cfg_gamma
        self.search = self.cfg_search

        # calculate analysis data rate
        if self.fResample > 0:
            self.rateANA = self.fResample >> self.levelR
        else:
            self.rateANA = self.inRate >> self.levelR

        self.nRES = self.l_high - self.l_low + 1

        # load WAT filter directory and set MRAcatalog
        if not self.filter_dir:
            if os.environ.get('HOME_WAT_FILTERS') is None:
                self.filter_dir = os.path.abspath(".")
            else:
                self.filter_dir = os.environ['HOME_WAT_FILTERS']

        self.MRAcatalog = f"{self.filter_dir}/{self.wdmXTalk}"

        # calculate TDRate
        if self.fResample > 0:
            self.TDRate = (self.fResample >> self.levelR) * self.upTDF
        else:
            self.TDRate = (self.inRate >> self.levelR) * self.upTDF
        self.TDRate = float(self.TDRate)

        # derive number of IFOs and DQFs
        self.nIFO = len(self.ifo)
        self.nDQF = len(self.DQF)

        # convert DQF to object
        for dqf in self.DQF:
            self.dq_files.append(DQFile(dqf[0], dqf[1], dqf[2], dqf[3], dqf[4], dqf[5]))

        self.max_delay = max_delay(self.ifo)

        self.WDM_level = [int(self.l_high + self.l_low - i) for i in range(self.l_low, self.l_high + 1)]

    def get_lag_buffer(self):
        """
        Resolve the lag buffer from the current ``lagFile`` / ``lagMode`` settings.

        * When ``lagMode`` is ``'r'`` (read), the contents of ``lagFile`` are
          read into ``lagBuffer`` and ``lagMode`` is updated to ``'s'``
          (string) so the buffer can be passed directly to the network.
        * Otherwise, ``lagBuffer`` is set to ``lagFile`` and ``lagMode`` is
          set to ``'w'`` (write).
        """
        if self.lagMode == "r":
            with open(self.lagFile, "r") as f:
                self.lagBuffer = f.read()
            self.lagMode = 's'
        else:
            self.lagBuffer = self.lagFile
            self.lagMode = 'w'

    @staticmethod
    def check_file(file_name):
        """
        Helper function to check if file exists

        Parameters
        ----------
        file_name : str
            Path to the file

        Raises
        ------
        FileNotFoundError
        """
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File {file_name} does not exist")

    @staticmethod
    def check_xtalk_file(file_name: str) -> bool:
        """
        Helper function to check if xtalk file exists and download it if it does not exist

        Parameters
        ----------
        file_name : str
            Path to the xtalk file

        Returns
        -------
        bool
            True if the file exists or has been downloaded successfully, False otherwise

        Raises
        ------
        FileNotFoundError
            If the file does not exist and cannot be downloaded
        """
        if not os.path.isfile(file_name):
            if check_and_download_xtalk_data(str(pathlib.Path(file_name).name), str(pathlib.Path(file_name).parent)):
                return True
            else:
                raise FileNotFoundError(f"File {file_name} does not exist")
        return True

    def check_lagStep(self):
        """
        Verify that ``lagStep``, ``segEdge``, and ``segMLS`` are compatible
        with WDM pixel parity.

        The WDM circular buffer uses odd/even pixel distinction tracked by the
        MRA catalog.  To prevent mixing between odd and even pixels during lag
        shifts, each of these time intervals must be an integer multiple of
        ``2 * dt_max``, where ``dt_max = 1 / rate_min`` and
        ``rate_min = rateANA >> l_high``.

        Raises
        ------
        ValueError
            If ``rate_min`` is not an integer, or if any of ``lagStep``,
            ``segEdge``, or ``segMLS`` fail the parity check.
        """
        rate_min = self.rateANA >> self.l_high
        dt_max = 1. / rate_min
        if rate_min % 1:
            logger.error("rate min=%s (Hz) is not integer", rate_min)
            raise ValueError("rate min=%s (Hz) is not integer", rate_min)
        if int(self.lagStep * rate_min + 0.001) & 1:
            logger.error("lagStep=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.lagStep,
                         2 * dt_max)
            raise ValueError("lagStep=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.lagStep,
                             2 * dt_max)
        if int(self.segEdge * rate_min + 0.001) & 1:
            logger.error("segEdge=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segEdge,
                         2 * dt_max)
            raise ValueError("segEdge=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segEdge,
                             2 * dt_max)
        if int(self.segMLS * rate_min + 0.001) & 1:
            logger.error("segMLS=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segMLS,
                         2 * dt_max)
            raise ValueError("segMLS=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segMLS,
                             2 * dt_max)

    def check_analyze_injection_only(self) -> None:
        """
        Validate that ``analyze_injection_only`` is compatible with the current configuration.

        When enabled, injections must be configured and time-slide lags
        must not be used (``lag_size`` must be 1).

        Raises
        ------
        ValueError
            If ``analyze_injection_only`` is True but no injections are
            configured, or ``lag_size > 1``.
        """
        if not getattr(self, 'analyze_injection_only', False):
            return
        if not self.injection:
            raise ValueError(
                "analyze_injection_only requires injections to be configured "
                "in the 'injection' block"
            )
        lag_size = getattr(self, 'lagSize', None) or getattr(self, 'lag_size', 1)
        if lag_size > 1:
            raise ValueError(
                f"analyze_injection_only is incompatible with lag_size={lag_size}. "
                "Time-slide background estimation cannot be used with injection-only "
                "analysis. Set lag_size to 1 or disable analyze_injection_only."
            )

    def check_MRA_catalog(self):
        """
        Validate that the MRA catalog is consistent with the configured
        resolution range and, if tagged, update WDM filter parameters.

        The catalog file at ``self.MRAcatalog`` is read and its layer list is
        checked against the expected WDM layers for every resolution level
        between ``l_low`` and ``l_high``.  The number of matching layers must
        equal ``nRES``.

        If the catalog contains a non-zero ``tag`` field, ``WDM_beta_order``
        and ``WDM_precision`` are updated from the catalog's ``beta_order`` and
        ``precision`` metadata fields, overriding any values from the YAML file.

        Raises
        ------
        ValueError
            If the catalog layer counts do not match the configured resolution
            range.
        """
        logger.info("Checking MRA catalog")
        metadata = read_catalog_metadata(self.MRAcatalog)
        layers = metadata['layers'].tolist() if hasattr(metadata['layers'], 'tolist') else [int(x) for x in metadata['layers']]
        n_res = int(metadata['nRes'])

        check_layers = 0
        for i in range(self.l_low, self.l_high + 1):
            level = self.l_high + self.l_low - i
            expected_layers = 2 ** level if level > 0 else 0
            for j in range(n_res):
                if expected_layers == int(layers[j]):
                    check_layers += 1

        if check_layers != self.nRES:
            logger.error("analysis layers do not match the MRA catalog")
            logger.error("analysis layers : ")
            for level in range(self.l_high, self.l_low - 1, -1):
                layers_level = 1 << level if level > 0 else 0
                logger.error("level : %s layers : %s", level, layers_level)

            logger.error("MRA catalog layers : ")
            for i in range(n_res):
                logger.error("layers : %s", int(layers[i]))
            raise ValueError("analysis layers do not match the MRA catalog")

        if float(metadata.get('tag', 0.0)) != 0.0:
            logger.info(
                "MRA catalog has tag %s, updating beta order and precision from MRA catalog",
                metadata.get('tag', 0.0),
            )
            self.WDM_beta_order = int(metadata.get('beta_order', self.WDM_beta_order or 0))
            self.WDM_precision = int(metadata.get('precision', self.WDM_precision or 0))

    @staticmethod
    def get_precision(cluster_size_threshold, healpix_order):
        """
        Encode cluster-size threshold and HEALPix order into a single integer
        precision value used by the likelihood processor.

        Parameters
        ----------
        cluster_size_threshold : int
            Maximum number of pixels per cluster to process in full detail.
        healpix_order : int
            HEALPix order for sky-map resolution.

        Returns
        -------
        int
            Combined precision value: ``cluster_size_threshold + 65536 * healpix_order``.
        """
        return cluster_size_threshold+65536*healpix_order
