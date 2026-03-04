"""Abstract base class for pycWB catalogs.

Any catalog implementation must subclass :class:`BaseCatalog` and implement
all abstract methods.  Code that calls the catalog should be typed against
:class:`BaseCatalog` so that the concrete implementation can be swapped by
changing only the import.

Example — future JSON implementation::

    class CatalogJSON(BaseCatalog):
        DEFAULT_EXTENSION = ".json"
        DEFAULT_FILENAME  = "catalog.json"

        @classmethod
        def create(cls, filename, config, jobs): ...
        @classmethod
        def open(cls, filename): ...
        ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union


class BaseCatalog(ABC):
    """Interface contract for all pycWB trigger catalog implementations.

    Subclasses must declare:

    - :attr:`DEFAULT_EXTENSION` – file extension including the leading dot,
      e.g. ``".parquet"`` or ``".json"``.
    - :attr:`DEFAULT_FILENAME`  – bare filename (without path) used as the
      default when callers do not specify one, e.g. ``"catalog.parquet"``.

    Subclasses must implement all abstract methods below.
    """

    #: File extension (including leading dot) for this backend.
    DEFAULT_EXTENSION: str = ""

    #: Default bare filename for this backend.
    DEFAULT_FILENAME: str = "catalog"

    # ------------------------------------------------------------------
    # Construction — class-methods rather than __init__ so concrete
    # implementors control file creation vs open semantics clearly.
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def create(cls, filename: str, config, jobs: list) -> "BaseCatalog":
        """Initialise a new, empty catalog on disk and return the instance.

        Parameters
        ----------
        filename : str
            Destination path for the catalog file.
        config : Config
            Pipeline configuration to embed in the catalog.
        jobs : list
            Job segment list to embed in the catalog.
        """

    @classmethod
    @abstractmethod
    def open(cls, filename: str) -> "BaseCatalog":
        """Open an existing catalog file and return the instance.

        Parameters
        ----------
        filename : str
            Path to an existing catalog file.

        Raises
        ------
        FileNotFoundError
            If *filename* does not exist.
        """

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def version(self) -> str:
        """pycwb version string recorded when the catalog was created."""

    @property
    @abstractmethod
    def config(self) -> dict:
        """Pipeline configuration dict stored in the catalog."""

    @property
    @abstractmethod
    def jobs(self) -> list:
        """List of job-segment dicts stored in the catalog."""

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @abstractmethod
    def add_triggers(self, triggers: Union["Trigger", list]) -> None:  # noqa: F821
        """Append one or more :class:`~pycwb.types.trigger.Trigger` objects.

        Parameters
        ----------
        triggers : Trigger | list[Trigger]
            One or more trigger objects to persist.
        """

    @abstractmethod
    def add_events(self, events) -> None:
        """Convert legacy ``Event`` objects and append as triggers.

        Parameters
        ----------
        events : Event | list[Event]
            One or more legacy event objects.
        """

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @abstractmethod
    def triggers(self, deduplicate: bool = True):
        """Return all trigger records.

        The return type is backend-dependent (e.g. ``pyarrow.Table`` for
        the Parquet backend), but should support iteration and column access.

        Parameters
        ----------
        deduplicate : bool
            Remove duplicate ``(job_id, id)`` pairs, keeping the last entry.
        """

    @abstractmethod
    def filter(self, *conditions) -> object:
        """Return trigger records matching *all* conditions.

        Each condition may be a string expression or a backend-native
        predicate object.

        Parameters
        ----------
        *conditions
            Filtering conditions (string expressions or native predicates).
        """

    @abstractmethod
    def query(self, sql: str) -> object:
        """Run a SQL query against the trigger records.

        The trigger table is accessible as ``"triggers"`` inside the SQL.

        Parameters
        ----------
        sql : str
            SQL statement.
        """

    @abstractmethod
    def live_time(self, filters: Optional[list] = None) -> list:
        """Return per-lag livetime dicts.

        Each dict contains at least ``"shift"``, ``"livetime"``, ``"lag"``.

        Parameters
        ----------
        filters : list of str, optional
            Python boolean expressions applied to each livetime dict.
        """
