"""Bounded reusable CUDA storage with exact-size synchronous host transfers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver

DEFAULT_BUDGET = 64 * 1024**2
"""Default resident byte budget of one :class:`Workspace`."""


class Workspace:
    """Named power-of-two device slots reused across synchronous stage calls.

    Parameters
    ----------
    budget : int, optional
        Maximum resident bytes over all slots. Growing a slot beyond the budget
        raises ``MemoryError`` instead of allocating silently.

    Notes
    -----
    Every stage call is synchronous, so a slot may be handed to the next call
    as soon as the previous download has returned. Slots are never shared
    between threads.
    """

    def __init__(self, budget: int = DEFAULT_BUDGET) -> None:
        self.budget = int(budget)
        self.slots: dict[str, Any] = {}

    def reserve(self, name: str, count: int, dtype: Any) -> Any:
        """Return a device array for ``name`` holding at least ``count`` items.

        Parameters
        ----------
        name : str
            Slot name.
        count : int
            Required element count; the slot grows to the next power of two.
        dtype : numpy dtype-like
            Element type. A dtype change reallocates the slot.

        Returns
        -------
        numba device array
            Flat array of capacity ``>= count``; callers use only the first
            ``count`` elements.

        Raises
        ------
        ValueError
            For negative counts or object dtypes.
        MemoryError
            If the resident bytes would exceed the budget.
        """
        dtype = np.dtype(dtype)
        if count < 0 or dtype.hasobject:
            raise ValueError("Invalid device workspace request")
        old = self.slots.get(name)
        if old is not None and old.dtype == dtype and old.size >= count:
            return old
        capacity = max(1, 1 << (max(1, int(count)) - 1).bit_length())
        retained = sum(slot.nbytes for key, slot in self.slots.items() if key != name)
        if retained + capacity * dtype.itemsize > self.budget:
            raise MemoryError("CUDA workspace exceeds its resident byte budget")
        result = cuda.device_array(capacity, dtype)
        self.slots[name] = result
        return result

    def upload(self, name: str, data: np.ndarray) -> Any:
        """Copy ``data`` into slot ``name`` and return the slot."""
        data = np.ascontiguousarray(data)
        destination = self.reserve(name, data.size, data.dtype)
        if data.nbytes:
            driver.host_to_device(destination, data, data.nbytes)
        return destination

    @staticmethod
    def download(source: Any, shape: tuple[int, ...]) -> np.ndarray:
        """Copy the first ``shape`` elements of ``source`` to a new host array.

        Raises
        ------
        ValueError
            If ``shape`` needs more bytes than the slot holds.
        """
        result = np.empty(shape, source.dtype)
        if result.nbytes > source.nbytes:
            raise ValueError("Download exceeds its device allocation")
        if result.nbytes:
            driver.device_to_host(result, source, result.nbytes)
        return result
