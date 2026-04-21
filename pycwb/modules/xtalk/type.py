from dataclasses import dataclass
import numpy as np
from .monster import load_catalog, getXTalk_pixels_numba, getXTalk
from pycwb.types.network_pixel import Pixel

@dataclass
class XTalk:
    """
    lookup_table: np.ndarray  # shape: (n_layers, n_pixels, n_times), stores lookup indices for crosstalk coefficients

    Attributes:
        coeff (np.ndarray): 1D array of arrays, where each element is an array of crosstalk coefficients for a layer.
        lookup_table (np.ndarray): Lookup table for crosstalk.
        layers (np.ndarray): Array of layer indices.
        nRes (int): Number of resolutions; indicates how many distinct resolution levels are present in the crosstalk data.
    """
    coeff: np.ndarray
    lookup_table: np.ndarray
    layers: np.ndarray
    nRes: int

    @classmethod
    def load(cls, fn, dump=True):
        """
        Load the crosstalk catalog file, if the file is in .npz format, then load the file as npz, otherwise load the bin
        file and parse it. If parameter dump is True, then save the parsed data into a .npz file with the same name as the
        input file in current working directory.
        """
        xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(fn, dump)
        return cls(xtalk_coeff, xtalk_lookup_table, layers, nRes)

    def get_xtalk_pixels(self, pixels, check=True):
        """
        Get the crosstalk coefficients for the given pixels.

        Accepts either a ``list[Pixel]`` (legacy) or a
        :class:`~pycwb.types.pixel_arrays.PixelArrays` instance.
        Returns a ``(cluster_xtalk_lookup, cluster_xtalk)`` tuple.
        """
        from pycwb.types.pixel_arrays import PixelArrays
        if isinstance(pixels, PixelArrays):
            pix_mat = np.column_stack([pixels.layers, pixels.time]).astype(np.int64)
        else:
            pix_mat = np.array([[pix.layers, pix.time] for pix in pixels])
        cluster_xtalk_lookup, cluster_xtalk = getXTalk_pixels_numba(
            pix_mat, check, self.layers, self.coeff, self.lookup_table
        )
        return cluster_xtalk_lookup, cluster_xtalk

    def get_xtalk(self, pix1, pix2):
        """Get the crosstalk coefficients for the given pixel pair.

        ``pix1`` / ``pix2`` can be ``Pixel`` objects **or** plain
        ``(layers, time)`` tuples / named objects with those attributes.
        """
        if isinstance(pix1, tuple):
            l1, t1 = pix1
        else:
            l1, t1 = pix1.layers, pix1.time
        if isinstance(pix2, tuple):
            l2, t2 = pix2
        else:
            l2, t2 = pix2.layers, pix2.time
        return getXTalk(l1, t1, l2, t2, self.layers, self.coeff, self.lookup_table)



    