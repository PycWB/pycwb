from dataclasses import dataclass
import numpy as np
from .monster import load_catalog, getXTalk_pixels, getXTalk
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
        """
        cluster_xtalk_lookup, cluster_xtalk = getXTalk_pixels(pixels, check, self.layers, self.coeff, self.lookup_table)
        return cluster_xtalk_lookup, cluster_xtalk

    def get_xtalk(self, pix1: Pixel, pix2: Pixel):
        """
        Get the crosstalk coefficients for the given pixel pair.

        Args:
            pix1 (Pixel): A Pixel object.
            pix2 (Pixel): A Pixel object.

        Raises:
            AttributeError: If either pix1 or pix2 does not have 'layer' or 'time' attributes.
        """
        return getXTalk(pix1.layers, pix1.time, pix2.layers, pix2.time, self.layers, self.coeff, self.lookup_table)



    