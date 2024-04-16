import struct
import numpy as np
import pathlib
from numba import njit


def load_catalog(fn, dump=True):
    """
    Load the crosstalk catalog file, if the file is in .npz format, then load the file as npz, otherwise load the bin
    file and parse it. If parameter dump is True, then save the parsed data into a .npz file with the same name as the
    input file in current working directory.

    Parameters
    ----------
    fn : str or Path
        The file name of the crosstalk catalog file
    dump : bool
        If True, save the parsed data into a .npz file with the same name as the input file in current working directory

    Returns
    -------
    xtalk_coeff : np.ndarray
        The crosstalk coefficients
    xtalk_lookup_table : np.ndarray
        The lookup table for crosstalk coefficients
    layers : np.ndarray
        The number of layers for each resolution
    nRes : int
        The number of resolutions
    """

    # if ext of fn is .npz, then load the file as npz
    if pathlib.Path(fn).suffix == ".npz":
        print(f"Loading {fn}")
        data = np.load(fn)
        return data['xtalk_coeff'], data['xtalk_lookup_table'], data['layers'], data['nRes']

    # Check if there is converted file
    # if ext of fn is .bin, search if there is a .npz file with the same name
    # under the same directory and working directory
    if pathlib.Path(fn).suffix == ".bin" or pathlib.Path(fn).suffix == ".xbin":
        print(f"A .bin file is detected, searching for .npz file with the same name.")
        # npz_fn = fn.replace(".bin", ".npz")
        npz_fn = pathlib.Path(fn).with_suffix(".npz")
        if pathlib.Path(npz_fn).exists():
            print(f".npz file found: {npz_fn}, loading the catalog from the .npz file.")
            return load_catalog(npz_fn)

        # npz_fn = pathlib.Path(fn).name.replace(".bin", ".npz")
        # if pathlib.Path(npz_fn).exists():
        #     print(f".npz file found: {npz_fn}, loading the catalog from the .npz file.")
        #     return load_catalog(npz_fn)

    with open(fn, "rb") as f:
        data = f.read()  # Read the entire file into memory

    offset = 0

    def unpack(fmt):
        nonlocal offset
        result = struct.unpack_from(fmt, data, offset)
        offset += struct.calcsize(fmt)
        return result

    nRes = int(unpack('f')[0])

    if nRes < 0:
        nRes = -nRes
        tag, BetaOrder, precision, KWDM = unpack('4f')
    else:
        tag, BetaOrder, precision, KWDM = 0, 0, 0, 0

    layers = [int(unpack('f')[0]) for _ in range(nRes)]
    max_layers = max(layers)
    lookup_table = np.zeros((nRes, nRes, max_layers + 1, 2, 2), dtype=np.int32)
    xtalk_coeff = []
    entry_index = 0
    for i in range(nRes):
        for j in range(i + 1):
            for k in range(layers[i] + 1):
                for l in range(2):
                    oa_size = int(unpack('f')[0])
                    oa_data = np.frombuffer(data, dtype=np.dtype([('index', 'i'), ('CC', '4f')]), count=oa_size,
                                            offset=offset).tolist()
                    offset += oa_size * struct.calcsize('i4f')
                    lookup_table[i, j, k, l, 0] = entry_index
                    for entry in oa_data:
                        xtalk_coeff.append(
                            np.array([entry[0], entry[1][0], entry[1][1], entry[1][2], entry[1][3]], dtype=np.float32))
                        entry_index += 1
                    lookup_table[i, j, k, l, 1] = entry_index

    if dump:
        # dump to current working directory
        # filename = pathlib.Path(fn).name.replace(".bin", ".npz")
        filename = pathlib.Path(fn).with_suffix(".npz")
        np.savez(filename, xtalk_coeff=xtalk_coeff, xtalk_lookup_table=lookup_table, layers=layers,
                 nRes=nRes)
    return np.array(xtalk_coeff), lookup_table, np.array(layers), nRes


@njit(cache=True)
def getXTalk(nLayer1, indx1, nLayer2, indx2, layers, xtalk_coeff, xtalk_lookup_table):
    r1, r2 = -1, -1  # Default values if the condition is not met
    for i, layer in enumerate(layers):
        if layer == nLayer1 - 1:
            r1 = i
        if layer == nLayer2 - 1:
            r2 = i

    if r1 == -1 or r2 == -1:
        raise ValueError(f"Resolution not found: {nLayer1} {nLayer2} in layers")

    # Simplify swapping logic
    swap = r1 < r2
    if swap:
        r1, r2, indx1, indx2 = r2, r1, indx2, indx1

    # Compute index values
    layer1 = layers[r1]
    time1, freq1 = divmod(indx1, layer1 + 1)
    odd = time1 & 1
    index = indx2 - (time1 - odd) * (layer1 // layers[r2]) * (layers[r2] + 1)

    # Vector retrieval and processing
    ret = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)  # Preset array
    entry_index = xtalk_lookup_table[r1][r2][freq1][odd]
    for item in xtalk_coeff[entry_index[0]:entry_index[1]]:
        if index == int(item[0]):
            ret[0] = item[1]
            ret[1] = item[2]
            ret[2] = item[3]
            ret[3] = item[4]

            if swap:
                ret[1], ret[2] = ret[2], ret[1]
            break

    return ret


@njit(cache=True)
def getXTalk_pixels_np(pixels, check, layers, xtalk_coeff, xtalk_lookup_table):
    n_pix = len(pixels)

    clusterCC = np.empty((n_pix * n_pix, 8), dtype=np.float32)
    clusterCC_lookup = np.empty((n_pix, 2), dtype=np.int32)

    index_counter = 0
    for i, pixi in enumerate(pixels):
        # tmp_data = []
        clusterCC_lookup[i, 0] = index_counter
        for j, pixj in enumerate(pixels):
            tmpOvlp = getXTalk(pixi[0], pixi[1], pixj[0], pixj[1], layers, xtalk_coeff, xtalk_lookup_table)
            if tmpOvlp[0] > 2:
                continue

            M = j + 1
            # N = 0 if i == j else len(tmp_data) // 8 + 1
            clusterCC[index_counter][0] = float(M - 1)
            clusterCC[index_counter][1] = tmpOvlp[0] ** 2 + tmpOvlp[1] ** 2
            clusterCC[index_counter][2] = tmpOvlp[2] ** 2 + tmpOvlp[3] ** 2
            clusterCC[index_counter][3] = clusterCC[i * n_pix + j][1] + clusterCC[i * n_pix + j][2]
            clusterCC[index_counter][4] = tmpOvlp[0]
            clusterCC[index_counter][5] = tmpOvlp[2]
            clusterCC[index_counter][6] = tmpOvlp[1]
            clusterCC[index_counter][7] = tmpOvlp[3]
            index_counter += 1

        clusterCC_lookup[i, 1] = index_counter

        # sizeCC.append(len(tmp_data) // 8)
        # clusterCC.append(np.array(tmp_data, dtype=np.float32))
    return clusterCC_lookup, clusterCC[0:index_counter]


def getXTalk_pixels(pixels, check, layers, xtalk_coeff, xtalk_lookup_table):
    pixels_np = np.array([[pix.layers, pix.time] for pix in pixels])

    clusterCC_lookup, clusterCC = getXTalk_pixels_np(pixels_np, check, layers, xtalk_coeff, xtalk_lookup_table)

    return clusterCC_lookup, clusterCC
