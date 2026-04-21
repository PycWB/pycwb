import struct
import logging
import numpy as np
import pathlib
from numba import njit, prange


logger = logging.getLogger(__name__)


def _read_catalog_header_and_layers(data):
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
    return {
        'nRes': nRes,
        'tag': tag,
        'beta_order': BetaOrder,
        'precision': precision,
        'kwdm': KWDM,
        'layers': np.array(layers, dtype=np.int32),
        'offset': offset,
    }


def read_catalog_metadata(fn):
    fn = pathlib.Path(fn)

    if fn.suffix == ".npz":
        data = np.load(fn)
        return {
            'nRes': int(data['nRes']),
            'tag': float(data['tag']) if 'tag' in data.files else 0.0,
            'beta_order': float(data['beta_order']) if 'beta_order' in data.files else 0.0,
            'precision': float(data['precision']) if 'precision' in data.files else 0.0,
            'kwdm': float(data['kwdm']) if 'kwdm' in data.files else 0.0,
            'layers': np.array(data['layers'], dtype=np.int32),
        }

    with open(fn, "rb") as f:
        data = f.read()
    metadata = _read_catalog_header_and_layers(data)
    metadata.pop('offset', None)
    return metadata


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
        logger.info("Loading %s", fn)
        data = np.load(fn)
        return data['xtalk_coeff'], data['xtalk_lookup_table'], data['layers'], data['nRes']

    # Check if there is converted file
    # if ext of fn is .bin, search if there is a .npz file with the same name
    # under the same directory and working directory
    if pathlib.Path(fn).suffix == ".bin" or pathlib.Path(fn).suffix == ".xbin":
        logger.info("A .bin file is detected, searching for .npz file with the same name.")
        # npz_fn = fn.replace(".bin", ".npz")
        npz_fn = pathlib.Path(fn).with_suffix(".npz")
        if pathlib.Path(npz_fn).exists():
            logger.info(".npz file found: %s, loading the catalog from the .npz file.", npz_fn)
            return load_catalog(npz_fn)

        # npz_fn = pathlib.Path(fn).name.replace(".bin", ".npz")
        # if pathlib.Path(npz_fn).exists():
        #     print(f".npz file found: {npz_fn}, loading the catalog from the .npz file.")
        #     return load_catalog(npz_fn)

    with open(fn, "rb") as f:
        data = f.read()  # Read the entire file into memory

    metadata = _read_catalog_header_and_layers(data)
    offset = metadata['offset']
    nRes = metadata['nRes']
    tag = metadata['tag']
    BetaOrder = metadata['beta_order']
    precision = metadata['precision']
    KWDM = metadata['kwdm']
    layers = metadata['layers'].tolist()

    def unpack(fmt):
        nonlocal offset
        result = struct.unpack_from(fmt, data, offset)
        offset += struct.calcsize(fmt)
        return result

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
                 nRes=nRes, tag=tag, beta_order=BetaOrder, precision=precision, kwdm=KWDM)
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
def getXTalk_pixels_numba(pixels, check, layers, xtalk_coeff, xtalk_lookup_table):
    n_pix = len(pixels)

    # First pass: count valid (non-filtered) pairs so we can allocate exact memory.
    # The original O(N²) dense pre-allocation (n_pix*n_pix rows) caused OOM for
    # large clusters (e.g. 216 MB at N=2600, 3.2 GB at N=10000).
    count = 0
    for i in range(n_pix):
        for j in range(n_pix):
            tmpOvlp = getXTalk(pixels[i][0], pixels[i][1], pixels[j][0], pixels[j][1],
                               layers, xtalk_coeff, xtalk_lookup_table)
            if tmpOvlp[0] <= 2:
                count += 1

    clusterCC = np.empty((count, 8), dtype=np.float32)
    clusterCC_lookup = np.empty((n_pix, 2), dtype=np.int32)

    # Second pass: fill entries into the exact-size array.
    index_counter = 0
    for i in range(n_pix):
        clusterCC_lookup[i, 0] = index_counter
        for j in range(n_pix):
            tmpOvlp = getXTalk(pixels[i][0], pixels[i][1], pixels[j][0], pixels[j][1],
                               layers, xtalk_coeff, xtalk_lookup_table)
            if tmpOvlp[0] > 2:
                continue

            clusterCC[index_counter][0] = float(j)
            clusterCC[index_counter][1] = tmpOvlp[0] ** 2 + tmpOvlp[1] ** 2
            clusterCC[index_counter][2] = tmpOvlp[2] ** 2 + tmpOvlp[3] ** 2
            # col[3] = total squared overlap; read from the row just written (was a bug:
            # the original read clusterCC[i*n_pix+j] which is uninitialized memory
            # whenever index_counter != i*n_pix+j due to sparse filtering).
            clusterCC[index_counter][3] = clusterCC[index_counter][1] + clusterCC[index_counter][2]
            clusterCC[index_counter][4] = tmpOvlp[0]   # xt[0]
            clusterCC[index_counter][5] = tmpOvlp[2]   # xt[2]
            clusterCC[index_counter][6] = tmpOvlp[1]   # xt[1]
            clusterCC[index_counter][7] = tmpOvlp[3]   # xt[3]
            index_counter += 1

        clusterCC_lookup[i, 1] = index_counter

    return clusterCC_lookup, clusterCC


@njit(cache=True, parallel=True)
def _compute_null_likelihood_numba(
    null_k_set, like_k_set,
    pn, pN, ps, pS,
    gn, ec,
    xtalks_lookup, xtalks,
    null_mask, like_mask,
    null_out, like_out,
):
    """Compute per-pixel null and likelihood statistics via sparse xtalk sum.

    Replaces the O(N²) Python-dispatch loops in fill_detection_statistic.
    Both loops are fully compiled and data-parallel (prange over pixels).

    The original Python code had inner loops over ``null_k_set`` / ``like_k_set``
    (the filtered subsets), not over all cluster pixels.  ``null_mask`` /
    ``like_mask`` reproduce that filtering: only neighbours j where
    ``null_mask[j]`` (resp. ``like_mask[j]``) is True are accumulated.

    Column layout of xtalks (from getXTalk_pixels_numba):
        col 0 : neighbour pixel index j
        col 4 : xt[0]   col 6 : xt[1]   col 5 : xt[2]   col 7 : xt[3]

    Parameters
    ----------
    null_k_set, like_k_set : int64 arrays
        Core pixel indices to process (pre-filtered: gn > 0 / ec > 0).
    pn, pN, ps, pS : float64 arrays, shape (n_ifo, n_pix)
        Per-pixel amplitude arrays.
    gn, ec : float64 arrays, shape (n_pix,)
        Gaussian noise correction / coherent energy per pixel.
    xtalks_lookup : int64 array, shape (n_pix, 2)
        CSR row-pointer: xtalks[xtalks_lookup[i,0]:xtalks_lookup[i,1]] are
        the entries for pixel i.
    xtalks : float32 array, shape (N_pairs, 8)
        Sparse xtalk coefficient table.
    null_mask, like_mask : bool arrays, shape (n_pix,)
        Boolean membership masks for the inner-sum sets.  null_mask[j] is
        True iff j is in null_k_set; like_mask[j] iff j is in like_k_set.
    null_out, like_out : float64 arrays, shape (n_pix,)
        Output arrays — written in place for pixels in the respective sets.
    """
    n_ifo = pn.shape[0]

    # --- Null loop ---
    for ii in prange(len(null_k_set)):
        i = null_k_set[ii]
        if gn[i] <= 0.0:
            continue
        acc = 0.0
        start = xtalks_lookup[i, 0]
        end   = xtalks_lookup[i, 1]
        for m in range(start, end):
            j = int(xtalks[m, 0])
            # Mirror original: inner sum was over null_k_set only
            if not null_mask[j]:
                continue
            xt0 = float(xtalks[m, 4])
            xt1 = float(xtalks[m, 6])
            xt2 = float(xtalks[m, 5])
            xt3 = float(xtalks[m, 7])
            d = 0.0
            for ifo in range(n_ifo):
                d += (xt0 * pn[ifo, i] * pn[ifo, j]
                      + xt1 * pn[ifo, i] * pN[ifo, j]
                      + xt2 * pN[ifo, i] * pn[ifo, j]
                      + xt3 * pN[ifo, i] * pN[ifo, j])
            acc += d
        null_out[i] = acc

    # --- Likelihood loop (identical structure) ---
    for ii in prange(len(like_k_set)):
        i = like_k_set[ii]
        if ec[i] <= 0.0:
            continue
        acc = 0.0
        start = xtalks_lookup[i, 0]
        end   = xtalks_lookup[i, 1]
        for m in range(start, end):
            j = int(xtalks[m, 0])
            # Mirror original: inner sum was over like_k_set only
            if not like_mask[j]:
                continue
            xt0 = float(xtalks[m, 4])
            xt1 = float(xtalks[m, 6])
            xt2 = float(xtalks[m, 5])
            xt3 = float(xtalks[m, 7])
            d = 0.0
            for ifo in range(n_ifo):
                d += (xt0 * ps[ifo, i] * ps[ifo, j]
                      + xt1 * ps[ifo, i] * pS[ifo, j]
                      + xt2 * pS[ifo, i] * ps[ifo, j]
                      + xt3 * pS[ifo, i] * pS[ifo, j])
            acc += d
        like_out[i] = acc
