import struct
import numpy as np
from numba import njit, jit


def load_catalog(fn):
    """
    Load cross-talk catalog from file

    Parameters
    ----------

    fn : str
        filename

    Returns
    -------

    catalog : list
        cross-talk catalog
    layers : list[int]
        number of layers for each resolution
    nRes : int
        number of resolutions
    """
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

    catalog = []
    for i in range(nRes):
        catalog_row = []
        for j in range(i + 1):
            layer_data = []
            for k in range(layers[i] + 1):
                xtalk_arrays = []
                for l in range(2):
                    oa_size = int(unpack('f')[0])
                    oa_data = np.frombuffer(data, dtype=np.dtype([('index', 'i'), ('CC', '4f')]), count=oa_size,
                                            offset=offset).tolist()
                    offset += oa_size * struct.calcsize('i4f')
                    xtalk_arrays.append(oa_data)
                layer_data.append(xtalk_arrays)
            catalog_row.append(layer_data)
        catalog.append(catalog_row)

    return catalog, layers, nRes


def getXTalk(nLayer1, indx1, nLayer2, indx2, layers, catalog):
    # Simplify the search for layer indices
    r1, r2 = layers.index(nLayer1 - 1) if nLayer1 - 1 in layers else -1, \
        layers.index(nLayer2 - 1) if nLayer2 - 1 in layers else -1

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
    vector = catalog[r1][r2][freq1][odd]
    for item in vector:
        if index == item[0]:
            ret[:4] = item[1]
            if swap:
                ret[1], ret[2] = ret[2], ret[1]
            break

    return ret


def getXTalk_pixels(pixels, check, layers, catalog):
    sizeCC = []
    clusterCC = []

    for i, pixi in enumerate(pixels):
        if pixi is None or (check and not pixi.td_amp.size):
            continue

        tmp_data = []
        for j, pixj in enumerate(pixels):
            if pixj is None or (check and not pixj.td_amp.size):
                continue

            tmpOvlp = getXTalk(pixi.layers, pixi.time, pixj.layers, pixj.time, layers, catalog)
            if tmpOvlp[0] > 2:
                continue

            M = j + 1
            N = 0 if i == j else len(tmp_data) // 8 + 1

            tmp_data.extend([
                float(M - 1),
                tmpOvlp[0]**2 + tmpOvlp[1]**2,
                tmpOvlp[2]**2 + tmpOvlp[3]**2,
                sum(tmp_data[1::8]) + tmpOvlp[0]**2 + tmpOvlp[1]**2,
                tmpOvlp[0],
                tmpOvlp[2],
                tmpOvlp[1],
                tmpOvlp[3]
            ])

        sizeCC.append(len(tmp_data) // 8)
        clusterCC.append(np.array(tmp_data, dtype=np.float32))

    return sizeCC, clusterCC

# @njit(cache=True)
# def getXTalk(nLayer1, indx1, nLayer2, indx2, nRes, layers, catalog):
#     # Assuming ret is a dictionary representing the struct xtalk
#     # ret = {'index': 1, 'CC': [3.0, 3.0, 3.0, 3.0]}
#     ret = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
#
#     r1, r2 = -1, -1
#     for i, layer in enumerate(layers):
#         if nLayer1 == layer + 1:
#             r1 = i
#             break
#
#     for i, layer in enumerate(layers):
#         if nLayer2 == layer + 1:
#             r2 = i
#             break
#
#     if r1 == -1 or r2 == -1:
#         print(f"monster::getXTalk : resolution not found {nRes} {nLayer1} {nLayer2} {r1} {r2} {layers[0]} {layers[6]}")
#         raise ValueError("Resolution not found")
#
#     swap = False
#     if r1 < r2:
#         r1, r2 = r2, r1
#         indx1, indx2 = indx2, indx1
#         swap = True
#
#     time1 = indx1 // (layers[r1] + 1)
#     freq1 = indx1 % (layers[r1] + 1)
#
#     odd = time1 & 1
#
#     index = indx2 - (time1 - odd) * (layers[r1] // layers[r2]) * (layers[r2] + 1)
#
#     vector = catalog[r1][r2][freq1][odd]
#     for item in vector:
#         if index == item[0]:
#             for i in range(4):
#                 ret[i] = item[1][i]
#             if swap:
#                 # ret['CC'][1], ret['CC'][2] = ret['CC'][2], ret['CC'][1]
#                 ret[1], ret[2] = ret[2], ret[1]
#             break
#
#     return ret
#
#
# def getXTalk_pixels(pixels, check, nRes, layers, catalog):
#     """
#     Get cross-talk for each pixel in the given pixel list
#
#     Parameters
#     ----------
#
#     pixels : list[Pixel]
#         list of pixels
#     check : bool
#         if True, check if pixel has time-delay data
#     nRes : int
#         number of resolutions
#     layers : list[int]
#         number of layers for each resolution
#     catalog : list
#         cross-talk catalog
#
#     Returns
#     -------
#
#     sizeCC : list[int]
#         number of cross-talk elements for each pixel
#     clusterCC : list[np.ndarray]
#         cross-talk elements for each pixel
#     """
#
#     # turn pixels into numpy array
#     # pixels_np = np.array([[pix.layers, pix.time] for pix in pixels])
#     # layers = np.array(layers, dtype=np.int32)
#
#     pI = []
#     sizeCC = []
#     clusterCC = []
#
#     for i, pixi in enumerate(pixels):
#         if pixi is None:
#             print("monster::netcluster error: NULL pointer")
#             exit(1)
#         if check and len(pixi.td_amp) == 0:
#             continue
#         pI.append(i)
#
#         N = M = K = 0
#         tmp_data = np.zeros(8 * len(pixels), dtype=np.float32)
#
#         for j, pixj in enumerate(pixels):
#             if pixj is None:
#                 print("monster::netcluster error: NULL pointer")
#                 exit(1)
#             if check and len(pixj.td_amp) == 0:
#                 continue
#             M += 1
#             tmpOvlp = getXTalk(pixi.layers, pixi.time, pixj.layers, pixj.time, nRes, layers, catalog)
#             if tmpOvlp[0] > 2:
#                 continue
#
#             N = 0 if i == j else K + 1
#             K += 1 if i != j else 0
#
#             index = N * 8
#             tmp_data[index:index+8] = [
#                 float(M - 1),
#                 tmpOvlp[0]**2 + tmpOvlp[1]**2,
#                 tmpOvlp[2]**2 + tmpOvlp[3]**2,
#                 tmp_data[index + 1] + tmp_data[index + 2],
#                 tmpOvlp[0],
#                 tmpOvlp[2],
#                 tmpOvlp[1],
#                 tmpOvlp[3]
#             ]
#
#         N += 1
#         p8 = tmp_data[:N * 8]
#         sizeCC.append(N)
#         clusterCC.append(p8)
#
#     return sizeCC, clusterCC



# catalog, layers, nRes = load_catalog(fn)
# getXTalk_pixels(pixels, True, nRes, layers, catalog)