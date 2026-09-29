"""Apply the reference likelihood TD-pixel limit without losing cluster volume."""
from copy import copy
import numpy as np


def select_likelihood_pixels(cluster, batch):
    """Return a working cluster and restoration state for cWB's BATCH limit.

    cwb2G::Likelihood calls loadTDampSSE(..., BATCH, BATCH), selecting the
    loudest BATCH pixels before sky scanning. BATCH=0 disables this limit.
    Retain input order within the selected set, as monster::getXTalk does.
    """
    limit = int(batch)
    pixels = cluster.pixel_arrays
    if limit <= 0 or len(pixels) <= limit:
        return cluster, None
    selected = np.sort(np.argsort(-pixels.likelihood, kind='stable')[:limit])
    working = copy(cluster)
    working.pixel_arrays = pixels[selected]
    return working, (pixels, selected)


def restore_likelihood_pixels(cluster, state):
    """Restore excluded non-core pixels so event volume retains its meaning."""
    if state is None:
        return cluster
    full, selected = state
    fitted = cluster.pixel_arrays
    full.core[:] = False
    full.likelihood[:] = 0
    full.null[:] = 0
    for name in ('core', 'likelihood', 'null'):
        getattr(full, name)[selected] = getattr(fitted, name)
    for name in ('noise_rms', 'wave', 'w_90', 'asnr', 'a_90'):
        getattr(full, name)[:, selected] = getattr(fitted, name)
    cluster.pixel_arrays = full
    return cluster
