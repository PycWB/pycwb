import numpy as np

def wavearray_max(a, b):
    """Pure version of max function returning new wavearray"""
    # Validate array compatibility
    if (len(a.data) == 0 or len(b.data) == 0 or
        a.slice.size != b.slice.size or
        a.slice.start != b.slice.start or
        a.slice.stride != b.slice.stride):
        raise ValueError("wavearray.max(): illegal input array")

    # Create new output array
    result = wavearray(np.empty_like(a.data))
    result.slice = a.slice.copy()
    
    K = a.slice.stride
    I = a.slice.start
    N = a.slice.size
    
    # Perform max operation
    for n in range(0, N, K):
        idx = I + n
        result.data[idx] = max(a.data[idx], b.data[idx])
    
    return result

def max_energy(ts, wavelet, dT, N, pattern, hist=None):
    """
    This function computes the maximum energy of a time series within a specified time delay and wave packet pattern.
    """
    if wavelet.wave_type != "WDMT":
        raise ValueError("wseries::maxEnergy(): illegal wavelet")

    # Initial transform
    current_max = WSeries.forward(ts, wavelet, 0)
    xx = ts.copy()
    K = int(ts.rate * abs(dT))
    shape = 1.0

    if pattern != 0:
        # Initialize with pattern processing
        tmp = WSeries.forward(ts, wavelet)
        tmp = tmp.set_freq_range(current_max.getlow(), current_max.gethigh())
        shape = wdm_packet(tmp, pattern, 'E')
        current_max = tmp
        current_max.data = np.zeros_like(current_max.data)

        for k in range(N, K+1, N):
            # Positive time shift
            xx.data = np.roll(ts.data, -k)
            tmp = WSeries.forward(xx, wavelet)
            tmp = tmp.set_freq_range(current_max.getlow(), current_max.gethigh())
            tmp = wdm_packet(tmp, pattern, 'E')
            current_max = wavearray_max(current_max, tmp)

            # Negative time shift
            xx.data = np.roll(ts.data, k)
            tmp = WSeries.forward(xx, wavelet)
            tmp = tmp.set_freq_range(current_max.getlow(), current_max.gethigh())
            tmp = wdm_packet(tmp, pattern, 'E')
            current_max = wavearray_max(current_max, tmp)
    else:
        # Single pixel processing
        for k in range(N, K+1, N):
            xx.data = np.roll(ts.data, -k)
            tmp = WSeries.forward(xx, wavelet, 0)
            current_max = wavearray_max(current_max, tmp)

            xx.data = np.roll(ts.data, k)
            tmp = WSeries.forward(xx, wavelet, 0)
            current_max = wavearray_max(current_max, tmp)

    # Boundary handling
    M = tmp.max_layer() + 1
    current_max = zero_layer(current_max, 0.1)
    current_max = zero_layer(current_max, M-1)

    m = abs(pattern)
    if m in {5, 6, 9}:
        current_max = zero_layer(current_max, 1)
        current_max = zero_layer(current_max, M-2)

    if not pattern:
        return current_max, 1.0
    
    return current_max, gamma_to_gauss(hist)