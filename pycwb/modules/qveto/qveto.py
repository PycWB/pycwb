import numpy as np


def get_qveto(wf, NTHR=1, ATHR=7.58859):
    """
    Compute Qveto and Qfactor for a given waveform.
    
    Parameters:
    wf (np.ndarray): Input waveform (1D array).
    NTHR (int): Neighbor threshold for peak inclusion (default=2).
    ATHR (float): Amplitude threshold ratio for external peaks (default=10).
    
    Returns:
    tuple: (Qveto, Qfactor) metrics.
    """
    wf = np.asarray(wf, dtype=np.float64)

    ## Resample waveform to 4 times the original length using FFT
    n = len(wf)
    # Real FFT (forward transform)
    spec = np.fft.rfft(wf)
    # Pad spectrum to length 2*n + 1 for output size 4*n
    new_spec = np.zeros(2 * n + 1, dtype=complex)
    new_spec[:len(spec)] = spec
    # Inverse FFT to get resampled waveform (4*n points)
    x = np.fft.irfft(new_spec, n=4 * n)


    # Find zero crossings and extract segment maxima
    signs = np.sign(x)
    # Identify zero crossings (where sign changes)
    crossings = np.where(np.diff(signs, prepend=signs[0]) != 0)[0]
    # If no crossings found, return zeros
    if len(crossings) < 2:
        return (0.0, 0.0)
    
    # Compute max absolute value in each segment
    a = []
    start_idx = 0
    for end_idx in crossings[1:]:
        segment = x[start_idx:end_idx]
        a.append(np.max(np.abs(segment)))
        start_idx = end_idx
    # Handle last segment
    if start_idx < len(x):
        segment = x[start_idx:]
        a.append(np.max(np.abs(segment)))
    a = np.array(a)
    
    # Find global maximum and its index
    if len(a) == 0:
        return (0.0, 0.0)
    imax = np.argmax(a)
    amax = a[imax]
    
    # Compute Qveto (energy ratio)
    indices = np.arange(len(a))
    # Mask for peaks inside the NTHR window
    mask_in = (np.abs(indices - imax) <= NTHR)
    ein = np.sum(a[mask_in] ** 2)
    # Mask for peaks outside the window and above amplitude threshold
    mask_out = ~mask_in & (a > amax / ATHR)
    eout = np.sum(a[mask_out] ** 2)
    Qveto = eout / ein if ein > 0 else 0.0
    
    # Compute Qfactor (using neighboring peaks)
    if imax < 1 or imax >= len(a) - 1:
        Qfactor = 0.0
    else:
        R = (a[imax - 1] + a[imax + 1]) / (2.0 * amax)
        if R <= 0:
            Qfactor = 0.0
        else:
            # Ensure the logarithm argument is positive
            Qfactor = np.sqrt(-(np.pi ** 2) / (2 * np.log(R)))
    
    return (float(Qveto), float(Qfactor))