import numpy as np

from pycwb.modules.qveto.qveto import get_qveto


def _reference_get_qveto(wf, NTHR=1, ATHR=7.58859):
    wf = np.asarray(wf, dtype=np.float64)

    n = len(wf)
    spec = np.fft.rfft(wf)
    new_spec = np.zeros(2 * n + 1, dtype=complex)
    new_spec[:len(spec)] = spec
    x = np.fft.irfft(new_spec, n=4 * n)

    signs = np.sign(x)
    crossings = np.where(np.diff(signs, prepend=signs[0]) != 0)[0]
    if len(crossings) < 2:
        return (0.0, 0.0)

    a = []
    start_idx = 0
    for end_idx in crossings[1:]:
        segment = x[start_idx:end_idx]
        a.append(np.max(np.abs(segment)))
        start_idx = end_idx
    if start_idx < len(x):
        segment = x[start_idx:]
        a.append(np.max(np.abs(segment)))
    a = np.array(a)

    if len(a) == 0:
        return (0.0, 0.0)
    imax = np.argmax(a)
    amax = a[imax]

    indices = np.arange(len(a))
    mask_in = (np.abs(indices - imax) <= NTHR)
    ein = np.sum(a[mask_in] ** 2)
    mask_out = ~mask_in & (a > amax / ATHR)
    eout = np.sum(a[mask_out] ** 2)
    Qveto = eout / ein if ein > 0 else 0.0

    if imax < 1 or imax >= len(a) - 1:
        Qfactor = 0.0
    else:
        R = (a[imax - 1] + a[imax + 1]) / (2.0 * amax)
        if R <= 0:
            Qfactor = 0.0
        else:
            Qfactor = np.sqrt(-(np.pi ** 2) / (2 * np.log(R)))

    return (float(Qveto), float(Qfactor))


def test_get_qveto_matches_reference_implementation():
    rng = np.random.default_rng(1234)

    for n in (8, 9, 64, 256, 1024):
        for _ in range(10):
            wf = rng.normal(size=n)
            if n > 10:
                wf[rng.integers(0, n)] = 0.0
            for nthr in (-1, 0, 0.5, 1, 2, 5):
                actual = get_qveto(wf, NTHR=nthr)
                expected = _reference_get_qveto(wf, NTHR=nthr)
                assert np.allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_get_qveto_returns_zero_without_zero_crossings():
    assert get_qveto(np.ones(32)) == (0.0, 0.0)
    assert get_qveto(np.zeros(32)) == (0.0, 0.0)
