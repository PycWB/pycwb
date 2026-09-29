"""Numba regression kernels; arithmetic follows the cWB LPE regression path.

The dispatcher imports this backend only when Numba is selected. Availability
and the JAX fallback are retained for environments where Numba cannot import.
"""

import numpy as np

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _cap_witness_numba(real, imag, fraction=1.0):
        """Cap normalized witness samples in place; retain both phase ratios."""
        if fraction >= 1.0:
            return
        energy = real * real + imag * imag
        kth = int(fraction * len(real) - 1)
        if not 0 <= kth < len(real):
            raise ValueError("Regression cap fraction has no valid order statistic")
        threshold = 5.0 * np.partition(energy, kth)[kth]
        for i in range(len(real)):
            if energy[i] > threshold:
                scale = np.sqrt(threshold / energy[i])
                real[i] *= scale
                imag[i] *= scale

    @njit(cache=True)
    def _numba_percentile_mean(arr, fraction, edge_samples, stride):
        ff = abs(fraction)
        if ff > 1.0:
            ff = 1.0

        if stride > 1:
            arr2 = arr[::stride]
            nn = edge_samples // stride
        else:
            arr2 = arr
            nn = edge_samples

        n = arr2.shape[0]
        if n == 0:
            return 0.0

        if nn < 0:
            nn = 0

        if nn == 0 or 2 * nn >= n - 2:
            core = arr2
        else:
            core = arr2[nn : n - nn]

        core_count = core.shape[0]
        mean_all = np.mean(arr2)
        if core_count <= 0:
            return mean_all

        keep = int(core_count * ff)
        if keep < 1:
            keep = 1
        if keep > core_count:
            keep = core_count
        if keep >= core_count:
            return np.mean(core)

        abs_core = np.abs(core)
        # np.partition gives O(n) selection vs O(n log n) for np.sort
        threshold = np.partition(abs_core, keep - 1)[keep - 1]

        select_sum = 0.0
        select_count = 0
        for i in range(core_count):
            if abs_core[i] <= threshold:
                select_sum += core[i]
                select_count += 1

        if select_count > 0:
            return select_sum / select_count
        return mean_all

    @njit(cache=True)
    def _numba_rotated_products(real, imag, lag, boundary):
        n = real.shape[0]
        start = boundary
        end = n - boundary
        size = end - start

        ww = np.empty(size, dtype=np.float64)
        WW = np.empty(size, dtype=np.float64)

        if lag < 0:
            for i in range(size):
                j = start + i
                rn = real[j]
                in_ = imag[j]
                jm = j - lag
                rm = real[jm]
                im = imag[jm]
                ww[i] = rn * rm + in_ * im
                WW[i] = im * rn - rm * in_
        else:
            for i in range(size):
                j = start + i
                jn = j + lag
                rn = real[jn]
                in_ = imag[jn]
                rm = real[j]
                im = imag[j]
                ww[i] = rn * rm + in_ * im
                WW[i] = im * rn - rm * in_

        return ww, WW

    @njit(cache=True)
    def _numba_build_matrix(acf, ccf, K, K2, fltr):
        size = 2 * (2 * K + 1)
        matrix = np.zeros((size, size), dtype=np.float64)
        half = size // 2

        for ii in range(-K, K + 1):
            for jj in range(-K, K + 1):
                idx = ii - jj + K2
                aa = acf[idx]
                cc = ccf[idx]
                if ii == 0 or jj == 0:
                    aa = aa * fltr
                    cc = cc * fltr

                r = ii + K
                c = jj + K
                matrix[r, c] = aa
                matrix[r, c + half] = cc
                matrix[r + half, c] = -cc
                matrix[r + half, c + half] = aa

        return matrix

    @njit(cache=True)
    def _numba_cross_products(target_real, target_imag, real, imag, lag, boundary):
        """Target/witness products with cWB's lag and quadrature conventions."""
        size = len(real) - 2 * boundary
        ww = np.empty(size, dtype=np.float64)
        WW = np.empty(size, dtype=np.float64)
        for i in range(size):
            j = i + boundary
            witness_index = j + max(lag, 0)
            target_index = j + max(-lag, 0)
            wr, wi = real[witness_index], imag[witness_index]
            tr, ti = target_real[target_index], target_imag[target_index]
            ww[i] = wr * tr + wi * ti
            WW[i] = ti * wr - tr * wi
        return ww, WW

    @njit(cache=True)
    def _numba_process_one_layer(
        target_real,
        target_imag,
        real,
        imag,
        K,
        K2,
        K4,
        half,
        fm,
        edge_samples,
        fltr,
        eigen_threshold,
        eigen_num,
        regulator_code,
        apply_threshold,
        rate_tf,
        edge_seconds,
        stride,
        apply_fraction=1.0,
    ):
        n_time = real.shape[0]

        target_power = target_real * target_real + target_imag * target_imag
        target_norm = np.sqrt(_numba_percentile_mean(target_power, fm, edge_samples, stride))
        valid_target = np.isfinite(target_norm) and (target_norm > 0.0)
        safe_target = target_norm if valid_target else 1.0
        power = real * real + imag * imag
        norm0_sq = _numba_percentile_mean(power, fm, edge_samples, stride)
        norm0 = np.sqrt(norm0_sq)
        valid_norm = np.isfinite(norm0) and (norm0 > 0.0)
        safe_norm = norm0 if valid_norm else 1.0
        base = safe_norm * safe_norm

        # ROOT regression::setMatrix fills products for j in [K, n-K], then calls
        # ww.mean(fm) which trims nn=edge_samples from the FULL n-sample array.
        # That means the effective trim on the product region is (edge_samples - K).
        # Likewise for ACF/CCF with boundary K2: effective trim is (edge_samples - K2).
        edge_v = edge_samples - K
        if edge_v < 0:
            edge_v = 0
        edge_m = edge_samples - K2
        if edge_m < 0:
            edge_m = 0

        v_cross = np.zeros((K4,), dtype=np.float64)
        for lag in range(-K, K + 1):
            ww, WW = _numba_cross_products(target_real, target_imag, real, imag, lag, K)
            idx = K + lag
            v0 = _numba_percentile_mean(ww, fm, edge_v, stride) / safe_target / safe_norm
            v1 = _numba_percentile_mean(WW, fm, edge_v, stride) / safe_target / safe_norm
            scale = fltr if lag == 0 else 1.0
            v_cross[idx] = v0 * scale
            v_cross[idx + half] = v1 * scale

        lag_count = 2 * K2 + 1
        acf = np.zeros((lag_count,), dtype=np.float64)
        ccf = np.zeros((lag_count,), dtype=np.float64)
        for lag in range(-K2, K2 + 1):
            ww, WW = _numba_rotated_products(real, imag, lag, K2)
            idx = lag + K2
            acf[idx] = _numba_percentile_mean(ww, fm, edge_m, stride) / base
            # ROOT matrix WW = x_m*xQ_n - xQ_m*x_n (opposite sign from cross-vector WW)
            # _numba_rotated_products returns WW matching cross-vector sign, so negate here
            ccf[idx] = -_numba_percentile_mean(WW, fm, edge_m, stride) / base

        matrix = _numba_build_matrix(acf, ccf, K, K2, fltr)
        evals, evecs = np.linalg.eigh(matrix)

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        th = (-eigen_threshold * evals[0]) if (eigen_threshold < 0.0) else (eigen_threshold + 1.0e-12)
        nlast = 0
        for i in range(K4):
            if evals[i] >= th:
                nlast = i
        if nlast < 1:
            nlast = 1

        ne = K4 if eigen_num <= 0 else (eigen_num - 1)
        if ne > K4 - 1:
            ne = K4 - 1
        if ne < 1:
            ne = 1
        if nlast > ne:
            nlast = ne

        last_s = (1.0 / evals[nlast]) if evals[nlast] > 0.0 else 0.0
        last_m = (1.0 / evals[0]) if evals[0] > 0.0 else 0.0
        if regulator_code == 1:
            last = last_s
        elif regulator_code == 2:
            last = last_m
        else:
            last = 0.0

        lam = np.empty((K4,), dtype=np.float64)
        for i in range(K4):
            inv = (1.0 / evals[i]) if evals[i] > 0.0 else 0.0
            lam[i] = inv if i <= nlast else last

        vv = np.dot(evecs.T, v_cross) * lam
        aa = np.dot(evecs, vv)
        filt00 = aa[: 2 * K + 1]
        filt90 = aa[half : half + 2 * K + 1]

        qq = real / safe_norm
        QQ = imag / safe_norm
        _cap_witness_numba(qq, QQ, apply_fraction)
        nn = np.zeros((n_time,), dtype=np.float64)
        NN = np.zeros((n_time,), dtype=np.float64)

        for center in range(K, n_time - K):
            val = 0.0
            VAL = 0.0
            for k in range(-K, K + 1):
                coeff_idx = k + K
                x = center + k
                val += qq[x] * filt00[coeff_idx] - QQ[x] * filt90[coeff_idx]
                VAL += qq[x] * filt90[coeff_idx] + QQ[x] * filt00[coeff_idx]
            nn[center] = val
            NN[center] = VAL

        kk = int(rate_tf * edge_seconds)
        if kk < K:
            kk = K
        kk += 1
        if kk < 0:
            kk = 0
        if kk > n_time:
            kk = n_time
        s0 = kk
        s1 = n_time - kk
        if s1 < s0:
            s1 = s0
        valid_range = s1 > s0

        count = s1 - s0
        if count < 1:
            count = 1

        nn_mean = 0.0
        NN_mean = 0.0
        for i in range(s0, s1):
            nn_mean += nn[i]
            NN_mean += NN[i]
        nn_mean /= count
        NN_mean /= count

        nn_var = 0.0
        NN_var = 0.0
        for i in range(s0, s1):
            d0 = nn[i] - nn_mean
            d1 = NN[i] - NN_mean
            nn_var += d0 * d0
            NN_var += d1 * d1
        nn_var /= count
        NN_var /= count
        layer_power = nn_var + NN_var

        included = valid_target and valid_norm and valid_range and (layer_power >= apply_threshold * apply_threshold)
        noise = np.zeros((n_time,), dtype=np.complex128)
        if included:
            for i in range(n_time):
                noise[i] = (nn[i] + 1j * NN[i]) * target_norm

        return noise, included

    @njit(cache=True, parallel=True)
    def _numba_process_layers(
        target_real_layers,
        target_imag_layers,
        real_layers,
        imag_layers,
        K,
        K2,
        K4,
        half,
        fm,
        edge_samples,
        fltr,
        eigen_threshold,
        eigen_num,
        regulator_code,
        apply_threshold,
        rate_tf,
        edge_seconds,
        stride,
        apply_fraction=1.0,
    ):
        n_layers = real_layers.shape[0]
        n_time = real_layers.shape[1]
        noise_layers = np.zeros((n_layers, n_time), dtype=np.complex128)
        include_mask = np.zeros((n_layers,), dtype=np.bool_)

        for i in prange(n_layers):
            noise, included = _numba_process_one_layer(
                target_real_layers[i],
                target_imag_layers[i],
                real_layers[i],
                imag_layers[i],
                K,
                K2,
                K4,
                half,
                fm,
                edge_samples,
                fltr,
                eigen_threshold,
                eigen_num,
                regulator_code,
                apply_threshold,
                rate_tf,
                edge_seconds,
                stride,
                apply_fraction,
            )
            noise_layers[i, :] = noise
            include_mask[i] = included

        return noise_layers, include_mask
