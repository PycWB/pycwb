// Wavelet-packet energy for one WDM shift; see packet_energy.py for layout setup.
// z is complex128 [time, frequency], exposed as interleaved double real/imaginary
// values. out has the same flattened time-major cell order. One thread owns a cell.
// Explicit fma calls mirror reference contractions; implicit contraction is disabled
// by the CUDA compiler options. Reassociating these expressions changes rounding.

extern "C" __global__ void packet_energy(
    const double* z, double* out, int nf, int nt, int pattern,
    int jb, int je, int low, int high, int accumulate) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nf * nt;
    if (j >= size) return;

    // Native pattern numbers select frequency/time/diagonal neighbours.
    // An offset of 1 changes frequency; nf changes time. Unused slots
    // remain zero and deliberately revisit the center in the fixed loop.
    int p[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double mean = 1.0;
    if (pattern == 1) {
        p[1] = 1;
        p[2] = -1;
    }
    if (pattern == 2) {
        p[1] = nf;
        p[2] = -nf;
    }
    if (pattern == 3) {
        p[1] = nf+1;
        p[2] = -nf-1;
    }
    if (pattern == 4) {
        p[1] = -nf+1;
        p[2] = nf-1;
    }
    if (pattern == 5) {
        p[1] = nf+1;
        p[2] = -nf-1;
        p[3] = 2*nf+2;
        p[4] = -2*nf-2;
    }
    if (pattern == 6) {
        p[1] = -nf+1;
        p[2] = nf-1;
        p[3] = -2*nf+2;
        p[4] = 2*nf-2;
    }
    if (pattern == 7) {
        p[1] = 1;
        p[2] = -1;
        p[3] = nf;
        p[4] = -nf;
    }
    if (pattern == 8) {
        p[1] = nf+1;
        p[2] = -nf+1;
        p[3] = nf-1;
        p[4] = -nf-1;
    }
    if (pattern == 9) {
        p[1] = 1;
        p[2] = -1;
        p[3] = nf;
        p[4] = -nf;
        p[5] = nf+1;
        p[6] = nf-1;
        p[7] = -nf+1;
        p[8] = -nf-1;
    }

    // mean is the packet size (3, 5, or 9), not a running average.
    // Other accepted patterns retain the single-cell normalization.
    if (pattern >= 1 && pattern <= 4) mean = 3.0;
    if (pattern >= 5 && pattern <= 8) mean = 5.0;
    if (pattern == 9) mean = 9.0;
    int f = j % nf;
    double energy = 0.0;

    // The reference preserves the nonnegative real component at time edges.
    // Interior cells outside the inclusive [low, high] band remain zero.
    if (j < jb || j >= je) energy = fmax(0.0, z[2*j]);
    else if (f >= low && f <= high) {
        // Accumulate real/imaginary cross products and squared amplitudes.
        // Neighbour indices clamp against the flat array, as in the reference.
        double ss = 0.0, ee = 0.0, EE = 0.0;
        #pragma unroll
        for (int n = 1; n<9; ++n) {
            int k = j+p[n];
            k = k<0 ? 0 : (k >= size ? size-1 : k);
            double r = z[2*k], i = z[2*k+1];

            // XLA's CPU code contracts singly-used products, while repeated
            // center products remain separately rounded and reused.
            if (n == 2 && p[n] != 0) {
                int k1 = j+p[1];
                k1 = k1<0 ? 0 : (k1 >= size ? size-1 : k1);
                double r1 = z[2*k1], i1 = z[2*k1+1];
                ss = fma(r1, i1, r*i);
                ee = fma(r1, r1, r*r);
                EE = fma(i1, i1, i*i);
            }
            else if (n>1 && p[n] != 0) {
                ss = fma(r, i, ss);
                ee = fma(r, r, ee);
                EE = fma(i, i, EE);
            }
            else {
                ss = ss+r*i;
                ee = ee+r*r;
                EE = EE+i*i;
            }
        }
        double r = z[2*j], i = z[2*j+1];

        // The eight neighbour slots already include repeated centers when unused.
        // For smaller packets, mean-8 corrects their multiplicity; for nine
        // cells, add the center once. This is not ordinary averaging.
        if (mean == 9.0) {
            ss = fma(r, i, ss);
            ee = fma(r, r, ee);
            EE = fma(i, i, EE);
        }
        else {
            ss = fma(r*i, mean-8.0, ss);
            ee = fma(r*r, mean-8.0, ee);
            EE = fma(i*i, mean-8.0, EE);
        }

        // The 2x2 quadrature Gram matrix gives two principal amplitudes.
        // The native guards limit roundoff before square roots; retain them.
        double cc = ee-EE, ss2 = ss*2.0;
        double nn = sqrt(fma(cc, cc, ss2*ss2)), sum = ee+EE;
        if (sum<nn) nn = sum;
        double a1 = sqrt(fmax((sum+nn)/2.0, 0.0));
        double a2 = sqrt(fmax((sum-nn)/2.0, 0.0));
        double aa = a1+a2;
        energy = mean == 1.0 ? sum/2.0 : aa*aa/4.0;
    }

    // The first shift overwrites out; subsequent ordered launches update
    // the resident maximum. A single writer per cell avoids atomics.
    out[j] = accumulate ? fmax(out[j], energy) : energy;
}
