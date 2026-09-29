// Time-delay interpolation for sparse WDM pixels, using the native filter tables.
// plane is a padded FP32 [time, cached frequency] view; offset is its first band.
// T0/Tx are FP64 [delay row, tap]; phase is [band, signed delay + radius, 4].
// Filter sums stay ordered in FP64; only final returned amplitudes narrow to FP32.
// See td_vectors.py/phase_table for CPU-generated phases and support validation.

// Phase factors are computed by the native CPU math functions, not CUDA libm.
__device__ double amplitude(int n, int m, int d, const float* plane,
    const double* T0, const double* Tx, const double* phase, int M, int taps,
    int J, int bands, int offset, int radius, bool quad) {
    // WDM signs depend on time parity, time+band parity, and quadrature.
    // The DC/Nyquist bands (0/M) suppress one parity contribution.
    bool odd = (n&1) != 0, oddmn = ((m+n)&1) != 0, cancel = odd^quad;
    bool edge = m == 0 || m == M;
    int win = 2*taps+1;

    // Four phase factors: cos(phase), sin(phase), sin(low), sin(high).
    // Do not replace these uploaded values with CUDA trigonometric calls.
    const double *p = phase+((long long) m*(2*radius+1)+d+radius)*4;

    // Same-band interpolation: separate even/odd tap sums, preserving the
    // native order within each sum. The padded plane includes tap support.
    double se = 0.0, so = 0.0;
    for (int k = 0; k<win; k++) {
        double v = double(plane[(long long)(n+k)*bands+m-offset])*T0[(long long)(d+J)*win+k];
        if (k%2 == 0) se += v;
        else so += v;
    }
    if (edge && cancel) se = 0.0;
    if (edge && !cancel) so = 0.0;
    double result = odd?se*p[0]-so*p[1]:se*p[0]+so*p[1];

    // Add leakage from the lower neighbouring band through Tx. Near the
    // DC/Nyquist boundaries, parity cancellation and sqrt(2) normalization
    // match the native edge-band treatment.
    if (m>0) {
        double even = 0.0, od = 0.0;
        for (int k = 0; k<win; k++) {
            double v = double(plane[(long long)(n+k)*bands+m-1-offset])*Tx[(long long)(d+J)*win+k];
            if (k%2 == 0) even += v;
            else od += v;
        }
        if (m == 1 && cancel) even = 0.0;
        if (m == 1 && !cancel) od = 0.0;
        double term = odd?(even-od)*p[2]:(even+od)*p[2];
        if (m == 1 || m == M) term *= 1.4142135623730951;
        if (oddmn) result -= term;
        else result += term;
    }

    // Add the upper neighbour with its own phase/sign convention.
    // The cached frequency interval must cover both neighbours when present.
    if (m<M) {
        double even = 0.0, od = 0.0;
        for (int k = 0; k<win; k++) {
            double v = double(plane[(long long)(n+k)*bands+m+1-offset])*Tx[(long long)(d+J)*win+k];
            if (k%2 == 0) even += v;
            else od += v;
        }
        if (m == M-1 && cancel) even = 0.0;
        if (m == M-1 && !cancel) od = 0.0;
        double term = odd?(even+od)*p[3]:(even-od)*p[3];
        if (m == 0 || m == M-1) term *= 1.4142135623730951;
        if (oddmn) result -= term;
        else result += term;
    }
    return edge && cancel?0.0:result;
}
extern "C" __global__ void td_vectors(const int* indices, const float* p0,
    const float* p9, const double* T0, const double* Tx, const double* phase,
    int np, int M, int taps, int K, int J, int stride, int bands, int offset,
    int radius, float* out) {
    // One thread owns one (pixel, requested delay) and computes both phases.
    // indices uses the full time-major WDM grid, even for a cached band subset.
    int t = blockIdx.x*blockDim.x+threadIdx.x, half = 2*K+1;
    if (t >= np*half) return;
    int pixel = t/half, ki = t%half, idx = indices[pixel];
    int n = idx/(M+1), m = idx%(M+1), delay = (ki-K)*stride;

    // cWB getTDvecSSE uses even TF-bin shifts, with remainder in [-J,J).
    int shift = 2*(int) floor((double)(delay+J)/(2*J));

    // Split the requested delay into an even time-bin shift and a filter
    // remainder. floor, rather than truncation, handles negative delays.
    int d = delay-shift*J, ne = n-shift;
    double a = amplitude(ne, m, d, p0, T0, Tx, phase, M, taps, J, bands, offset, radius, false);
    double b = amplitude(ne, m, d, p9, T0, Tx, phase, M, taps, J, bands, offset, radius, true);

    // Output is [pixel, 2*(2*K+1)]: all phase-0 delays followed by phase-90.
    // Each thread writes two unique slots; no synchronization is required.
    out[(long long) pixel*half*2+ki] = float(a);
    out[(long long) pixel*half*2+half+ki] = float(b);
}
