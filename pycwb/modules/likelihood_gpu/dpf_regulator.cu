// Dominant-polarisation-frame (DPF) index for each selected sky direction.
// Reference: likelihoodWP/dpf_regulator.py, compute_dpf_index.
// fp0/fx0 are [sky, detector], rms is [pixel, detector], and out is [valid sky].
// The wrapper validates 2/3 detectors; the three-element arrays are thread-local.
// Parallelism is across skies, not pixels: changing the sum order changes rounding.

// One thread per sky direction; ordered FP32 pixel/detector arithmetic.
extern "C" __global__ void dpf_index(
    const float* fp0, const float* fx0, const float* rms,
    const long long* skies, int n_valid, int n_pix, int n_ifo, double* out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_valid) return;

    // skies maps the compact launch index to the full antenna-pattern sky grid.
    int sky = skies[k];
    float NI = 0.f;
    unsigned int NN = 0;
    const float eps = 1.e-9f;
    for (int p = 0; p<n_pix; ++p) {
        float f[3], F[3];
        float ff = 0.f, FF = 0.f, fF = 0.f;

        // Noise-weight both antenna patterns and form their 2x2 Gram matrix.
        // f/F denote plus/cross; ff/FF are squared norms and fF is their dot product.
        for (int d = 0; d<n_ifo; ++d) {
            f[d] = rms[p*n_ifo+d]*fp0[sky*n_ifo+d];
            F[d] = rms[p*n_ifo+d]*fx0[sky*n_ifo+d];
            ff += f[d]*f[d];
            FF += F[d]*F[d];
            fF += F[d]*f[d];
        }

        // Diagonalize the Gram matrix using the native half-angle rotation.
        // Keep the sign branch and epsilon placements identical to the reference.
        float si = 2.f*fF, co = ff-FF, AP = ff+FF;
        float nn = sqrtf(co*co+si*si), cc = co/(nn+eps), fp = (AP+nn)/2.f;
        float rs = sqrtf((1.f-cc)/2.f);
        float rc = sqrtf((1.f+cc)/2.f);
        if (!(si>0.f)) rc = -rc;
        for (int d = 0; d<n_ifo; ++d) {
            float a = f[d]*rc+F[d]*rs, b = F[d]*rc-f[d]*rs;
            f[d] = a;
            F[d] = b;
        }

        // Remove the residual cross projection onto the dominant plus direction.
        // This explicit orthogonalization is retained even after the DPF rotation.
        float fFn = 0.f;
        for (int d = 0; d<n_ifo; ++d) fFn += f[d]*F[d];
        fFn = fFn/(fp+eps);
        float fx = 0.f, ni = 0.f;
        for (int d = 0; d<n_ifo; ++d) {
            F[d] -= f[d]*fFn;
            fx += F[d]*F[d];
            float sq = f[d]*f[d];
            ni += sq*sq;
        }

        // ni is the normalized fourth moment of the plus response over detectors.
        // Accumulate the cross norm divided by ni in the original pixel order.
        ni = ni/(fp*fp+eps);
        NI += fx/(ni+eps);
        NN += (fp>0.f);
    }

    // Only the final per-sky ratio/square root widens to FP64. The host then
    // counts indices above gamma_regulator and constructs the scalar regulator.
    out[k] = sqrt((double) NI/((double) NN+0.01));
}
