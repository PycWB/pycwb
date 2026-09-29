// Filter contraction feeding the native WDM forward FFT (the FFT stays on CPU).
// signal/filter are FP64; out is [time, fft_size], with fft_size = 2*M.
// One thread evaluates one output sample in one time window. The Python wrapper
// validates padding and filter block counts; this kernel assumes those bounds.

// Experimental equation-23 prefilter. CPU FFT remains the numerical reference.
extern "C" __global__ void wdm_prefilter(
    const double* signal, const double* filter, double* out,
    int nfilt, int fft_size, int stride, int nt, int mode) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= nt*fft_size) return;
    int t = index/fft_size, f = index%fft_size;

    // The padded signal center advances by stride samples per time bin.
    // f labels a pre-FFT sample, not a physical output-frequency bin.
    int center = nfilt+t*stride;
    double sum = 0.0;
    double tail = 0.0;
    for (int begin = 0; begin<nfilt-1; begin += fft_size) {
        int end = begin+fft_size;

        // Pair positive/negative filter support around the current center.
        // The mirrored filter index end-f is intentional at the block boundary.
        double a = signal[center+begin+f], b = filter[begin+f];
        double c = signal[center-end+f], d = filter[end-f];
        double value;

        // Modes preserve alternative CPU contraction orders: left product
        // fused, right product fused, or separately rounded products.
        // Only mode 6 has the documented six-block numerical validation.
        if (mode%3 == 1) value = fma(a, b, c*d);
        else if (mode%3 == 2) value = fma(c, d, a*b);
        else value = a*b+c*d;

        // CPU YNN's six-block reduction uses a four-term accumulator
        // followed by a two-term accumulator, then adds the two.
        if (mode == 6 && begin >= 4*fft_size) tail = tail+value;
        else sum = sum+value;
    }
    if (mode == 6) sum = sum+tail;

    // The endpoint filter tap falls outside the full-block loop and belongs
    // only to the first pre-FFT sample. Modes >=3 fuse this final addition.
    if (f == 0) {
        double a = filter[nfilt-1], b = signal[center+nfilt-1];
        sum = mode >= 3 ? fma(a, b, sum) : sum+a*b;
    }
    out[index] = sum;
}
