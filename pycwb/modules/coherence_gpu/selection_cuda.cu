// Two-pass counterpart of coherence_native/selection.py: align support, then
// select and compact pixels. maps is [detector, frequency, time] in FP64;
// support is [frequency, time]. Each pass assigns one thread to one map cell.
// The wrapper launches both passes in order and sorts sparse rows on the host.

extern "C" __global__ void align_support(const double* maps, const long long* shifts,
    const short* veto, int nd, int nf, int nt, int start, int length, int ib,
    double eo, double* support, unsigned char* live) {
    int p = blockIdx.x*blockDim.x+threadIdx.x;
    if (p >= nf*nt) return;
    int f = p/nt, t = p%nt;
    bool active = t >= start && t<start+length;
    double value = 0.0;
    if (active) {
        for (int d = 0; d<nd; d++) {
            // The host normalizes shifts modulo length. Wrap only inside the valid
            // interval; detector padding must not enter the circular lag shift.
            int src = start+(t-start+shifts[d])%length;
            active = active && (veto[src] != 0);
            value += maps[((long long) d*nf+f)*nt+src];
        }
    }

    // Liveness depends only on shifted time/veto values, so frequency zero
    // is the sole writer for each live[t] entry (no atomic operation needed).
    if (f == 0) live[t] = active;

    // Threshold and cap support used by the neighbour test. The +0.1 marker
    // distinguishes values strictly above 2*eo from those exactly on the cap.
    if (value<eo) value = 0.0;
    else if (value>2.0*eo) value = 2.0*eo+0.1;
    support[p] = active && f >= ib?value:0.0;
}

// Neighbour reads reproduce the native rolled support map at array edges.
// Offsets are small enough to require at most one wrap on each axis.
__device__ double at(const double* support, int nf, int nt, int f, int t) {
    if (f<0) f += nf;
    if (f >= nf) f -= nf;
    if (t<0) t += nt;
    if (t >= nt) t -= nt;
    return support[(long long) f*nt+t];
}
extern "C" __global__ void select_sparse(const double* support, const double* maps,
    const long long* shifts, int nd, int nf, int nt, int start, int length, int ib, int ie,
    int margin, double eo, unsigned int capacity, unsigned int* count,
    long long* frequency, long long* time, double* total, double* detector_energy,
    long long* detector_index) {
    int p = blockIdx.x*blockDim.x+threadIdx.x;
    if (p >= nf*nt) return;
    int f = p/nt, t = p%nt;
    if (f<ib || f >= ie || t<margin || t >= nt-margin) return;
    double center = support[p];
    if (center<eo) return;

    // ct/cb collect the immediate diagonal neighbourhood; ht/hb extend it
    // by a second time/frequency step. Frequency guards match the native
    // edge stencil rather than adding every wrapped second neighbour.
    double ct = at(support, nf, nt, f+1, t)+at(support, nf, nt, f, t+1)+at(support, nf, nt, f+1, t+1);
    double cb = at(support, nf, nt, f-1, t)+at(support, nf, nt, f, t-1)+at(support, nf, nt, f-1, t-1);
    double ht = at(support, nf, nt, f+1, t+2)+(f<nf-2?at(support, nf, nt, f+2, t+2)+at(support, nf, nt, f+2, t+1):0.0);
    double hb = at(support, nf, nt, f-1, t-2)+(f >= 2?at(support, nf, nt, f-2, t-2)+at(support, nf, nt, f-2, t-1):0.0);
    double em = 2.0*eo, eh = em*em;

    // A pixel survives if any support combination reaches the threshold,
    // or its own support reaches 2*eo. Preserve the strict comparisons.
    if ((ct+cb)*center<eh && (ct+ht)*center<eh && (cb+hb)*center<eh && center<em) return;

    // Reserve a unique sparse row. The counter includes overflowed rows so
    // the host can reject the entire result instead of silently truncating it.
    // Atomic reservation order is nondeterministic; the host restores map order.
    unsigned int row = atomicAdd(count, 1u);
    if (row >= capacity) return;
    frequency[row] = f;
    time[row] = t;
    double sum = 0.0;
    for (int d = 0; d<nd; d++) {
        int src = t;
        if (t >= start && t<start+length && length>0) src = start+(t-start+shifts[d])%length;

        // Persist original detector energies, not capped support. total sums
        // raw values; individual detector energies clamp negatives to zero.
        double raw = maps[((long long) d*nf+f)*nt+src];
        sum += raw;
        detector_energy[(long long) row*nd+d] = raw>0.0?raw:0.0;

        // Native detector pixel indices use time-major order, unlike the
        // frequency-major map storage used to read raw above.
        detector_index[(long long) row*nd+d] = (long long) src*nf+f;
    }
    total[row] = sum;
}
