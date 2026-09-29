// Score the trials sampled on the CPU by likelihoodWP/chirp_bootstrap.py.
// Classification is parallel over [micropixel, local trial], with trial contiguous.
// Reduction is parallel only over trials, keeping ordered FP64 pixel sums.
// The host launches classify then reduce in the same stream; no cross-kernel
// shared memory or atomics are needed. Sampling/final metadata remain on the CPU.

// Each seeded trial has an independent score; preserve ordered pixel sums.
extern "C" __global__ void bootstrap_classify(const double* x, const double* f,
    const double* weights, const double* slopes, const double* mergers,
    const unsigned char* valid, const double* frequency_powers, int n,
    int trial_start, int trial_count, unsigned char* out) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= trial_count*n) return;

    // Trial batches reuse a bounded mask buffer; trial_start indexes global
    // slopes/mergers while local trial indexes the current packed mask rows.
    int trial = trial_start+index%trial_count, i = index/trial_count;
    out[index] = 0;
    if (!valid[trial]) return;
    double sl = slopes[trial], tm = mergers[trial];

    // Mask bits: 1/2 contribute to total upper/lower weights; 4/8 contribute
    // to on-track upper/lower weights; 16 marks an on-track micropixel.
    // weights is kept in this kernel's argument contract but used by reduce.
    unsigned char mask = 16;

    // Measure offsets from the fitted merger. The curve is
    // f = 128 * (slope * t)^(-3/8); sign handles either fitted slope direction.
    double t = x[i]-tm, freq = f[i], sign = sl<0.0?-1.0:1.0;

    // Test the two opposite micropixel corners (1/64 s and 4 Hz half-widths).
    // frequency_powers holds CPU-computed (f/128)^(-8/3) for -4, +4, 0 Hz.
    // Keep the positivity guards before evaluating inverse powers.
    double t1 = t-sign/64.0, f1 = freq-4.0;
    double dt1 = f1>0.0?t1-frequency_powers[i*3]/sl:0.0;
    double df1 = t1*sign>0.0?f1-128.0*pow(sl*t1, -3.0/8.0):0.0;
    if (sign*dt1 >= 0.0 && df1 >= 0.0) {
        out[index] = 1;
        return;
    }
    t1 = t+sign/64.0;
    f1 = freq+4.0;
    dt1 = f1>0.0?t1-frequency_powers[i*3+1]/sl:0.0;
    df1 = t1*sign>0.0?f1-128.0*pow(sl*t1, -3.0/8.0):0.0;
    if (sign*dt1 <= 0.0 && df1 <= 0.0) {
        out[index] = 2;
        return;
    }

    // Neither corner excludes this pixel: score it as on-track and use
    // the center to classify its side. Equality may set both side bits.
    double dt = freq>0.0?t-frequency_powers[i*3+2]/sl:0.0;
    double df = t*sign>0.0?freq-128.0*pow(sl*t, -3.0/8.0):0.0;
    if (sign*dt >= 0.0 && df >= 0.0) {
        mask |= 5;
    }
    if (sign*dt <= 0.0 && df <= 0.0) {
        mask |= 10;
    }
    out[index] = mask;
}

// Produce [score, selected_count, symmetry] for each global trial.
extern "C" __global__ void bootstrap_reduce(const unsigned char* masks,
    const double* weights, const unsigned char* valid, int n,
    int trial_start, int trial_count, double* out) {
    int local_trial = blockIdx.x*blockDim.x+threadIdx.x;
    if (local_trial >= trial_count) return;
    int trial = trial_start+local_trial;
    double *o = out+trial*3;

    // Invalid fitted trials keep a negative score so they cannot win.
    // Valid trials with no selected energy are evaluated normally below.
    o[0] = -1.0;
    o[1] = 0.0;
    o[2] = 0.0;
    if (!valid[trial]) return;
    double upper = 0.0, lower = 0.0, ut = 0.0, lt = 0.0, score = 0.0;
    int count = 0;

    // Walk every micropixel in native order; a tree reduction could alter
    // trial scores and therefore which seeded trial wins.
    for (int i = 0; i<n; i++) {
        unsigned char m = masks[i*trial_count+local_trial];
        double w = weights[i];
        if (m&1) ut += w;
        if (m&2) lt += w;
        if (m&4) upper += w;
        if (m&8) lower += w;
        if (m&16) {
            score += w;
            count++;
        }
    }

    // Penalize imbalance within the track for the selection score. The
    // returned symmetry instead uses all upper/lower weights (ut/lt).
    double balance = upper+lower>0.0?1.0-fabs((upper-lower)/(upper+lower)):0.0;
    o[0] = score*balance;
    o[1] = double(count);
    o[2] = ut+lt>0.0?1.0-fabs((ut-lt)/(ut+lt)):0.0;
}
