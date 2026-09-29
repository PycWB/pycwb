// Subnetwork cuts: evaluate skies independently, then select a cluster's best sky.
// Reference: super_cluster_native/sub_net_cut.py and the native DPF helpers.
// FP/FX: [sky, detector]; rms: [pixel, detector]; ml: [detector, sky];
// td0/td9: [delay, detector, pixel]. Wrappers validate 2/3 detectors and bounds.
// FP32 accumulations, explicit FP64 promotions, and threshold inequalities below
// preserve native evaluation order; they must not be algebraically simplified.

// Ordered subnet sky evaluation. Disabled contraction preserves CPU FP32 steps.
__device__ void subnet_one(int s,
    const float* FP, const float* FX, const float* rms, const float* td0, const float* td9,
    const int* ml, int nsky, int npix, int nd, int offset, float threshold, float Es, double subcut,
    double* scores) {

    // One thread owns this sky row. Early rejection leaves all six scores
    // zero; only positive ranking scores participate in best-sky selection.
    double* out = scores+6*s;
    for (int z = 0; z<6; z++) out[z] = 0.;

    // First pass: sum both quadratures across detectors. Removing the
    // loudest detector leaves the energy supported by the remaining subnet.
    float mgt = 0.f, Egt = 0.f, Lsgt = 0.f, Lngt = 0.f, Eo = 0.f, Ls = 0.f, Ln = 0.f;
    for (int p = 0; p<npix; p++) {
        float en = 0.f, em = 0.f;
        for (int d = 0; d<nd; d++) {
            int idx = ((ml[d*nsky+s]+offset)*nd+d)*npix+p;
            float a = td0[idx], b = td9[idx], e = a*a+b*b;
            en += e;
            if (e>em) em = e;
        }
        float sub = en-em;

        // The native calculation has two distinct admission rules: >= supplies
        // final statistics, while > supplies the preliminary subcut. Keep both.
        if (en >= threshold) {
            Eo += en;
            Ls += sub;
            if (sub>Es) Ln += en;
        }
        if (en>threshold) {
            mgt += 1.f;
            Egt += en;
            Lsgt += sub;
            if (sub>Es) Lngt += en;
        }
    }

    // Apply the preliminary subnetwork cut before the more expensive DPF
    // projection. Negative subcut disables this cut, not the Eo energy gate.
    float aa = Lsgt*Lngt/((Egt+0.01f)-Lsgt);
    int mcut = (int)(2.0*(double) mgt+0.01);
    if (subcut >= 0. && ((double) aa-mcut)/((double) aa+mcut+(double)1.e-16f)<subcut) return;
    if (Eo <= 0.f) return;

    // Second pass: reconstruct coherent likelihood Lo for admitted pixels.
    // m counts the inclusive-threshold pixels used by the final statistics.
    int m = 0;
    float Lo = 0.f;
    for (int p = 0; p<npix; p++) {
        float v0[3], v9[3], en = 0.f;
        for (int d = 0; d<nd; d++) {
            int idx = ((ml[d*nsky+s]+offset)*nd+d)*npix+p;
            v0[d] = td0[idx];
            v9[d] = td9[idx];
            en += v0[d]*v0[d]+v9[d]*v9[d];
        }
        if (en<threshold) continue;
        m++;

        // Construct noise-weighted plus/cross responses, rotate into the DPF,
        // and explicitly remove the residual cross projection onto plus.
        float f[3], F[3], ff = 0.f, FF = 0.f, fF = 0.f;
        for (int d = 0; d<nd; d++) {
            f[d] = rms[p*nd+d]*FP[s*nd+d];
            F[d] = rms[p*nd+d]*FX[s*nd+d];
            ff += f[d]*f[d];
            FF += F[d]*F[d];
            fF += F[d]*f[d];
        }
        float si = 2.f*fF, co = ff-FF, AP = ff+FF, nn = sqrtf(co*co+si*si);
        float cc = co/(nn+1.e-9f),
            fp = (AP+nn)/2.f,
            rs = sqrtf((1.f-cc)/2.f),
            rc = sqrtf((1.f+cc)/2.f);
        if (!(si>0.f)) rc = -rc;
        for (int d = 0; d<nd; d++) {
            float a = f[d]*rc+F[d]*rs, b = F[d]*rc-f[d]*rs;
            f[d] = a;
            F[d] = b;
        }
        float ffn = 0.f;
        for (int d = 0; d<nd; d++) ffn += f[d]*F[d];
        ffn = ffn/(fp+1.e-9f);
        for (int d = 0; d<nd; d++) F[d] -= f[d]*ffn;

        // The native BLAS sdot accumulates rounded FP32 products in FP64.
        // Keep that separate from the ordered FP32 DPF reductions above.
        double dxp = 0., dXP = 0., dxx = 0., dXX = 0., dgp = 0., dgx = 0.;
        for (int d = 0; d<nd; d++) {
            dxp += (double)(f[d]*v0[d]);
            dXP += (double)(f[d]*v9[d]);
            dxx += (double)(F[d]*v0[d]);
            dXX += (double)(F[d]*v9[d]);
            dgp += (double)(f[d]*f[d]);
            dgx += (double)(F[d]*F[d]);
        }
        float xp = (float) dxp,
            XP = (float) dXP,
            xx = (float) dxx,
            XX = (float) dXX,
            gp = (float) dgp,
            gx = (float) dgx;
        gp += 1.e-12f;
        gx += 1.e-12f;
        xp = xp*xp+XP*XP;
        xx = xx*xx+XX*XX;
        Lo += xp/gp+xx/gx;
    }

    // Finalize the six-column contract consumed by SubnetScan:
    // [ranking AA, selected energy Eo, preliminary subnet statistic,
    // pixel count m, final subnet statistic, residual-plus-count term em].
    float AA = (float)((double) aa/((double)(fabsf(aa)+fabsf(Eo-Lo))+(double)(2*m)*(double)(Eo-Ln)/(double) Eo));
    float ee = Ls*Eo/(Eo-Ls);
    double em = (double) fabsf(Eo-Lo)+2*m;
    out[0] = (double) AA;
    out[1] = (double) Eo;
    out[2] = ((double) aa-m)/((double) aa+m);
    out[3] = (double) m;
    out[4] = (double) ee/((double) ee+em);
    out[5] = em;
}

extern "C" __global__ void subnet_scan(
    const float* FP, const float* FX, const float* rms, const float* td0, const float* td9,
    const int* ml, int nsky, int npix, int nd, int offset, float threshold, float Es, double subcut, double* scores) {
    int s = blockIdx.x*blockDim.x+threadIdx.x;
    if (s<nsky) subnet_one(s, FP, FX, rms, td0, td9, ml, nsky, npix, nd, offset, threshold, Es, subcut, scores);
}
extern "C" __global__ void subnet_batch(
    const float* FP, const float* FX, const float* rms, const float* td0, const float* td9,
    const int* ml, const int* pixels, const int* pixel_offsets, const int* td_offsets, const int* delays,
    int nc, int nsky, int nd, float threshold, float Es, double subcut, double* scores) {
    // Batch mode assigns one thread to each (cluster, sky). Geometry is
    // shared; pixel/TD offsets locate each cluster in concatenated buffers.
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int c = index/nsky, s = index%nsky;
    if (c >= nc) return;

    // pixel_offsets counts pixels (multiply by nd for rms); td_offsets
    // counts scalar amplitudes. Each cluster has its own delay-axis center.
    subnet_one(s, FP, FX, rms+pixel_offsets[c]*nd, td0+td_offsets[c], td9+td_offsets[c], ml,
        nsky, pixels[c], nd, delays[c]/2, threshold, Es, subcut, scores+c*nsky*6);
}

// One block per cluster with exactly 128 threads: the shared arrays below and the
// step=64 reduction start are sized for that block, see SUBNET_BEST_THREADS in subnet_scan.py.
extern "C" __global__ void subnet_best(const double* scores, int nc, int nsky, double* result) {
    int c = blockIdx.x, t = threadIdx.x;
    if (c >= nc) return;
    __shared__ double bests[128];
    __shared__ int indices[128];

    // Each thread scans strided sky rows and retains its first strict
    // maximum. Zero is the native no-winner baseline.
    double best = 0.;
    int idx = 0;
    for (int s = t; s<nsky; s += 128) {
        double value = scores[(c*nsky+s)*6];
        if (value>best) {
            best = value;
            idx = s;
        }
    }

    // Publish every thread's candidate before the shared-memory reduction.
    // Every thread must reach each barrier, including those without sky rows.
    bests[t] = best;
    indices[t] = idx;
    __syncthreads();
    for (int step = 64; step; step /= 2) {
        // Prefer the larger score; on ties keep the lower sky index (first maximum).
        if (t<step && (bests[t+step]>bests[t] || (bests[t+step] == bests[t] && indices[t+step]<indices[t]))) {
            bests[t] = bests[t+step];
            indices[t] = indices[t+step];
        }

        // The next reduction level reads candidates written by this level.
        __syncthreads();
    }

    // One thread publishes [sky index, six score columns]. If no score is
    // positive, the complete result remains zero, matching the host contract.
    if (t == 0) {
        double* out = result+c*7;
        for (int i = 0; i<7; i++) out[i] = 0.;
        if (bests[0]>0.) {
            int s = indices[0];
            out[0] = (double) s;
            for (int i = 0; i<6; i++) out[i+1] = scores[(c*nsky+s)*6+i];
        }
    }
}
