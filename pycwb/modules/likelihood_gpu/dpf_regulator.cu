// One thread per sky direction; ordered FP32 pixel/detector arithmetic.
extern "C" __global__ void dpf_index(
    const float* fp0, const float* fx0, const float* rms,
    const long long* skies, int n_valid, int n_pix, int n_ifo, double* out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_valid) return;
    int sky = skies[k];
    float NI = 0.f;
    unsigned int NN = 0;
    const float eps = 1.e-9f;
    for (int p=0; p<n_pix; ++p) {
        float f[3], F[3];
        float ff=0.f, FF=0.f, fF=0.f;
        for (int d=0; d<n_ifo; ++d) {
            f[d] = rms[p*n_ifo+d]*fp0[sky*n_ifo+d];
            F[d] = rms[p*n_ifo+d]*fx0[sky*n_ifo+d];
            ff += f[d]*f[d]; FF += F[d]*F[d]; fF += F[d]*f[d];
        }
        float si=2.f*fF, co=ff-FF, AP=ff+FF;
        float nn=sqrtf(co*co+si*si), cc=co/(nn+eps), fp=(AP+nn)/2.f;
        float rs=sqrtf((1.f-cc)/2.f);
        float rc=sqrtf((1.f+cc)/2.f);
        if (!(si>0.f)) rc=-rc;
        for (int d=0;d<n_ifo;++d) {
            float a=f[d]*rc+F[d]*rs, b=F[d]*rc-f[d]*rs;
            f[d]=a;F[d]=b;
        }
        float fFn=0.f;
        for (int d=0;d<n_ifo;++d) fFn+=f[d]*F[d];
        fFn = fFn/(fp+eps);
        float fx=0.f, ni=0.f;
        for (int d=0;d<n_ifo;++d) {
            F[d]-=f[d]*fFn;
            fx+=F[d]*F[d];
            float sq=f[d]*f[d];
            ni+=sq*sq;
        }
        ni=ni/(fp*fp+eps);
        NI+=fx/(ni+eps);
        NN+=(fp>0.f);
    }
    out[k]=sqrt((double)NI/((double)NN+0.01));
}
