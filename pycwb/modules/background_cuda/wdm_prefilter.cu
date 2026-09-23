// Experimental equation-23 prefilter. CPU FFT remains the numerical reference.
extern "C" __global__ void wdm_prefilter(
    const double* signal, const double* filter, double* out,
    int nfilt, int fft_size, int stride, int nt, int mode) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=nt*fft_size) return;
    int t=index/fft_size, f=index%fft_size;
    int center=nfilt+t*stride;
    double sum=0.0;
    double tail=0.0;
    for (int begin=0; begin<nfilt-1; begin+=fft_size) {
        int end=begin+fft_size;
        double a=signal[center+begin+f], b=filter[begin+f];
        double c=signal[center-end+f], d=filter[end-f];
        double value;
        if (mode%3==1) value=fma(a,b,c*d);
        else if (mode%3==2) value=fma(c,d,a*b);
        else value=a*b+c*d;
        // CPU YNN's six-block reduction uses a four-term accumulator
        // followed by a two-term accumulator, then adds the two.
        if (mode==6 && begin>=4*fft_size) tail=tail+value;
        else sum=sum+value;
    }
    if (mode==6) sum=sum+tail;
    if (f==0) {
        double a=filter[nfilt-1], b=signal[center+nfilt-1];
        sum=mode>=3 ? fma(a,b,sum) : sum+a*b;
    }
    out[index]=sum;
}
