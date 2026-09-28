// Phase factors are computed by the native CPU math functions, not CUDA libm.
__device__ double amplitude(int n, int m, int d, const float* plane,
 const double* T0, const double* Tx, const double* phase, int M, int taps,
 int J, int bands, int offset, int radius, bool quad) {
 bool odd=(n&1)!=0, oddmn=((m+n)&1)!=0, cancel=odd^quad;
 bool edge=m==0||m==M;
 int win=2*taps+1;
 const double *p=phase+((long long)m*(2*radius+1)+d+radius)*4;
 double se=0.0,so=0.0;
 for(int k=0;k<win;k++) {
  double v=double(plane[(long long)(n+k)*bands+m-offset])*T0[(long long)(d+J)*win+k];
  if(k%2==0) se+=v; else so+=v;
 }
 if(edge&&cancel) se=0.0;
 if(edge&&!cancel) so=0.0;
 double result=odd?se*p[0]-so*p[1]:se*p[0]+so*p[1];
 if(m>0) {
  double even=0.0,od=0.0;
  for(int k=0;k<win;k++) {
   double v=double(plane[(long long)(n+k)*bands+m-1-offset])*Tx[(long long)(d+J)*win+k];
   if(k%2==0) even+=v; else od+=v;
  }
  if(m==1&&cancel) even=0.0;
  if(m==1&&!cancel) od=0.0;
  double term=odd?(even-od)*p[2]:(even+od)*p[2];
  if(m==1||m==M) term*=1.4142135623730951;
  if(oddmn) result-=term; else result+=term;
 }
 if(m<M) {
  double even=0.0,od=0.0;
  for(int k=0;k<win;k++) {
   double v=double(plane[(long long)(n+k)*bands+m+1-offset])*Tx[(long long)(d+J)*win+k];
   if(k%2==0) even+=v; else od+=v;
  }
  if(m==M-1&&cancel) even=0.0;
  if(m==M-1&&!cancel) od=0.0;
  double term=odd?(even+od)*p[3]:(even-od)*p[3];
  if(m==0||m==M-1) term*=1.4142135623730951;
  if(oddmn) result-=term; else result+=term;
 }
 return edge&&cancel?0.0:result;
}
extern "C" __global__ void td_vectors(const int* indices, const float* p0,
 const float* p9, const double* T0, const double* Tx, const double* phase,
 int np, int M, int taps, int K, int J, int stride, int bands, int offset,
 int radius, float* out) {
 int t=blockIdx.x*blockDim.x+threadIdx.x, half=2*K+1;
 if(t>=np*half) return;
 int pixel=t/half, ki=t%half, idx=indices[pixel];
 int n=idx/(M+1),m=idx%(M+1),delay=(ki-K)*stride;
 // cWB getTDvecSSE uses even TF-bin shifts, with remainder in [-J,J).
 int shift=2*(int)floor((double)(delay+J)/(2*J));
 int d=delay-shift*J, ne=n-shift;
 double a=amplitude(ne,m,d,p0,T0,Tx,phase,M,taps,J,bands,offset,radius,false);
 double b=amplitude(ne,m,d,p9,T0,Tx,phase,M,taps,J,bands,offset,radius,true);
 out[(long long)pixel*half*2+ki]=float(a);
 out[(long long)pixel*half*2+half+ki]=float(b);
}
