// Each seeded trial has an independent score; preserve ordered pixel sums.
extern "C" __global__ void bootstrap_classify(const double* x, const double* f,
 const double* weights, const double* slopes, const double* mergers,
 const unsigned char* valid, const double* frequency_powers, int n,
 int trial_start, int trial_count, unsigned char* out) {
 int index=blockIdx.x*blockDim.x+threadIdx.x;
 if(index>=trial_count*n) return;
 int trial=trial_start+index%trial_count, i=index/trial_count;
 out[index]=0;
 if(!valid[trial]) return;
 double sl=slopes[trial], tm=mergers[trial];
 unsigned char mask=16;
   double t=x[i]-tm, freq=f[i],sign=sl<0.0?-1.0:1.0;
   double t1=t-sign/64.0, f1=freq-4.0;
   double dt1=f1>0.0?t1-frequency_powers[i*3]/sl:0.0;
   double df1=t1*sign>0.0?f1-128.0*pow(sl*t1,-3.0/8.0):0.0;
   if(sign*dt1>=0.0&&df1>=0.0) {out[index]=1; return;}
   t1=t+sign/64.0; f1=freq+4.0;
   dt1=f1>0.0?t1-frequency_powers[i*3+1]/sl:0.0;
   df1=t1*sign>0.0?f1-128.0*pow(sl*t1,-3.0/8.0):0.0;
   if(sign*dt1<=0.0&&df1<=0.0) {out[index]=2; return;}
   double dt=freq>0.0?t-frequency_powers[i*3+2]/sl:0.0;
   double df=t*sign>0.0?freq-128.0*pow(sl*t,-3.0/8.0):0.0;
   if(sign*dt>=0.0&&df>=0.0) {mask|=5;}
   if(sign*dt<=0.0&&df<=0.0) {mask|=10;}
 out[index]=mask;
}
extern "C" __global__ void bootstrap_reduce(const unsigned char* masks,
 const double* weights, const unsigned char* valid, int n,
 int trial_start, int trial_count, double* out) {
 int local_trial=blockIdx.x*blockDim.x+threadIdx.x;
 if(local_trial>=trial_count) return;
 int trial=trial_start+local_trial;
 double *o=out+trial*3;
 o[0]=-1.0; o[1]=0.0; o[2]=0.0;
 if(!valid[trial]) return;
 double upper=0.0,lower=0.0,ut=0.0,lt=0.0,score=0.0;
 int count=0;
 for(int i=0;i<n;i++) {
  unsigned char m=masks[i*trial_count+local_trial]; double w=weights[i];
  if(m&1) ut+=w;
  if(m&2) lt+=w;
  if(m&4) upper+=w;
  if(m&8) lower+=w;
  if(m&16) {score+=w; count++;}
 }
 double balance=upper+lower>0.0?1.0-fabs((upper-lower)/(upper+lower)):0.0;
 o[0]=score*balance;
 o[1]=double(count);
 o[2]=ut+lt>0.0?1.0-fabs((ut-lt)/(ut+lt)):0.0;
}
