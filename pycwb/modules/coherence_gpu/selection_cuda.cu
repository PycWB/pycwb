extern "C" __global__ void align_support(const double* maps, const long long* shifts,
 const short* veto, int nd, int nf, int nt, int start, int length, int ib,
 double eo, double* support, unsigned char* live) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;
 if(p>=nf*nt) return;
 int f=p/nt,t=p%nt;
 bool active=t>=start&&t<start+length;
 double value=0.0;
 if(active) {
  for(int d=0;d<nd;d++) {
   int src=start+(t-start+shifts[d])%length;
   active=active&&(veto[src]!=0);
   value+=maps[((long long)d*nf+f)*nt+src];
  }
 }
 if(f==0) live[t]=active;
 if(value<eo) value=0.0;
 else if(value>2.0*eo) value=2.0*eo+0.1;
 support[p]=active&&f>=ib?value:0.0;
}
__device__ double at(const double* support,int nf,int nt,int f,int t) {
 if(f<0) f+=nf; if(f>=nf) f-=nf;
 if(t<0) t+=nt; if(t>=nt) t-=nt;
 return support[(long long)f*nt+t];
}
extern "C" __global__ void select_sparse(const double* support,const double* maps,
 const long long* shifts,int nd,int nf,int nt,int start,int length,int ib,int ie,
 int margin,double eo,unsigned int capacity,unsigned int* count,
 long long* frequency,long long* time,double* total,double* detector_energy,
 long long* detector_index) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;
 if(p>=nf*nt) return;
 int f=p/nt,t=p%nt;
 if(f<ib||f>=ie||t<margin||t>=nt-margin) return;
 double center=support[p];
 if(center<eo) return;
 double ct=at(support,nf,nt,f+1,t)+at(support,nf,nt,f,t+1)+at(support,nf,nt,f+1,t+1);
 double cb=at(support,nf,nt,f-1,t)+at(support,nf,nt,f,t-1)+at(support,nf,nt,f-1,t-1);
 double ht=at(support,nf,nt,f+1,t+2)+(f<nf-2?at(support,nf,nt,f+2,t+2)+at(support,nf,nt,f+2,t+1):0.0);
 double hb=at(support,nf,nt,f-1,t-2)+(f>=2?at(support,nf,nt,f-2,t-2)+at(support,nf,nt,f-2,t-1):0.0);
 double em=2.0*eo,eh=em*em;
 if((ct+cb)*center<eh&&(ct+ht)*center<eh&&(cb+hb)*center<eh&&center<em) return;
 unsigned int row=atomicAdd(count,1u);
 if(row>=capacity) return;
 frequency[row]=f;time[row]=t;
 double sum=0.0;
 for(int d=0;d<nd;d++) {
  int src=t;
  if(t>=start&&t<start+length&&length>0) src=start+(t-start+shifts[d])%length;
  double raw=maps[((long long)d*nf+f)*nt+src];
  sum+=raw;
  detector_energy[(long long)row*nd+d]=raw>0.0?raw:0.0;
  detector_index[(long long)row*nd+d]=(long long)src*nf+f;
 }
 total[row]=sum;
}
