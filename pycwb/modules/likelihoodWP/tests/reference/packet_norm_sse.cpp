// Small SSE oracle retaining network.hh::_avx_norm_ps arithmetic and lane order.
#include <xmmintrin.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
int main(int argc,char**argv){
 std::ifstream in(argv[1],std::ios::binary);int32_t D,P,J;in.read((char*)&D,4);in.read((char*)&P,4);in.read((char*)&J,4);
 auto read=[&](auto &v){in.read((char*)v.data(),v.size()*sizeof(v[0]));};
 std::vector<float>p(D*P),q(D*P),cc(J*8),mask(P),energy(D);std::vector<int64_t>lookup(P*2);
 read(p);read(q);read(cc);read(lookup);read(mask);read(energy);
 std::vector<float>g(D,0),rn(P,0),qn(D*P,0),snr(D);
 for(int m=0;m<P;m++)if(mask[m]>0)for(int d=0;d<D;d++){
  __m128 x=_mm_setzero_ps();
  for(int64_t j=lookup[2*m];j<lookup[2*m+1];j++){
   int n=int(cc[j*8]);__m128 a=_mm_setr_ps(p[d*P+n],p[d*P+n],q[d*P+n],q[d*P+n]);
   x=_mm_add_ps(x,_mm_mul_ps(_mm_loadu_ps(&cc[j*8+4]),a));
  }
  float u=p[d*P+m],v=q[d*P+m],h[4],xx[4];
  _mm_storeu_ps(h,_mm_mul_ps(x,_mm_setr_ps(u,v,u,v)));_mm_storeu_ps(xx,x);
  float t=h[0]+h[1]+h[2]+h[3];t=t>0?t:0;g[d]+=t;
  float e=(u*u+v*v)/(t+1.e-12f);qn[d*P+m]=e>=1?0:e;
  u=xx[0]+xx[2];v=xx[1]+xx[3];rn[m]+=u*u+v*v;
 }
 for(int d=0;d<D;d++){g[d]=g[d]<2?2:g[d];snr[d]=energy[d]*2/g[d];}
 std::ofstream out(argv[2],std::ios::binary);
 for(auto &v:{snr,g,rn,qn})out.write((char*)v.data(),v.size()*sizeof(float));
}
