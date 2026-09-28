// Native sky_scratch arithmetic, fused per pixel and ordered across pixels.
extern "C" __global__ void likelihood_scan(
 const float* FP,const float* FX,const float* rms,const float* td0,const float* td9,
 const int* ml,const long long* skies,int nvalid,int nsky,int npix,int nd,int offset,
 const float* reg,double threshold,double netCC,float* maps) {
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nvalid)return;
 int sky=skies[k];float* out=maps+12*k;
 for(int z=0;z<11;z++)out[z]=0.f;out[11]=-1.e12f;
 float EEdata=0.f,EC=0.f,GN=0.f,RN=0.f;
 double LL=0.,Lr=0.,NN=0.;int active=0;
 float pw=0.f,cw=0.f,ew=0.f;
 for(int p=0;p<npix;p++) {
  float v0[3],v9[3],f[3],F[3],ps[3],pS[3];
  float e0=0.f,e9=0.f,ff=0.f,FF=0.f,fF=0.f;
  for(int d=0;d<nd;d++){
   int idx=((ml[d*nsky+sky]+offset)*nd+d)*npix+p;
   v0[d]=td0[idx];v9[d]=td9[idx];e0+=v0[d]*v0[d];e9+=v9[d]*v9[d];
   f[d]=rms[p*nd+d]*FP[sky*nd+d];F[d]=rms[p*nd+d]*FX[sky*nd+d];
   ff+=f[d]*f[d];FF+=F[d]*F[d];fF+=F[d]*f[d];
  }
  float et=e0+e9+1.e-12f;int mk0=((double)et>threshold);et=(float)((double)et*mk0);EEdata+=et;
  float si=2.f*fF,co=ff-FF,ap=ff+FF,nn=sqrtf(co*co+si*si);
  float cc=co/(nn+1.e-9f),fp=(ap+nn)/2.f;
  float rs=sqrtf((1.f-cc)/2.f),rc=sqrtf((1.f+cc)/2.f);if(!(si>0.f))rc=-rc;
  for(int d=0;d<nd;d++){float a=f[d]*rc+F[d]*rs,b=F[d]*rc-f[d]*rs;f[d]=a;F[d]=b;}
  float fn=0.f;for(int d=0;d<nd;d++)fn+=f[d]*F[d];fn=fn/(fp+1.e-9f);
  float fx=0.f,ni=0.f;
  for(int d=0;d<nd;d++){F[d]-=f[d]*fn;fx+=F[d]*F[d];float sq=f[d]*f[d];ni+=sq*sq;}
  ni=ni/(fp*fp+1.e-9f);
  float xp=0.f,XP=0.f,xx=0.f,XX=0.f;
  for(int d=0;d<nd;d++){xp+=v0[d]*f[d];XP+=v9[d]*f[d];xx+=v0[d]*F[d];XX+=v9[d]*F[d];}
  float rf=sqrtf(ni*(xp*xp+XP*XP)/(et+1.e-9f))*reg[0]-fp;
  if(!(rf>0.f))rf=0.f;
  // Integer mask division promotes this stage to FP64 in native Numba.
  double uf=(double)mk0/(double)(fp+rf+1.e-9f);
  double h=(double)xp*uf,H=(double)XP*uf;h=h*h+H*H;
  float Hf=xx*xx+XX*XX;
  double vf=sqrt((double)Hf/(h+(double)1.e-9f));
  float R=0.1f+reg[1]/(et+1.e-9f);vf=vf*(double)R-(double)fx;if(!(vf>0.))vf=0.;
  vf=(double)mk0/((double)fx+vf+(double)1.e-9f);
  float au=(float)((double)xp*uf),AU=(float)((double)XP*uf),av=(float)((double)xx*vf),AV=(float)((double)XX*vf);
  float mask=(float)(uf*(double)fp+vf*(double)fx+mk0-1.);
  active+=mk0;
  float aa=0.f,AA=0.f,aA=0.f;
  for(int d=0;d<nd;d++){
   ps[d]=f[d]*au+F[d]*av;pS[d]=f[d]*AU+F[d]*AV;
   aa+=ps[d]*ps[d];AA+=pS[d]*pS[d];aA+=ps[d]*pS[d];
  }
  float osi=aA*2.f,oco=aa-AA;
  float onn=sqrtf(oco*oco+osi*osi),occ=oco/(onn+1.e-21f);
  float sn=sqrtf((1.f-occ)/2.f),cs=sqrtf((1.f+occ)/2.f);if(!(osi>0.f))cs=-cs;
  float c=0.f,C=0.f,ss=0.f,SS=0.f,rr=0.f,RR=0.f,xs=0.f,XS=0.f;
  for(int d=0;d<nd;d++){
   float s0=ps[d]*cs+pS[d]*sn,x0=v0[d]*cs+v9[d]*sn;
   float s9=pS[d]*cs-ps[d]*sn,x9=v9[d]*cs-v0[d]*sn;
   float a=s0*x0,A=s9*x9;xs+=a;XS+=A;c+=a*a;C+=A*A;ss+=s0*s0;SS+=s9*s9;
   float dr=ps[d]-v0[d],dR=pS[d]-v9[d];rr+=dr*dr;RR+=dR*dR;
  }
  int mk=(mask>=0.f);
  c=c/(xs*xs+1.e-9f);C=C/(XS*XS+1.e-9f);
  double ll=(double)mk*(double)(ss+SS);
  double sc=(double)ss*(1.-(double)c),Sc=(double)SS*(1.-(double)C);
  float ec=(float)((double)mk*(sc+Sc)),gn=(float)((double)mk*2.*(double)mask),rn=(float)((double)mk*(double)(rr+RR));
  double a=2.*(double)fabsf(ec);float A=rn+gn+1.e-9f;double cr=(double)ec/(a+(double)A);
  Lr+=ll*cr;LL+=ll;EC+=ec;GN+=gn;RN+=rn;NN+=(ec>1.e-9f);
  if(mask>0.f){ew+=et;pw+=fp*et;cw+=fx*et;}
 }
 float ellipticity=2.f*(float)(Lr/(LL+0.001));
 float noise=(GN+RN)/2.f,total=EEdata/2.f;
 double dis=(double)noise/((double)(nd*active)+sqrt((double)active));
 double corrnoise=dis>1.?dis:1.;
 double correlation=(double)EC/((double)EC+(double)noise*corrnoise-active*(nd-1));
 if(!isfinite(ellipticity)||(double)ellipticity<netCC)return;
 float likelihood=total>0.f?total-noise:0.f;
 double stat=(double)likelihood*correlation;
 pw=ew>0.f?pw/ew:0.f;cw=ew>0.f?cw/ew:0.f;
 out[0]=sqrtf(pw+cw);out[1]=pw>0.f?sqrtf(cw/pw):0.f;
 out[2]=total-noise;out[3]=noise;out[4]=EC;out[5]=(float)correlation;out[6]=(float)stat;
 out[7]=(float)dis;out[8]=(float)corrnoise;out[9]=ellipticity;out[10]=(float)NN;out[11]=(float)stat;
}
