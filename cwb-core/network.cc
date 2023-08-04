/*
# Copyright (C) 2019 Sergey Klimenko, Gabriele Vedovato
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


// ++++++++++++++++++++++++++++++++++++++++++++++
// S. Klimenko, University of Florida, Gainesville, FL
// G.Vedovato,  INFN,  Sezione  di  Padova, Italy
// WAT network class
// ++++++++++++++++++++++++++++++++++++++++++++++

#define NETWORK_CC
#include <time.h>
#include <iostream>
#include <stdexcept>
#include <xmmintrin.h>
#include <immintrin.h> 
#include "TRandom3.h"
#include "TMath.h" 
#include <fstream>
#include "Meyer.hh"
#include "injection.hh"
#include "network.hh"
#include "watplot.hh"       // remove
#include "TComplex.h"

using namespace std;

ClassImp(network)

// constructors

network::network() : 
   nRun(0), nLag(1), nSky(0), mIFO(0), rTDF(0), Step(0.), Edge(0.), gNET(0.), aNET(0.), iNET(0), 
  eCOR(0.), norm(1.), e2or(0.), acor(sqrt(2.)), pOUT(false), EFEC(true), local(true), optim(true), 
  delta(0.), gamma(0.), precision(0.02), pSigma(4.), penalty(1.), netCC(-1.), netRHO(0.), 
   eDisbalance(true), MRA(false), wfsave(false), pattern(1), _WDM(false), _LIKE(' ')  
{
   this->ifoList.clear();
   this->ifoName.clear();
   this->segList.clear();
   this->mdcList.clear();
   this->livTime.clear();
   this->mdcTime.clear();
   this->mdcType.clear();
   this->mdc__ID.clear();
}


network::network(const network& value)
{
   *this = value;
}

// destructor

network::~network()
{
  return;
}

//**************************************************************************
//:select TF samples by value of the network excess energy: 2-8 detectors
//**************************************************************************
long network::getNetworkPixels(int LAG, double Eo, double norm, TH1F* hist)
{
// 2G analysis algorithm for selection of significant network pixles
// works with WDM/wavelet energy TF maps
// LAG - time shift lag defining how detectors are shifted wrt each other.
// Eo  - pixel energy threshold
// norm  - dummy
// hist- pointer to a diagnostic histogram showing network pixel energy. 

   size_t nIFO = this->ifoList.size();       // number of detectors

   if(nIFO>NIFO) {
      cout<<"network::getNetworkPixels(): " 
          <<"invalid number of detectors or\n";
      return 0;
   }
   if(getifo(0)->getTFmap()->w_mode != 1) {
      cout<<"network::getNetworkPixels(): invalid whitening mode.\n"; 
      return 0;
   } 

   WSeries<double>* pTF = this->getifo(0)->getTFmap(); // pointer to first TF map
   WSeries<double> MAP; MAP = *pTF; MAP=0.;            // initialize TF map
   wavearray<double>* hTS = this->getifo(0)->getHoT(); // pointer to first TS data
   
   int i,j,k,m,n,NN,jj,nM,jE,jb,je,J,K;
   
   double Em = 2*Eo;                                   // maximum (sole pixel) threshold
   double Eh = Em*Em;                                  // halo energy^2
   double R  = pTF->wrate();                           // pixel layer rate
   double r  = hTS->rate();                            // TS rate
   int N  = pTF->size();                               // size of TF array
   int M  = hTS->size();                               // size of TS array
   int I  = pTF->maxLayer()+1;                         // number of layers
   int II = pTF->maxLayer()-1;                         // number of layers - 2
   int jB = int(this->Edge*R+0.001);                   // number of samples in the edges
   if(jB&1) {cout<<"getNetworkPixels(1): WDM parity violation\n"; exit(1);}
  
   if(jB < 3) {
      cout<<"network::getNetworkPixels(): insufficient data edge length.\n"; 
      exit(1);
   } 

   netpixel pix(nIFO); 
   pix.core = true;
   pix.rate = R;
   pix.layers = I;
   
   int     in[NIFO];                                    // pixel time index
   int     IN[NIFO];                                    // pixel time index
   double* PDATA; 
   double* pmap; 
   double* pdata[NIFO];                                 // pointers to data
   double* pp[5];                                       // pointers to sorted F-arrays
   for(n=0; n<nIFO; n++) {                              // pointers to data
      pdata[n] = getifo(n)->getTFmap()->data;
   }

   long nPix = 0;
   size_t count = 0;                              // live pixel counter  
   double a,b,E,Ct,Cb,Ht,Hb;

   if(hist) {pixeLHood = *pTF; pixeLHood=-1.;}   
   if(this->veto.size() != M) {                   // set veto array if it is not set
      veto.resize(M); veto = 1; 
   }
   short* pveto = this->veto.data;                // pointer to veto
  
   this->wc_List[LAG].clear();                    // clear netcluster structure
   this->livTime[LAG] = 0.;                       // clear live time counters
   this->wc_List[LAG].setlow(pTF->getlow());
   this->wc_List[LAG].sethigh(pTF->gethigh());

   a  = 1.e10; nM = 0;                            // master detector    
   for(n=0; n<nIFO; n++) {
      b = this->getifo(n)->lagShift.data[LAG];    // shift in seconds
      if(a>b) { a = b; nM = n; }
   }
   
   for(n=0; n<nIFO; n++) {
      b = this->getifo(n)->lagShift.data[LAG];    // shift in seconds
      K = int((b-a)*R+0.001);                     // time shift wrt reference
      if(K&1) {cout<<"getNetworkPixels(2): WDM parity violation\n"; exit(1);}
      in[n] = IN[n] = K+jB;                       // time index of first pixel in the layer 
   }
   
   int ib=1;
   int ie=I;
   for(i=0; i<I; i++) {                           // select bandwidth
      if(pTF->frequency(i) <= pTF->gethigh()) ie=i;
      if(pTF->frequency(i) <= pTF->getlow())  ib=i+1;     
   }
   if(ie>I-1) ie = I-1;                           // required by catalog
   if(ib<1)   ib = 1;                             // required by catalog

   slice S = pTF->getSlice(0);
   jE = S.size()-jB;                              // last good sample in the layer
   NN = jE-jB;                                    // #of good samples in the layer
   if(jE&1) {cout<<"getNetworkPixels(3): WDM parity violation\n"; exit(1);}

   //cout<<r<<" "<<R<<" "<<I<<" "<<jB<<" "<<this->veto.size()<<endl;
   //cout<<ib<<" "<<ie<<" "<<NN<<" "<<jB<<" "<<jE<<endl;

   for(jj=0; jj<NN; jj++) {                       // loop over time stamps

      double VETO = 1.;
      pmap = MAP.data+(jj+jB)*I;                  // pointer to 0 F sample in MAP
      for(n=0; n<nIFO; n++) {
	 if(in[n] >= jE) in[n] -= NN;             // go to jB sample
	 jb = int(in[n]*r/R+0.01);                // first veto index
	 je = int((in[n]+1)*r/R+0.01);            // last veto index
	 while(jb<je) if(!pveto[jb++]) VETO=0.;   // set veto value
	 PDATA = &(pdata[n][in[n]*I]);            // pointer to 0 F sample
	 for(i=0; i<I; i++) pmap[i]+=*PDATA++;    // sum energy
	 in[n]++;                                 // increment index pointer
      }
      
      for(i=0; i<I; i++) {
	 pmap[i] *= VETO;
	 if(pmap[i]<Eo || i<ib) pmap[i]=0.;       // zero sub-threshold pixels 
	 if(pmap[i]>Em) pmap[i]=Em+0.1;           // degrade loud pixels 
      }
      count += VETO;                              // count live time
   }

   for(jj=0; jj<NN; jj++) {                        // loop over time stamps

      pmap = MAP.data+(jj+jB)*I;                   // pointer to 0 F sample in MAP
      for(n=0; n<nIFO; n++) {
	 if(IN[n] >= jE) IN[n] -= NN;              // go to jB sample
      }
      for(n=0; n<5; n++) pp[n]=pmap+(n-2)*I;       // initialize set of pointers
      for(i=ib; i<ie; i++) {
	 if((E=pp[2][i])<Eo) continue;             // skip subthreshold pixels
	 Ct = pp[2][i+1]+pp[3][ i ]+pp[3][i+1];    // top core
	 Cb = pp[2][i-1]+pp[1][ i ]+pp[1][i-1];    // bottom core
	 Ht = pp[4][i+1];                          // top halo
	 Ht+= i<II? pp[4][i+2]+pp[3][i+2]:0.;      // top halo
	 Hb = pp[0][i-1];                          // bottom halo
	 Hb+= i>1 ? pp[0][i-2]+pp[1][i-2]:0.;      // bottom halo

	 if((Ct+Cb)*E<Eh && 
	    (Ct+Ht)*E<Eh && 
	    (Cb+Hb)*E<Eh && 
	    E<Em) continue;
	 
         E = 0;
	 for(n=0; n<nIFO; n++) {
	    j = IN[n]*I+i;                          // sample index
	    pix.data[n].index = j;
	    pix.data[n].asnr = sqrt(pdata[n][j]);
	    E += pdata[n][j];
	 }
	 j = IN[nM]*I+i;                            // reference sample index
	 if(hist) hist->Fill(E);         
	 if(hist) pixeLHood.data[j] = E;         	
	 pix.time = j;
	 pix.frequency = i;                
	 pix.likelihood = E;
	 pix.phi = 1;                               // set pixel mark 1 (will be owerriden in likelihood)
	 wc_List[LAG].append(pix);	            // save pixels in wc_List
         nPix++;
      } 
      for(n=0; n<nIFO; n++) IN[n]++;                // increment IN
   }        

// set metadata in wc_List
   this->wc_List[LAG].start = pTF->start();  
   this->wc_List[LAG].stop  = pTF->stop();
   this->wc_List[LAG].rate  = pTF->rate();
   this->livTime[LAG] = count/R;                    // live time depends on resolution 
   
   if(nPix) this->setRMS();
  
   return nPix;
}


void network::test_sse(int n, int m) {
   wavearray<float> a(n*4);
   wavearray<float> b(n*4);
   wavearray<float> x(n*4);
   float qq[4];
   float rq[4];
   float sq[4];
   float* pa = a.data;
   float* pb = b.data;
   float* px = x.data;
   __m128* _pa = (__m128*) pa;
   __m128* _pb = (__m128*) pb;
   __m128* _px = (__m128*) px;

   for(int i=0; i<n*4; i++) {
      a.data[i]=i/float(n*4*m);
      b.data[i]=i/float(n*4*m);
      x.data[i]=i*float(m)/float(n*4);
   }

   for(int i=0; i<n; i++) {
      *(_pa+i) = _mm_sqrt_ps(*(_px+i));
      *(_pa+i) = _mm_div_ps(_mm_set1_ps(1.),*(_pa+i));
      *(_pb+i) = _mm_rsqrt_ps(*(_px+i));
      _mm_storeu_ps(qq,*(_px+i));
      _mm_storeu_ps(sq,*(_pa+i));
      _mm_storeu_ps(rq,*(_pb+i));
      printf("%d  %9.7f  %9.7f  %9.7f \n",i,qq[0],sq[0],rq[0]);
      printf("%d  %9.7f  %9.7f  %9.7f \n",i,qq[1],sq[1],rq[1]);
      printf("%d  %9.7f  %9.7f  %9.7f \n",i,qq[2],sq[2],rq[2]);
      printf("%d  %9.7f  %9.7f  %9.7f \n",i,qq[3],sq[3],rq[3]);
   }

   return;
}


long network::likelihoodWP(char mode, int lag, int iID, TH2F* hist, char* Search)
{
//  Likelihood analysis with packets
//  mode: analysis mode: 
//  OPTRES analyses, if upper case and optim=true
//  MRA analysis in low case or optim=false
//        r - un-modeled
//        i - iota - wave: no,partial dispersion correction
//        p - Psi - wave (no dispersion correction)
//      l,s - linear, loose linear
//      c,g - circular. loose circular
//      e,b - elliptical (no dispersion correction), b=p for now
//   iID: cluster ID, if negative - sky error regions are calculated 
//   lag: lag index
// hist: chirp histogram: If not needed, TGraphErrors* hist=NULL
// Search: if Search = ""/cbc/bbh/imbhb then mchirp is reconstructed
// shold be used as input
// return number of processed pixels
// Negative gamma regulator turns on the AP prior for sky localization
//

   if(!this->wc_List[lag].size()) return 0;

   this->like('2');
   this->wdm(true);
   this->tYPe = mode;

   bool cirwave = mode=='g' || mode=='G' || mode=='c' || mode=='C';
   bool linwave = mode=='l' || mode=='L' || mode=='s' || mode=='S';
   bool iotwave = mode=='i' || mode=='l' || mode=='e' || mode=='c' ||
                  mode=='I' || mode=='L' || mode=='E' || mode=='C';
   bool psiwave = mode=='l' || mode=='e' || mode=='p' ||
                  mode=='L' || mode=='E' || mode=='P';
   bool mureana = mode=='i' || mode=='e' || mode=='c' ||
                  mode=='r' || mode=='p' || mode=='b' ||
                  mode=='l' || mode=='s' || mode=='g';
   bool rndwave = mode=='r' || mode=='R';

   bool   prior = this->gamma<0 ? true  : false;  	 // gamma<0 : antenna pattern prior is used
   bool m_chirp = this->optim   ? false : mureana;

   if(!this->optim) mureana = true;

   size_t nIFO = this->ifoList.size();
   size_t ID = abs(iID);
  
   if(nIFO>NIFO) {
      cout<<"network::likelihoodAVX(): invalid network.\n";
      exit(0);
   }

   float   En = 2*acor*acor*nIFO;                             // network energy threshold in the sky loop
   float gama = this->gamma*this->gamma*2./3.;                // gamma regulator for x componet
   float deta = fabs(this->delta);   if(deta>1) deta=1;       // delta regulator for + component
   float REG[2]; REG[0] = deta*sqrt(2);
   float netEC = this->netRHO*this->netRHO*2;                 // netEC/netRHO threshold

   static const __m128 _oo = _mm_set1_ps(1.e-16);             // nusance parameter
   static const __m128 _sm = _mm_set1_ps(-0.f);               // sign mask: -0.f = 1 << 31
   static const __m128 _En = _mm_set1_ps(En);                 // network threshold

   float aa,AA,Lo,Eo,Co,No,Ep,Lp,Np,Cp,Ec,Dc,To,Fo,Em,Lm,Rc,Mo,Mw,Eh;
   float STAT,ee,EE,cc,ff,FF,Lw,Ew,Cw,Nw,Gn,rho,norm,ch,CH,Cr,Mp,N;
   float penalty,ecor;	// used in the definition of XGB rho0 (XGB.rho0)
   float xrho;		// original 2G definition (XGB.rho0)

   size_t i,j,k,l,m,Vm,lm,V,V4,V44,id,K,M;
   size_t L  = this->index.size();             // total number of source locations 
   wavearray<short> skyMMcc(L);
   short* mm = this->skyMask.data;
   short* MM = skyMMcc.data;
   bool skymaskcc = (skyMaskCC.size()==L);
   int f_ = NIFO/4;

   float  vvv[8];
   float* v00[NIFO];
   float* v90[NIFO];
   float*  pe[NIFO];
   float*  pa[NIFO];
   float*  pA[NIFO];
   float  *pd[NIFO], *pD[NIFO];
   float  *ps[NIFO], *pS[NIFO];
   float  *pn[NIFO], *pN[NIFO];
   short*  ml[NIFO];
   double* FP[NIFO];
   double* FX[NIFO];
   double  xx[NIFO];

   std::vector<float*> _vtd;              // vectors of TD amplitudes
   std::vector<float*> _vTD;              // vectors of TD amplitudes
   std::vector<float*> _eTD;              // vectors of TD energies
   std::vector<float*> _AVX;              // vectors for network pixel statistics
   std::vector<float*> _APN;              // vectors for noise and antenna patterns
   std::vector<float*> _DAT;              // vectors for packet amplitudes
   std::vector<float*> _SIG;              // vectors for packet amplitudes
   std::vector<float*> _NUL;              // vectors for packet amplitudes
   std::vector<float*> _TMP;              // temp array for _avx_norm_ps() function 

   for(i=0; i<NIFO; i++) {
      if(i<nIFO) {
         ml[i] = getifo(i)->index.data;
         FP[i] = getifo(i)->fp.data;
         FX[i] = getifo(i)->fx.data;
      }
      else {
         ml[i] = getifo(0)->index.data;
         FP[i] = getifo(0)->fp.data;
         FX[i] = getifo(0)->fx.data;
      }
   }

   // allocate buffers
   std::vector<int> pI;                      // buffer for pixel IDs
   std::vector<int> pJ;                      // buffer for pixel index
   wavearray<double> cid;                    // buffers for cluster ID
   wavearray<double> cTo;                    // buffers for cluster time
   wavearray<float>  S_snr(NIFO);            // energy SNR of signal
   wavearray<float>  D_snr(NIFO);            // energy SNR of data time series 
   wavearray<float>  N_snr(NIFO);            // energy of null streams
   netpixel* pix;
   std::vector<int>* vint;
   std::vector<int>* vtof;
   netcluster* pwc = &this->wc_List[lag];
   
   size_t count = 0;
   size_t tsize = 0;

   std::map<int,float> vLr;		     // resolution map

   // initialize parameters to manage big clusters 
   int precision = int(fabs(this->precision));
   int csize = precision%65536;	             // get number of pixels threshold per level 
   int healpix  = this->nSkyStat.getOrder(); // get healpix order of likelihood skymap
   int order = (precision-csize)/65536;      // get resampled order
   wavearray<short> BB(L);BB=1;     	     // index array for setting sky mask
   bool bBB = false;
   if(healpix && csize && order && order<healpix) {
      skymap rsm(order); 		     // resampled skymap 
      for(int l=0;l<rsm.size();l++) {
        int m = this->nSkyStat.getSkyIndex(rsm.getTheta(l),rsm.getPhi(l));
        BB[m]=0;
      }
      for(int l=0;l<L;l++) BB[l] = BB[l] ? 0 : 1;
   }

//+++++++++++++++++++++++++++++++++++++++
// loop over clusters
//+++++++++++++++++++++++++++++++++++++++

   cid = pwc->get((char*)"ID",  0,'S',0);                 // get cluster ID
   cTo = pwc->get((char*)"time",0,'L',0);                 // get cluster time
   
   K = cid.size();
   for(k=0; k<K; k++) {                                   // loop over clusters 
      id = size_t(cid.data[k]+0.1);
      
      if(pwc->sCuts[id-1] != -2) continue;                // skip rejected/processed clusters 

      vint = &(pwc->cList[id-1]);                         // pixel list
      vtof = &(pwc->nTofF[id-1]);                         // TofFlight configurations
      V = vint->size();
      if(!V) continue;

      pI = wdmMRA.getXTalk(pwc, id);
      V = pI.size();
      if(!V) continue;

      bBB = (V>this->wdmMRA.nRes*csize) ? true : false;	  // check big cluster size condition

      if(ID==id) {
         this->nSensitivity = 0.;
         this->nAlignment = 0.;
         this->nNetIndex = 0.;
         this->nDisbalance = 0.;
         this->nLikelihood = 0.;
         this->nNullEnergy = 0.;
         this->nCorrEnergy = 0.;
         this->nCorrelation = 0.;
         this->nSkyStat = 0.;
         this->nEllipticity = 0.;
         this->nPolarisation = 0.;
         this->nProbability = 0.;
      }                                                
      this->nAntenaPrior = 0.;

      pix = pwc->getPixel(id,pI[0]);
      tsize = pix->tdAmp[0].size();
      if(!tsize || tsize&1) {                       // tsize%1 = 1/0 = power/amplitude
         cout<<"network::likelihoodWP() error: wrong pixel TD data\n";
         exit(1);
      }
      
      tsize /= 2;
 
      if(!(V=pI.size())) continue;
      V4  = V + (V%4 ? 4 - V%4 : 0);
      V44 = V4 + 4;
      pJ.clear();
      for(j=0; j<V4; j++) pJ.push_back(0);                          

      float* ptmp;                                     // allocate aligned arrays
      if(_vtd.size()) _avx_free_ps(_vtd);              // array for 00 amplitudes
      if(_vTD.size()) _avx_free_ps(_vTD);              // array for 90 amplitudes
      if(_eTD.size()) _avx_free_ps(_eTD);              // array for pixel energy
      if(_APN.size()) _avx_free_ps(_APN);              // container for noise rms and antenna patterns 
      if(_DAT.size()) _avx_free_ps(_DAT);              // container for data packet amplitudes 
      if(_SIG.size()) _avx_free_ps(_SIG);              // container for signal packet amplitudes  
      if(_NUL.size()) _avx_free_ps(_NUL);              // container for null packet amplitudes  
      for(i=0; i<NIFO; i++) {                          
	 ptmp = (float*)_mm_malloc(tsize*V4*sizeof(float),32);
	 for(j=0; j<tsize*V4; j++) ptmp[j]=0; _vtd.push_back(ptmp);   // array of aligned vectors
	 ptmp = (float*)_mm_malloc(tsize*V4*sizeof(float),32);
	 for(j=0; j<tsize*V4; j++) ptmp[j]=0; _vTD.push_back(ptmp);   // array of aligned vectors
	 ptmp = (float*)_mm_malloc(tsize*V4*sizeof(float),32);
	 for(j=0; j<tsize*V4; j++) ptmp[j]=0; _eTD.push_back(ptmp);   // array of aligned vectors
	 ptmp = (float*)_mm_malloc((V4*3+16)*sizeof(float),32);
	 for(j=0; j<(V4*3+16); j++) ptmp[j]=0; _APN.push_back(ptmp);  // concatenated arrays {f+}{fx}{rms}{a+,A+,ax,AX}
	 ptmp = (float*)_mm_malloc((V4*3+8)*sizeof(float),32);
	 for(j=0; j<(V4*3+8); j++) ptmp[j]=0; _DAT.push_back(ptmp);   // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
	 ptmp = (float*)_mm_malloc((V4*3+8)*sizeof(float),32);
	 for(j=0; j<(V4*3+8); j++) ptmp[j]=0; _SIG.push_back(ptmp);   // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
	 ptmp = (float*)_mm_malloc((V4*3+8)*sizeof(float),32);
	 for(j=0; j<(V4*3+8); j++) ptmp[j]=0; _NUL.push_back(ptmp);   // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
      }

      // data arrays for polar coordinates storage : [0,1] = [radius,angle]
      for(i=0;i<2;i++) {
        this->p00_POL[i].resize(V4); this->p00_POL[i]=0.;
        this->p90_POL[i].resize(V4); this->p90_POL[i]=0.;
        this->r00_POL[i].resize(V4); this->r00_POL[i]=0.;
        this->r90_POL[i].resize(V4); this->r90_POL[i]=0.;
      }

      for(i=0; i<NIFO; i++) {                          // set up zero delay and packet pointers                   
         pa[i] = _vtd[i] + (tsize/2)*V4;
         pA[i] = _vTD[i] + (tsize/2)*V4;
         pe[i] = _eTD[i] + (tsize/2)*V4;
	 pd[i] = _DAT[i]; pD[i] = _DAT[i]+V4; 
	 ps[i] = _SIG[i]; pS[i] = _SIG[i]+V4; 
	 pn[i] = _NUL[i]; pN[i] = _NUL[i]+V4; 
      }

      this->a_00.resize(NIFO*V44); this->a_00=0.;      // array for pixel amplitudes in sky loop
      this->a_90.resize(NIFO*V44); this->a_90=0.;      // array for pixel amplitudes in sky loop
      this->rNRG.resize(V4);       this->rNRG=0.;
      this->pNRG.resize(V4);       this->pNRG=0.;

      __m128* _aa  = (__m128*) this->a_00.data;         // set pointer to 00 array
      __m128* _AA  = (__m128*) this->a_90.data;         // set pointer to 90 array
      
      if(_AVX.size()) _avx_free_ps(_AVX);
      float* p_et = (float*)_mm_malloc(V4*sizeof(float),32);      // 0
      for(j=0; j<V4; j++) p_et[j]=0; _AVX.push_back(p_et);
      float* pMSK = (float*)_mm_malloc(V44*sizeof(float),32);     // 1  - pixel mask
      for(j=0; j<V44; j++) pMSK[j]=0; _AVX.push_back(pMSK);   pMSK[V4]=nIFO;
      float* p_fp = (float*)_mm_malloc(V44*sizeof(float),32);     // 2- |f+|^2 (0:V4), +norm (V4:V4+4)
      for(j=0; j<V44; j++) p_fp[j]=0; _AVX.push_back(p_fp);   
      float* p_fx = (float*)_mm_malloc(V44*sizeof(float),32);     // 3- |fx|^2 (0:V4), xnorm (V4:V4+4)
      for(j=0; j<V44; j++) p_fx[j]=0; _AVX.push_back(p_fx);   
      float* p_si = (float*)_mm_malloc(V4*sizeof(float),32);      // 4
      for(j=0; j<V4; j++) p_si[j]=0; _AVX.push_back(p_si);
      float* p_co = (float*)_mm_malloc(V4*sizeof(float),32);      // 5
      for(j=0; j<V4; j++) p_co[j]=0; _AVX.push_back(p_co);
      float* p_uu = (float*)_mm_malloc((V4+16)*sizeof(float),32); // 6 - 00+ unit vector(0:V4), norm(V4), cos(V4+4)
      for(j=0; j<V4+16; j++) p_uu[j]=0; _AVX.push_back(p_uu);	                                                  
      float* p_UU = (float*)_mm_malloc((V4+16)*sizeof(float),32); // 7 - 90+ unit vector(0:V4), norm(V4), sin(V4+4)
      for(j=0; j<V4+16; j++) p_UU[j]=0; _AVX.push_back(p_UU);
      float* p_vv = (float*)_mm_malloc((V4+16)*sizeof(float),32); // 8- 00x unit vector(0:V4), norm(V4), cos(V4+4)
      for(j=0; j<V4+16; j++) p_vv[j]=0; _AVX.push_back(p_vv);		                                              
      float* p_VV = (float*)_mm_malloc((V4+16)*sizeof(float),32); // 9- 90x unit vector(0:V4), norm(V4), sin(V4+4)
      for(j=0; j<V4+16; j++) p_VV[j]=0; _AVX.push_back(p_VV);
      float* p_au = (float*)_mm_malloc(V4*sizeof(float),32);      // 10  
      for(j=0; j<V4; j++) p_au[j]=0; _AVX.push_back(p_au);
      float* p_AU = (float*)_mm_malloc(V4*sizeof(float),32);      // 11
      for(j=0; j<V4; j++) p_AU[j]=0; _AVX.push_back(p_AU);
      float* p_av = (float*)_mm_malloc(V4*sizeof(float),32);      // 12
      for(j=0; j<V4; j++) p_av[j]=0; _AVX.push_back(p_av);
      float* p_AV = (float*)_mm_malloc(V4*sizeof(float),32);      // 13
      for(j=0; j<V4; j++) p_AV[j]=0; _AVX.push_back(p_AV);
      float* p_uv = (float*)_mm_malloc(V4*4*sizeof(float),32);    // 14 special array for GW norm calculation
      for(j=0; j<V4*4; j++) p_uv[j]=0; _AVX.push_back(p_uv);
      float* p_ee = (float*)_mm_malloc(V4*sizeof(float),32);      // 15 + energy array
      for(j=0; j<V4; j++) p_ee[j]=0; _AVX.push_back(p_ee);
      float* p_EE = (float*)_mm_malloc(V4*sizeof(float),32);      // 16 x energy array 
      for(j=0; j<V4; j++) p_EE[j]=0; _AVX.push_back(p_EE);
      float* pTMP=(float*)_mm_malloc(V4*4*NIFO*sizeof(float),32); // 17 temporary array for _avx_norm_ps()
      for(j=0; j<V4*4*NIFO; j++) pTMP[j]=0; _AVX.push_back(pTMP);
      float* p_ni = (float*)_mm_malloc(V4*sizeof(float),32);      // 18 + network index 
      for(j=0; j<V4; j++) p_ni[j]=0; _AVX.push_back(p_ni);
      float* p_ec = (float*)_mm_malloc(V4*sizeof(float),32);      // 19 + coherent energy 
      for(j=0; j<V4; j++) p_ec[j]=0; _AVX.push_back(p_ec);
      float* p_gn = (float*)_mm_malloc(V4*sizeof(float),32);      // 20 + Gaussian noise correction 
      for(j=0; j<V4; j++) p_gn[j]=0; _AVX.push_back(p_gn);
      float* p_ed = (float*)_mm_malloc(V4*sizeof(float),32);      // 21 + energy disbalance
      for(j=0; j<V4; j++) p_ed[j]=0; _AVX.push_back(p_ed);
      float* p_rn = (float*)_mm_malloc(V4*sizeof(float),32);      // 22 + sattelite noise in TF domain
      for(j=0; j<V4; j++) p_rn[j]=0; _AVX.push_back(p_rn);

      
      this->pList.clear(); 
      for(j=0; j<V; j++) {                             // loop over selected pixels 
         pix = pwc->getPixel(id,pI[j]);
	 this->pList.push_back(pix);                   // store pixel pointers for MRA

         double rms = 0.;
         for(i=0; i<nIFO; i++) {
            xx[i] = 1./pix->data[i].noiserms;
            rms += xx[i]*xx[i];                        // total inverse variance
         }

         rms = sqrt(rms);
         for(i=0; i<nIFO; i++) {
	    _APN[i][V4*2+j]  =(float)xx[i]/rms;        // noise array for AVX processing
            for(l=0; l<tsize; l++) {
               aa = pix->tdAmp[i].data[l];             // copy TD 00 data
               AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data
               _vtd[i][l*V4+j] = aa;                   // copy 00 data
               _vTD[i][l*V4+j] = AA;                   // copy 90 data
               _eTD[i][l*V4+j] = aa*aa+AA*AA;          // copy power
            }
         }
      }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// sky loop
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      __m256 _CC;
      size_t lb  = 0;
      size_t le  = L-1;
      double sky = 0.;
      STAT=-1.e12; lm=0; Em=Lm=ff=FF=0;

      skyMMcc = 0;
      for(l=lb; l<=le; l++) {	                        // loop over sky locations
         if(!mm[l]) continue;                           // skip delay configurations
         if(bBB && !BB[l]) continue;                    // skip delay configurations : big clusters
	 
         if(skymaskcc) {                                // transform l into celestial coordinates lc
           skymap* sm = &(this->nSkyStat);
           double gT = cTo.data[k]+pwc->start;          // trigger gps time
           double RA = sm->phi2RA(sm->getPhi(l),gT);    // phi -> RA
           int lc=this->getIndex(sm->getTheta(l),RA);   // get sky index in celestial coordinates
           if (!skyMaskCC.data[lc]) continue;
         }
         MM[l] = 1; FF += 1;                            // set final skymap
	 aa = _avx_dpf_ps(FP,FX,l,_APN,_AVX,V4);        // calculate DPF f+,fx and their norms
	 if(aa > gama) ff += 1;
      }
      REG[1] = (FF*FF/(ff*ff+1.e-9)-1)*En;              // setup x regulator
	 
  optsky:
      
      AA = 0.;                                          // initialize sky statistic
      for(l=lb; l<=le; l++) {	                        // loop over sky locations
         skyProb.data[l] = -1.e12;
         if(!MM[l]) continue;                           // apply sky mask
	 pnt_(v00, pa, ml, (int)l, (int)V4);            // pointers to first pixel 00 data 
         pnt_(v90, pA, ml, (int)l, (int)V4);            // pointers to first pixel 90 data 
	 Eo=_avx_loadata_ps(v00,v90,pd,pD,En,_AVX,V4);  // calculate data stats and store in _AVX

	 _avx_dpf_ps(FP,FX,l,_APN,_AVX,V4);             // calculate DPF f+,fx and their norms
	 _avx_cpf_ps(v00,v90,ps,pS,V4);                 // copy data for GW reconstruction 
	 Mo = _avx_GW_ps(ps,pS,_APN,REG,_AVX,V4);       // gw strain packet, return number of selected pixels 
	 
         if(lb==le) _avx_saveGW_ps(ps,pS,V);            // save gw strain packet into a_00,a_90
	    
	 Lo  = _avx_ort_ps(ps,pS,_AVX,V4);              // othogonalize signal amplitudes
	 _CC = _avx_stat_ps(pd,pD,ps,pS,_AVX,V4);       // coherent statistics
	 _mm256_storeu_ps(vvv,_CC);                     // extract coherent statistics
	 Cr = vvv[0];                                   // cc statistics
	 Ec = vvv[1];                                   // signal coherent energy in TF domain
	 Mp = vvv[2];                                   // signal energy disbalance in TF domain
	 No = vvv[3];                                   // total noise in TF domain
	 CH = No/(nIFO*Mo+sqrt(Mo));                    // chi2 in TF domain
	 cc = CH>1 ? CH : 1;                            // noise correction factor in TF domain
	 Co = Ec/(Ec+No*cc-Mo*(nIFO-1));                // network correlation coefficient in TF	 
	 
	 if(Cr<netCC) continue;

	 aa = Eo>0. ? Eo-No : 0.;                        // likelihood skystat
	 AA = aa*Co;                                     // x-correlation skystat
	 skyProb.data[l] = this->delta<0 ? aa : AA;
	 
	 ff = FF = ee = 0.;
	 for(j=0; j<V; j++) { 
	    if(pMSK[j]<=0) continue;
	    ee += p_et[j];                             // total energy
	    ff += p_fp[j]*p_et[j];                     // |f+|^2
	    FF += p_fx[j]*p_et[j];                     // |fx|^2 
	 }
	 ff = ee>0. ? ff/ee  : 0.;
	 FF = ee>0. ? FF/ee  : 0.;
	 this->nAntenaPrior.set(l, sqrt(ff+FF));

	 if(ID==id) {  
	    this->nSensitivity.set(l, sqrt(ff+FF));
	    this->nAlignment.set(l, ff>0 ? sqrt(FF/ff):0);  
	    this->nLikelihood.set(l, Eo-No);              
	    this->nNullEnergy.set(l, No);                
	    this->nCorrEnergy.set(l, Ec);                      
	    this->nCorrelation.set(l,Co); 
	    this->nSkyStat.set(l,AA);                         
	    this->nProbability.set(l, skyProb.data[l]);          
	    this->nDisbalance.set(l,CH);               
	    this->nNetIndex.set(l,cc);    
	    this->nEllipticity.set(l,Cr);                                       
	    this->nPolarisation.set(l,Mp);                            
	 }

	 if(AA>=STAT) {STAT=AA; lm=l; Em=Eo-Eh;}
	 if(skyProb.data[l]>sky) sky=skyProb.data[l];            // find max of skyloc stat

	 if(lb!=le) continue;

	 Eo = _avx_packet_ps(pd,pD,_AVX,V4);            // get data packet
	 Lo = _avx_packet_ps(ps,pS,_AVX,V4);            // get signal packet
	 D_snr = _avx_norm_ps(pd,pD,_AVX,V4);           // data packet energy snr	 
	 S_snr = _avx_norm_ps(pS,pD,p_ec,V4);           // set signal norms, return signal SNR	 
	 Ep = D_snr[0];
	 Lp = S_snr[0];

	 _CC = _avx_noise_ps(pS,pD,_AVX,V4);            // get G-noise correction 
	 _mm256_storeu_ps(vvv,_CC);                     // extract coherent statistics
	 Gn = vvv[0];                                   // gaussian noise correction
	 Ec = vvv[1];                                   // core coherent energy in TF domain
	 Dc = vvv[2];                                   // signal-core coherent energy in TF domain
	 Rc = vvv[3];                                   // EC normalization
	 Eh = vvv[4];                                   // satellite energy in TF domain

	 N = _avx_setAMP_ps(pd,pD,_AVX,V4)-1;           // set data packet amplitudes

	 _avx_setAMP_ps(ps,pS,_AVX,V4);                 // set signal packet amplitudes
	 _avx_loadNULL_ps(pn,pN,pd,pD,ps,pS,V4);        // load noise TF domain amplitudes
	 D_snr = _avx_norm_ps(pd,pD,_AVX,-V4);          // data packet energy snr	 
	 N_snr = _avx_norm_ps(pn,pN,_AVX,-V4);          // noise packet energy snr	 
	 Np = N_snr.data[0];                            // time-domain NULL
	 Em = D_snr.data[0];                            // time domain energy
	 Lm = Em-Np-Gn;                                 // time domain signal energy
	 norm = Em>0 ? (Eo-Eh)/Em : 1.e9;               // norm
	 if(norm<1) norm = 1;                           // corrected norm
	 Ec  /= norm;                                   // core coherent energy in time domain
	 Dc  /= norm;                                   // signal-core coherent energy in time domain
	 ch = (Np+Gn)/(N*nIFO);                         // chi2
         if(this->netRHO>=0) {	// original 2G
	   cc = ch>1 ? ch : 1;                          // rho correction factor
	   rho  = Ec>0 ? sqrt(Ec*Rc/2.) : 0.;           // cWB detection stat 
         } else {		// (XGB.rho0)
           penalty = ch;
           ecor = Ec;
           rho = sqrt(ecor/(1+penalty*(max((float)1.,penalty)-1)));
           // original 2G rho statistic: only for test
           cc = ch>1 ? ch : 1;                          // rho correction factor
           xrho  = Ec>0 ? sqrt(Ec*Rc/2.) : 0.;          // cWB detection stat
	 }
         // save projection on network plane in polar coordinates
         // The Dual Stream Transform (DSP) is applied to v00,v90 
	 _avx_pol_ps(v00,v90,p00_POL,p90_POL,_APN,_AVX,V4);        
         // save DSP components in polar coordinates
	 _avx_pol_ps(v00,v90,r00_POL,r90_POL,_APN,_AVX,V4);        
      }

      if(le-lb) {lb=le=lm; goto optsky;}                // process all pixels at opt sky location

      if(this->netRHO>=0) {	// original 2G
        if(Lm<=0.||(Eo-Eh)<=0.||Ec*Rc/cc<netEC||N<1) {
          pwc->sCuts[id-1]=1; count=0;                   // reject cluster 
          pwc->clean(id); continue;                                         
        }
      } else {			// (XGB.rho0)
        if(Lm<=0.||(Eo-Eh)<=0.||rho<fabs(this->netRHO)||N<1) {
          pwc->sCuts[id-1]=1; count=0;                   // reject cluster 
          pwc->clean(id); continue;                    
        }
      }    

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// detection statistics at selected sky location
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      vint = &(pwc->cList[id-1]);                       // pixel list
      for(j=0; j<vint->size(); j++) {                   // initialization for all pixels
	 pix = pwc->getPixel(id,j);
	 pix->core = false;                         
	 pix->likelihood = 0.;
	 pix->null = 0;
      }
      
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// detection statistics at selected sky location
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      M = Mw = 0;                                        // add denoised pixels
      for(j=0; j<V; j++) {                               // loop over pixels
         pix = pwc->getPixel(id,pI[j]);
	 if(pMSK[j]>0) {                                 // Mo - EP pixels: stored in size[0]
	    pix->core = true;
	    pix->likelihood = -(p_ee[j]+p_EE[j])/2;      // negative total pixel energy
	 }

         for(i=0; i<nIFO; i++) {
            pix->setdata(double(pd[i][j]),'W',i);        // 00 whitened
            pix->setdata(double(pD[i][j]),'U',i);        // 90 whitened
            pix->setdata(double(ps[i][j]),'S',i);        // 00 reconstructed whitened response
            pix->setdata(double(pS[i][j]),'P',i);        // 90 reconstructed whitened response
         }
      }

      for(j=0; j<V; j++) {                               // loop over pixels
         pix = pwc->getPixel(id,pI[j]);
         if(!pix->core) continue;
	 if(p_gn[j]<=0) continue;                        // skip satellites
	 Mw += 1.;                                       // event size stored in size[1]
         for(k=0; k<V; k++) {                            // loop over xtalk components
            netpixel* xpix = pwc->getPixel(id,pI[k]);
	    struct xtalk xt = wdmMRA.getXTalk(pix->layers, pix->time, xpix->layers, xpix->time);
            if(!xpix->core || p_gn[k]<=0 || xt.CC[0]>2) continue;
            for(i=0; i<nIFO; i++) {
              pix->null += xt.CC[0]*pn[i][j]*pn[i][k];
              pix->null += xt.CC[1]*pn[i][j]*pN[i][k];
              pix->null += xt.CC[2]*pN[i][j]*pn[i][k];
              pix->null += xt.CC[3]*pN[i][j]*pN[i][k];
            }
	 }
	 
	 if(p_ec[j]<=0) continue;                         // skip incoherent pixels
	 M += 1;                                          // M - signal size: stored in volume[1]
	 pix->likelihood = 0;                             // total pixel energy
         for(k=0; k<V; k++) {                             // loop over xtalk components
            netpixel* xpix = pwc->getPixel(id,pI[k]);
	    struct xtalk xt = wdmMRA.getXTalk(pix->layers, pix->time, xpix->layers, xpix->time);
            if(!xpix->core || p_ec[k]<=0 || xt.CC[0]>2) continue;
            for(i=0; i<nIFO; i++) {
              pix->likelihood += xt.CC[0]*ps[i][j]*ps[i][k];
              pix->likelihood += xt.CC[1]*ps[i][j]*pS[i][k];
              pix->likelihood += xt.CC[2]*pS[i][j]*ps[i][k];
              pix->likelihood += xt.CC[3]*pS[i][j]*pS[i][k];
            }
         }
      }

      // subnetwork statistic
      double Emax = 0;
      double Nmax = 0;
      for(j=1; j<=nIFO; j++) {                            // loop over detectors
	 if(S_snr[j]>Emax) Emax=S_snr[j];                 // detector with max energy 
      }
      double Esub = S_snr.data[0]-Emax; 
      Esub = Esub*(1+2*Rc*Esub/Emax);
      Nmax = Gn+Np-N*(nIFO-1);
      
      //if(hist) hist->Fill(pwc->cData[id-1].skycc,pwc->cData[id-1].netcc);
 
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// fill in detection statistics, prepare output data
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// fill in backward delay configuration
       
      vtof->clear();
      NETX (vtof->push_back(ml[0][lm]); ,
            vtof->push_back(ml[1][lm]); ,
            vtof->push_back(ml[2][lm]); ,
            vtof->push_back(ml[3][lm]); ,
            vtof->push_back(ml[4][lm]); ,
            vtof->push_back(ml[5][lm]); ,
            vtof->push_back(ml[6][lm]); ,
            vtof->push_back(ml[7][lm]); )

      // need to fix a problem below
      if((wfsave)||(mdcListSize() && !lag)) {     // if wfsave=false only simulated wf are saved
        if(this->getMRAwave(id,lag,'S',0,true)) { // reconstruct whitened shifted pd->waveForm 
          detector* pd;
          for(i=0; i<nIFO; i++) {                 // loop over detectors
            pd = this->getifo(i);
            pd->RWFID.push_back(id);              // save cluster ID
            WSeries<double>* wf = new WSeries<double>;
            *wf = pd->waveForm;
            wf->start(pwc->start+pd->waveForm.start());
            pd->RWFP.push_back(wf);
          }
        }
        if(this->getMRAwave(id,lag,'s',0,true)) { // reconstruct strain shifted pd->waveForm
          detector* pd;
          for(i=0; i<nIFO; i++) {                 // loop over detectors
            pd = this->getifo(i);
            pd->RWFID.push_back(-id);             // save cluster -ID
            WSeries<double>* wf = new WSeries<double>;
            *wf = pd->waveForm;
            wf->start(pwc->start+pd->waveForm.start());
            pd->RWFP.push_back(wf);
          }
        }
      }
      
      Lw = Ew = To = Fo = Nw = ee = norm = 0.;
      for(i=0; i<nIFO; i++) {              		            
         detector* d = this->getifo(i);
         d->sSNR = d->xSNR = d->null = d->enrg = 0.;
      }

      this->getMRAwave(id,lag,'W',0);
      this->getMRAwave(id,lag,'S',0);
      for(i=0; i<nIFO; i++) {              		            
	 detector* d = this->getifo(i);
	 d->waveNull = d->waveBand;  
	 d->waveNull-= d->waveForm;   	                             
	 float sSNR = d->get_SS();
	 float xSNR = d->get_XS();
	 float null = d->get_NN();
	 float enrg = d->get_XX();
	 d->sSNR += sSNR;
	 d->xSNR += xSNR;
	 d->null += null;
	 d->enrg += enrg;
	 To += sSNR*d->getWFtime(); 
	 Fo += sSNR*d->getWFfreq(); 
	 Lw += sSNR;
	 Ew += enrg;
	 Nw += null;
      }
      To  /= Lw; Fo /= Lw;
      ch   = (Nw+Gn)/(N*nIFO);                               // chi2
      cc   = ch>1 ? 1+(ch-1)*2*(1-Rc) : 1;                   // Cr correction factor
      Cr   = Ec*Rc/(Ec*Rc+(Dc+Nw+Gn)*cc-N*(nIFO-1));         // reduced network correlation coefficient	 
      cc   = ch>1 ? ch : 1;                                  // rho correction factor
      Cp   = Ec*Rc/(Ec*Rc+(Dc+Nw+Gn)-N*(nIFO-1));            // network correlation coefficient	 
      norm = (Eo-Eh)/Ew;

      pwc->cData[id-1].norm    = norm*2;                     // packet norm  (saved in norm) 
      pwc->cData[id-1].skyStat = 0;                          //  
      pwc->cData[id-1].skySize = Mw;                         // event size in the skyloop    (size[1])
      pwc->cData[id-1].netcc   = Cp;                         // network cc                   (netcc[0])
      pwc->cData[id-1].skycc   = Cr;                         // reduced network cc           (netcc[1])
      pwc->cData[id-1].subnet  = Esub/(Esub+Nmax);           // sub-network statistic        (netcc[2])
      pwc->cData[id-1].SUBNET  = Co;                         // sky cc                       (netcc[3])
      pwc->cData[id-1].likenet = Lw;                         // waveform likelihood 
      pwc->cData[id-1].netED   = Nw+Gn+Dc-N*nIFO;            // residual NULL energy         (neted[0])
      pwc->cData[id-1].netnull = Nw+Gn;                      // packet NULL                  (neted[1])
      pwc->cData[id-1].energy  = Ew;                         // energy in time domain        (neted[2])
      pwc->cData[id-1].likesky = Em;                         // energy in the loop           (neted[3])
      pwc->cData[id-1].enrgsky = Eo;                         // TF-domain all-res energy     (neted[4])
      pwc->cData[id-1].netecor = Ec;                         // packet (signal) coherent energy                        
      pwc->cData[id-1].normcor = Ec*Rc;                      // normalized coherent energy                        
      if(this->netRHO>=0) {	// original 2G
      	pwc->cData[id-1].netRHO  = rho/sqrt(cc);             // reduced rho - stored in rho[0]
      	pwc->cData[id-1].netrho  = rho;                      // chirp rho   - stored in rho[1]
      } else {			// (XGB.rho0)
	pwc->cData[id-1].netRHO  = -rho;               	     // reduced rho - stored in rho[0] with negative value in order to inform netevent.cc that it is XGB.rho0
	pwc->cData[id-1].netrho  = xrho/sqrt(cc);            // original 2G rho - stored in rho[1], only for test	
      }
      pwc->cData[id-1].cTime   = To;
      pwc->cData[id-1].cFreq   = Fo;
      pwc->cData[id-1].theta   = nLikelihood.getTheta(lm);
      pwc->cData[id-1].phi     = nLikelihood.getPhi(lm);
      pwc->cData[id-1].gNET    = sqrt(ff+FF);
      pwc->cData[id-1].aNET    = sqrt(FF/ff);
      pwc->cData[id-1].iNET    = 0;                          // degrees of freedom
      pwc->cData[id-1].nDoF    = N;                          // degrees of freedom
      pwc->cData[id-1].skyChi2 = CH;
      pwc->cData[id-1].Gnoise  = Gn;
      pwc->cData[id-1].iota    = 0;
      pwc->cData[id-1].psi     = 0;
      pwc->cData[id-1].ellipticity = 0.;

      cc = pwc->cData[id-1].netcc;
      if(hist) {
	 printf("rho=%4.2f|%4.2f cc: %5.3f|%5.3f|%5.3f subnet=%4.3f|%4.3f \n",
	        rho,rho*sqrt(Cp),Co,Cp,Cr,pwc->cData[id-1].subnet,pwc->cData[id-1].SUBNET);
	 printf(" L: %5.1f|%5.1f|%5.1f E: %5.1f|%5.1f|%5.1f|%5.1f N: %4.1f|%4.1f|%4.1f|%4.1f|%4.1f \n",
	        Lw,Lp,Lo,Ew,Ep,Eo,Em,Nw,Np,Rc,Eh,No);
	 printf("id|lm %3d|%6d  Vm|m=%3d|%3d|%3d|%3d T|F: %6.3f|%4.1f (t,p)=(%4.1f|%4.1f) \n",
		int(id),int(lm),int(V),int(Mo),int(Mw),int(M),To,Fo,nLikelihood.getTheta(lm),nLikelihood.getPhi(lm)); 
	 cout<<" L: |"; for(i=1; i<nIFO+1; i++) {printf("%5.1f|",S_snr[i]);} 
	 cout<<" E: |"; for(i=1; i<nIFO+1; i++) {printf("%5.1f|",D_snr[i]);} 
	 cout<<" N: |"; for(i=1; i<nIFO+1; i++) {printf("%5.1f|",N_snr[i]);}
	 cout<<endl<<" dof|G|G+R "; printf("%5.1f|%5.1f|%5.1f r[1]=%4.1f",N,Gn,Nw+Gn,REG[1]);
	 printf(" norm=%3.1f chi2 %3.2f|%3.2f Rc=%3.2f, Dc=%4.1f\n",norm,ch,CH,Rc,Dc);
		     //     cout<<" r1="<<REG[1]<<" norm="<<norm<<" chi2="<<ch<<"|"<<CH<<" Rc="<<Rc<<" Dc="<<Dc<<endl;
	 //hist->Fill(pwc->cData[id-1].subnet,pwc->cData[id-1].SUBNET);
      }
      count++;                                                        

// calculation of error regions

      pwc->p_Ind[id-1].push_back(Mo);
      double T = To+pwc->start;                          // trigger time
      std::vector<float> sArea;
      pwc->sArea.push_back(sArea);
      pwc->p_Map.push_back(sArea);

      double var = norm*Rc*sqrt(Mo)*(1+fabs(1-CH)); 
      
      if(iID<=0 || ID==id) { 
	 getSkyArea(id,lag,T,var);       // calculate error regions
      }
// calculation of chirp mass

      pwc->cData[id-1].mchirp = 0;
      pwc->cData[id-1].mchirperr = 0;
      pwc->cData[id-1].tmrgr = 0;
      pwc->cData[id-1].tmrgrerr = 0;
      pwc->cData[id-1].chi2chirp = 0;

      // It works only for MRA.
      if(m_chirp) {
        if(this->netRHO>=0) {
           ee = pwc->mchirp(id);		// original mchirp 2G
           cc = Ec/(fabs(Ec)+ee);         	// chirp cc 
           printf("mchirp_2g : %d %g %.2e %.3f %.3f %.3f %.3f \n\n",
                  int(id),cc,pwc->cData[id-1].mchirp,
                  pwc->cData[id-1].mchirperr, pwc->cData[id-1].tmrgr,
      	          pwc->cData[id-1].tmrgrerr, pwc->cData[id-1].chi2chirp);
        } else {				// Enabled only for Search=CBC/BBH/IMBHB
           if(m_chirp && (TString(Search)=="CBC"||TString(Search)=="BBH"||TString(Search)=="IMBHB")) {
              ee = pwc->mchirp_upix(id, this->nRun);		// mchirp micropixel version
           }
        }
      }

      if(ID==id && !EFEC) {   
	 this->nSensitivity.gps = T;
	 this->nAlignment.gps   = T;
	 this->nDisbalance.gps  = T;
	 this->nLikelihood.gps  = T;
	 this->nNullEnergy.gps  = T;
	 this->nCorrEnergy.gps  = T;
	 this->nCorrelation.gps = T;
	 this->nSkyStat.gps     = T;
	 this->nEllipticity.gps = T;
	 this->nPolarisation.gps= T;
	 this->nNetIndex.gps    = T;
      }

      pwc->sCuts[id-1] = -1;
      pwc->clean(id);
   } // end of loop over clusters
   
   if(_vtd.size()) _avx_free_ps(_vtd); 
   if(_vTD.size()) _avx_free_ps(_vTD); 
   if(_eTD.size()) _avx_free_ps(_eTD); 
   if(_AVX.size()) _avx_free_ps(_AVX);
   if(_APN.size()) _avx_free_ps(_APN);              // container for antenna patterns and noise RMS 
   if(_DAT.size()) _avx_free_ps(_DAT);              // container for data packet amplitudes 
   if(_SIG.size()) _avx_free_ps(_SIG);              // container for signal packet amplitudes  
   if(_NUL.size()) _avx_free_ps(_NUL);              // container for null packet amplitudes  
  
   return count;
}



long network::subNetCut(int lag, float subnet, float subcut, float subnorm, TH2F* hist)
{
// sub-network cut with dsp regulator
//  lag: lag index
//  subnet: sub network threshold 
//  subcut: sub network threshold in the skyloop (enabled only if >=0)
// hist: diagnostic histogram
// return number of processed pixels

   if(!this->wc_List[lag].size()) return 0;

   size_t nIFO = this->ifoList.size();
  
   if(nIFO>NIFO) {
      cout<<"network::subNetCut(): invalid network.\n";
      exit(0);
   }

   float   En = 2*acor*acor*nIFO;            // network energy threshold in the sky loop
   float   Es = 2*e2or;                      // subnet energy threshold in the sky loop

   subnet = fabs(subnet);                    // sub network threshold
   
   __m128 _En = _mm_set1_ps(En);
   __m128 _Es = _mm_set1_ps(Es);
   __m128 _oo = _mm_set1_ps(1.e-12);
   __m128 _0  = _mm_set1_ps(0.);
   __m128 _05 = _mm_set1_ps(0.5);
   __m128 _1  = _mm_set1_ps(1.);
   __m128* _pe[NIFO];

   int f_ = NIFO/4;
   int l,lm,Vm;
   float Lm,Em,Am,Lo,Eo,Co,Lr,Er,ee,em,To;
   float cc,aa,AA,rHo,stat,Ls,Ln,EE;
   float Lt;				    // store total coherent energy for all resolution levels

   size_t i,j,k,m,V,V4,id,K,M;
   int  Lsky = int(this->index.size());             // total number of source locations 
   short* mm = this->skyMask.data;

   float  vvv[NIFO] _ALIGNED;
   float* v00[NIFO] _ALIGNED;
   float* v90[NIFO] _ALIGNED;
   float*  pe[NIFO] _ALIGNED;
   float*  pa[NIFO] _ALIGNED;
   float*  pA[NIFO] _ALIGNED;
   short*  ml[NIFO] _ALIGNED;
   double* FP[NIFO] _ALIGNED;
   double* FX[NIFO] _ALIGNED;
   double  xx[NIFO] _ALIGNED;

   for(i=0; i<NIFO; i++) {
      if(i<nIFO) {
         ml[i] = getifo(i)->index.data;
         FP[i] = getifo(i)->fp.data;
         FX[i] = getifo(i)->fx.data;
      }
      else {
         ml[i] = getifo(0)->index.data;
         FP[i] = getifo(0)->fp.data;
         FX[i] = getifo(0)->fx.data;
      }
   }

   // allocate buffers
   std::vector<int> pI;                      // buffer for pixel IDs
   wavearray<double> cid;                    // buffers for cluster ID
   wavearray<double> cTo;                    // buffers for cluster time
   netpixel* pix;
   std::vector<int>* vint;
   netcluster* pwc = &this->wc_List[lag];
   
   size_t count = 0;
   size_t tsize = 0;

//+++++++++++++++++++++++++++++++++++++++
// loop over clusters
//+++++++++++++++++++++++++++++++++++++++

   cid = pwc->get((char*)"ID",  0,'S',0);                 // get cluster ID
   cTo = pwc->get((char*)"time",0,'L',0);                 // get cluster time
   
   K = cid.size();
   for(k=0; k<K; k++) {                                   // loop over clusters 
      id = size_t(cid.data[k]+0.1);
      if(pwc->sCuts[id-1] != -2) continue;                // skip rejected/processed clusters 
      vint = &(pwc->cList[id-1]);                         // pixel list
      V = vint->size();                                   // pixel list size
      if(!V) continue;

      pI = wdmMRA.getXTalk(pwc, id);

      V = pI.size();                                      // number of loaded pixels
      if(!V) continue;

      pix = pwc->getPixel(id,pI[0]);
      tsize = pix->tdAmp[0].size();
      if(!tsize || tsize&1) {                          // tsize%1 = 1/0 = power/amplitude
         cout<<"network::subNetCut() error: wrong pixel TD data\n";
         exit(1);
      }    
      tsize /= 2;
      V4 = V + (V%4 ? 4 - V%4 : 0);

      //cout<<En<<" "<<Es<<" "<<lag<<" "<<id<<" "<<V4<<" "<<" "<<tsize<<endl;
     
      std::vector<wavearray<float> > vtd;              // vectors of TD amplitudes
      std::vector<wavearray<float> > vTD;              // vectors of TD amplitudes
      std::vector<wavearray<float> > eTD;              // vectors of TD energies

      wavearray<float> tmp(tsize*V4); tmp=0;           // aligned array for TD amplitudes 
      wavearray<float>  fp(NIFO*V4);  fp=0;            // aligned array for + antenna pattern 
      wavearray<float>  fx(NIFO*V4);  fx=0;            // aligned array for x antenna pattern 
      wavearray<float>  nr(NIFO*V4);  nr=0;            // aligned array for inverse rms 
      wavearray<float>  Fp(NIFO*V4);  Fp=0;            // aligned array for pattern 
      wavearray<float>  Fx(NIFO*V4);  Fx=0;            // aligned array for patterns 
      wavearray<float>  am(NIFO*V4);  am=0;            // aligned array for TD amplitudes 
      wavearray<float>  AM(NIFO*V4);  AM=0;            // aligned array for TD amplitudes 
      wavearray<float>  bb(NIFO*V4);  bb=0;            // temporary array for MRA amplitudes 
      wavearray<float>  BB(NIFO*V4);  BB=0;            // temporary array for MRA amplitudes 
      wavearray<float>  xi(NIFO*V4);  xi=0;            // 00 array for reconctructed responses 
      wavearray<float>  XI(NIFO*V4);  XI=0;            // 90 array for reconstructed responses
      wavearray<float>  ww(NIFO*V4);  ww=0;            // 00 array for phase-shifted data vectors 
      wavearray<float>  WW(NIFO*V4);  WW=0;            // 90 array for phase-shifted data vectors
      wavearray<float>  u4(NIFO*4);   u4=0;            // temp array  
      wavearray<float>  U4(NIFO*4);   U4=0;            // temp array  

      __m128* _Fp = (__m128*) Fp.data;
      __m128* _Fx = (__m128*) Fx.data;
      __m128* _am = (__m128*) am.data;
      __m128* _AM = (__m128*) AM.data;
      __m128* _xi = (__m128*) xi.data;
      __m128* _XI = (__m128*) XI.data;
      __m128* _fp = (__m128*) fp.data;
      __m128* _fx = (__m128*) fx.data;
      __m128* _nr = (__m128*) nr.data; 
      __m128* _ww = (__m128*) ww.data;
      __m128* _WW = (__m128*) WW.data;
      __m128* _bb = (__m128*) bb.data;
      __m128* _BB = (__m128*) BB.data;

      for(i=0; i<NIFO; i++) {                          
         vtd.push_back(tmp);                           // array of aligned energy vectors
         vTD.push_back(tmp);                           // array of aligned energy vectors
         eTD.push_back(tmp);                           // array of aligned energy vectors
      }

      for(i=0; i<NIFO; i++) {                          // set up zero deley pointers                   
         pa[i] = vtd[i].data + (tsize/2)*V4;
         pA[i] = vTD[i].data + (tsize/2)*V4;
         pe[i] = eTD[i].data + (tsize/2)*V4; 
      }

      this->a_00.resize(NIFO*V4); this->a_00=0.;
      this->a_90.resize(NIFO*V4); this->a_90=0.;
      this->rNRG.resize(V4);      this->rNRG=0.;
      this->pNRG.resize(V4);      this->pNRG=0.;

      __m128* _aa = (__m128*) this->a_00.data;         // set pointer to 00 array
      __m128* _AA = (__m128*) this->a_90.data;         // set pointer to 90 array

      this->pList.clear();
      for(j=0; j<V; j++) {                             // loop over selected pixels 
         pix = pwc->getPixel(id,pI[j]);                // get pixel pointer
	 pList.push_back(pix);                         // store pixel pointers for MRA

         double rms = 0.;
         for(i=0; i<nIFO; i++) {
            xx[i] = 1./pix->data[i].noiserms;
            rms += xx[i]*xx[i];                        // total inverse variance
         }

         for(i=0; i<nIFO; i++) {
            nr.data[j*NIFO+i]=(float)xx[i]/sqrt(rms);  // normalized 1/rms
            for(l=0; l<tsize; l++) {
               aa = pix->tdAmp[i].data[l];             // copy TD 00 data
               AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data
               vtd[i].data[l*V4+j] = aa;               // copy 00 data
               vTD[i].data[l*V4+j] = AA;               // copy 90 data
               eTD[i].data[l*V4+j] = aa*aa+AA*AA;      // copy power
            }
         }
      }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// first sky loop
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      int lb = 0;
      int le = Lsky-1;
      bool mra = false;
      double suball=0;
      double submra=0;
      bool skymaskcc = (skyMaskCC.size()==Lsky);

      stat=Lm=Em=Am=EE=0.; lm=Vm= -1;    

  skyloop:

      for(l=lb; l<=le; l++) {	                      // loop over sky locations
         if(!mm[l] || l<0) continue;                  // skip delay configurations

         if(skymaskcc) {                              // transform l into celestial coordinates lc
           skymap* sm = &(this->nSkyStat);
           double gT = cTo.data[k]+pwc->start;        // trigger gps time
           double RA = sm->phi2RA(sm->getPhi(l),gT);  // phi -> RA
           int lc=this->getIndex(sm->getTheta(l),RA); // get sky index in celestial coordinates
           if (!skyMaskCC.data[lc]) continue;
         }
            
         _sse_point_ps(_pe, pe, ml, int(l), (int)V4); // point _pe to energy vectors
                                     
         __m128 _msk;
         __m128 _E_o = _mm_setzero_ps();              // total network energy
         __m128 _E_n = _mm_setzero_ps();              // network energy above the threshold
         __m128 _E_s = _mm_setzero_ps();              // subnet energy above the threshold
         __m128 _M_m = _mm_setzero_ps();              // # of pixels above threshold
         __m128* _rE = (__m128*) rNRG.data;           // m128 pointer to energy array     
         __m128* _pE = (__m128*) pNRG.data;           // m128 pointer to energy array     

         for(j=0; j<V4; j+=4) {                                // loop over selected pixels 
            *_rE = _sse_sum_ps(_pe);                           // get pixel energy     
            _msk = _mm_and_ps(_1,_mm_cmpge_ps(*_rE,_En));      // E>En  0/1 mask
	    _M_m = _mm_add_ps(_M_m,_msk);                      // count pixels above threshold
	    *_pE = _mm_mul_ps(*_rE,_msk);                      // zero sub-threshold pixels 
            _E_o = _mm_add_ps(_E_o,*_pE);                      // network energy
            _sse_minSNE_ps(_rE,_pe,_pE);                       // subnetwork energy with _pe increment
            _E_s = _mm_add_ps(_E_s,*_pE);                      // subnetwork energy
            _msk = _mm_and_ps(_1,_mm_cmpge_ps(*_pE++,_Es));    // subnet energy > Es 0/1 mask 
            _E_n = _mm_add_ps(_E_n,_mm_mul_ps(*_rE++,_msk));   // network energy
         }

         _mm_storeu_ps(vvv,_E_n);
         Ln = vvv[0]+vvv[1]+vvv[2]+vvv[3];             // network energy above subnet threshold
         _mm_storeu_ps(vvv,_E_o);
         Eo = vvv[0]+vvv[1]+vvv[2]+vvv[3]+0.01;        // total network energy
         _mm_storeu_ps(vvv,_E_s);
         Ls = vvv[0]+vvv[1]+vvv[2]+vvv[3];             // subnetwork energy
         _mm_storeu_ps(vvv,_M_m);
	 m = 2*(vvv[0]+vvv[1]+vvv[2]+vvv[3])+0.01;     // pixels above threshold

	 aa = Ls*Ln/(Eo-Ls);
         if(subcut>=0) if((aa-m)/(aa+m)<subcut) continue;
 
         pnt_(v00, pa, ml, (int)l, (int)V4);           // pointers to first pixel 00 data 
         pnt_(v90, pA, ml, (int)l, (int)V4);           // pointers to first pixel 90 data 
         float* pfp = fp.data;                         // set pointer to fp
         float* pfx = fx.data;                         // set pointer tp fx
         float* p00 = this->a_00.data;                 // set pointer for 00 array
         float* p90 = this->a_90.data;                 // set pointer for 90 array

         m = 0;
         for(j=0; j<V; j++) { 
            int jf = j*f_;                             // source sse pointer increment 
	    cpp_(p00,v00);  cpp_(p90,v90);             // copy amplitudes with target increment
            cpf_(pfp,FP,l); cpf_(pfx,FX,l);            // copy antenna with target increment
	    _sse_zero_ps(_xi+jf);                      // zero MRA amplitudes
            _sse_zero_ps(_XI+jf);                      // zero MRA amplitudes
            _sse_cpf_ps(_am+jf,_aa+jf);                // duplicate 00
            _sse_cpf_ps(_AM+jf,_AA+jf);                // duplicate 90
            if(rNRG.data[j]>En) m++;                   // count superthreshold pixels 
         }

	 __m128* _pp = (__m128*) am.data;              // point to multi-res amplitudes
         __m128* _PP = (__m128*) AM.data;              // point to multi-res amplitudes

         if(mra) {                                     // do MRA
	    _sse_MRA_ps(xi.data,XI.data,En,m);         // get principle components
            _pp = (__m128*) xi.data;                   // point to PC amplitudes
            _PP = (__m128*) XI.data;                   // point to PC amplitudes
	 }

	 m = 0; Ls=Ln=Eo=0;
	 for(j=0; j<V; j++) { 
	    int jf = j*f_;                             // source sse pointer increment 
	    int mf = m*f_;                             // target sse pointer increment
	    _sse_zero_ps(_bb+jf);                      // reset array for MRA amplitudes
	    _sse_zero_ps(_BB+jf);                      // reset array for MRA amplitudes
	    ee = _sse_abs_ps(_pp+jf,_PP+jf);           // total pixel energy
	    if(ee<En) continue;
	    _sse_cpf_ps(_bb+mf,_pp+jf);                // copy 00 amplitude/PC
	    _sse_cpf_ps(_BB+mf,_PP+jf);                // copy 90 amplitude/PC
	    _sse_cpf_ps(_Fp+mf,_fp+jf);                // copy F+
	    _sse_cpf_ps(_Fx+mf,_fx+jf);                // copy Fx
	    _sse_mul_ps(_Fp+mf,_nr+jf);                // normalize f+ by rms
	    _sse_mul_ps(_Fx+mf,_nr+jf);                // normalize fx by rms
	    m++;
	    em = _sse_maxE_ps(_pp+jf,_PP+jf);          // dominant pixel energy 
	    Ls += ee-em; Eo += ee;                     // subnetwork energy, network energy
	    if(ee-em>Es) Ln += ee;                     // network energy above subnet threshold
	 }
         if(Eo<=0) continue;

         size_t m4 = m + (m%4 ? 4 - m%4 : 0);
          _E_n = _mm_setzero_ps();                     // + likelihood

         for(j=0; j<m4; j+=4) {                                   
            int jf = j*f_;
	    _sse_dpf4_ps(_Fp+jf,_Fx+jf,_fp+jf,_fx+jf);                // go to DPF
            _E_s = _sse_like4_ps(_fp+jf,_fx+jf,_bb+jf,_BB+jf);        // std likelihood
            _E_n = _mm_add_ps(_E_n,_E_s);                             // total likelihood
         }
         _mm_storeu_ps(vvv,_E_n);

         Lo = vvv[0]+vvv[1]+vvv[2]+vvv[3];
	 AA = aa/(fabs(aa)+fabs(Eo-Lo)+2*m*(Eo-Ln)/Eo);        //  subnet stat with threshold
	 ee = Ls*Eo/(Eo-Ls);
	 em = fabs(Eo-Lo)+2*m;                                 //  suball NULL
	 ee = ee/(ee+em);                                      //  subnet stat without threshold
	 aa = (aa-m)/(aa+m);                    
	 if(!mra) Lt=Lo;				       //  store total coherent energy for all resolution levels

         if(AA>stat && !mra) {
	    stat=AA; Lm=Lo; Em=Eo; Am=aa; lm=l; Vm=m; suball=ee; EE=em;
	 }  
       }

      if(!mra && lm>=0) {mra=true; le=lb=lm; goto skyloop;}    // get MRA principle components
      
      pwc->sCuts[id-1] = -1; 
      pwc->cData[id-1].likenet = Lm; 
      pwc->cData[id-1].energy = Em; 
      pwc->cData[id-1].theta = nLikelihood.getTheta(lm);
      pwc->cData[id-1].phi = nLikelihood.getPhi(lm);
      pwc->cData[id-1].skyIndex = lm;

      rHo = 0.;
      if(mra) {
	 submra = Ls*Eo/(Eo-Ls);                                     // MRA subnet statistic
	 submra/= fabs(submra)+fabs(Eo-Lo)+2*(m+6);                  // MRA subnet coefficient 
	 To = 0;
	 pwc->p_Ind[id-1].push_back(lm);
	 for(j=0; j<vint->size(); j++) { 
	    pix = pwc->getPixel(id,j);
	    pix->theta = nLikelihood.getTheta(lm);
	    pix->phi   = nLikelihood.getPhi(lm);
	    To += pix->time/pix->rate/pix->layers;
	    if(j==0&&mra) pix->ellipticity = submra;                 // subnet MRA propagated to L-stage
	    if(j==0&&mra) pix->polarisation = fabs(Eo-Lo)+2*(m+6);   // submra NULL propagated to L-stage
	    if(j==1&&mra) pix->ellipticity = suball;                 // subnet all-sky propagated to L-stage
	    if(j==1&&mra) pix->polarisation = EE;                    // suball NULL propagated to L-stage
	 }
	 To /= vint->size();
	 rHo = sqrt(Lo*Lo/(Eo+2*m)/2);                               // estimator of coherent amplitude
      }
 
      if(hist && rHo>fabs(this->netRHO)) 
	 for(j=0;j<vint->size();j++) hist->Fill(suball,submra);

      if(fmin(suball,submra)>subnet && rHo>fabs(this->netRHO) && Em>subnorm*Eo) {
         count += vint->size();	
	 if(hist) {
	    printf("lag|id %3d|%3d rho=%5.2f To=%5.1f stat: %5.3f|%5.3f|%5.3f ",
		   int(lag),int(id),rHo,To,suball,submra,stat);
	    printf("E: %6.1f|%6.1f L: %6.1f|%6.1f|%6.1f pix: %4d|%4d|%3d|%2d \n",
		   Em,Eo,Lm,Lo,Ls,int(vint->size()),int(V),Vm,int(m));
	 }
      }         
      else pwc->sCuts[id-1]=1;

// clean time delay data

      V = vint->size();
      for(j=0; j<V; j++) {                           // loop over pixels           
         pix = pwc->getPixel(id,j);
         pix->core = true;
         if(pix->tdAmp.size()) pix->clean(); 
      } 
   }                                                 // end of loop over clusters
   return count;
}


long network::likelihood2G(char mode, int lag, int iID, TH2F* hist)
{
// 2G likelihood analysis
//  mode: analysis mode: 
//  OPTRES analyses, if upper case and optim=true
//  MRA analysis in low case or optim=false
//        r - un-modeled
//        i - iota - wave: no,partial dispersion correction
//        p - Psi - wave (no dispersion correction)
//      l,s - linear, loose linear
//      c,g - circular. loose circular
//      e,b - elliptical (no dispersion correction), b=p for now
//   iID: cluster ID, if negative - sky error regions are calculated 
//   lag: lag index
// hist: chirp histogram: If not needed, TGraphErrors* hist=NULL
// shold be used as input
// return number of processed pixels
// Negative gamma regulator turns on the AP prior for sky localization
//
// Delta Regulator:
// constraint   w(weak)     g(circ)    h(hard)              
//     D0       1----------0.5--------0.5                   // value of D0 (00-phase) threshold
//   |delta|:   0----------0.5---------1                    // value of delta regulator
//     D9       1-----------1---------0.5                   // value of D9 (90-phase) threshold

   if(!this->wc_List[lag].size()) return 0;

   this->like('2');
   this->wdm(true);
   this->tYPe = mode;

   bool cirwave = mode=='g' || mode=='G' || mode=='c' || mode=='C';
   bool linwave = mode=='l' || mode=='L' || mode=='s' || mode=='S';
   bool iotwave = mode=='i' || mode=='l' || mode=='e' || mode=='c' ||
                  mode=='I' || mode=='L' || mode=='E' || mode=='C';
   bool psiwave = mode=='l' || mode=='e' || mode=='p' ||
                  mode=='L' || mode=='E' || mode=='P';
   bool mureana = mode=='i' || mode=='e' || mode=='c' ||
                  mode=='r' || mode=='p' || mode=='b' ||
                  mode=='l' || mode=='s' || mode=='g';
   bool rndwave = mode=='r' || mode=='R';

   bool   prior = this->gamma<0 ? true  : false;  	 // gamma<0 : antenna pattern prior is used
   bool m_chirp = this->optim   ? false : mureana;

   if(!this->optim) mureana = true;

   size_t nIFO = this->ifoList.size();
   size_t ID = abs(iID);
  
   if(nIFO>NIFO) {
      cout<<"network::likelihood2G(): invalid network.\n";
      exit(0);
   }

   float   En = 2*acor*acor*nIFO;                           // network energy threshold in the sky loop
   float   Es = 2*e2or;                                     // subnet energy threshold 
   float gama = fabs(this->gamma);                          // gamma regulator - hard/kill
   float deta = fabs(this->delta);                          // delta regulator - weak/circular/hard
   if(gama<=0) gama = 1.e-24;                               // limit gamma 
   if(gama>=1) gama = 0.999999;                             // limit gamma 

// delta regulator:
// constraint   w(weak)     g(circ)    h(hard)              // w - no constraint
//     DI       1----------0.5--------0.5                   // value of DI (00-phase) threshold
//   |delta|:   0----------0.5---------1                    // value of delta regulator
//     DQ       1-----------1---------0.5                   // value of DQ (90-phase) threshold

   static const __m128 _D0 = _mm_set1_ps(deta<0.5?1-deta:0.5); 
   static const __m128 _D9 = _mm_set1_ps(deta<0.5?1:1.5-deta); 

   static const __m128 _oo = _mm_set1_ps(1.e-16);           // nusance parameter
   static const __m128 _sm = _mm_set1_ps(-0.f);             // sign mask: -0.f = 1 << 31
   static const __m128 _En = _mm_set1_ps(En);               // network threshold
   static const __m128 _rG = _mm_set1_ps(-1./log(gama));    // regulator-gamma threshold           
   static const __m128 _kG = _mm_set1_ps(gama);             // kill-gamma threshold      
   static const __m128 _PW = _mm_set1_ps(psiwave?0:1);      // flag for psiwave option
   static const __m128 _01 = _mm_set1_ps(0.1);
   static const __m128 _05 = _mm_set1_ps(0.5);
   static const __m128 _09 = _mm_set1_ps(0.9);
   static const __m128 _1  = _mm_set1_ps(1.0+1.e-16);
   static const __m128 _2  = _mm_set1_ps(2.0);
   static const __m128 _4  = _mm_set1_ps(4.0);

   __m128* _pe[NIFO];

   int f_ = NIFO/4;
   float NRG,Lm,Em,Lo,Eo,No,Nm,cc,Cm,Co,Do,To,Fo,Ln,Ns;
   float STAT,CHR,aa,AA,ee,em,EE,ff,FF,Lr,Cr,ss,Ls,Nc,gg;
   double eLp, s2p, c2p;
   int   optR = 0;                            // optimal resolution (used by SRA)

   size_t i,j,k,l,m,Vm,lm,V,V4,V44,id,K,M;
   size_t L = this->index.size();             // total number of source locations 
   short* mm = this->skyMask.data;
   bool skymaskcc = (skyMaskCC.size()==L);

   float  vvv[NIFO] _ALIGNED;
   float  uuu[NIFO] _ALIGNED;
   float* v00[NIFO] _ALIGNED;
   float* v90[NIFO] _ALIGNED;
   float*  pe[NIFO] _ALIGNED;
   float*  pa[NIFO] _ALIGNED;
   float*  pA[NIFO] _ALIGNED;
   short*  ml[NIFO] _ALIGNED;
   double* FP[NIFO] _ALIGNED;
   double* FX[NIFO] _ALIGNED;
   double  xx[NIFO] _ALIGNED;

   for(i=0; i<NIFO; i++) {
      if(i<nIFO) {
         ml[i] = getifo(i)->index.data;
         FP[i] = getifo(i)->fp.data;
         FX[i] = getifo(i)->fx.data;
      }
      else {
         ml[i] = getifo(0)->index.data;
         FP[i] = getifo(0)->fp.data;
         FX[i] = getifo(0)->fx.data;
      }
        }

   // allocate buffers
   std::vector<int> pI;                      // buffer for pixel IDs
   std::vector<int> pJ;                      // buffer for pixel index
   wavearray<double> cid;                    // buffers for cluster ID
   wavearray<double> cTo;                    // buffers for cluster time
   netpixel* pix;
   std::vector<int>* vint;
   std::vector<int>* vtof;
   std::vector<int> pRate;
   netcluster* pwc = &this->wc_List[lag];
   
   size_t count = 0;
   size_t tsize = 0;

   std::map<int,float> vLr;		     // resolution map

//+++++++++++++++++++++++++++++++++++++++
// loop over clusters
//+++++++++++++++++++++++++++++++++++++++

   cid = pwc->get((char*)"ID",  0,'S',0);                 // get cluster ID
   cTo = pwc->get((char*)"time",0,'L',0);                 // get cluster time
   
   K = cid.size();
   for(k=0; k<K; k++) {                                   // loop over clusters 
      id = size_t(cid.data[k]+0.1);

      if(pwc->sCuts[id-1] != -2) continue;                // skip rejected/processed clusters 

      vint = &(pwc->cList[id-1]);                         // pixel list
      vtof = &(pwc->nTofF[id-1]);                         // TofFlight configurations
      V = vint->size();
      if(!V) continue;

      pI = wdmMRA.getXTalk(pwc, id);
      V = pI.size();
      if(!V) continue;

      if(ID==id) {
         this->nSensitivity = 0.;
         this->nAlignment = 0.;
         this->nNetIndex = 0.;
         this->nDisbalance = 0.;
         this->nLikelihood = 0.;
         this->nNullEnergy = 0.;
         this->nCorrEnergy = 0.;
         this->nCorrelation = 0.;
         this->nSkyStat = 0.;
         this->nEllipticity = 0.;
         this->nPolarisation = 0.;
         this->nProbability = 0.;
      }                                                
      this->nAntenaPrior = 0.;

      pix = pwc->getPixel(id,pI[0]);
      tsize = pix->tdAmp[0].size();
      if(!tsize || tsize&1) {                       // tsize%1 = 1/0 = power/amplitude
         cout<<"network::likelihood2G() error: wrong pixel TD data\n";
         exit(1);
      }
      
      tsize /= 2;
 
      if(!(V=pI.size())) continue;
      V4  = V + (V%4 ? 4 - V%4 : 0);
      V44 = V4 + 4;
      pJ.clear();
      for(j=0; j<V4; j++) pJ.push_back(0);                          

      //cout<<En<<" "<<Es<<" "<<lag<<" "<<id<<" "<<V4<<" "<<" "<<tsize<<endl;
     
      std::vector<wavearray<float> > vtd;              // vectors of TD amplitudes
      std::vector<wavearray<float> > vTD;              // vectors of TD amplitudes
      std::vector<wavearray<float> > eTD;              // vectors of TD energies

      wavearray<float> tmp(tsize*V4); tmp=0;           // aligned array for TD amplitudes 
      wavearray<float>  nr(NIFO*V44); nr=0;            // aligned array for inverse rms 
      wavearray<float>  fp(NIFO*V44); fp=0;            // aligned array for + antenna pattern 
      wavearray<float>  fx(NIFO*V44); fx=0;            // aligned array for x antenna pattern 
      wavearray<float>  ep(NIFO*V44); ep=0;            // aligned array for + unity vector 
      wavearray<float>  ex(NIFO*V44); ex=0;            // aligned array for x unity vector 
      wavearray<float>  Fp(NIFO*V44); Fp=0;            // aligned array for F+ patterns 
      wavearray<float>  Fx(NIFO*V44); Fx=0;            // aligned array for Fx patterns 
      wavearray<float>  am(NIFO*V44); am=0;            // aligned array for pixel amplitudes 
      wavearray<float>  AM(NIFO*V44); AM=0;            // aligned array for pixel amplitudes 
      wavearray<float>  bb(NIFO*V44); bb=0;            // temporary array for MRA amplitudes 
      wavearray<float>  BB(NIFO*V44); BB=0;            // temporary array for MRA amplitudes 
      wavearray<float>  xi(NIFO*V44); xi=0;            // 00 array for reconctructed responses 
      wavearray<float>  XI(NIFO*V44); XI=0;            // 90 array for reconstructed responses
      wavearray<float>  ww(NIFO*V44); ww=0;            // 00 array for phase-shifted data vectors 
      wavearray<float>  WW(NIFO*V44); WW=0;            // 90 array for phase-shifted data vectors
      wavearray<float>  xp(NIFO*V44); xp=0;            // 00 array for network projection 
      wavearray<float>  XP(NIFO*V44); XP=0;            // 90 array for network projection

      // data arrays for polar coordinates storage : [0,1] = [radius,angle]
      for(i=0;i<2;i++) {
        this->p00_POL[i].resize(V4); this->p00_POL[i]=0.;
        this->p90_POL[i].resize(V4); this->p90_POL[i]=0.;
        this->r00_POL[i].resize(V4); this->r00_POL[i]=0.;
        this->r90_POL[i].resize(V4); this->r90_POL[i]=0.;
      }

      __m128* _Fp  = (__m128*) Fp.data;
      __m128* _Fx  = (__m128*) Fx.data;
      __m128* _am  = (__m128*) am.data;
      __m128* _AM  = (__m128*) AM.data;
      __m128* _xi  = (__m128*) xi.data;
      __m128* _XI  = (__m128*) XI.data;
      __m128* _xp  = (__m128*) xp.data;
      __m128* _XP  = (__m128*) XP.data;
      __m128* _ww  = (__m128*) ww.data;
      __m128* _WW  = (__m128*) WW.data;
      __m128* _bb  = (__m128*) bb.data;
      __m128* _BB  = (__m128*) BB.data;
      __m128* _fp  = (__m128*) fp.data;
      __m128* _fx  = (__m128*) fx.data;
      __m128* _nr  = (__m128*) nr.data;
      __m128* _ep  = (__m128*) ep.data;                 // point to + unity vector
      __m128* _ex  = (__m128*) ex.data;                 // point to x unity vector

      __m128* _fp4 = _fp+V4*f_;
      __m128* _fx4 = _fx+V4*f_;
      __m128* _uu4 = _am+V4*f_;
      __m128* _vv4 = _AM+V4*f_;
      __m128* _bb4 = _bb+V4*f_;
      __m128* _BB4 = _BB+V4*f_;

      for(i=0; i<NIFO; i++) {                          
         vtd.push_back(tmp);                           // array of aligned energy vectors
         vTD.push_back(tmp);                           // array of aligned energy vectors
         eTD.push_back(tmp);                           // array of aligned energy vectors
      }

      for(i=0; i<NIFO; i++) {                          // set up zero deley pointers                   
         pa[i] = vtd[i].data + (tsize/2)*V4;
         pA[i] = vTD[i].data + (tsize/2)*V4;
         pe[i] = eTD[i].data + (tsize/2)*V4; 
      }

      wavearray<float>  siDPF(V4); siDPF=0;            // temporary array for DPF sin 
      wavearray<float>  coDPF(V4); coDPF=0;            // temporary array for DPF cos 
      wavearray<float>  siORT(V4); siORT=0;            // temporary array for ort4 sin 
      wavearray<float>  coORT(V4); coORT=0;            // temporary array for ort4 cos 
      wavearray<float>  siPHS(V4); siPHS=0;            // temporary array for Phase sin 
      wavearray<float>  coPHS(V4); coPHS=0;            // temporary array for Phase cos 
      wavearray<float>   chir(V4);  chir=0;            // chirality array 
      wavearray<float>   q2Q2(V4);  q2Q2=0;            // energy array q^2+Q^2 
      wavearray<float>   ec00(V4);  ec00=0;            // 00-phase coherent energy array 
      wavearray<float>   EC90(V4);  EC90=0;            // 90-phase coherent energy array 
      wavearray<float>    zzz(V4);   zzz=0;            // temporary array 
      wavearray<float>    yyy(V4);   yyy=0;            // temporary array 
      wavearray<float>    xxx(V4);   xxx=0;            // temporary array
      wavearray<float>    rrr(V4);   rrr=0;            // regulator flag array

      this->a_00.resize(NIFO*V44); this->a_00=0.;      // array for pixel amplitudes in sky loop
      this->a_90.resize(NIFO*V44); this->a_90=0.;      // array for pixel amplitudes in sky loop
      this->rNRG.resize(V4);       this->rNRG=0.;
      this->pNRG.resize(V4);       this->pNRG=0.;

      __m128* _aa  = (__m128*) this->a_00.data;         // set pointer to 00 array
      __m128* _AA  = (__m128*) this->a_90.data;         // set pointer to 90 array


      this->pList.clear(); pRate.clear();
      for(j=0; j<V; j++) {                             // loop over selected pixels 
         pix = pwc->getPixel(id,pI[j]);
	 this->pList.push_back(pix);                   // store pixel pointers for MRA
	 pRate.push_back(int(pix->rate+0.5));          // store pixel rates for MRA

	 if(vLr.find(pRate[j]) == vLr.end())           // initialize vLr map
	    vLr[pRate[j]] = 0.;

         double rms = 0.;
         for(i=0; i<nIFO; i++) {
            xx[i] = 1./pix->data[i].noiserms;
            rms += xx[i]*xx[i];                        // total inverse variance
         }

         rms = sqrt(rms);
         for(i=0; i<nIFO; i++) {
            nr.data[j*NIFO+i]=(float)xx[i]/rms;        // normalized 1/rms
            for(l=0; l<tsize; l++) {
               aa = pix->tdAmp[i].data[l];             // copy TD 00 data
               AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data
               vtd[i].data[l*V4+j] = aa;               // copy 00 data
               vTD[i].data[l*V4+j] = AA;               // copy 90 data
               eTD[i].data[l*V4+j] = aa*aa+AA*AA;      // copy power
            }
         }
      }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// sky loop
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      STAT=0.; lm=0; Vm=0; 
      double skystat = 0.;
      size_t lb = 0;
      size_t le = L-1;
      bool mra = false;
 
   optsky:

      AA = 0.;                                         // initialize sky statistic
      for(l=lb; l<=le; l++) {	                       // loop over sky locations
         if(!mra) skyProb.data[l] = 0.;
         if(!mm[l]) continue;                          // skip delay configurations

         if(skymaskcc) {                               // transform l into celestial coordinates lc
           skymap* sm = &(this->nSkyStat);
           double gT = cTo.data[k]+pwc->start;         // trigger gps time
           double RA = sm->phi2RA(sm->getPhi(l),gT);   // phi -> RA
           int lc=this->getIndex(sm->getTheta(l),RA);  // get sky index in celestial coordinates
           if (!skyMaskCC.data[lc]) continue;
         }
            
         pnt_(v00, pa, ml, (int)l, (int)V4);           // pointers to first pixel 00 data 
         pnt_(v90, pA, ml, (int)l, (int)V4);           // pointers to first pixel 90 data 
         float* pfp = fp.data;                         // set pointer to fp
         float* pfx = fx.data;                         // set pointer to fx
         float* p00 = this->a_00.data;                 // set pointer for 00 array
         float* p90 = this->a_90.data;                 // set pointer for 90 array
	 float mxLr = 0.;
	       optR = 0;
 
         for(j=0; j<V; j++) { 
            cpp_(p00,v00);  cpp_(p90,v90);             // copy amplitudes with target increment
            cpf_(pfp,FP,l); cpf_(pfx,FX,l);            // copy antenna with target increment
	    if(!this->optim || !mra) continue;         // skip if not optimal resolution or !mra  
	    if(vLr[pRate[j]] <= mxLr) continue;        // skip small Lr
	    mxLr = vLr[pRate[j]];                      // update maximum Lr
	    optR = pRate[j];                           // select optimal rate
	 }

         m = 0;
         for(j=0; j<V; j++) { 
            int jf = j*f_;                             // source sse pointer increment 
            _sse_zero_ps(_xi+jf);                      // zero MRA amplitudes
            _sse_zero_ps(_XI+jf);                      // zero MRA amplitudes
            rNRG.data[j] = 1;                          // SRA flag
	    if(optR && optR!=pRate[j]) {               // clear non optimal rate amplitudes
	       _sse_zero_ps(_aa+jf);
	       _sse_zero_ps(_AA+jf);
	       rNRG.data[j] = 0;                       // exclude these resolutions
	    }
            _sse_cpf_ps(_am+jf,_aa+jf);                // duplicate 00
            _sse_cpf_ps(_AM+jf,_AA+jf);                // duplicate 90
            ee = _sse_abs_ps(_aa+jf,_AA+jf);           // total pixel energy / quadrature
	    if(ee>En) m++; else ee=0.;                 // count core pixels
            rNRG.data[j]*= ee;                         // init residual energy array
            pNRG.data[j] = rNRG.data[j];               // init residual energy array
         }

	 __m128* _pp = (__m128*) am.data;              // point to multi-res amplitudes
         __m128* _PP = (__m128*) AM.data;              // point to multi-res amplitudes

         if(mra && mureana) {                          // do MRA
            _sse_mra_ps(xi.data,XI.data,En,m);         // get principle components
            _pp = (__m128*) xi.data;                   // point to PC amplitudes
            _PP = (__m128*) XI.data;                   // point to PC amplitudes
         }

         m = 0; Em = 0.;
         for(j=0; j<V; j++) { 
            int jf = j*f_;                             // source sse pointer increment 
            int mf = m*f_;                             // target sse pointer increment
	    pJ[j] = 0;
            _sse_zero_ps(_bb+jf);                      // reset array for MRA amplitudes
            _sse_zero_ps(_BB+jf);                      // reset array for MRA amplitudes
	    ee = pNRG.data[j];                         // total pixel energy
            if(ee<En) continue;
            _sse_cpf_ps(_bb+mf,_pp+jf);                // copy 00 amplitude/PC
            _sse_cpf_ps(_BB+mf,_PP+jf);                // copy 90 amplitude/PC
            _sse_cpf_ps(_Fp+mf,_fp+jf);                // copy F+
            _sse_cpf_ps(_Fx+mf,_fx+jf);                // copy Fx
            _sse_mul_ps(_Fp+mf,_nr+jf);                // normalize f+ by rms
            _sse_mul_ps(_Fx+mf,_nr+jf);                // normalize fx by rms
            pJ[m++]= j;                                // store pixel index
         }

         size_t m4 = m + (m%4 ? 4 - m%4 : 0);
         __m128 _ll,_LL,_ec,_EC,_ee,_EE,_NI,_s2,_c2,_AX,_NN,_FF,_QQ,_ie,_gg;
	 __m128 _en,_EN,_ed,_ED,_cc,_ss,_ni,_si,_co,_ax,_nn,_ff,_mm,_IE,_GG;

	 __m128* _siP = (__m128*) siPHS.data;          // phase sin
	 __m128* _coP = (__m128*) coPHS.data;	       // phase cos
	 __m128* _siO = (__m128*) siORT.data;          // ort4 sin
	 __m128* _coO = (__m128*) coORT.data;	       // ort4 cos
	 __m128* _siD = (__m128*) siDPF.data;	       // DPF sin
	 __m128* _coD = (__m128*) coDPF.data;	       // DPF cos
	 __m128* _nrg = (__m128*)  q2Q2.data;	       // energy (1+e^2)*(q^2+Q^2)
	 __m128* _chr = (__m128*)  chir.data;	       // chirality (sign of e)

	 if(mra) {                                
  	    _pp = (__m128*) xi.data;                   // point to PC amplitudes
  	    _PP = (__m128*) XI.data;                   // point to PC amplitudes
	 }

// test sky location

         Lo = Ln = Co = Eo = 0.;
         for(j=0; j<m4; j+=4) {
            int jf = j*f_;                             // sse index increment

            __m128* _pbb = _bb+jf;
            __m128* _pBB = _BB+jf;
            __m128* _pxi = _pp+jf;
            __m128* _pXI = _PP+jf;
            __m128* _pxp = _xp+jf;
            __m128* _pXP = _XP+jf;
            __m128* _pww = _ww+jf;
            __m128* _pWW = _WW+jf;
            __m128* _pfp = _fp+jf;
            __m128* _pfx = _fx+jf;
            __m128* _pFp = _Fp+jf;
            __m128* _pFx = _Fx+jf;

// do transformations 

	    _sse_ort4_ps(_pFp,_pFx,_siD,_coD);                 // get DPF sin and cos
	    _sse_rot4p_ps(_pFp,_coD,_pFx,_siD,_pfp);           // get DPF fp=Fp*c+Fx*s  
	    _sse_rot4m_ps(_pFx,_coD,_pFp,_siD,_pfx);           // get DPF fx=Fx*c-Fp*s 	    
            _sse_pnp4_ps(_pfp,_pfx,_pbb,_pBB,_pxp,_pXP);       // projection on network plane
            _sse_ort4_ps(_pxp,_pXP,_siO,_coO);                 // dual-stream phase sin and cos
            _sse_rot4p_ps(_pxp,_coO,_pXP,_siO,_pxi);           // get 00 standard response  
            _sse_rot4m_ps(_pXP,_coO,_pxp,_siO,_pXI);           // get 90 standard response 
            _sse_rot4p_ps(_pbb,_coO,_pBB,_siO,_pww);           // get 00 phase data vector 
            _sse_rot4m_ps(_pBB,_coO,_pbb,_siO,_pWW);           // get 90 phase data vector
            _coO++; _siO++; _coD++; _siD++;                    // increment to next 4 pixels

// save projection on network plane and standard response in polar coordinates
            if(le==lb && (optR==0)) {
              _sse_pol4_ps(_pfp, _pfx, _pxp, p00_POL[0].data+j, p00_POL[1].data+j);
              _sse_pol4_ps(_pfp, _pfx, _pXP, p90_POL[0].data+j, p90_POL[1].data+j);
              _sse_pol4_ps(_pfp, _pfx, _pxi, r00_POL[0].data+j, r00_POL[1].data+j);
              _sse_pol4_ps(_pfp, _pfx, _pXI, r90_POL[0].data+j, r90_POL[1].data+j);
            }

// standard statistics

            _ee = _sse_abs4_ps(_pww);                          // 00 energy
            _ll = _sse_abs4_ps(_pxi);                          // standard 00 likelihood
            _ec = _sse_ecoh4_ps(_pww,_pxi,_ll);                // 00 coherent energy
            _EE = _sse_abs4_ps(_pWW);                          // 90 energy
            _LL = _sse_abs4_ps(_pXI);                          // standard 00 likelihood
            _EC = _sse_ecoh4_ps(_pWW,_pXI,_LL);                // 90 coherent energy            
            _cc = _mm_and_ps(_mm_cmpge_ps(_ec,_05),_1);        // 00-phase denoising  
            _ss = _mm_and_ps(_mm_cmpgt_ps(_EC,_05),_cc);       // 90-phase denoising 

            _ec = _sse_rotp_ps(_ec,_cc,_EC,_ss);               // total coherent energy            
            _ll = _sse_rotp_ps(_ll,_cc,_LL,_ss);               // total likelihood            
            _mm = _sse_rotp_ps(_1,_cc,_1,_ss);                 // Gaussian noise            
            _nn = _mm_add_ps(_mm_add_ps(_ee,_EE),_2);          // total energy
	    _cc = _mm_add_ps(_ec,_mm_sub_ps(_nn,_ll));
	    _cc = _mm_div_ps(_ec,_mm_add_ps(_cc,_mm));         // network correlation coefficient	    

            _mm_storeu_ps(vvv,_mm_add_ps(_ee,_EE));
            Eo += vvv[0]+vvv[1]+vvv[2]+vvv[3];                 // network null energy
            _mm_storeu_ps(vvv,_ll);
            Lo += vvv[0]+vvv[1]+vvv[2]+vvv[3];                 // network likelihood
            _mm_storeu_ps(vvv,_mm_mul_ps(_ll,_cc));
            Ln += vvv[0]+vvv[1]+vvv[2]+vvv[3];                 // network likelihood
            _mm_storeu_ps(vvv,_ec);
            Co += vvv[0]+vvv[1]+vvv[2]+vvv[3];                 // network coherent energy
            _mm_storeu_ps(vvv,_mm_mul_ps(_ec,_ll));
	    if(le==lb && !mra) {		               // optimal resolution likelihood
	       vLr[pRate[pJ[j+0]]] += vvv[0];
	       vLr[pRate[pJ[j+1]]] += vvv[1];
	       vLr[pRate[pJ[j+2]]] += vvv[2];
	       vLr[pRate[pJ[j+3]]] += vvv[3];
	    }
	 }
	 Ln = Eo>0 ? Ln/Eo : 0;
         if(Ln<this->netCC) continue;                          // skip
	 _IE = _mm_set1_ps(1-Co/Lo);                           // global event index

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// reconstruction loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	 __m128* _xx = (__m128*) xxx.data;	               // store |f+|
	 __m128* _yy = (__m128*) yyy.data;	               // store |fx|
	 __m128* _zz = (__m128*) zzz.data;	               // store total amplitude
	 __m128* _rr = (__m128*) rrr.data;	               // store regulator flag

	 for(j=0; j<m4; j+=4) {
	    int jf = j*f_;                                     // sse index increment
	    __m128* _pfp = _fp+jf;
	    __m128* _pfx = _fx+jf;
	    *_xx = _sse_abs4_ps(_pfp);                         // |f+|^2 
	    *_yy = _sse_abs4_ps(_pfx);                         // |fx|^2 
	    _ee = _mm_add_ps(_mm_sqrt_ps(*_xx++),_oo);         // |f+| + eps 
	    _EE = _mm_add_ps(_mm_sqrt_ps(*_yy++),_oo);         // |fx| + eps 
	    _sse_cpf4_ps(_ep+jf,_pfp,_mm_div_ps(_1,_ee));      // store + unity vector in ep
	    _sse_cpf4_ps(_ex+jf,_pfx,_mm_div_ps(_1,_EE));      // store x unity vector in ex
	 }						       
	 
	 _xx  = (__m128*) xxx.data;	                       // store 1/|f+|
	 _yy  = (__m128*) yyy.data;	                       // store 1/|fx|
	 _zz  = (__m128*) zzz.data;	                       // store (1-e*e)*(qq+QQ)*|fx|^2
	 _rr  = (__m128*) rrr.data;	                       // regulator flag
	 _siP = (__m128*) siPHS.data;                          // phase sin
	 _coP = (__m128*) coPHS.data;	                       // phase cos
	 _siD = (__m128*) siDPF.data;                          // DPF sin
	 _coD = (__m128*) coDPF.data;	                       // DPF cos
	 _nrg = (__m128*)  q2Q2.data;	                       // energy (1+e^2)*(q^2+Q^2)
	 
	 __m128 linw = linwave ? _oo : _1;                     
	 __m128 cirw = cirwave ? _oo : _1;
	 __m128  _ch = _mm_setzero_ps();                       // chirality
	 __m128 _eqQ = _mm_setzero_ps();                       // e*(qq+QQ)*|fx|^2
	 __m128 _qQ2 = _mm_setzero_ps();                       // 2*(qq+QQ)*|fx|^2

	 _c2 = _mm_setzero_ps();                               // cos[-2p] polarization angle
	 _s2 = _mm_setzero_ps();                               // sin[-2p]
	 _cc = _mm_setzero_ps();                               // cos[t]  average vector angle
	 _ss = _mm_setzero_ps();                               // sin[t]
	 
	 for(j=0; j<m4; j+=4) {                                // produce NEW pattern
	    int jf = j*f_;                                     // sse index increment	    
	    __m128* _pfp = _fp+jf;
	    __m128* _pfx = _fx+jf;
	    __m128* _pep = _ep+jf;
	    __m128* _pex = _ex+jf;
	    __m128* _pxp = _xp+jf;
	    __m128* _pXP = _XP+jf;
	    __m128* _pxi = _pp+jf;
	    __m128* _pXI = _PP+jf;
	    
	    _ee = _mm_div_ps(_1,_mm_add_ps(*_xx,_oo));         // 1/|f+|^2 
	    _co = _mm_mul_ps(_sse_dot4_ps(_pfp,_pxi),_ee);     // (fp,x)/|f+|^2
	    _si = _mm_mul_ps(_sse_dot4_ps(_pfp,_pXI),_ee);     // (fp,X)/|f+|^2
	    _sse_rot4p_ps(_pxi,&_co,_pXI,&_si,_bb4);           // y 00 phase pattern  
	    _sse_rot4m_ps(_pXI,&_co,_pxi,&_si,_BB4);           // Y 90 phase pattern 	       
	    _NN = _sse_rotp_ps(_co,_co,_si,_si);               // (xp*xp+XP*XP) - norm^2
	    _nn = _mm_div_ps(_1,_mm_add_ps(_NN,_oo));          // 1/(xp*xp+XP*XP) - 1/norm^2
	    *_siP = _mm_mul_ps(_si,_nn);                       // normalize siP (used later)
	    *_coP = _mm_mul_ps(_co,_nn);                       // normalize coP (used later)

            if(le==lb && (optR==0)) {                          // save polargrams
              __m128 _snn = _mm_sqrt_ps(_nn);
              _sse_cpf4_ps(_uu4,_bb4,_snn);                    // normalize bb4
              _sse_cpf4_ps(_vv4,_BB4,_snn);                    // normalize BB4
              _sse_pol4_ps(_ep+jf, _ex+jf, _uu4, r00_POL[0].data+j, r00_POL[1].data+j);
              _sse_pol4_ps(_ep+jf, _ex+jf, _vv4, r90_POL[0].data+j, r90_POL[1].data+j);
            }
	    
	    _en = _sse_dot4_ps(_pex,_bb4);                     // (ex,y)=-(1-e*e)*(qq+QQ)*S[2d-2p]*|fx|/2
	    _EN = _sse_dot4_ps(_pex,_BB4);                     // e*(qq+QQ)*|fx|
	    _ch = _mm_add_ps(_ch,_EN);                         // total e*(qq+QQ)*|fx|
	    _ll = _mm_and_ps(_mm_cmpgt_ps(_EN,_oo),_1);        // 1 if e positive
	    *_chr = _mm_sub_ps(_mm_mul_ps(_ll,_2),_1);         // +1/-1 e positive/negative
	    
	    _ll = _sse_abs4_ps(_pxi);                          // 00 energy in ORT pattern
	    _ni = _sse_ind4_ps(_pfp,*_xx);                     // f+ index
	    _NI = _sse_ind4_ps(_pfx,*_yy);                     // fx index
	    _ie = _sse_ind4_ps(_pxi,_ll);                      // normalized incoherent energy
	    _ff = _mm_div_ps(_mm_add_ps(*_xx,*_yy),_ni);       // (|f+|^2+|fx|^2)/ni

	    _LL = _sse_rotp_ps(_en,_en,_EN,_EN);               // (fx,y)^2+(fx,Y)^2
	    _LL = _mm_mul_ps(_LL,_nn);                         // {(1+e*e)-(1-e*e)*C[2d-2p]}*|fx|^2*(qq+QQ)/2 
	    _FF = _mm_mul_ps(*_yy,_ee);                        // FF = |fx|^2/|f+|^2 - alignment factor
	    _gg = _mm_mul_ps(_01,_ff);                         // 0.1*ff
	    _NN = _mm_mul_ps(_NN,*_xx);                        // {(1+e*e)+(1-e*e)*C[2d-2p]}*|f+|^2*(qq+QQ)/2 
	    _ll = _mm_mul_ps(_NN,_mm_add_ps(_FF,_gg));         // {(1+e*e)+(1-e*e)*C[2d-2p]}*|f+|^2*(qq+QQ)/2 * (FF+0.1*ff) 
	    _co = _mm_sub_ps(_ll,_LL);                         // C2*(1-e*e)*(qq+QQ)*|fx|^2
	    _si = _mm_mul_ps(_2,_sse_dot4_ps(_pfx,_bb4));      //-S2*(1-e*e)*(qq+QQ)*|fx|^2
	    *_zz = _sse_rotp_ps(_co,_co,_si,_si);              // [(1-e*e)*(qq+QQ)*|fx|^2]^2
	    *_zz = _mm_add_ps(_mm_sqrt_ps(*_zz),_oo);          //  (1-e*e)*(qq+QQ)*|fx|^2
	    _ll = _mm_mul_ps(_NN,_FF);                         // {(1+e*e)+(1-e*e)*C[2d-2p]}*|fx|^2*(qq+QQ)/2 
	    _QQ = _mm_add_ps(_mm_add_ps(_ll,_LL),*_zz);        // 2*(qq+QQ)*|fx|^2
	    _ff = _mm_sqrt_ps(_mm_mul_ps(_ff,_05));            // sqrt[(|f+|^2+|fx|^2)/ni/2]
	    _FF = _mm_sqrt_ps(_FF);                            // a = FF = |fx|/|f+| - alignment factor
	                                                       // CHANGE MEANING OF _ll and _LL
	    _ax = _mm_add_ps(_1,_mm_div_ps(_co,*_zz));         // 2*sin(d-p)^2 (solution for cos[2d-2p] ~ -1)
	    _ax = _mm_sqrt_ps(_mm_mul_ps(_ax,_05));            // sin(d-p)
	    _AX = _sse_dot4_ps(_pfx,_BB4);                     // e*(qq+QQ)*|fx|^2
	   _qQ2 = _mm_add_ps(_qQ2,_QQ);                        // store amplitude term
	   _eqQ = _mm_add_ps(_eqQ,_AX);                        // store ellipticity term
	    _AX = _mm_div_ps(_mm_mul_ps(_2,_AX),_QQ);          // e
	    _GG = _mm_sqrt_ps(_sse_rotp_ps(_ax,_ax,_AX,_AX));  // sqrt(e*e + sin(d-p)^2) - gamma regulator
	    _gg = _mm_mul_ps(_mm_sub_ps(_05,_ni),
			     _mm_sub_ps(_1,_FF));              // network index correction
	    _GG = _mm_sub_ps(_mm_sub_ps(_09,_GG),_gg);         // 0.9-sqrt(e*e + sin(d-p)^2) - (0.5-ni)*(1-a) 
	    _GG = _mm_mul_ps(_mm_mul_ps(_IE,_rG),_GG);         // gamma regulator value
	    _GG = _mm_mul_ps(_mm_mul_ps(_IE,_4),_GG);          // enhanced gamma regulator value

	    _gg = _mm_mul_ps(_mm_sub_ps(_IE,_ff),_kG);         // AP threshold       
	    _gg = _mm_mul_ps(_ni,_gg);                         // network index correction
	    _nn = _mm_and_ps(_mm_cmpgt_ps(_ff,_gg),_1);        // AP regulator --> kill flag
	    _ee = _mm_mul_ps(_sse_dot4_ps(_pep,_bb4),_nn);     // regulated + amplitude:  DO NOT UPDATE _ee !!!
	    _sse_cpf4_ps(_pxi,_pep,_ee);                       // store + projection

	    _mm = _mm_and_ps(_mm_cmpgt_ps(_ff,_GG),_nn);       // gamma regulator condition   

	    _gg = _mm_andnot_ps(_sm,_mm_sub_ps(_NI,_ni));      // |NI-ni|
	    _gg = _mm_sub_ps(_IE,_mm_mul_ps(_gg,_FF));         // IE-|NI-ni| * |fx|/|f+|
	    _nn = _mm_and_ps(_mm_cmplt_ps(_gg,_D0),_mm);       // 00-phase regulator flag
	    _mm = _mm_and_ps(_mm_cmplt_ps(_gg,_D9),_mm);       // 90-phase regulator flag
	    _nn = _mm_mul_ps(_nn,cirw);                        // zero nn if circular wave
	    _mm = _mm_mul_ps(_mm,linw);                        // zero mm if linear wave
	    _EE = _mm_mul_ps(_en,_nn);                         // 00 x-projection regulator: DO NOT UPDATE _EE !!!
	    _sse_add4_ps(_pxi,_pex,_EE);                       // updated 00-phase response
	    _sse_cpf4_ps(_pXI,_BB4,_mm);                       // updated 90-phase response    

	    _nn = _mm_mul_ps(_nn,_PW);                         // kill 1 dof for psiwave
	   *_rr = _mm_add_ps(_mm_add_ps(_mm,_nn),_1);          // store G-noise bias
	    _coP++;_siP++;_chr++;_xx++;_yy++;_zz++,_rr++;      // advance pointers
	    
	    _cc = _mm_add_ps(_cc,_ee);                         // + a*a* cos(t)
	    _ss = _mm_add_ps(_ss,_EE);                         // + a*a* sin(t)	      

	    _ll = _sse_rotm_ps(*_coD,*_coD,*_siD,*_siD);       // cos(2d)
	    _LL = _sse_rotp_ps(*_siD,*_coD,*_siD,*_coD);       // sin(2d)	    
	    _ec = _sse_rotm_ps(_co,_ll,_si,_LL);               // C[-2p] term (si -> -si)
	    _EC = _sse_rotp_ps(_si,_ll,_co,_LL);               // S[-2p] term (si -> -si)
	    _c2 = _mm_add_ps(_c2,_ec);                         // accumulate C[-2p]
	    _s2 = _mm_sub_ps(_s2,_EC);                         // accumulate S[-2p] (si -> -si)
	    _coD++;_siD++;                                     // advance pointers
	 }
	 
	 _mm_storeu_ps(vvv,_c2);
	 c2p = vvv[0]+vvv[1]+vvv[2]+vvv[3];                    // cos[-2p]
	 _mm_storeu_ps(vvv,_s2);
	 s2p = vvv[0]+vvv[1]+vvv[2]+vvv[3];                    // sin[-2p]
	 gg = sqrt(c2p*c2p+s2p*s2p+1.e-16);
	 _si  = _mm_set1_ps(s2p/gg);                           // sin[-2p]
	 _co  = _mm_set1_ps(c2p/gg);                           // cos[-2p]
	 
	 if(psiwave) {                                         // reconstruct p-wave
	    _zz  = (__m128*) zzz.data;	                       // store z-amplitude
	    _siD = (__m128*) siDPF.data;                       // DPF sin
	    _coD = (__m128*) coDPF.data;	               // DPF cos
	 
	    _mm_storeu_ps(vvv,_cc);
	     cc = (vvv[0]+vvv[1]+vvv[2]+vvv[3]);               // cos[t]
	    _mm_storeu_ps(vvv,_ss);
	     ss = (vvv[0]+vvv[1]+vvv[2]+vvv[3]);               // sin[t]
	     gg = sqrt(cc*cc+ss*ss+1.e-16);
	    _si = _mm_set1_ps(ss/gg);                          // sin[-2p]
	    _co = _mm_set1_ps(cc/gg);                          // cos[-2p]
	 
	    for(j=0; j<m4; j+=4) {                             // fix chirality and average vector
	       int jf = j*f_;                                  // sse index increment	    
	       __m128* _pxi = _pp+jf;
	       __m128* _pXI = _PP+jf;		  
	    	    
	       _sse_rot4p_ps(_ep+jf,&_co,_ex+jf,&_si,_bb4);    // rotate e+,ex by theta
	       _sse_cpf4_ps(_pxi,_bb4,_sse_dot4_ps(_pxi,_bb4));// 0 residual: do not enforce sign
	   
/*  dispersion correction code
	       _cc = _sse_rotm_ps(*_coD,*_coD,*_siD,*_siD);       // cos(2d) 
	       _ss = _sse_rotp_ps(*_siD,*_coD,*_siD,*_coD);       // sin(2d)
	       _ss = _sse_rotp_ps(_si,_cc,_ss,_co);               // S[2d-2p] term
	       _sse_cpf4_ps(_pxi,_pep,_sse_dot4_ps(_pep,_pxi));   // store + projection
	       _sse_sub4_ps(_pxi,_pex,_mm_mul_ps(_ss,*_zz));      // subtract 0-phase x response
	       _coD++;_siD++;_zz++;                               // advance pointers
*/
	    }
	 }

	 // orthogonalize responces before calculate sky statistics 

	 _siO = (__m128*) siORT.data;                          // ort4 sin
	 _coO = (__m128*) coORT.data;	                       // ort4 cos
	 _siP = (__m128*) siPHS.data;                          // phase sin
	 _coP = (__m128*) coPHS.data;	                       // phase cos
	 _chr = (__m128*) chir.data;	                       // chirality (sign of ellipticity)

	 _mm_storeu_ps(vvv,_eqQ);
	 _mm_storeu_ps(uuu,_qQ2);
	 eLp = 2.*(vvv[0]+vvv[1]+vvv[2]+vvv[3]);               // average ellipticity
	 eLp/= uuu[0]+uuu[1]+uuu[2]+uuu[3]+1.e-16;             // average ellipticity
	 _mm_storeu_ps(vvv,_ch);
	 CHR = vvv[0]+vvv[1]+vvv[2]+vvv[3];                    // average chirality
	  ff = CHR>0. ? 1. : -1.;                              // chirality
	 _ch = _mm_set1_ps(ff);
	 _gg = rndwave ? _oo : _mm_set1_ps(0.5);               // check or not chirality
	 	 
	 for(j=0; j<m4; j+=4) {                                // Orthogonalize
	    int jf = j*f_;                                     // sse index increment	    
	    __m128* _pbb = _bb+jf;
	    __m128* _pBB = _BB+jf;
	    __m128* _pxp = _xp+jf;
	    __m128* _pXP = _XP+jf;
	    __m128* _pxi = _pp+jf;
	    __m128* _pXI = _PP+jf;
	    
	    _ee = _mm_sub_ps(_mm_mul_ps(_ch,*_chr),_1);        // -1 or 0
	    _sse_add4_ps(_pXI,_pXI,_mm_mul_ps(_ee,_gg));       // set chirality 
	    
	    _co = _sse_rotm_ps(*_coO,*_coP,*_siO,*_siP);       // cos(ort4+phase)
	    _si = _sse_rotp_ps(*_siO,*_coP,*_siP,*_coO);       // sin(ort4+phase)
	    _sse_rot4m_ps(_pxi,&_co,_pXI,&_si,_pxp);           // get 00 phase response  
	    _sse_rot4p_ps(_pXI,&_co,_pxi,&_si,_pXP);           // get 90 phase response 	       
	    _sse_ort4_ps(_pxp,_pXP,_siO,_coO);                 // dual-stream phase sin and cos
	    _sse_rot4p_ps(_pxp,_coO,_pXP,_siO,_pxi);           // get 00 standard response  
	    _sse_rot4m_ps(_pXP,_coO,_pxp,_siO,_pXI);           // get 90 standard response 
	    _sse_rot4p_ps(_pbb,_coO,_pBB,_siO,_ww+jf);         // get 00 phase data vector 
	    _sse_rot4m_ps(_pBB,_coO,_pbb,_siO,_WW+jf);         // get 90 phase data vector
	    _coO++; _siO++; _coP++; _siP++; _chr++;            // increment to next 4 pixels
	 }
	 
// calculation of likelihood statistics

	 Nm = Lo;

	 _rr = (__m128*) rrr.data;	                     // regulator flag

         Lo = Co = Eo = Lr = Cr = Do = 0.;
         for(j=0; j<m4; j+=4) {
            int jf = j*f_;                                   // sse index increment

            __m128* _pbb = _bb+jf;
            __m128* _pBB = _BB+jf;
            __m128* _pxi = _pp+jf;
            __m128* _pXI = _PP+jf;
            __m128* _pww = _ww+jf;
            __m128* _pWW = _WW+jf;

            _ee = _sse_abs4_ps(_pbb);                        // 00 total energy
            _ll = _mm_add_ps(_sse_abs4_ps(_pxi),_oo);        // standard 00 likelihood
            _ed = _sse_ed4_ps(_pww,_pxi,_ll);                // 00 energy disbalance
            _ec = _sse_ecoh4_ps(_pww,_pxi,_ll);              // coherent energy

            _EE = _sse_abs4_ps(_pBB);                        // 90 total energy
            _LL = _mm_add_ps(_sse_abs4_ps(_pXI),_oo);        // standard 90 likelihood
            _ED = _sse_ed4_ps(_pWW,_pXI,_LL);                // 90 energy disbalance
            _EC = _sse_ecoh4_ps(_pWW,_pXI,_LL);              // coherent energy

	    _ll = _mm_add_ps(_ll,_LL);                       // total signal energy
	    _ec = _mm_add_ps(_ec,_EC);                       // total coherent energy
	    _ed = _mm_add_ps(_ed,_ED);                       // total energy disbalance
	    _ee = _mm_add_ps(_ee,_EE);                       // total energy

            _mm_storeu_ps(vvv,_ee);
            Eo += vvv[0]+vvv[1]+vvv[2]+vvv[3];               // total network energy
            _mm_storeu_ps(vvv,_ec);
            Co += vvv[0]+vvv[1]+vvv[2]+vvv[3];               // network coherent energy
            _mm_storeu_ps(vvv,_ed);
            Do += vvv[0]+vvv[1]+vvv[2]+vvv[3];               // network energy disbalance
	    _mm_storeu_ps(vvv,_ll);
            Lo += vvv[0]+vvv[1]+vvv[2]+vvv[3];               // network likelihood

// calculate sky statistics

            _en = _mm_andnot_ps(_sm,_mm_sub_ps(_ee,_ll));    // pixel null energy
            _en = _mm_add_ps(_en,*_rr); _rr++;               // Gaussian bias added	    
            _EC = _mm_andnot_ps(_sm,_ec);                    // | coherent energy |
            _cc = _mm_add_ps(_mm_add_ps(_EC,_ed),_en);       // |C|+null+ed
            _cc = _mm_div_ps(_mm_sub_ps(_ec,_ed),_cc);       // network correlation

            _mm_storeu_ps(vvv,_mm_mul_ps(_ll,_cc));
            Lr += vvv[0]+vvv[1]+vvv[2]+vvv[3];               // reduced likelihood
            _mm_storeu_ps(vvv,_mm_mul_ps(_ec,_cc));
            Cr += vvv[0]+vvv[1]+vvv[2]+vvv[3];               // reduced coherent energy
	   
            // _mm_storeu_ps(vvv,_gg);
            // _mm_storeu_ps(uuu,_ff);
            // if(hist && (le-lb) && vvv[0]!=0) hist->Fill(vvv[0], uuu[0]);
            // if(hist && (le-lb) && vvv[1]!=0) hist->Fill(vvv[1], uuu[1]);
            // if(hist && (le-lb) && vvv[2]!=0) hist->Fill(vvv[2], uuu[2]);
            // if(hist && (le-lb) && vvv[3]!=0) hist->Fill(vvv[3], uuu[3]);
         }

         aa = Eo>0. ? Lo/Eo : 0.;                                 // detection skystat
         AA = Eo>0. ? Lr/Eo : 0.;                                 // detection skystat
         if(!mra) skyProb.data[l] = this->delta<0 ? aa : AA;
 
         if(ID==id || mra) {                                      // antenna patterns              
            float ll,LL,Et;
            ff = FF = Et = Nm = 0.;
            for(j=0; j<m; j++) { 
               int jf = j*f_;                                     // sse pointer increment          
               ee = _sse_abs_ps(_bb+jf,_BB+jf);                   // total energy
               if(ee<En) continue;                                // skip sub-threshold PC 
               ff += _sse_abs_ps(_fp+jf)*ee;                      // |f+|^2
               FF += _sse_abs_ps(_fx+jf)*ee;                      // |fx|^2 
               ll = _sse_mul_ps(_pp+jf,_pp+jf,_bb4)+1.e-12;       // 00 likelihood 
               LL = _sse_mul_ps(_PP+jf,_PP+jf,_BB4)+1.e-12;       // 90 likelihood 
               Nm+= _sse_abs_ps(_bb4)/ll+_sse_abs_ps(_BB4)/LL;    // network index*L 
               Et+= ll+LL;
            }
            Nm = Et>0.&&Nm>0 ? Et/Nm : 0.;
            ff = Eo>0. ? 2*ff/Eo  : 0.;
            FF = Eo>0. ? 2*FF/Eo  : 0.;
         }

         if(ID==id && !mra) {                                     // fill skymaps
	    Eo += 0.001; Cr += 0.001;
            this->nAntenaPrior.set(l, sqrt(ff+FF));		  // fill sensitivity
            this->nSensitivity.set(l, sqrt(ff+FF));
            this->nAlignment.set(l, sqrt(FF/ff));  
            this->nLikelihood.set(l, Lo/Eo);              
            this->nNullEnergy.set(l, (Eo-Lo)/Eo);                
            this->nCorrEnergy.set(l, Cr/Eo);                      
            this->nCorrelation.set(l, Ln);                       
            this->nSkyStat.set(l, AA);                         
            this->nProbability.set(l, skyProb.data[l]);          
            this->nDisbalance.set(l, 2*Do/Eo);               
            this->nEllipticity.set(l, eLp);                                       
            this->nPolarisation.set(l, -atan2(s2p,c2p)*180./PI/4.);                            
            this->nNetIndex.set(l, Nm);                                        
         }

         if(prior && !mra && ID!=id) {                            // used in getSkyArea with prior
            ff = FF = 0.;
            for(j=0; j<m; j++) { 
               int jf = j*f_;                                     // sse pointer increment          
               ee = _sse_abs_ps(_bb+jf,_BB+jf);                   // total energy
               if(ee<En) continue;                                // skip sub-threshold PC 
               ff += _sse_abs_ps(_fp+jf)*ee;                      // |f+|^2
               FF += _sse_abs_ps(_fx+jf)*ee;                      // |fx|^2 
            }
            ff = Eo>0. ? 2*ff/Eo  : 0.;
            FF = Eo>0. ? 2*FF/Eo  : 0.;
            this->nAntenaPrior.set(l, sqrt(ff+FF));		  // fill sensitivity
         }

         if(AA>STAT && !mra) {STAT=AA; lm=l; Vm=m;} 
      }
      
      if(STAT==0. || (mra && AA<=0.)) {
         pwc->sCuts[id-1]=1; count=0;                            // reject cluster 
         pwc->clean(id); continue;                                         
      }    

      if(le-lb) {lb=le=lm; goto optsky;}                         // process all pixels at opt sky location

      double Em_all,Ln_all,Ls_all,Lr_all;
      double Eo_all,Lo_all,Co_all,Do_all,cc_all;
      float GNoise = 0.;

      if(!mra) {                                       // all resolution pixels
	 Em_all=Ls_all=Ln_all = 0;
	 Eo_all = Eo;                                  // multiresolution energy
	 Lo_all = Lo;                                  // multiresolution likelihood
	 Lr_all = Lr;                                  // reduced likelihood
	 Co_all = Co;                                  // multiresolution coherent energy
	 Do_all = Do*2;                                // multiresolution ED
         pwc->cData[id-1].skySize = m;                 // event size in the skyloop
	 pwc->cData[id-1].likesky = Lo;                // multires Likelihood - stored in neted[3]
         vint = &(pwc->cList[id-1]);                   // pixel list
         for(j=0; j<vint->size(); j++) {               // initialization for all pixels
            pix = pwc->getPixel(id,j);
            pix->core = false;                         
            pix->likelihood = 0.; 
         }
         
         for(j=0; j<Vm; j++) {                         // loop over significant pixels
            pix = pwc->getPixel(id,pI[pJ[j]]);
            int jf = j*f_;                             // source sse pointer increment 
            ee = _sse_abs_ps(_bb+jf,_BB+jf);           
            pix->likelihood = ee/2;                    // total pixel energy
	    em = _sse_maxE_ps(_bb+jf,_BB+jf);          // dominant pixel likelihood 
	    Ls_all += ee-em;                           // subnetwork Energy
	    Em_all += em;                              // maximum detector energy
	    if(ee-em>Es) Ln_all += ee;                 // reduced network energy
	    GNoise += rrr.data[j];                     // counter for G-noise bias
         }

	 pwc->cData[id-1].skyStat = Lr/Eo;             // all-resolution sky statistic (saved in norm) 
	 cc_all = Lr/Lo;                               // multiresolution cc

	 Ns = Eo_all-Lo_all+Do+GNoise;                 // NULL stream with G-noise correction
	 gg = Ls_all*Ln_all/Em_all;                    // L: all-sky subnet "energy"
	 pwc->cData[id-1].SUBNET = gg/(fabs(gg)+Ns);   // like2G sub-network statistic (saved in netcc[3])

         mra=true; goto optsky;                        // process mra pixels at opt sky location
      }

      if(AA<this->netCC || !m) {
         pwc->sCuts[id-1]=1; count=0;                  // reject cluster 
         pwc->clean(id); continue;                                         
      }    

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// detection statistics at selected sky location
// wavelet domain: netcc, ecor
// time domain: energy, likelihood, xSNR, sSNR, neted
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      double Em_mra,Ln_mra,Ls_mra;
      double Eo_mra,Lo_mra,Co_mra;

      M=m; m=0; GNoise=0.;
      Em_mra=Ln_mra=Ls_mra = 0;
      for(j=0; j<M; j++) {                          // loop over principle components
         pix = pwc->getPixel(id,pI[pJ[j]]);
         int jf = j*f_;                             // source sse pointer increment 
         float* psi = siORT.data+j;
         float* pco = coORT.data+j;
         __m128* _pxi = _xi+jf;
         __m128* _pXI = _XI+jf;
         ee = _sse_abs_ps(_bb+jf,_BB+jf);           // total pixel energy
	 em = _sse_maxE_ps(_bb+jf,_BB+jf);          // dominant pixel energy 
	 Em_mra += em;                              // maximum detector energy
	 Ls_mra += ee-em;                           // subnetwork energy
	 if(ee-em>Es) Ln_mra += ee;                 // reduced network energy
	 GNoise += rrr.data[j];                     // counter for G-noise bias
         if(em>0) m++;                              // counter for subnet pixels
         pix->core = true;

         _sse_rotm_ps(_pxi,pco,_pXI, psi,_bb4);     // invDSP 00 response 
         _sse_rotp_ps(_pXI,pco,_pxi, psi,_BB4);     // invDSP 90 response  

         for(i=0; i<nIFO; i++) {              		            
            pix->setdata(double(bb.data[j*NIFO+i]),'W',i);    // store 00 whitened PC
            pix->setdata(double(BB.data[j*NIFO+i]),'U',i);    // store 90 whitened PC
            pix->setdata(double(bb.data[V4*NIFO+i]),'S',i);   // 00 reconstructed whitened response
            pix->setdata(double(BB.data[V4*NIFO+i]),'P',i);   // 90 reconstructed whitened response
         }
      }
      
      if(!m) {                                         // zero reconstructed response 
         pwc->sCuts[id-1]=1; count=0;                  // reject cluster 
         pwc->clean(id); continue;                                         
      }       

      Em=Eo; Lm=Lo; Do*=2;                             // copy all-pixel statistics
      Eo_mra=Eo; Lo_mra=Lo; Co_mra=Co;
 
      pwc->cData[id-1].netcc = Lr/Eo;                  // network cc with mres correction (saved in netcc[0])
      Nc = Eo-Lo+Do/2+GNoise;                          // NULL stream with correction
      gg = Ls_mra*Ln_mra/Em_mra;                       // L: MRA subnet "energy"
      pwc->cData[id-1].subnet = gg/(fabs(gg)+Nc);      // mra/sra sub-energy statistic (saved in netcc[2])
      pwc->cData[id-1].skycc = Co/(fabs(Co)+Nc);       // classic SRA/MRA cc (saved in netcc[1])

      //if(hist) hist->Fill(pwc->cData[id-1].skycc,pwc->cData[id-1].netcc);
 
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// fill in detection statistics, prepare output data
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// fill in backward delay configuration
       
      vtof->clear();
      NETX (vtof->push_back(ml[0][lm]); ,
            vtof->push_back(ml[1][lm]); ,
            vtof->push_back(ml[2][lm]); ,
            vtof->push_back(ml[3][lm]); ,
            vtof->push_back(ml[4][lm]); ,
            vtof->push_back(ml[5][lm]); ,
            vtof->push_back(ml[6][lm]); ,
            vtof->push_back(ml[7][lm]); )

      // need to fix a problem below
      if((wfsave)||(mdcListSize() && !lag)) {     // if wfsave=false only simulated wf are saved
        int m0d = mureana ? 0 : 1;
        if(this->getMRAwave(id,lag,'S',m0d,true)) {  // reconstruct whitened shifted pd->waveForm 
          detector* pd;
          for(i=0; i<nIFO; i++) {                 // loop over detectors
            pd = this->getifo(i);
            pd->RWFID.push_back(id);              // save cluster ID
            WSeries<double>* wf = new WSeries<double>;
            *wf = pd->waveForm;
            wf->start(pwc->start+pd->waveForm.start());
            pd->RWFP.push_back(wf);
          }
        }
        if(this->getMRAwave(id,lag,'s',m0d,true)) {   // reconstruct strain shifted pd->waveForm
          detector* pd;
          for(i=0; i<nIFO; i++) {                 // loop over detectors
            pd = this->getifo(i);
            pd->RWFID.push_back(-id);             // save cluster -ID
            WSeries<double>* wf = new WSeries<double>;
            *wf = pd->waveForm;
            wf->start(pwc->start+pd->waveForm.start());
            pd->RWFP.push_back(wf);
          }
        }
      }
      
      Lo = Eo = To = Fo = No = 0.;
      for(i=0; i<nIFO; i++) {              		            
         detector* d = this->getifo(i);
         d->sSNR = d->xSNR = d->null = d->enrg = 0.;
      }

      int two = mureana ? 1 :  2;
      int m0d = mureana ? 0 : -1;
      while(m0d < 2) {
         this->getMRAwave(id,lag,'W',m0d);
         this->getMRAwave(id,lag,'S',m0d);
         for(i=0; i<nIFO; i++) {              		            
            detector* d = this->getifo(i);
            d->waveNull = d->waveBand;
            d->waveNull-= d->waveForm; 
            float sSNR = d->get_SS()/two;
            float xSNR = d->get_XS()/two;
            float null = d->get_NN()/two;
            float enrg = d->get_XX()/two;
            d->sSNR += sSNR;
            d->xSNR += xSNR;
            d->null += null;
            d->enrg += enrg;
            To += sSNR*d->getWFtime();
            Fo += sSNR*d->getWFfreq();
            Lo += sSNR;
            Eo += enrg;
            No += null;
         }
	 m0d += 2;
      }
      To /= Lo; Fo /= Lo;

      gg = Lo/Lo_mra;
      Co = Co*gg;
      Cr = Cr*gg;
      Do = Do*gg;
      Nc = Nc*gg;

      pwc->cData[id-1].likenet = Lo;
      pwc->cData[id-1].energy  = Eo;                         // energy of the event - stored in neted[2]
      pwc->cData[id-1].enrgsky = Eo_all;                     // energy in the skyloop - stored in neted[4]
      pwc->cData[id-1].netecor = Co;
      pwc->cData[id-1].netnull = No+GNoise/2;                // NULL with Gauss correction - stored in neted[1]
      pwc->cData[id-1].netED   = Do;                         // network energy disbalance - stored in neted[0]
      pwc->cData[id-1].netRHO  = sqrt(Co*cc_all/(nIFO-1.));  // signal rho  - stored in rho[0]
      pwc->cData[id-1].netrho  = sqrt(Cr/(nIFO-1.));         // reguced rho - stored in rho[1]
      pwc->cData[id-1].cTime   = To;
      pwc->cData[id-1].cFreq   = Fo;
      pwc->cData[id-1].theta   = nLikelihood.getTheta(lm);
      pwc->cData[id-1].phi     = nLikelihood.getPhi(lm);
      pwc->cData[id-1].gNET    = sqrt(ff+FF);
      pwc->cData[id-1].aNET    = sqrt(FF/ff);
      pwc->cData[id-1].iNET    = Nm;
      pwc->cData[id-1].iota    = Ns;
      pwc->cData[id-1].psi     = -atan2(s2p,c2p)*180./PI/4.;
      pwc->cData[id-1].ellipticity = eLp;

      if(this->optim) pwc->cRate[id-1][0] = optR;            // update optimal resolution

      if(sqrt(Co/(nIFO-1.))<this->netRHO || pwc->cData[id-1].skycc<this->netCC) {
         pwc->sCuts[id-1]=1; count=0;     // reject cluster 
         pwc->clean(id); continue;                                         
      }                                                   

      cc = pwc->cData[id-1].skycc;
      if(hist) {
	 printf("id|lm %3d|%6d rho=%4.2f cc: %5.3f|%5.3f|%5.3f|%5.3f \n",
		int(id),int(lm),sqrt(Co/(nIFO-1)),STAT,cc,pwc->cData[id-1].netcc,AA);
	 printf(" (t,p)=(%4.1f|%4.1f)  T|F: %6.3f|%4.1f L: %5.1f|%5.1f|%5.1f E: %5.1f|%5.1f|%5.1f \n",
		nLikelihood.getTheta(l),nLikelihood.getPhi(l),To,Fo,Lo,Lo_mra,Lo_all,Eo,Em,Eo_all);
	 printf(" D|N: %4.1f|%4.1f|%4.1f Vm|m=%3d|%3d subnet=%4.3f|%4.3f \n",
		Do,No,Nc,int(Vm),int(M),pwc->cData[id-1].subnet,pwc->cData[id-1].SUBNET);
	 hist->Fill(pwc->cData[id-1].subnet,pwc->cData[id-1].SUBNET);
     }
      count++;                                                        

// calculation of error regions

      pwc->p_Ind[id-1].push_back(m);
      double T = To+pwc->start;                            // trigger time
      std::vector<float> sArea;
      pwc->sArea.push_back(sArea);
      pwc->p_Map.push_back(sArea);

      skyProb *= Lo;
      //double rMs = Ns/(nIFO*Vm);
      double rMs = this->delta<0 ? 0 : 2;                                    
      if(iID<=0 || ID==id) getSkyArea(id,lag,T,rMs);       // calculate error regions

// calculation of chirp mass

      pwc->cData[id-1].mchirp = 0;
      pwc->cData[id-1].mchirperr = 0;
      pwc->cData[id-1].tmrgr = 0;
      pwc->cData[id-1].tmrgrerr = 0;
      pwc->cData[id-1].chi2chirp = 0;

      if(m_chirp) {                                        // work only for MRA 
         ee = pwc->mchirp(id);
         cc = Co_all/(fabs(Co_all)+ee);                    // chirp cc 
         printf("mchirp : %d %g %.2e %.3f %.3f %.3f %.3f \n\n",
                int(id),cc,pwc->cData[id-1].mchirp,
       	        pwc->cData[id-1].mchirperr, pwc->cData[id-1].tmrgr,
	        pwc->cData[id-1].tmrgrerr, pwc->cData[id-1].chi2chirp);
      }

      if(ID==id && !EFEC) {   
	 this->nSensitivity.gps = T;
	 this->nAlignment.gps   = T;
	 this->nDisbalance.gps  = T;
	 this->nLikelihood.gps  = T;
	 this->nNullEnergy.gps  = T;
	 this->nCorrEnergy.gps  = T;
	 this->nCorrelation.gps = T;
	 this->nSkyStat.gps     = T;
	 this->nEllipticity.gps = T;
	 this->nPolarisation.gps= T;
	 this->nNetIndex.gps    = T;
      }
      
      pwc->sCuts[id-1] = -1;
      pwc->clean(id);
   } // end of loop over clusters
   
   return count;
}

//: operator =

network& network::operator=(const network& value)
{
   this->wfsave  = value.wfsave;
   this->MRA     = value.MRA;
   this->nRun    = value.nRun;
   this->nLag    = value.nLag;
   this->nSky    = value.nSky;
   this->mIFO    = value.mIFO;
   this->Step    = value.Step;
   this->Edge    = value.Edge;
   this->gNET    = value.gNET;
   this->aNET    = value.aNET;
   this->iNET    = value.iNET;
   this->eCOR    = value.eCOR;
   this->e2or    = value.e2or;
   this->acor    = value.acor;
   this->norm    = value.norm;
   this->pOUT    = false;
   this->local   = value.local;
   this->EFEC    = value.EFEC;
   this->optim   = value.optim;
   this->delta   = value.delta;
   this->gamma   = value.gamma;
   this->penalty = value.penalty;
   this->netCC   = value.netCC;
   this->netRHO  = value.netRHO;
   this->pSigma   = value.pSigma;
   this->ifoList = value.ifoList;
   this->precision=value.precision;

   this->NDM.clear();     this->NDM=value.NDM;
   this->ifoList.clear(); this->ifoList=value.ifoList;
   this->ifoName.clear(); this->ifoName=value.ifoName;
   this->wc_List.clear(); this->wc_List=value.wc_List;
   this->segList.clear(); this->segList=value.segList;
   this->mdcList.clear(); this->mdcList=value.mdcList;
   this->livTime.clear(); this->livTime=value.livTime;
   this->mdcTime.clear(); this->mdcTime=value.mdcTime;
   this->mdcType.clear(); this->mdcType=value.mdcType;
   this->mdc__ID.clear(); this->mdc__ID=value.mdc__ID;

   return *this;
}


//**************************************************************************
//: add detector to the network  
//**************************************************************************
size_t network::add(detector* d) {

   if(ifoList.size()==NIFO) {
     cout << "network::add - Error : max number of detectors is " << NIFO << endl;
     exit(1);
   }

   size_t i,n;
   vectorD v; v.clear();
   this->ifoList.push_back(d); 
   this->ifoName.push_back(d->Name);

   n = ifoList.size();
   d->ifoID = n-1;
   for(i=0; i<n; i++) {
      v.push_back(0.);
      if(i<n-1) this->NDM[i].push_back(0);
      else      this->NDM.push_back(v);
   }

//   cout<<"size="<<NDM.size()<<" size_0="<<NDM[0].size()<<endl;
   return ifoList.size();
}

//**************************************************************************
// calculate WaveBurst pattern threshold for a given black pixel probability
//**************************************************************************
double network::THRESHOLD(double p, double shape) {
// calculate WaveBurst energy threshold for a given black pixel probability p
// and single detector Gamma distribution shape. TF data should contain pixel energy
   int N = ifoListSize();
   WSeries<double>* pw = &(getifo(0)->TFmap);
   size_t M  = pw->maxLayer()+1;
   size_t nL = size_t(Edge*pw->wrate()*M);
   size_t nR = pw->size() - nL - 1;
   wavearray<double> w = *pw;
   for(int i=1; i<N; i++) w += getifo(i)->TFmap;
   double amp, avr, bbb, alp;   
   avr = bbb = 0.; 
   int nn = 0;
   for(int i=nL; i<nR; i++) {                              // get Gamma shape & mean
      amp = (double)w.data[i];
      if(amp>N*100) amp = N*100.;
      if(amp>0.001) {avr+=amp; bbb+=log(amp); nn++;}
   }
   avr = avr/nn;                                           // Gamma mean
   alp = log(avr)-bbb/nn;
   alp = (3-alp+sqrt((alp-3)*(alp-3)+24*alp))/12./alp;     // Gamma shape
   bbb = p*alp/shape;                                      // corrected bpp
   //cout<<bbb<<" "<<avr<<" "<<alp<<" "<<shape<<" "<<iGamma(alp,bbb)<<endl;
   return avr*iGamma(alp,bbb)/alp/2;
}

//**************************************************************************
// calculate WaveBurst energy threshold for a given black pixel probability
//**************************************************************************
double network::THRESHOLD(double p) {
// calculate WaveBurst energy threshold for a given black pixel probability p
// TF data should contain pixel energy
   int N = ifoListSize();
   WSeries<double>* pw = &(getifo(0)->TFmap);
   size_t M  = pw->maxLayer()+1;
   size_t nL = size_t(Edge*pw->wrate()*M);
   size_t nR = pw->size() - nL;
   wavearray<double> w = *pw;
   for(int i=1; i<N; i++) w += getifo(i)->TFmap;
   double p10 = p*10.;
   double p00 = 0.0;
   double fff = w.wavecount(0.0001)/double(w.size());
   double v10 = w.waveSplit(nL,nR,nR-int(p10*fff*(nR-nL)));
   double val = w.waveSplit(nL,nR,nR-int(p*fff*(nR-nL)));
   double med = w.waveSplit(nL,nR,nR-int(0.2*fff*(nR-nL)));
   double m   = 1.;
   while(p00<0.2) {p00 = 1-Gamma(N*m,med); m+=0.01;}
   if(m>1) m -= 0.01;
   printf("\nm\tM\tbpp\t0.2(D)\t0.2(G)\t0.01(D)\t0.01(G)\tbpp(D)\tbpp(G)\tN*log(m)\tfff\n");
   printf("%g\t%d\t%g\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t\t%.3f\n\n",
           m,M,p,med,iGamma(N*m,0.2),v10,iGamma(N*m,p10),val,iGamma(N*m,p),N*log(m),fff);
   return (iGamma(N*m,p)+val)*0.3+N*log(m);
}


//**************************************************************************
// calculate WaveBurst threshold as a function of resolution and maximum delay (used by 1G)
//**************************************************************************
double network::threshold(double p, double t)
{
  size_t I;
  size_t K = ifoListSize();
  double n = 1.;
  if(getifo(0) && t>0. && K) {
    I = getifo(0)->TFmap.maxLayer()+1;  // number of wavelet layers
    n = 2*t*getifo(0)->TFmap.rate()/I + 1;
    if(!getifo(0)->TFmap.size()) n = 1.;
  }
  else if(t < 0.) n = -t;               // special case: n=-t
  return sqrt(iGamma1G(K/2.,1.-(p/2.)/pow(n,K/2.))/K);
}

void network::printwc(size_t n) { 
   netcluster* p = this->getwc(n);
   int iTYPE = this->MRA ? 0 : 1; 
   wavearray<double> cid = p->get((char*)"ID",0,'S',iTYPE);
   wavearray<double> vol = p->get((char*)"volume",0,'S',iTYPE);
   wavearray<double> siz = p->get((char*)"size",0,'S',iTYPE);
   wavearray<double> lik = p->get((char*)"likelihood",0,'S',iTYPE);
   wavearray<double> rat = p->get((char*)"rate",0,'S',iTYPE);
   wavearray<double> tim = p->get((char*)"time",1,'L',0);
   wavearray<double> T_o = p->get((char*)"time",0,'L',0);
   wavearray<double> frq = p->get((char*)"frequency",1,'L',0);
   wavearray<double> rho = p->get((char*)"subrho",0,'S',0);
   wavearray<double> sub = p->get((char*)"subnet",0,'S',0);

   if(cid.size() != rho.size()) 
      cout<<"wrong size "<<cid.size()<<" "<<rho.size()<<endl;

   for(size_t i=0; i<cid.size(); i++){       
      printf("%2d %5.0f vol=%4.0f size=%4.0f like=%5.1e rho=%5.1f ", 
	     int(n),cid[i],vol[i],siz[i],lik[i],rho[i]);
      printf("sub=%3.2f rate=%4.0f time=%8.3f To=%8.3f freq=%5.0f\n",
	     sub[i],rat[i],tim[i],T_o[i],frq[i]); 
   }
}


//**************************************************************************
//: initialize network sky maps 
//**************************************************************************
void network::setSkyMaps(double sms,double t1,double t2,double p1,double p2)
{
   size_t i;
   detector* d;
   skymap temp(sms,t1,t2,p1,p2);
   size_t m = temp.size();
   size_t n = this->ifoList.size();
    
   nSensitivity = temp;
   nAlignment   = temp;
   nCorrelation = temp;
   nLikelihood  = temp;
   nNullEnergy  = temp;
   nPenalty     = temp;
   nCorrEnergy  = temp;
   nNetIndex    = temp;
   nDisbalance  = temp;
   nSkyStat     = temp;
   nEllipticity = temp;
   nProbability = temp;
   nPolarisation= temp;
   nAntenaPrior = temp;

   for(i=0; i<n; i++) { 
      d = ifoList[i]; 
      d->setTau(sms,t1,t2,p1,p2);
      d->setFpFx(sms,t1,t2,p1,p2);
   }
   skyProb.resize(m);
   skyENRG.resize(m);
   skyMask.resize(m); skyMask = 1;
   skyMaskCC.resize(0); 
   skyHole.resize(m); skyHole = 1.;
   index.resize(m);
   for(i=0; i<m; i++) index.data[i] = i; 
}

//**************************************************************************
//: initialize network sky maps (healpix)
//**************************************************************************
void network::setSkyMaps(int healpix_order) 
{
   size_t i;
   detector* d;
   skymap temp(healpix_order);  
   size_t m = temp.size();
   size_t n = this->ifoList.size();

   nSensitivity = temp;
   nAlignment   = temp;
   nCorrelation = temp;
   nLikelihood  = temp;
   nNullEnergy  = temp;
   nPenalty     = temp;
   nCorrEnergy  = temp;
   nNetIndex    = temp;
   nDisbalance  = temp;
   nSkyStat     = temp;
   nEllipticity = temp;
   nProbability = temp;
   nPolarisation= temp;
   nAntenaPrior = temp;

   for(i=0; i<n; i++) {
      d = ifoList[i];
      d->setTau(healpix_order);  
      d->setFpFx(healpix_order);  
   }
   skyProb.resize(m);
   skyENRG.resize(m);
   skyMask.resize(m); skyMask = 1;
   skyMaskCC.resize(0); 
   skyHole.resize(m); skyHole = 1.;
   index.resize(m);
   for(i=0; i<m; i++) index.data[i] = i; 
}

//**************************************************************************
// calculate delays in frame or in barycenter (B) or fermat frame (F)
//**************************************************************************
void network::setDelay(const char* frame) {
  size_t n,nn,m,mm;
  skymap s = ifoList[0]->tau;
  size_t N = this->ifoList.size();
  double t,tm,gg;

  if(N < 2) return;

  s = 0.;

  if(strstr(frame,"FL") || strstr(frame,"FS")) {           // "Fermat" frame
    tm = strstr(frame,"FS") ? 1. :  0.;
    gg = strstr(frame,"FS") ? 1. : -1.;
    nn = 0;
    mm = 1;
    for(n=0; n<N; n++) { 
      for(m=n+1; m<N; m++) { 
	s  = ifoList[n]->tau; 
	s -= ifoList[m]->tau;
	t  = gg*(s.max()-s.min());
	if(t < tm) { tm=t; nn = n; mm = m; }
      }
    }

    s = ifoList[nn]->tau; 
    s+= ifoList[mm]->tau; 
    s*= 0.5;
    mIFO = 99;
  }

  else if(strstr(frame,"BC")) {                // barycenter frame
    for(n=1; n<N; n++) s += ifoList[n]->tau;    
    s *= 1./N;
    mIFO = 99;
  }

  else {                                       // detector frame
    for(n=0; n<N; n++) { 
      if(strstr(frame,getifo(n)->Name)) this->mIFO = n;
    }
    s = ifoList[this->mIFO]->tau;    
  }


  for(n=0; n<N; n++) ifoList[n]->tau -= s;    
  return;
}

//**************************************************************************
// calculate maximum delay between detectors
//**************************************************************************
double network::getDelay(const char* name) {
  size_t i;
  size_t n = this->ifoList.size();
  double maxTau = -1.;
  double minTau =  1.;
  double tmax, tmin;

  if(n < 2) return 0.;

  tmax = tmin = 0.;
  for(i=0; i<n; i++) { 
    tmax = ifoList[i]->tau.max();
    tmin = ifoList[i]->tau.min();
    if(tmax > maxTau) maxTau = tmax;
    if(tmin < minTau) minTau = tmin;
  }
  if(strstr(name,"min")) return minTau;
  if(strstr(name,"max")) return maxTau;
  if(strstr(name,"MAX")) return fabs(maxTau)>fabs(minTau) ? fabs(maxTau) : fabs(minTau);
  return (maxTau-minTau)/2.;
}




//***************************************************************
//:set antenna pattern buffers in input detector 
//***************************************************************
void network::setAntenna(detector* di)
{
  size_t n = di->mFp.size();
  double a, b;

  di->fp.resize(n);
  di->fx.resize(n);
  di->ffp.resize(n);
  di->ffm.resize(n);
  di->fpx.resize(n);

  while(n-- >0) {
    a = di->mFp.get(n);
    b = di->mFx.get(n);
    di->fp.data[n] = a;
    di->fx.data[n] = b;
    di->ffp.data[n] = a*a+b*b;
    di->ffm.data[n] = a*a-b*b;
    di->fpx.data[n] = 2*a*b;
  }

  return;
}

//***************************************************************
//:set antenna patterns in the DPF
//***************************************************************
void network::setAntenna()
{
  size_t M = this->ifoList.size(); // number of detectors
  if(M > NIFO) return;

  detector* D[NIFO];

  for(size_t m=0; m<M; m++) {
    D[m] = this->getifo(m);
    if(D[m]->mFp.size() != D[0]->mFp.size()) {
       cout<<"network::setIndex(): invalid detector skymaps\n";
       return;
    }
    this->setAntenna(D[m]);
  }
  return;
}

//***************************************************************
//:set index array for delayed amplitudes, used with WDM delay filters
// time delay convention: t+tau - arrival time at the center of Earth
// ta1-tau0 - how much det1 should be delayed to be sinchronized with det0
///***************************************************************
void network::setDelayIndex(double rate)
{
  double t;
  int i,ii;
  size_t n,m,l,k;
  size_t N = ifoList.size();           // number of detectors

  double tt[NIFO][NIFO];                  
  double TT[NIFO];                  
  int    mm[NIFO][NIFO];                  

  if(N<2) {
    cout<<"network::setDelayIndex(): invalid network\n";
    return;
  }

  detector* dr[NIFO];
  for(n=0; n<N; n++) dr[n] = ifoList[n];

  size_t L = dr[0]->tau.size();               // skymap size  
  this->rTDF = rate;                          // effective time-delay rate

  //  if(pOUT) cout<<"filter size="<<this->filter.size()
  //	       <<" layers="<<I<<" delays="<<K<<" samples="<<dr[0]->nDFS<<endl;

  for(n=0; n<N; n++) {
    if(dr[n]->index.size() != L) {
       dr[n]->index.resize(L);
    }
  }

// calculate time interval the di detector is delayed to be 
// sinchronized with dr
// time delay > 0 - shift di right (future) 
// time delay < 0 - shift di left  (past)

  this->nPenalty = dr[0]->tau;
  this->nNetIndex = dr[0]->tau;

  for(l=0; l<L; l++){

// calculate time delay matrix
//  0 d01 d02  
// d10  0 d12  
// d20 d21 0

    for(n=0; n<N; n++) {
      for(m=0; m<N; m++) {
	t = dr[n]->tau.get(l)-dr[m]->tau.get(l);
	i = t>0 ? int(t*rTDF+0.5) : int(t*rTDF-0.5);
	mm[n][m] = i;
	tt[n][m] = t*rTDF;
      }
    }

    for(n=0; n<N; n++) {
      TT[n] = 0.;                           // max delay for n-th configuration
      for(m=0; m<N; m++) {
	for(k=0; k<N; k++) {
	  t = fabs(mm[n][k]-mm[n][m]-tt[m][k]);
	  if(TT[n] < t) TT[n] = t; 
	}
      }
    }      

    t = 20.; i = N;
    for(m=0; m<N; m++) {
      if(t>TT[m]) { t = TT[m]; k = m; }     // first best configuration
    }
    this->nPenalty.set(l,double(t));

    t = dr[k]->tau.get(l);
    if(mIFO<9) i = mm[k][this->mIFO];
    else       i = t>0 ? int(t*rTDF+0.5) : int(t*rTDF-0.5);
        
//  0 d01 d02      0  d01  d02 
// d10  0 d12  ->  0 d'01 d'02 
// d20 d21 0       0 d"01 d"02

    for(m=0; m<N; m++) {
      ii = mm[k][m]-i;                // convert to time delay with respect to master IFO
      dr[m]->index.data[l] = ii; 
      //if(m!=this->mIFO) this->nNetIndex.set(l,double(ii));
    }
  }
  return;
}

//**************************************************************************
//:selection of clusters based on: 
//  'C' - network correlation coefficient with incoherent energy
//  'c' - network correlation coefficient with null energy
//  'r' - rho(ecor)
//  'X' - correlated energy
//  'x' - correlated energy - null energy (reduced corr energy)
//  'l' - likelihood (biased)
//  'L' - likelihood (unbiased)
//  'A' - snr - null assymetry (double OR)
//  'E' - snr - null energy    (double OR)
//  'a' - snr - null assymetry (edon)
//  'e' - snr - null energy    (edon)
//**************************************************************************
size_t network::netcut(double CUT, char cut, size_t core_size, int TYPE)
{
  size_t M = this->ifoList.size(); // number of detectors
  if(!M) return 0; 

  size_t i,j,k,n,m,K;
  size_t count = 0;
  size_t ID;
  size_t I = this->ifoList[0]->TFmap.maxLayer()+1;
  int    R = int(this->ifoList[0]->getTFmap()->rate()/I+0.5);
  int type = this->optim ? R : TYPE; // likelihoodI

  wavearray<double> cid;      // buffers for cluster ID
  wavearray<double> siz;      // buffers for cluster size (core)
  wavearray<double> SIZ;      // buffers for cluster size (core+halo)
  vector<wavearray<double> > snr(M); // buffers for total normalized energy
  vector<wavearray<double> > nul(M); // biased null stream

  double xcut,xnul,xsnr,amin,emin,like,LIKE;
  bool skip=false;

  for(j=0; j<nLag; j++) {         // loop on time shifts 

    if(!this->wc_List[j].size()) continue;
    
    cid = this->wc_List[j].get((char*)"ID",0,'S',type);           // get cluster ID
    K = cid.size();    
    if(!K) continue;                                       // no clusters

    siz = this->wc_List[j].get((char*)"size",0,'S',type);         // get cluster C size
    SIZ = this->wc_List[j].get((char*)"SIZE",0,'S',type);         // get cluster C+H size

    for(m=0; m<M; m++) {          // loop on detectors 
      snr[m] = this->wc_List[j].get((char*)"energy",m+1,'S',type);  // get total energy
      nul[m] = this->wc_List[j].get((char*)"null",m+1,'W',type);    // get biased null stream
    }

    for(k=0; k<K; k++) {          // loop on clusters 
      
      if(siz.data[k] < core_size) continue;                  // skip small clusters
      i = ID = size_t(cid.data[k]+0.1);
      
      if(tYPe=='i' || tYPe=='s' || tYPe=='g' || tYPe=='r' ||
	 tYPe=='I' || tYPe=='S' || tYPe=='G' || tYPe=='R') { 
	skip = !this->SETNDM(i,j,true,type);      	     // fill network data matrix
      }
      else if(tYPe=='b' || tYPe=='B' || tYPe=='E') {
	skip = !this->setndm(i,j,true,type);                 // fill network data matrix
      }
      else {
        skip = true;
      }

      if(skip) {
	cout<<"network::netcut: - undefined NDM matrix"<<endl;
	this->wc_List[j].ignore(ID);                         // reject cluster
	count += size_t(SIZ.data[k]+0.1);                    // count rejected pixels
	continue;        
      }

      xnul = xsnr = like = LIKE = 0;
      emin = amin = 1.e99;
      for(n=0; n<M; n++) {
	xnul += this->getifo(n)->null;
	xsnr += snr[n].data[k];
	like += snr[n].data[k]-this->getifo(n)->null;
	xcut  = snr[n].data[k]-this->getifo(n)->null;
	if(xcut < emin) emin = xcut;
	xcut  = (xcut-this->getifo(n)->null)/snr[n].data[k];
	if(xcut < amin) amin = xcut;
	for(m=0; m<M; m++) { 
	  LIKE += this->getNDM(n,m); 
	}
      }

      if(cut == 'A' || cut == 'E') {    // double OR selection
	emin = amin = 1.e99;
	for(n=0; n<M; n++) {
	  xcut  = snr[n].data[k] - this->getifo(n)->null;
	  if(like-xcut < emin) emin = like-xcut; 
	  xcut  = 2*(like-xcut)/(xsnr-xcut) - 1.;
	  if(xcut < amin) amin = xcut; 
	}
      }
      
      xcut = 0;
      if(cut == 'c' || cut == 'C') xcut = this->eCOR/(xnul + fabs(this->eCOR));            // network x-correlation
      if(cut == 'x' || cut == 'X') xcut = this->eCOR/sqrt(fabs(this->eCOR)*M);             // network correlated amplitude
      if(cut == 'l' || cut == 'L') xcut = like;
      if(cut == 'a' || cut == 'A') xcut = amin; 
      if(cut == 'e' || cut == 'E') xcut = emin; 
      if(cut == 'r' || cut == 'R') xcut = this->eCOR/sqrt((xnul + fabs(this->eCOR))*M); 

      if(xcut < CUT) { 
	this->wc_List[j].ignore(ID);          // reject cluster
	count += size_t(SIZ.data[k]+0.1);     // count rejected pixels
	continue;
      }
      else
	if(cut=='X') cout<<xcut<<cut<<":  size="<<SIZ.data[k]<<" L="<<LIKE<<" ID="<<ID<<endl;


      if(pOUT) { 
	printf("%3d  %4d  %7.2e %7.2e %7.2e %7.2e %7.2e %7.2e   %7.2e\n",
	       (int)n,(int)i,getNDM(0,0),getNDM(1,1),getNDM(2,2),getNDM(0,1),getNDM(0,2),getNDM(1,2),
	       getNDM(0,0)+getNDM(1,1)+getNDM(2,2)+2*(getNDM(0,1)+getNDM(0,2)+getNDM(1,2)));
      }
    }
  }
  return count;
}



//**************************************************************************
//: set rank statistic for pixels in network netcluster structure
//**************************************************************************

size_t network::setRank(double T, double F)
{
  size_t j,m,n,V;
  size_t cOUNt = 0;

  size_t I = this->ifoList[0]->TFmap.maxLayer()+1;
  size_t R = size_t(this->ifoList[0]->getTFmap()->rate()/I+0.5);
  size_t N = this->ifoList[0]->getTFmap()->size();
  size_t M = this->ifoList.size();                        // number of detectors

  int window = int(T*R+0.5)*I/2;                          // rank half-window size
  int offset = int(this->Edge*ifoList[0]->getTFmap()->rate()+0.5);

  double amp, rank;
  int inD, frst, last, nB, nT;

// pointers

  double* ppp;
  std::vector<detector*> pDet; pDet.clear();
  std::vector<double*>   pDat; pDat.clear();

  for(m=0; m<M; m++) {
    pDet.push_back(ifoList[m]);
    pDat.push_back(ifoList[m]->getTFmap()->data);
  }

  size_t wmode = pDet[0]->getTFmap()->w_mode; 
  if(wmode != 1) return 0;

// get index arrays for delayed amplitudes

  wavearray<double> cid;   // buffers for cluster ID
  netpixel* pix;

  for(n=0; n<nLag; n++) {                  // loop over time shifts 
     V = this->wc_List[n].pList.size();
    nB = int(this->wc_List[n].getbpp()*2*window+0.5);
    
    if(!V) continue;
      
    for(j=0; j<V; j++) {   // loop over pixels 
	
      pix = &(this->wc_List[n].pList[j]);
      if(R != size_t(pix->rate+0.5)) continue;
	
//      printf("lag=%2d pixel=%4d ID=%4d  ",n,j,pix->clusterID);

      for(m=0; m<M; m++) {               // loop over detectors 

	inD = size_t(pix->getdata('I',m)+0.1);
	ppp = pDat[m]+inD;
	amp = *ppp;
	  
	*ppp = pix->getdata('S',m);    // pixel SNR
	if(inD-window < offset) {      // left boundary
	  frst = offset;
	  last = frst + 2*window;
	}
	else if(inD+window > int(N-offset)) {
	  last = N-offset;
	  frst = last - 2*window;	    
	}
	else {
	  frst = inD-window;
	  last = inD+window;
	}
	
	if( inD>int(N-offset) ||  inD<offset || 
	   frst>int(N-offset) || frst<offset ||
	   last>int(N-offset) || last<offset) { 
	  cout<<"network::setRank() error\n";
	  *ppp = amp;                     // restore pixel value in the array
	  pix->setdata(0.,'R',m);
	  continue;
	}
	
	rank = pDet[m]->getTFmap()->getSampleRankE(inD,frst,last);
	nT = last-frst+1;

	if(rank > nT-nB) rank = log(nB/double(nT+1-rank));
	else rank = 0.;

//	printf("%8.1e|%8.1e ",rank,amp);

	*ppp = amp;                       // restore pixel value in the array
	pix->setdata(rank,'R',m);
	cOUNt++;
	
      }
//      cout<<endl;
    }
  }
  return cOUNt;
}

// read earth skyMask coordinates from file (1G)
size_t network::setSkyMask(double f, char* file) {
  int i;
  size_t L = this->skyHole.size();
  size_t n = 0;
  size_t l;
  char   str[1024];
  FILE* in;
  char* pc;
  double a;

  if(!L) {
    cout<<endl<<"network::setSkyMask() - skymap size L=0"<<endl<<endl;
    exit(1);
  } else if(!file) {
    cout<<endl<<"network::setSkyMask() - NULL input skymask file"<<endl<<endl;
    exit(1);
  } else if(!strlen(file)) {
    cout<<endl<<"network::setSkyMask() - input skymask file not defined"<<endl<<endl;
    exit(1);
  } else if( (in=fopen(file,"r"))==NULL ) {
    cout << endl << "network::setSkyMask() - input skymask file '" 
         << file << "' not exist" << endl << endl;;
    exit(1);
  }

  while(fgets(str,1024,in) != NULL){

     if(str[0] == '#') continue;
     if((pc = strtok(str," \t")) == NULL) continue;
     if(pc) i = atoi(pc);                                       // sky index
     if((pc = strtok(NULL," \t")) == NULL) continue;
     if(pc && i>=0 && i<int(L)) {
       this->skyHole.data[i] = atof(pc);                        // skyProb
       this->nSkyStat.set(i, atof(pc));		 
       n++;
     } else {
       cout<<endl<<"network::setSkyMask() - "
           <<"skymask file contains index > max L="<<L<<endl<<endl;
       exit(1);
     }
  }
  if(n!=L) {
    cout<<endl<<"network::setSkyMask() - "
        <<"the number of indexes in the skymask file != L="<<L<<endl<<endl;
    exit(1);
  }
  a = skyHole.mean()*skyHole.size();
  skyHole *= a>0. ? 1./a : 0.;
  if(f==0.) { skyHole = 1.; return n; }

  double* p  = this->skyHole.data;
  double** pp = (double **)malloc(L*sizeof(double*));
  for(l=0; l<L; l++) pp[l] = p + l;

  skyProb.waveSort(pp,0,L-1);

  a = double(L);
  for(l=0; l<L; l++) { 
    a -= 1.; 
    *pp[l] = a/L<f ? 0. : 1.; 
    if(*pp[l] == 0.) this->nSkyStat.set(pp[l]-p,*pp[l]);		 
  }
  free(pp);
  return n;
}

// read celestial/earth skyMask coordinates from file
size_t network::setSkyMask(char* file, char skycoord) {

  int i;
  size_t L = this->skyHole.size();  
  size_t n = 0;
  char   str[1024];
  FILE* in;
  char* pc;
  double data=0;

  if(skycoord!='e' && skycoord!='c') {
    cout << "network::setSkyMask() - wrong input sky coordinates " 
         << " must be 'e'/'c' earth/celestial" << endl;;
    exit(1);
  }

  if(!L) {
    cout<<endl<<"network::setSkyMaskCC() - skymap size L=0"<<endl<<endl;
    exit(1);
  } else if(!file) {
    cout<<endl<<"network::setSkyMaskCC() - NULL input skymask file"<<endl<<endl;
    exit(1);
  } else if(!strlen(file)) {
    cout<<endl<<"network::setSkyMaskCC() - input skymask file not defined"<<endl<<endl;
    exit(1);
  } else if( (in=fopen(file,"r"))==NULL ) {
    cout << endl << "network::setSkyMaskCC() - input skymask file '" 
         << file << "' not exist" << endl << endl;;
    exit(1);
  }

  if(skycoord=='e') {skyMask.resize(L); skyMask = 1;} 
  if(skycoord=='c') {skyMaskCC.resize(L); skyMaskCC = 1;} 

  while(fgets(str,1024,in) != NULL){

    if(str[0] == '#') continue;
    if((pc = strtok(str," \t")) == NULL) continue;
    if(pc) i = atoi(pc);                                       // sky index
    if((pc = strtok(NULL," \t")) == NULL) continue;
    if(pc && i>=0 && i<int(L)){
      data = atof(pc);                      
      if(skycoord=='e') this->skyHole.data[i]=data;
      if(skycoord=='c') this->skyMaskCC.data[i]=data;
      n++;
    } else {
      cout<<endl<<"network::setSkyMask() - "
          <<"skymask file contains index > max L="<<L<<endl<<endl;
      exit(1);
    }
  }
  if(n!=L) {
    cout<<endl<<"network::setSkyMask() - "
        <<"the number of indexes in the skymask file != L="<<L<<endl<<endl;
    exit(1);
  }

  if(in!=NULL) fclose(in);  
  return n;
}

// read celestial/earth skyMask coordinates from skymap
size_t network::setSkyMask(skymap sm, char skycoord) {

  if(skycoord!='e' && skycoord!='c') {
    cout << "network::setSkyMask() - wrong input sky coordinates " 
         << " must be 'e'/'c' earth/celestial" << endl;;
    exit(1);
  }

  size_t L = this->skyHole.size();  
  if((int)sm.size()!=L) {
    cout << "network::setSkyMask() - wrong input skymap size " 
         << sm.size() << " instead of " << L << endl;;
    exit(1);
  }

  if(skycoord=='e') {
    skyMask.resize(L);  
    for(int i=0;i<L;i++) this->skyHole.data[i]=sm.get(i);
  }
  if(skycoord=='c') {
    skyMaskCC.resize(L);  
    for(int i=0;i<L;i++) this->skyMaskCC.data[i]=sm.get(i);
  }

  return L;
}

// read MDC log file with list of injections
size_t network::readMDClog(char* file, double gps, int nTime, int nName) {
  int i;
  size_t j;
  FILE* in;
  char   str[1024];
  char   STR[1024];
  char* p;
  bool save;
  double GPS;

  int imdcMap=0;
  std::map <string, int> mdcMap;	// used to check uniqueness of mdc types

  if( (in=fopen(file,"r"))==NULL ) {
    cout<<"network::readMDClog() - no file is found \n";
    exit(1);
  }

  while(fgets(str,1024,in) != NULL){

     if(str[0] == '#') continue;
     sprintf(STR,"%s",str);                    // copy string

// find and save injection gps time

     if((p = strtok(STR," \t")) == NULL) continue;

     for(i=1; i<nTime; i++) { 
       p = strtok(NULL," \t");                 // get gps time
       if(!p) break;
     }

     if(p) {
       GPS = atof(p);
       if(gps==0. || fabs(GPS-gps)<7200.) {
	 this->mdcList.push_back(str);
	 this->mdcTime.push_back(GPS);
       }
     }

// find and save injection type

     if((p = strtok(str," \t")) == NULL) continue;

     for(i=1; i<nName; i++) {
       p = strtok(NULL," \t");   	// get name
       if(!p) break;
     }

     if(p) if(mdcMap.find(p)==mdcMap.end())  mdcMap[p]=imdcMap++; 
  }

  // copy mdc type to mdcType vector
  // the data are sorted keeping the back compatibility with the 1G algorithm 
  this->mdcType.resize(mdcMap.size());
  std::map<std::string, int>::iterator iter;
  for (iter=mdcMap.begin(); iter!=mdcMap.end(); iter++) {
    this->mdcType[iter->second]=iter->first;
  }
  // print list
  for(int j=0;j<this->mdcType.size();j++) {
    int step=1;
    if(j<100) step=1;
    else if(j<10000) step=100;
    else step=1000;
    if(j%step==0) {
      printf("type %3d\t",(int)j); 
      cout<<" has been assigned to waveform "<<mdcType[j]<<endl;
    }
  }

  return this->mdcList.size();
}


// read file with segment list
size_t network::readSEGlist(char* file, int n) {
  int i;
  char   str[1024];
  char* p;
  FILE* in;
  waveSegment SEG; 
  SEG.index = 0;

  if( (in=fopen(file,"r"))==NULL ) {
    cout<<"network::readSEGlist(): specified segment file "<<file<<" does not exist\n";
    exit(1);
  }

  while(fgets(str,1024,in) != NULL){

     if(str[0] == '#') continue;

// find and save segment start time

     if((p = strtok(str," \t")) == NULL) continue;

     for(i=1; i<n; i++) { 
       p = strtok(NULL," \t");                 // get start
       if(!p) break;
     }

     if(p) { 
       SEG.index++;
       SEG.start = atof(p);
       p = strtok(NULL," \t");                 // get stop
       if(!p) continue;
       SEG.stop  = atof(p);
//       printf("%12.2f  %12.2f \n",SEG.start,SEG.stop);
       this->segList.push_back(SEG);
     }
  }
  return this->segList.size();
}


// set veto array
double network::setVeto(double Tw) {
// set veto array from the input list of DQ segments
// Tw - time window around injections
// 

  int j, jb, je, jm;
  size_t i,k;
  double gps, EE;
  double live = 0.;
  wavearray<short> w;
  this->mdc__ID.clear();

  size_t I = this->ifoList.size();
  if(Tw<2.) Tw = 2.;
  detector* d = this->ifoList[0];

  int    N = d->getTFmap()->size();               // TF data size
  int    M = d->getHoT()->size();                 // TS data size
  double R = (d->getTFmap()->pWavelet->m_WaveType==WDMT) ? // time series rate
             d->getHoT()->rate() : d->getTFmap()->rate();
  double S = d->getTFmap()->start();              // segment start time 
  double E = d->getTFmap()->stop();               // segment end time 
  size_t K = this->segList.size();                // segment list size
  size_t L = this->mdcList.size();                // injection list size
  size_t n = size_t(this->Edge*R+0.5);            // data offset
  int    W = int(Tw*R/2.+0.5);                    // injection window size
  
  if(M>2) N=M;                                    // use size of TS object
  if(!I || !N) return 0.;

  if(this->veto.size() != size_t(N)) {            // initialize veto array
    this->veto.resize(N);
  }
  this->veto = 0;
  w = this->veto;

  for(k=0; k<K; k++) {                            // loop over segmets
    gps = segList[k].start;
    if(gps<S) gps=S; 
    if(gps>E) gps=E;
    j  = int((gps-S)*R);                          // index in data array
    jb = j<0 ? 0 : j;
    gps = segList[k].stop;
    if(gps<S) gps=S; 
    if(gps>E) gps=E;
    j  = int((gps-S)*R);                          // index in data array
    je = j>N ? N : j;
    for(j=jb; j<je; j++) this->veto.data[j] = 1;
  }

  if(!K) this->veto = 1;                          // no segment list

  for(k=0; k<L; k++) {                            // loop over injections
    gps = mdcTime[k];                             // get LOG injection time
    if(gps == 0.) continue;
    
    if(d->HRSS.size()) {                          // get MDC injection time
      gps = EE = 0.;
      for(i=0; i<I; i++) {
	d = this->ifoList[i];
	gps += d->TIME.data[k]*d->ISNR.data[k];
	EE  += d->ISNR.data[k];
      }
      gps /= EE;
      mdcTime[k] = gps;
    }
    
    jm = int((gps-S)*R);                          // index in data array
    jb = jm-W; je = jm+W; 
    if(jb < 0) jb = 0;
    if(jb >=N) continue;
    if(je > N) je = N;
    if(je <=0) continue;
    if(je-jb < int(R)) continue;
    if(jm<jb || jm>je) continue;

    for(j=jb; j<je; j++) w.data[j] = 1;

    if(veto.data[jm]) this->mdc__ID.push_back(k); // save ID of selected injections
  }

  if(L) this->veto *= w;                          // apply injection mask
  live = 0.;
  for(k=n; k<N-n; k++) live+=this->veto.data[k];
  
  return live/R;
}

// get reconstructed detector responses
bool network::getwave(size_t ID, size_t lag, char atype)
{ 
  int n,m;
  size_t i,j,k,l;
  double R = 0;
  size_t M = this->ifoList.size();
  bool flag = true;

  netcluster* wc = this->getwc(lag);
  detector*   pd = this->getifo(0);

  Meyer<double> Me(256,1);
  WSeries<double> w(Me);
  WSeries<double> W;
  wavearray<double> x;  
  wavearray<double> id = wc->get((char*)"ID"); 
  std::vector<int> v;

  flag = false;
  for(j=0; j<id.size(); j++) { 
    if(id.data[j] == ID) { flag=true; break; }
  }  

  if(!flag) return false;
  flag = true;

  v = wc->nTofF[ID-1];                  // backward time delay configuration  
  k = pd->nDFS/pd->nDFL;                // up-sample factor
  l = size_t(log(k*1.)/log(2.)+0.1);    // wavelet level

// time-of-flight backward correction for reconstructed waveforms

  for(i=0; i<M; i++) {
    pd = this->getifo(i);
    if(pd->getwave(ID,*wc,atype,i) == 0.) { flag=false; break; } 

    R = pd->waveForm.rate();

    w.rate(R*k); x.rate(R);
    m = int(pd->waveForm.size());
    n = m/int(R+0.1)+4;                 // integer number of seconds
    n *= int(R+0.1);                    // total number of samples in layer 0
    if(n!=int(x.size())) x.resize(n);
    x = 0.; x.cpf(pd->waveForm,m,0,n/2);
    n *= k;                             // total number of up samples
    if(n!=int(w.size())) w.resize(n);
    w.setLevel(l);                      // set wavelet level
    w = 0.; w.putLayer(x,0);            // prepare for upsampling
    w.Inverse();                        // up-sample
    W = w; w = 0.;
    
    if(v[i]>0) {                        // time-shift waveform
      w.cpf(W,w.size()-v[i],0,v[i]);    // v[i] - backward time delay 
    }
    else {
      w.cpf(W,w.size()+v[i],-v[i]);
    }

    n = x.size();
    w.Forward(l);
    w.getLayer(x,0);
    pd->waveForm.cpf(x,m,n/2);          // align waveforms

    x  = pd->waveForm;                  // produce null stream
    w  = pd->waveNull;
    x *= -1.;                           
    n  = int((x.start()-w.start())*x.rate()+0.1);
    w.add(x,m,0,n);
    w.Forward(pd->TFmap.getLevel());

    pd->waveNull.resize(4*int(R+0.1)+x.size());
    pd->waveNull.setLevel(pd->TFmap.getLevel());
    pd->waveNull = 0.;
    pd->waveNull.start(x.start()-2.);

    n = int((pd->waveNull.start()-w.start())*w.rate()+0.1);
    m = pd->waveNull.size();
    if(n>=0) {
      if(n+m > (int)w.size()) m = w.size()-n;
      pd->waveNull.cpf(w,m,n,0);
    }
    else {
      m += n;
      if(m > (int)w.size()) m = w.size();
      pd->waveNull.cpf(w,m,0,-n);    
    }
  }
  return flag;
}

bool network::getMRAwave(size_t ID, size_t lag, char atype, int mode, bool tof)
{ 
// get MRA waveforms of type atype in time domain given lag nomber and cluster ID
// mode: -1/0/1 - return 90/mra/0 phase
// if tof = true, apply time-of-flight corrections
// fill in waveform arrays in the detector class
  size_t i,j;
  double R = 0;
  size_t nIFO = this->ifoList.size();

  netcluster* pwc = this->getwc(lag);
  wavearray<double> id = pwc->get((char*)"ID",0,'S',0); 

  bool signal = (abs(atype)=='W' || abs(atype)=='w') ? false : true;
  bool flag = false;

  for(j=0; j<id.size(); j++) { 
     if(size_t(id.data[j]+0.1) == ID) flag=true;
  }  
  if(!flag) return false;

  wavearray<double> x;
  std::vector<int> v;

  v = pwc->nTofF[ID-1];                  // backward time delay configuration  

  // time-of-flight backward correction for reconstructed waveforms

  for(i=0; i<nIFO; i++) {

    x = pwc->getMRAwave(this,ID,i,atype,mode);
    if(x.size() == 0.) {cout<<"zero length\n"; return false;}

// apply time delay

    if(tof) {
       double R = this->rTDF;                    	// effective time-delay rate
       double tShift = -v[i]/R; 

       x.FFTW(1);
       TComplex C;
       double df = x.rate()/x.size();
       for (int ii=0;ii<(int)x.size()/2;ii++) {
          TComplex X(x.data[2*ii],x.data[2*ii+1]);
          X=X*C.Exp(TComplex(0.,-2*PI*ii*df*tShift));  	// Time Shift
          x.data[2*ii]=X.Re();
          x.data[2*ii+1]=X.Im();
       }
       x.FFTW(-1);
    }

    if(signal) this->getifo(i)->waveForm = x;
    else       this->getifo(i)->waveBand = x;
 
  }
  return flag;
}

//**************************************************************************
// initialize wc_List for a selected TF area
//**************************************************************************
size_t network::initwc(double sTARt, double duration)
{
  size_t i,j,m,k;
  double a;
  size_t npix = 0;
  bool   save = false;

  size_t  I = this->ifoList[0]->TFmap.maxLayer()+1;
  size_t  R = size_t(this->ifoList[0]->getTFmap()->rate()/I+0.5);
  size_t  N = this->ifoList[0]->getTFmap()->size();
  size_t  M = this->ifoList.size();                    // number of detectors
  size_t jB = size_t(this->Edge*R)*I;                  // number of samples in the edges

// pointers

  std::vector<detector*> pDet; pDet.clear();
  std::vector<double*>   pDat; pDat.clear();
  std::vector<int>       pLag; pLag.clear();  

  netpixel pix(M);                       // initialize pixel for M detectors
  pix.clusterID = 0;                     // initialize cluster ID
  pix.rate = float(R);                   // pixel rate
  pix.core = true;                       // pixel core
  pix.neighbors.push_back(0);            // just one neighbor for each pixel 

  for(m=0; m<M; m++) {
    pDet.push_back(ifoList[m]);
    pDat.push_back(ifoList[m]->getTFmap()->data);
    pLag.push_back(int(ifoList[m]->sHIFt*R*I+0.5));
  }

  size_t il = size_t(2.*pDet[0]->TFmap.getlow()/R);    // low frequency boundary index
  size_t ih = size_t(2.*pDet[0]->TFmap.gethigh()/R);   // high frequency boundary index
  if(ih==0 || ih>=I) ih = I;

  size_t J = size_t(sTARt*R+0.1);                      // start index in the slice
  size_t K = size_t(duration*R+0.1);                   // number of pixels in the slice
  slice S;

  this->wc_List[0].clear();              // clear wc_List

//  cout<<"il="<<il<<"  ih"<<ih<<"  K="<<K<<" J="<<J<<endl;

  for(i=il; i<ih; i++){                  // loop over layers
    pix.frequency = i;
    S = pDet[0]->TFmap.getSlice(i);

    for(j=0; j<K; j++){                  // loop over pixels 
      pix.time = (J+j)*I + S.start();    // LTF pixel index in the map;

      if(pix.time >= N) { 
	cout<<"network::initwc() error - index out of limit \n";
	continue;
      }

      pix.likelihood = 0.; 
      save = true;
      for(m=0; m<M; m++) {               // loop over detectors  
	k = pix.time+pLag[m]; 
	if(k>=N) k -= N-jB; 
	if(!this->veto.data[k]) save = false;
	a = pDat[m][k];
	pix.likelihood += a*a/2.; 
	pix.setdata(a,'S',m);                         // set amplitude 
	pix.setdata(k,'I',m);                         // set index 
	pix.setdata(pDet[m]->getNoise(i,k),'N',m);    // set noise RMS 
      }
      pix.neighbors[0] = ++npix;
      if(save) this->wc_List[0].append(pix);
    }
    
  }

  wc_List[0].start = pDet[0]->TFmap.start();
  wc_List[0].stop  = N/R/I;
  wc_List[0].rate  = pDet[0]->TFmap.rate();

  if(npix) { 
    this->wc_List[0].pList[npix-1].neighbors[0]=0;
    this->wc_List[0].cluster();
  }
  return npix;
}


//**************************************************************************
//:select TF samples by value of the network likelihood: 2-NIFO detectors
//**************************************************************************
long network::coherence(double Eo, double Es, double factor)
{
  size_t nIFO = this->ifoList.size();       // number of detectors

  if(nIFO>NIFO || !this->filter.size() || this->filter[0].index.size()!=32) {
     cout<<"network::coherence(): \n" 
         <<"invalid number of detectors or\n"
	 <<"delay filter is not set\n";
    return 0;
  }
  if(getifo(0)->getTFmap()->w_mode != 1) {
    cout<<"network::coherence(): invalid whitening mode.\n"; 
    return 0;
  } 

  if(factor > 1.) factor = float(nIFO-1)/nIFO;
  Eo = nIFO*Eo*Eo;       // lognormal threshold on total pixel energy
  Es = nIFO*Es*Es;       // final threshold on pixel energy

  size_t i,j,k,m,n,l,NN;
  size_t jS,jj,nM,jE,LL;

  double R = this->ifoList[0]->getTFmap()->rate();
  size_t N = this->ifoList[0]->getTFmap()->size();
  size_t I = this->ifoList[0]->TFmap.maxLayer()+1;
  size_t M = this->filter.size()/I;                  // total number of delays 
  size_t K = this->filter[0].index.size();           // length of delay filter 
  size_t L = this->index.size();                     // total number of source locations 
  size_t jB = size_t(this->Edge*R/I)*I;              // number of samples in the edges
  double band = this->ifoList[0]->TFmap.rate()/I/2.;
  slice S = getifo(0)->getTFmap()->getSlice(0);      // 0 wavelet slice

  delayFilter* pv;
  netpixel pix(nIFO); 
  pix.core = true;
  pix.rate = R/I;
  pix.layers = I;
  
// pointers to data

  wavearray<int>    inTF; 
  wavearray<double> emax[NIFO]; 
  double* pdata[NIFO];
  double* pq;
  for(n=0; n<NIFO; n++) {
    emax[n].resize(S.size()); emax[n] = 0.;
    if(n >= nIFO) continue;
    pdata[n] = getifo(n)->getTFmap()->data;
  }
  inTF.resize(S.size()); inTF = 0;

// allocate buffers
  wavearray<double> eD[NIFO];       // array for delayed amplitudes^2
  double* pe[NIFO];
  int     in[NIFO];
  for(n=0; n<NIFO; n++) { eD[n].resize(M); eD[n] = 0.; pe[n] = eD[n].data; }

// get sky index arrays
  wavearray<short> inDEx[NIFO];
  short* ina;

  LL = 0;
  for(l=0; l<L; l++) if(skyMask.data[l]) LL++;
  for(n=0; n<NIFO; n++) { 
    inDEx[n].resize(LL);
    if(n>=nIFO) inDEx[n] = 0;
    else {
      k = 0;
      ina = this->getifo(n)->index.data;
      for(l=0; l<L; l++) {
	if(skyMask.data[l]) inDEx[n].data[k++] = ina[l];
      }
    }
  }

  NETX( 
  short* m0 = inDEx[0].data; ,
  short* m1 = inDEx[1].data; ,
  short* m2 = inDEx[2].data; ,
  short* m3 = inDEx[3].data; ,
  short* m4 = inDEx[4].data; ,
  short* m5 = inDEx[5].data; ,
  short* m6 = inDEx[6].data; ,
  short* m7 = inDEx[7].data; )

// buffer for wavelet layer delay filter
  double* pF = (double*)malloc(K*M*sizeof(double));
  int*    pJ =    (int*)malloc(K*M*sizeof(int));
  double* F;
  int*    J;

// sky time delays

  size_t M1[NIFO];
  size_t M2[NIFO];
  double t1,t2;
  size_t KK = this->filter.size()/I;         // number of time delay samples
  for(n=0; n<nIFO; n++) { 
    t1 = getifo(n)->tau.min()*R*getifo(0)->nDFS/I;
    t2 = getifo(n)->tau.max()*R*getifo(0)->nDFS/I;
    M1[n] = short(t1+KK/2+0.5)-1;
    M2[n] = short(t2+KK/2+0.5)+1;
  }

// time shifts

  long nZero = 0;
  bool skip = false;
  size_t Io = 0;                             // counter of number of layers  
  double a,b,E;

  this->pixeLHood = getifo(0)->TFmap;
  this->pixeLHood = 0.;

// set veto array if it is not set
  if(this->veto.size() != N) { veto.resize(N); veto = 1; }

  N -= jB;                                   // correction for left boundary

  for(k=0; k<nLag; k++) { 
    this->wc_List[k].clear();                // clear netcluster structure
    this->livTime[k] = 0.;                   // clear live time counters
    this->wc_List[k].setlow(getifo(0)->TFmap.getlow());
    this->wc_List[k].sethigh(getifo(0)->TFmap.gethigh());
  }

  for(i=1; i<I; i++) {                       // loop over wavelet layers

// select bandwidth
    a = i*band;
    if(a >= getifo(0)->TFmap.gethigh()) continue;
    a = (i+1)*band;
    if(a <= getifo(0)->TFmap.getlow()) continue; 
    
    Io++;
    
// set filter array for this layer
    for(m=0; m<M; m++){                    
      for(k=0; k<K; k++){             
	pv = &(filter[i*M+m]);
	pF[k+m*K] = double(pv->value[k]);
	pJ[k+m*K] =    int(pv->index[k]);
      }
    }

    S = getifo(0)->getTFmap()->getSlice(i);

    NN = S.size();
    for(n=0; n<NIFO; n++) {           // resize arrays for max amplitudes
      if(emax[n].size() != NN) emax[n].resize(NN); 
      emax[n] = 0; in[n] = 0;
    }      
    if(inTF.size() != NN) inTF.resize(NN); 

// apply delay filters to calculate emax[]

    jS = S.start()+jB;
    NN = NN - jB/I;
    jE = NN - jB/I;

    for(n=0; n<nIFO; n++) {

      jj = jB/I; 

      for(j=jS; j<N; j+=I) {              // loop over samples in the layer

	emax[n].data[jj] = 0.;
	inTF.data[jj] = j;              // store index in TF map
	
	F  = pF+M1[n]*K; 
	J  = pJ+M1[n]*K;
	pq = pdata[n]+j;

	b = 0.;
	for(m=M1[n]; m<M2[n]; m++){                   
	  a  = dot32(F,pq,J);           // delayed amplitude
	  a *= a;
	  if(b < a) b = a;      // max energy 
	  F+=K; J+=K;
	}
	emax[n].data[jj++]  = b;
      }
    }

// select shifted amplitudes
// regular case:             jS.......j*********.....N
// left boundary handling:   jS***.............j*****N shift becomes negative 

    for(k=0; k<nLag; k++) {              // over lags
      
      a  = 1.e10;
      nM = 0;                                     // master detector 

      for(n=0; n<nIFO; n++) {
	b = this->getifo(n)->lagShift.data[k];    // shift in seconds
	if(a>b) { a = b; nM = n; }
      }

      for(n=0; n<nIFO; n++) {
	b = this->getifo(n)->lagShift.data[k];    // shift in seconds
	in[n] = (int((b-a)*R)+jB)/I - 1;          // index of first pixel in emax -1
      }

      for(jj=jB/I; jj<NN; jj++) {                 // loop over samples in the emax
	
	m = 0; E = 0.;
	for(n=0; n<nIFO; n++) {
	  if((++in[n]) >= int(NN)) in[n] -= jE;   // check boundaries
	  m += this->veto.data[inTF.data[in[n]]]; // check data quality
	  E += emax[n].data[in[n]]; 
	}
	this->livTime[k] += float(m/nIFO);        // calculate live time for each lag
	if(E<Eo || m<nIFO) continue;

	skip = false;
	for(n=0; n<nIFO; n++) {
	  b = E - emax[n].data[in[n]];
	  if(b<Eo*factor) skip = true;
	}
	if(skip) continue;
	
// calculate delays again and run skyloop.

	for(n=0; n<nIFO; n++) {
	  F  = pF+M1[n]*K; 
	  J  = pJ+M1[n]*K;
	  pq = pdata[n]+inTF.data[in[n]];

	  for(m=M1[n]; m<M2[n]; m++){                   
	    a  = dot32(F,pq,J);              // delayed amplitude
	    pe[n][m]  = a*a;                 // delayed energy
	    F+=K; J+=K;
	  }
	}

	skip = true;
	for(l=0; l<LL; l++) {
	  double pet = 0.;
          NETX( 
          pet+=pe[0][m0[l]]; ,
          pet+=pe[1][m1[l]]; ,
          pet+=pe[2][m2[l]]; ,
          pet+=pe[3][m3[l]]; ,
          pet+=pe[4][m4[l]]; ,
          pet+=pe[5][m5[l]]; ,
          pet+=pe[6][m6[l]]; ,
          pet+=pe[7][m7[l]]; )
	  if(pet > Eo)  
	    { skip = false; break; }
	}
	if(skip) continue;

// save pixels in wc_List
	
	j = inTF.data[in[nM]];                      // index in TF
	pix.rate = float(R/I);
	pix.time = j;
	pix.core = E>Es ? true : false;
	pix.frequency = i;
	pix.likelihood = E;
	
	for(n=0; n<nIFO; n++) {
	  pix.data[n].index = inTF.data[in[n]];
	  pix.data[n].asnr = emax[n].data[in[n]];
	}
	
	wc_List[k].append(pix);
	if(!k) this->pixeLHood.data[j] = sqrt(E/nIFO);
	nZero++;
      }      
    }
  }

// set metadata in wc_List
  for(k=0; k<nLag; k++) {
    a = getifo(0)->getTFmap()->start();
    b = getifo(0)->getTFmap()->size()/R;
    this->wc_List[k].start = a;  
    this->wc_List[k].stop  = a+b;
    this->wc_List[k].rate  = R;
    this->livTime[k] *= double(I)/(Io*R);
  }
  
  if(nZero) this->setRMS();
  
  free(pF); free(pJ);

  return nZero;
}

// calculate sky error regions
void network::getSkyArea(size_t id, size_t lag, double To, double rMs) {
// calculate sky error regions
// new version designed for 2G analysis
//!param: cluster id
//!param: time lag
//!param: cluster time
//!param: rms correction: noise rms is 1+rMs

   int in,im,IN,IM;
   size_t i,j,l,m,k,K;
   size_t N = this->wc_List[lag].csize();
   size_t M = this->mdc__IDSize();
   size_t L = this->skyProb.size();
   size_t Lm = L-int(0.9999*L); 
   size_t nIFO = this->ifoList.size();        // number of detectors
   bool   prior = this->gamma<0?true:false;   // gamma<0  : antenna pattern prior is used
   skymap* sm = &(this->nSkyStat);
   
   if(Lm < 2) return; 
   if(nSky > long(L-Lm)) nSky = L-Lm;
   if(id>N) return;
   
   double th,ph,a;
   double sum = 0.;
   double vol = 0.;
   double co1 = cos(sm->theta_1*PI/180.);
   double co2 = cos(sm->theta_2*PI/180.);
   double phi = sm->phi_2-sm->phi_1;
   double s = fabs(phi*(co1-co2))*180/PI/sm->size();  // sky solid angle
   
   std::vector<float>* vf = &(this->wc_List[lag].sArea[id-1]);  
   size_t v[11]; 
   
   double* p  = this->skyProb.data;
   double** pp = (double **)malloc(L*sizeof(double*));
   for(l=0; l<L; l++) pp[l] = p + l;
   
   skyProb.waveSort(pp,0,L-1);
   
   double Po = *pp[L-1]; 
   double rms = fabs(rMs);                             // rms for method 2
   
   double smax=nAntenaPrior.max();		       // max sensitivity
 
   for(l=0; l<L; l++) {
      if(*pp[l] <= 0.) {*pp[l]=0.; continue;}
      *pp[l] = exp(-(Po - *pp[l])/2./rms);
      if(prior) *pp[l] *= pow(nAntenaPrior.get(int(pp[l]-p))/smax,4);
      sum += *pp[l];
   }
   if(prior) skyProb.waveSort(pp,0,L-1);

   for(l=0; l<L; l++) {
      p[l] /= sum;                     // normalize map
      nProbability.set(l,p[l]);        // fill in skyProb map
   }

   if(pOUT) cout<<rMs<<" "<<*pp[L-1]<<" "<<*pp[L-2]<<" "<<*pp[L-3]<<"\n";

   vf->clear();
   for(m=0; m<11; m++) { v[m] = 0; vf->push_back(0.); }
   
   vol = 0;
   for(l=L-1; l>Lm; l--){
      vol += *pp[l];
      for(m=size_t(vol*10.)+1; m<10; m++) v[m] += 1;
      if(vol >= 0.9) break;
   }
   
  for(m=1; m<10; m++) {
     (*vf)[m] = sqrt(v[m]*s);
     if(pOUT && !M) cout<<m<<" error region: "<<(*vf)[m]<<endl;
  }

  
// fill skyProb skymap 
  
  std::vector<float>* vP = &(this->wc_List[lag].p_Map[id-1]);
  std::vector<int>*   vI = &(this->wc_List[lag].p_Ind[id-1]);
  
  K = 0;
  sum = 0.;
  vP->clear();
  vI->clear();
  double pthr=0;
  // if nSky -> nSky is converted into a probability threshold nSky=-XYZ... -> pthr=0.XYZ... 
  if(nSky<0) {char spthr[1024];sprintf(spthr,"0.%d",int(abs(nSky)));pthr=atof(spthr);}
  for(l=L-1; l>Lm; l--){
     sum += *pp[l];
     if(nSky==0 && (K==1000 || sum > 0.99) && K>0) break; 
     else if(nSky<0 && sum > pthr && K>0) break; 
     else if(nSky>0 && K==nSky && K>0) break;
     K++;  
     vI->push_back(int(pp[l]-p));
     vP->push_back(float(*pp[l]));
  } 

// set injections if there are any

  if(!M) { free(pp); return; }
  
  double dT = 1.e13;
  double injTime = 1.e12;
  int injID = -1;
  int mdcID = -1;
  injection INJ(this->ifoList.size());

  for(m=0; m<M; m++) {
    mdcID = this->getmdc__ID(m);
    dT = fabs(To - this->getmdcTime(mdcID));
    if(dT<injTime && INJ.fill_in(this,mdcID)) { 
      injTime = dT; 
      injID = mdcID; 
      if(pOUT) printf("getSkyArea: %4d %12.4f %7.3f %f \n",int(m),To,dT,s);
    } 
  }
 
  if(INJ.fill_in(this,injID)) {

    th = INJ.theta[0];
    ph = INJ.phi[0]; 
    i  = this->getIndex(th,ph); 

    vI->push_back(int(i));
    vP->push_back(float(p[i]));

    vol = sum = 0.;
    for(l=L-1; l>Lm; l--){
      vol += s;
      sum += *pp[l];
      if(pp[l]-p == int(i)) break;
    }
    (*vf)[0]  = sqrt(vol);
    (*vf)[10] = sum;
    j = pp[L-1]-p;                                    // reference sky index at max

    if(pOUT) {
      printf("getSkyArea: %5d %12.4f %6.1f %6.1f %6.1f %6.1f %6.2f %6.2f %6.2f %7.5f, %e %d \n",
	     int(id),INJ.time[0]-this->getifo(0)->TFmap.start(),INJ.theta[0],INJ.phi[0],
	     sm->getTheta(j),sm->getPhi(j),(*vf)[0],(*vf)[5],(*vf)[9],(*vf)[10],p[i],int(i));
    }
  }
  
  free(pp);
  return;
}


// calculate sky error regions
void network::getSkyArea(size_t id, size_t lag, double To) {
  int in,im,IN,IM;
  size_t i,j,l,m,k,K;
  size_t N = this->wc_List[lag].csize();
  size_t L = this->skyProb.size();
  size_t Lm = L-int(0.9999*L); 
  skymap* sm = &(this->nSkyStat);

  if(Lm < 2) return; 
  if(nSky > long(L-Lm)) nSky = L-Lm;
  if(id>N) return;

  double a,th,ph,st,sp;
  double TH,PH,ST,SP,Eo;
  double sum = 0.;
  double vol = 0.;
  double co1 = cos(sm->theta_1*PI/180.);
  double co2 = cos(sm->theta_2*PI/180.);
  double phi = sm->phi_2-sm->phi_1;
  double s = fabs(phi*(co1-co2))*180/PI/sm->size();  // sky solid angle

  std::vector<float>* vf = &(this->wc_List[lag].sArea[id-1]);  
  size_t v[11]; 

  double* p  = this->skyProb.data;
  double** pp = (double **)malloc(L*sizeof(double*));
  for(l=0; l<L; l++) pp[l] = p + l;

  skyProb.waveSplit(pp,0,L-1,Lm);
  skyProb.waveSort(pp,Lm,L-1);

  Eo = *pp[L-1];                                    // max L
  for(l=L-1; l>=Lm; l--) *pp[l] -= Eo;
  for(l=0; l<Lm; l++) *pp[l] = -1.e10;

// correction for sky segmentation

  if(sm->getOrder()==0) {                               // is disabled for HEALPix skymap 
    for(l=Lm; l<L; l++){                
      j = pp[l]-p;                                      // reference sky index
      th = sm->getTheta(j);
      ph = sm->getPhi(j);
      st = sm->getThetaStep(j);
    
      for(in=-1; in<2; in++) {
        sp = sm->getPhiStep(getIndex(th+in*st,ph));
        for(im=-1; im<2; im++) {
      	  i = this->getIndex(th+in*st,ph+im*sp);        // neighbour sky index
	  if(p[i] >= p[j] || p[i] < -1.e9) continue;

	  TH = sm->getTheta(i);
	  PH = sm->getPhi(i);
	  ST = sm->getThetaStep(i);
	  m = 0; a = 0.;
	  for(IN=-1; IN<2; IN++) {
	  SP = sm->getPhiStep(getIndex(TH+IN*ST,PH));
	    for(IM=-1; IM<2; IM++) {
	      k = this->getIndex(TH+IN*ST,PH+IM*SP);    // neighbour sky index
	      if(p[i] >=p[k]) continue;
	      m++; a += p[k];
	    }
	  }
  	  if(m>3) p[i] = a/m;
        }
      }
    }
    skyProb.waveSplit(pp,0,L-1,Lm);
    skyProb.waveSort(pp,Lm,L-1);
  }

  skyENRG = 0.;
  Eo = *pp[L-2]*0.9;                                    // max L
  sum = vol = 0.;
  for(l=L-2; l>=Lm; l--) {  
    a = (Eo - *pp[l]);
    skyENRG.data[L-l-1] = a;
    if(a < pSigma || vol<1.) {
      sum += a/(L-l-1);
      vol += 1;
    }
    *pp[l] -= Eo;
  }

  if(pOUT) cout<<sum/vol<<" "<<*pp[L-2]<<" "<<*pp[L-3]<<" "<<*pp[0]<<"\n";

  Eo = sum/vol/2.;
  *pp[0] = exp(-30.);
  for(l=L-1; l>0; l--) {
    a = l>Lm ? (L-l-1)*Eo : 30.;
    if(a > 30.) a = 30.;
    *pp[l] = exp(-a);             // calculate skyProb map
  }

  if(pOUT) cout<<"norm: "<<(skyProb.mean()*L)<<endl;
  skyProb *= 1./(skyProb.mean()*L); 
  for(l=0; l<L; l++) nProbability.set(l,log10(p[l]));        // fill in skyProb map

  vf->clear();
  for(m=0; m<11; m++) { v[m] = 0; vf->push_back(0.); }

  sum = 0.;
  for(l=L-1; l>Lm; l--){
    for(m=size_t(sum*10.)+1; m<10; m++) v[m] += 1;
    sum += *pp[l];
    if(sum >= 0.9) break;
  }
  
  for(m=1; m<10; m++) (*vf)[m] = sqrt(v[m]*s); 
  
// fill skyProb skymap 
  
  std::vector<float>* vP = &(this->wc_List[lag].p_Map[id-1]);
  std::vector<int>*   vI = &(this->wc_List[lag].p_Ind[id-1]);
  
  K = 0;
  sum = 0.;
  vP->clear();
  vI->clear();
  double pthr=0;
  // if nSky -> nSky is converted into a probability threshold nSky=-XYZ... -> pthr=0.XYZ... 
  if(nSky<0) {char spthr[1024];sprintf(spthr,"0.%d",int(abs(nSky)));pthr=atof(spthr);}
  for(l=L-1; l>Lm; l--){
    sum += *pp[l]; 
    if(nSky==0 && (K==1000 || sum > 0.99) && K>0) break; 
    else if(nSky<0 && sum > pthr && K>0) break; 
    else if(nSky>0 && K==nSky && K>0) break;
    K++;  
    vI->push_back(int(pp[l]-p));
    vP->push_back(float(*pp[l]));
  } 

// set injections if there are any

  size_t M = this->mdc__IDSize();
  
  if(!M) { free(pp); return; }
  
  double dT = 1.e13;
  double injTime = 1.e12;
  int injID = -1;
  int mdcID = -1;
  injection INJ(this->ifoList.size());

  for(m=0; m<M; m++) {
    mdcID = this->getmdc__ID(m);
    dT = fabs(To - this->getmdcTime(mdcID));
    if(dT<injTime && INJ.fill_in(this,mdcID)) { 
      injTime = dT; 
      injID = mdcID; 
      if(pOUT) printf("getSkyArea: %4d %12.4f %7.3f %f \n",int(m),To,dT,s);
    } 
  }
 
  if(INJ.fill_in(this,injID)) {

    th = INJ.theta[0];
    ph = INJ.phi[0]; 
    i  = this->getIndex(th,ph); 

    vI->push_back(int(i));
    vP->push_back(float(p[i]));

    vol = sum = 0.;
    for(l=L-1; l>Lm; l--){
      vol += s;
      sum += *pp[l];
      if(pp[l]-p == int(i)) break;
    }
    (*vf)[0]  = sqrt(vol);
    (*vf)[10] = sum;
    j = pp[L-1]-p;                                    // reference sky index at max

    if(pOUT) {
      printf("getSkyArea: %5d %12.4f %6.1f %6.1f %6.1f %6.1f %6.2f %6.2f %6.2f %7.5f, %e %d \n",
	     int(id),INJ.time[0]-this->getifo(0)->TFmap.start(),INJ.theta[0],INJ.phi[0],
	     sm->getTheta(j),sm->getPhi(j),(*vf)[0],(*vf)[5],(*vf)[9],(*vf)[10],p[i],int(i));
    }
  }
  
  free(pp);
  return;
}

//**************************************************************************
// calculate network likelihood for constructed clusters: 2-NIFO detectors
// It is designed to replace likelihood functions for 2,3,4 and NIFO networks. 
// If the network has less then NIFO detectors, all arrays are allocated for NIFO 
// detectors anyway. The arrays are initialized so that the dummy detectors
// do not contribute into calculation of the coherent statistics.
// Both general and elliptical constraint can be executed 
//**************************************************************************
long network::likelihood(char type, double Ao, int ID, size_t lag, int ind, bool core)
{
  size_t nIFO = this->ifoList.size();
  this->tYPe = type;
  this->acor = Ao;
  if(nIFO>NIFO || !this->filter.size() || this->filter[0].index.size()!=32) {
    cout<<"network::likelihood(): invalid number of detectors or delay filter is not set.\n";
    return false;
  }
  if(type=='b' || type=='B' || type=='E')         
    return likelihoodB(type, Ao, ID, lag, ind, core);
  else 
    return likelihoodI(type, Ao, ID, lag, ind, core);
}


//**************************************************************************
// calculate network likelihood for constructed clusters: 2-NIFO detectors
// It is a new implementation for likelihood functions for networks of 2,3,4,NIFO 
// detectors with improved constraints. If the network has less then NIFO detectors, 
// all arrays are allocated for NIFO detectors anyway. The arrays are initialized 
// so that the dummy detectors do not contribute into calculation of the coherent 
// statistics. 
//**************************************************************************
long network::likelihoodB(char type, double Ao, int iID, size_t lag, int ind, bool core)
{
  this->like('B'); 

  size_t nIFO = this->ifoList.size();
  size_t ID = abs(iID);
  int   N_1 = nIFO>2 ? int(nIFO)-1 : 2;
  int   N_2 = nIFO>2 ? int(nIFO)-2 : 1;
  if(nIFO>NIFO || !this->filter.size() || this->filter[0].index.size()!=32) {
    cout<<"network::likelihood(): invalid number of detectors or delay filter is not set.\n";
    return false;
  }

// regulators hard <- soft <0> weak -> hard
//   gamma =    -1 <-       0       -> 1
//   delta =     0  ->  1

  double Eo    = nIFO*Ao*Ao;                    // energy threshold in the sky loop
  double soft  = delta>0 ? delta : 0.;
  double GAMMA = 1.-gamma*gamma;                // network regulator
  
  int  LOCAL = local ? 1 : 0;                   // ED minimization case
  double rho = this->netRHO*this->netRHO*nIFO;  // threshold on rho

  double gr,gp,gx,gR,gI,gc,gg,gP,gX,T,rm,Le,E,LPm;
  double STAT,P,EE,Et,Lo,Lm,hh,Xp,Xx,XX,Lp,Em,Ep;
  double S,co,si,cc,em,um,vm,uc,us,vc,vs,Co,Cp;
  NETX(double c0;,double c1;,double c2;,double c3;,double c4;,double c5;,double c6;,double c7;)
  double inet;  

  if(type=='E' && Eo<nIFO) Eo = double(nIFO);  
  if(!ID && ind>=0) ind = -1;  
  this->tYPe = type;
  this->acor = Ao;
  
  int    ii,II,IIm;
  size_t i,j,k,l,m,n,V,U,K,id;
  size_t I = this->ifoList[0]->TFmap.maxLayer()+1;
  size_t M = this->filter.size()/I;              // total number of delays 
  size_t L = ind<0 ? this->index.size() : ind+1; // total number of source locations 
  int    R = int(this->ifoList[0]->getTFmap()->rate()/I+0.5);

// pointers to data
  double* pdata[NIFO];
  double* qdata[NIFO];
  double* pq;
  for(n=0; n<nIFO; n++) {
    pdata[n] = getifo(n)->getTFmap()->data;
    qdata[n] = getifo(n)->getTFmap()->data;
  }

// buffer for wavelet layer delay filter
  std::vector<float>* F;
  std::vector<short>* J;
  
// get antenna patterns and index arrays
  double*  fp[NIFO];                     // f+ 
  double*  fx[NIFO];                     // fx
  double* ffp[NIFO];                     // f+f+ + fxfx 
  double* ffm[NIFO];                     // f+f+ - fxfx
  double* fpx[NIFO];                     // 2*f+fx
  wavearray<double> f00(L); f00 = 0.; // dummy zero array

  for(n=0; n<NIFO; n++) {
     fp[n] = n<nIFO ? getifo(n)->fp.data  : f00.data;
     fx[n] = n<nIFO ? getifo(n)->fx.data  : f00.data;
    ffp[n] = n<nIFO ? getifo(n)->ffp.data : f00.data;
    ffm[n] = n<nIFO ? getifo(n)->ffm.data : f00.data;
    fpx[n] = n<nIFO ? getifo(n)->fpx.data : f00.data;
  }

  short* mm = this->skyMask.data;
  NETX( 
  short* m0 = getifo(0)->index.data;                                  ,
  short* m1 = nIFO>1 ? getifo(1)->index.data : getifo(1)->index.data; ,
  short* m2 = nIFO>2 ? getifo(2)->index.data : getifo(1)->index.data; ,
  short* m3 = nIFO>3 ? getifo(3)->index.data : getifo(1)->index.data; ,
  short* m4 = nIFO>4 ? getifo(4)->index.data : getifo(1)->index.data; ,
  short* m5 = nIFO>5 ? getifo(5)->index.data : getifo(1)->index.data; ,
  short* m6 = nIFO>6 ? getifo(6)->index.data : getifo(1)->index.data; ,
  short* m7 = nIFO>7 ? getifo(7)->index.data : getifo(1)->index.data; )

// allocate buffers
  std::vector<size_t> pI;

  wavearray<double> aS[NIFO];       // single whitened amplitude
  wavearray<double> eS[NIFO];       // energy SNR
  wavearray<double> NV[NIFO];       // noise variance
  wavearray<double> NR[NIFO];       // noise rms
  wavearray<double> cid;         // buffers for cluster ID
  wavearray<double> cTo;         // buffers for cluster time
  wavearray<double> vol;         // buffers for cluster volume
  wavearray<double> xi(500);     // buffers for detector responses
  wavearray<short>  jU(2500);    // buffers for likelihood normalization factor
  wavearray<double> Fplus(2500); // buffers for F+
  wavearray<double> Fcros(2500); // buffers for Fx

  double* px;
  double* nv[NIFO];
  double* nr[NIFO];
  double* pe[NIFO];
  double* pa[NIFO];
  double  pp[NIFO];
  double  qq[NIFO];
  double  am[NIFO];
  double  Fp[NIFO];
  double  Fx[NIFO];
  double  ee[NIFO];
  double  rr[NIFO];
  double   e[NIFO];
  double   u[NIFO];
  double   v[NIFO];
  double   r[NIFO];

  U = 500;
  for(n=0; n<NIFO; n++) {
    NV[n].resize(U);   NV[n] = 0.; nv[n] = NV[n].data; 
    NR[n].resize(U);   NR[n] = 0.; nr[n] = NR[n].data; 
    aS[n].resize(U*M); aS[n] = 0.; 
    eS[n].resize(U*M); eS[n] = 0.;
    rr[n] = 1.;
  }

  netpixel* pix;
  std::vector<int>* vint;
  std::vector<int>* vtof;

  bool    skip;
  double  logbpp;
  size_t  count = 0;

  S = 0.;
  if(ID) { 
    this->pixeLHood = getifo(0)->TFmap; 
    this->pixeLHood = 0.; 
    this->pixeLNull = getifo(0)->TFmap; 
    this->pixeLNull = 0.; 
    this->nSensitivity = 0.;
    this->nAlignment = 0.;	
    this->nNetIndex = 0.;		 
    this->nDisbalance = 0.;		 
    this->nLikelihood = 0.;		 
    this->nNullEnergy = 0.;		 
    this->nCorrEnergy = 0.;		 
    this->nCorrelation = 0.;
    this->nSkyStat = 0.;	 
    this->nPenalty = 0.;	 
    this->nProbability = 0.;	 
    this->nAntenaPrior = 0.;	 
  }

//+++++++++++++++++++++++++++++++++++++++
// liklihood calculation for clusters
//+++++++++++++++++++++++++++++++++++++++

  for(n=lag; n<nLag; n++) {                  // loop over time shifts 

     if(!this->wc_List[n].size()) continue;
     logbpp = -log(this->wc_List[n].getbpp());

     cid = this->wc_List[n].get((char*)"ID",0,'S',optim ? R : -R);   // get cluster ID
     cTo = this->wc_List[n].get((char*)"time",0,'L',optim ? R : -R); // get cluster time

     K = cid.size();

     for(k=0; k<K; k++) {      // loop over clusters 

	id = size_t(cid.data[k]+0.1);

	if(ID && id!=ID) continue;

	vint = &(this->wc_List[n].cList[id-1]);           // pixel list
	vtof = &(this->wc_List[n].nTofF[id-1]);           // TofFlight configurations

	V = vint->size();
	if(!V) continue;

	pI.clear();

	for(j=0; j<V; j++) {   // loop over pixels 

	  pix = this->wc_List[n].getPixel(id,j);

	  if(!pix) {
	    cout<<"network::likelihood() error: NULL pointer"<<endl;
	    exit(1);
	  }

	  if(R != int(pix->rate+0.5)) continue;           // check rate 
	  if(!pix->core && core) continue;                // check core flag

	  pI.push_back(j);                                // save pixel index

	}

	V = pI.size();
	if(!V) continue;
	
	if(NV[0].size() < V) {                            // reallocate arrays           
	  U = V+100;
	  for(i=0; i<NIFO; i++) {
	    NV[i].resize(U);   NV[i] = 0.; nv[i] = NV[i].data;
	    NR[i].resize(U);   NR[i] = 0.; nr[i] = NR[i].data;
	    eS[i].resize(U*M); eS[i] = 0.; 
	    aS[i].resize(U*M); aS[i] = 0.; 
	  }
	  jU.resize(U); xi.resize(U*NIFO); 
	  Fplus.resize(U*NIFO); Fcros.resize(U*NIFO);
	}

	for(j=0; j<V; j++) {   // loop over selected pixels 

	  pix = this->wc_List[n].getPixel(id,pI[j]);

	  cc = 0.;
	  for(i=0; i<nIFO; i++) {
	    ee[i] = 1./pix->data[i].noiserms;
	    cc += ee[i]*ee[i];                          // total inverse variance
	  }

	  for(i=0; i<nIFO; i++) {
	    nv[i][j] = ee[i]*ee[i];                     // inverse variance
	    nr[i][j] = ee[i]/sqrt(cc);                  // normalized 1/rms
	    qdata[i] = pdata[i] + pix->data[i].index;   // pointer to data   
	  }

// apply delay filter
	   
	  for(i=0; i<nIFO; i++) {
	    pq = qdata[i];

	    for(m=0; m<M; m++){          // loop over delays                   
	      F = &(filter[pix->frequency*M+m].value); 
	      J = &(filter[pix->frequency*M+m].index);
	      l = m*V + j; 

	      gg = dot32(F,pq,J);        // apply filter 

	      eS[i].data[l] = gg*gg;
	      aS[i].data[l] = gg;
	    }
	  }
	}

        STAT = -1.e64; m=IIm=0; Lm=Em=LPm= 0.;

// Max Energy

	if(type == 'E') {

	  skip = true;
	  l = ind<0 ? 0 : ind;              // process selected index
	  for(; l<L; l++) {	            // loop over sky locations
	    if(!mm[l]) continue;            // skip identical delay configurations

	    NETX( 
            pe[0] = eS[0].data + m0[l]*V; ,
	    pe[1] = eS[1].data + m1[l]*V; ,
	    pe[2] = eS[2].data + m2[l]*V; ,
	    pe[3] = eS[3].data + m3[l]*V; ,
	    pe[4] = eS[4].data + m4[l]*V; ,
	    pe[5] = eS[5].data + m5[l]*V; ,
	    pe[6] = eS[6].data + m6[l]*V; ,
	    pe[7] = eS[7].data + m7[l]*V; )

	    NETX(c0=0.;,c1=0.;,c2=0.;,c3=0.;,c4=0.;,c5=0.;,c6=0.;,c7=0.;) 
            i=0;  
	    for(j=0; j<V; j++) {            // loop over selected pixels 
              double pet = 0.;
              NETX( 
              pet+=pe[0][j]; ,
              pet+=pe[1][j]; ,
              pet+=pe[2][j]; ,
              pet+=pe[3][j]; ,
              pet+=pe[4][j]; ,
              pet+=pe[5][j]; ,
              pet+=pe[6][j]; ,
              pet+=pe[7][j]; )
	      if(pet > Eo) { 
		NETX( 
                c0 += pe[0][j]; ,
		c1 += pe[1][j]; ,
		c2 += pe[2][j]; ,
		c3 += pe[3][j]; ,
		c4 += pe[4][j]; ,
		c5 += pe[5][j]; ,
		c6 += pe[6][j]; ,
		c7 += pe[7][j]; )
		i++;
	      }
	    }
	   
	    E = 0.; NETX(E+=c0;,E+=c1;,E+=c2;,E+=c3;,E+=c4;,E+=c5;,E+=c6;,E+=c7;) E-=i*nIFO; // correction for dummy detectors  
	    if(E>STAT) { m = l; STAT = E; Lm = E; }  // maximize energy
 	    E += i;
	    if(NETX(E-c0>e2or &&,E-c1>e2or &&,E-c2>e2or &&,E-c3>e2or &&,E-c4>e2or &&,E-c5>e2or &&,E-c6>e2or &&,E-c7>e2or  &&) true) skip = false; 
	  }
	  if(skip) { this->wc_List[n].ignore(id); continue; }
	}

// constraint: total (XkSk-ekSkSk)/Ek = 0 

	else {
	  
	  l = ind<0 ? 0 : ind;                 // process selected index
	  for(; l<L; l++) {	               // loop over sky locations
	    
	    skyProb.data[l] = 0.;

            if(skyMaskCC.size()==L) {
              // transform l in celestial coordinates cc_l and check the skyMaskCC
              skymap* sm = &(this->nSkyStat);
              double th = sm->getTheta(l);
              double ph = sm->getPhi(l);
              double gpsT = cTo.data[k]+getifo(0)->TFmap.start();          // trigger time
              double ra = sm->phi2RA(ph,  gpsT);
              int cc_l = this->getIndex(th,ra);
              if (skyMaskCC.data[cc_l]<=0) continue;  
            }

	    if(!mm[l] && ind<0) continue;      // skip sky configurations

	    NETX( 
            pa[0] = aS[0].data + m0[l]*V; ,     // single whitening 
	    pa[1] = aS[1].data + m1[l]*V; ,
	    pa[2] = aS[2].data + m2[l]*V; ,
	    pa[3] = aS[3].data + m3[l]*V; ,
	    pa[4] = aS[4].data + m4[l]*V; ,
	    pa[5] = aS[5].data + m5[l]*V; ,
	    pa[6] = aS[6].data + m6[l]*V; ,
	    pa[7] = aS[7].data + m7[l]*V; )

// weak constraint 
//  transformation to PCF, 
//  u = [X x (f+ x fx)] x (f+ x fx)
//  u/(|u|*|X|) is a unity vector along the projection of X
//  u = Fp*uc + Fx*us
	      		      			       
	    
	    EE=Lo=Co=Ep=Lp=Cp=inet = 0.; II=0;
	    
	    for(j=0; j<V; j++) {               // loop over selected pixels 

	      Et = dotx(pa,j,pa,j);	       // total energy     
	      ii = int(Et>Eo);
	      EE+= Et*ii;

	      mulx(fp,l,nr,j,Fp);
	      mulx(fx,l,nr,j,Fx);

	      gp = dotx(Fp,Fp)+1.e-12;         // fp^2
	      gx = dotx(Fx,Fx)+1.e-12;         // fx^2
	      gI = dotx(Fp,Fx);                // fp*fx
	      Xp = dotx(Fp,pa,j);              // (X*f+)
	      Xx = dotx(Fx,pa,j);              // (X*fx)       
	      uc = Xp*gx - Xx*gI;              // u cos of rotation to PCF
	      us = Xx*gp - Xp*gI;              // u sin of rotation to PCF
	      um = rotx(Fp,uc,Fx,us,u);        // calculate u and return its norm
	      hh = dotx(u,pa,j,e);             // (u*X) - not normalized         
	      Le = hh*hh/um;
	      cc = Le-dotx(e,e)/um;
	      ii*= int(cc>0);

	      Lp+= Le*cc*ii/(Et-Le+1+cc);     // effective likelihood	      
	      Ep+= Et*ii;
	      Lo+= Le*ii;                      // baseline Likelihood
	      Co+= cc*ii;                      // coherent energy
	      II+= ii;
	      mulx(u,hh*ii/um,xi.data+j*NIFO);

	    }

	    cc = Ep>0 ? Co/(Ep-Lo+II+fabs(Co)) : 0.;
	    if(type=='B') Lp = Lo*cc; 

	    skyProb.data[l] = Lp/EE;

	    if(Lp>LPm) { LPm=Lp; Em=EE; IIm=II; }

	    cc = EE>0 ? Co/(EE-Lo+II+fabs(Co)) : 0.;
	    if(cc<this->netCC || cc*Co<rho) continue;

//  weak and hard regulators used for estimation of u 
//  v = u x (f+ x fx) is a vector perpendicular to u
//  u = Fp*uc + Fx*us, v = Fx*vc - Fp*vs

	    Lo=Co=gP=gX = 0.;

	    for(j=0; j<V; j++) {                // loop over selected pixels 

	      NETX( 
              am[0] = pa[0][j]; ,
	      am[1] = pa[1][j]; ,
	      am[2] = pa[2][j]; ,
	      am[3] = pa[3][j]; ,
	      am[4] = pa[4][j]; ,
	      am[5] = pa[5][j]; ,
	      am[6] = pa[6][j]; ,
	      am[7] = pa[7][j]; )

	      px = xi.data+j*NIFO;
	      if(dotx(px,px)<=0) continue;
	      Et = dotx(am,am);

	      mulx(fp,l,nr,j,Fp);
	      mulx(fx,l,nr,j,Fx);

	      gp = dotx(Fp,Fp)+1.e-12;          // fp^2
	      gx = dotx(Fx,Fx)+1.e-12;          // fx^2
	      gI = dotx(Fp,Fx);                 // fp*fx
	      gr = (gp+gx)/2.;                  // sensitivity
	      gR = (gp-gx)/2.;                  // real part of gc
	      gc = sqrt(gR*gR+gI*gI);           // norm of complex antenna pattern

	      Xp = dotx(Fp,am);                 // (X*f+)
	      Xx = dotx(Fx,am);                 // (X*fx)       
	      uc = Xp*gx - Xx*gI;               // u cos of rotation to PCF
	      us = Xx*gp - Xp*gI;               // u sin of rotation to PCF
	      vc = gp*uc + gI*us;               // (u*f+)/|u|^2 - 'cos' for v
	      vs = gx*us + gI*uc;               // (u*fx)/|u|^2 - 'sin' for v
	      um = rotx(Fp,uc,Fx,us,u);         // calculate new response vector
	      vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24; // calculate complementary vector

// regulator

	      ii = netx(u,um,v,vm,GAMMA);        // network index
	      inet += ii*Et;
	      if(ii<N_2 && gamma<0) continue;    // superclean selection cut

	      gP+= (gr+gc)*Et;                   // + sensitivity
	      gX+= (gr-gc)*Et;                   // x sensitivity
	      gg = (gp+gx)*soft;
	      uc = Xp*(gx+gg) - Xx*gI;           // u cos of rotation to PCF
	      us = Xx*(gp+gg) - Xp*gI;           // u sin of rotation to PCF

	      if(ii<N_1 && gamma!=0) { 
		uc = Xp*(gc+gR)+Xx*gI;
		us = Xx*(gc-gR)+Xp*gI;
	      }

	      vc = gp*uc + gI*us;                // (u*f+)/|u|^2 - 'cos' for v
	      vs = gx*us + gI*uc;                // (u*fx)/|u|^2 - 'sin' for v
	      um = rotx(Fp,uc,Fx,us,u);          // calculate new response vector
	      vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;  // calculate complementary vector

// energy disbalance ratio vectors

	      NETX ( 
              rr[0] = LOCAL*am[0]/(am[0]*am[0]+2.)+1-LOCAL; ,
	      rr[1] = LOCAL*am[1]/(am[1]*am[1]+2.)+1-LOCAL; ,
	      rr[2] = LOCAL*am[2]/(am[2]*am[2]+2.)+1-LOCAL; ,
	      rr[3] = LOCAL*am[3]/(am[3]*am[3]+2.)+1-LOCAL; ,
	      rr[4] = LOCAL*am[4]/(am[4]*am[4]+2.)+1-LOCAL; ,
	      rr[5] = LOCAL*am[5]/(am[5]*am[5]+2.)+1-LOCAL; ,
	      rr[6] = LOCAL*am[6]/(am[6]*am[6]+2.)+1-LOCAL; ,
	      rr[7] = LOCAL*am[7]/(am[7]*am[7]+2.)+1-LOCAL; )

	      hh = dotx(u,am)/um;
              NETX (
	      pp[0] = rr[0]*(am[0]-hh*u[0])*u[0]; ,
	      pp[1] = rr[1]*(am[1]-hh*u[1])*u[1]; ,
	      pp[2] = rr[2]*(am[2]-hh*u[2])*u[2]; ,
	      pp[3] = rr[3]*(am[3]-hh*u[3])*u[3]; ,
	      pp[4] = rr[4]*(am[4]-hh*u[4])*u[4]; ,
	      pp[5] = rr[5]*(am[5]-hh*u[5])*u[5]; ,
	      pp[6] = rr[6]*(am[6]-hh*u[6])*u[6]; ,
	      pp[7] = rr[7]*(am[7]-hh*u[7])*u[7]; )
		
	      gg = dotx(v,am)/um;
	      hh*= 2.;
              NETX (
	      qq[0] = rr[0]*((hh*u[0]-am[0])*v[0]+u[0]*u[0]*gg); ,
	      qq[1] = rr[1]*((hh*u[1]-am[1])*v[1]+u[1]*u[1]*gg); ,
	      qq[2] = rr[2]*((hh*u[2]-am[2])*v[2]+u[2]*u[2]*gg); ,
	      qq[3] = rr[3]*((hh*u[3]-am[3])*v[3]+u[3]*u[3]*gg); ,
	      qq[4] = rr[4]*((hh*u[4]-am[4])*v[4]+u[4]*u[4]*gg); ,
	      qq[5] = rr[5]*((hh*u[5]-am[5])*v[5]+u[5]*u[5]*gg); ,
	      qq[6] = rr[6]*((hh*u[6]-am[6])*v[6]+u[6]*u[6]*gg); ,
	      qq[7] = rr[7]*((hh*u[7]-am[7])*v[7]+u[7]*u[7]*gg); )

	      co = dotx(qq,qq)/vm+dotx(pp,pp)/um+1.e-24;     // cos term
	      si = dotx(pp,qq);                              // sin term
              if(!eDisbalance) {co=1.;si=0.;}  
	      em = rotx(u,co,v,si/vm,e);                     // calculate rotated vector e

// second iteration
	      
	      rm = rotx(v,co,u,-si/um,r)+1.e-24;      // calculate rotated vector v
	      hh = dotx(e,am)/em;
	     
              NETX ( 
	      pp[0] = rr[0]*(am[0]-hh*e[0])*e[0]; ,
	      pp[1] = rr[1]*(am[1]-hh*e[1])*e[1]; ,
	      pp[2] = rr[2]*(am[2]-hh*e[2])*e[2]; ,
	      pp[3] = rr[3]*(am[3]-hh*e[3])*e[3]; ,
	      pp[4] = rr[4]*(am[4]-hh*e[4])*e[4]; ,
	      pp[5] = rr[5]*(am[5]-hh*e[5])*e[5]; ,
	      pp[6] = rr[6]*(am[6]-hh*e[6])*e[6]; ,
	      pp[7] = rr[7]*(am[7]-hh*e[7])*e[7]; )
	      
	      gg = dotx(r,am)/em;
	      hh*= 2.;
              NETX (
	      qq[0] = rr[0]*((hh*e[0]-am[0])*r[0]+e[0]*e[0]*gg); ,
	      qq[1] = rr[1]*((hh*e[1]-am[1])*r[1]+e[1]*e[1]*gg); ,
	      qq[2] = rr[2]*((hh*e[2]-am[2])*r[2]+e[2]*e[2]*gg); ,
	      qq[3] = rr[3]*((hh*e[3]-am[3])*r[3]+e[3]*e[3]*gg); ,
	      qq[4] = rr[4]*((hh*e[4]-am[4])*r[4]+e[4]*e[4]*gg); ,
	      qq[5] = rr[5]*((hh*e[5]-am[5])*r[5]+e[5]*e[5]*gg); ,
	      qq[6] = rr[6]*((hh*e[6]-am[6])*r[6]+e[6]*e[6]*gg); ,
	      qq[7] = rr[7]*((hh*e[7]-am[7])*r[7]+e[7]*e[7]*gg); )
	      
	      co = dotx(qq,qq)/rm+dotx(pp,pp)/em+1.e-24;     // cos term
	      si = dotx(pp,qq);                              // sin term
              if(!eDisbalance) {co=1.;si=0.;}  
	      em = rotx(e,co,r,si/rm,e);                     // calculate ED vector
	      hh = dotx(e,am,ee);                            // ee[i] = e[i]*am[i]
	      Lo+= hh*hh/em;                                 // corrected L
	      Co+= (hh*hh-dotx(ee,ee))/em;                   // coherent energy		

	    }

// <x*xi> penalty factor and asymmetry

            NETX(pp[0]=0.;,pp[1]=0.;,pp[2]=0.;,pp[3]=0.;,pp[4]=0.;,pp[5]=0.;,pp[6]=0.;,pp[7]=0.;)
            NETX(qq[0]=0.001;,qq[1]=0.001;,qq[2]=0.001;,qq[3]=0.001;,qq[4]=0.001;,qq[5]=0.001;,qq[6]=0.001;,qq[7]=0.001;)
            NETX(ee[0]=0.;,ee[1]=0.;,ee[2]=0.;,ee[3]=0.;,ee[4]=0.;,ee[5]=0.;,ee[6]=0.;,ee[7]=0.;) 

	    for(j=0; j<V; j++) {                     // loop over selected pixels 	      
	      px = xi.data+j*NIFO;
	      addx(px,px,qq);
	      addx(px,pa,j,pp);
	      addx(pa,j,pa,j,ee);
	    }
	    
	    S = 0.;
            NETX (
            S+= fabs(pp[0]-qq[0]); ,
	    S+= fabs(pp[1]-qq[1]); ,
	    S+= fabs(pp[2]-qq[2]); ,
	    S+= fabs(pp[3]-qq[3]); ,
	    S+= fabs(pp[4]-qq[4]); ,
	    S+= fabs(pp[5]-qq[5]); ,
	    S+= fabs(pp[6]-qq[6]); ,
	    S+= fabs(pp[7]-qq[7]); )

	    NETX ( 
	    pp[0] /= sqrt(qq[0]); ,
	    pp[1] /= sqrt(qq[1]); ,
	    pp[2] /= sqrt(qq[2]); ,
	    pp[3] /= sqrt(qq[3]); ,
	    pp[4] /= sqrt(qq[4]); ,
	    pp[5] /= sqrt(qq[5]); ,
	    pp[6] /= sqrt(qq[6]); ,
	    pp[7] /= sqrt(qq[7]); )
	    
	    hh = 0.;
            NETX (
            hh+= pp[0]; ,
            hh+= pp[1]; ,
            hh+= pp[2]; ,
            hh+= pp[3]; ,
            hh+= pp[4]; ,
            hh+= pp[5]; ,
            hh+= pp[6]; ,
            hh+= pp[7]; )
	    gg = 0.;
            NETX (
            gg+= sqrt(ee[0]); ,
            gg+= sqrt(ee[1]); ,
            gg+= sqrt(ee[2]); ,
            gg+= sqrt(ee[3]); ,
            gg+= sqrt(ee[4]); ,
            gg+= sqrt(ee[5]); ,
            gg+= sqrt(ee[6]); ,
            gg+= sqrt(ee[7]); )

	    P  = hh/gg;                                     // Pearson's correlation penalty
	    cc = Co/(EE-Lo+fabs(Co));                       // network correlation coefficient
	    XX = Lo*cc/EE;                                  // sky statistic

	    skyProb.data[l] *= P;

	    if(XX>=STAT) { m=l; STAT=XX; Lm=Lo; }
	    
	    if(ID) {                            // fill in skymaps
	      this->nAntenaPrior.set(l, gP/EE);
	      this->nSensitivity.set(l, gP/EE);
	      this->nAlignment.set(l, gX/EE);	
	      this->nNetIndex.set(l, inet/EE);		 
	      this->nDisbalance.set(l, S);		 
	      this->nLikelihood.set(l, Lo);		 
	      this->nNullEnergy.set(l, (EE-Lo));		 
	      this->nCorrEnergy.set(l, Co);		 
	      this->nCorrelation.set(l, cc);
	      this->nSkyStat.set(l, XX);		 
	      this->nPenalty.set(l, P);		 
	      this->nProbability.set(l,skyProb.data[l]);		 
	    }	    
	  }

	  if(STAT<=0.001 && ind<0) {
	    this->wc_List[n].ignore(id);  // reject cluster
	    continue;
	  }
	}
	
// final calculation of likelihood for selected sky location
	
	Lo = E = 0.;
	l = ind<0 ? m : ind;

        NETX (
	pa[0] = aS[0].data + m0[l]*V; ,    // single whitening 
	pa[1] = aS[1].data + m1[l]*V; ,
	pa[2] = aS[2].data + m2[l]*V; ,
	pa[3] = aS[3].data + m3[l]*V; ,
	pa[4] = aS[4].data + m4[l]*V; ,
	pa[5] = aS[5].data + m5[l]*V; ,
	pa[6] = aS[6].data + m6[l]*V; ,
	pa[7] = aS[7].data + m7[l]*V; )
	    
// initialize detector energy array

	for(j=0; j<V; j++) {               // loop over selected pixels 

	  Et = 0.; U = 1;
	  for(i=0; i<nIFO; i++) {
	    am[i] = pa[i][j];
	    Fp[i] = fp[i][l]*nr[i][j];
	    Fx[i] = fx[i][l]*nr[i][j];
	    Et += am[i]*am[i];
	    if(!j) ee[i] = 0.;
	  }

	  gp = dotx(Fp,Fp)+1.e-12;          // (fp^2)
	  gx = dotx(Fx,Fx)+1.e-12;          // (fx^2)
	  gI = dotx(Fp,Fx);                 // (fp*fx)
	  Xp = dotx(Fp,am);                 // (X*f+)
	  Xx = dotx(Fx,am);                 // (X*fx)       
	  gR = (gp-gx)/2.;
	  gc = sqrt(gR*gR+gI*gI);           // norm of complex antenna pattern
	  uc = Xp*gx - Xx*gI;               // u cos
	  us = Xx*gp - Xp*gI;               // u sin
	  vc = gp*uc + gI*us;               // (u*f+)/|u|^2 - 'cos' for v
	  vs = gx*us + gI*uc;               // (u*fx)/|u|^2 - 'sin' for v
	  um = rotx(Fp,uc,Fx,us,u);         // calculate u and return its norm
	  vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24; // calculate v and return its norm
	  hh = dotx(u,am,e);                // GW amplitude

	  if((hh*hh-dotx(e,e))/um<=0.) U=0;

// regulator

	  ii = 0;
	  for(i=0; i<nIFO; i++) {
	    if(u[i]*u[i]/um > 1-GAMMA) ii++;
	    if(u[i]*u[i]/um+v[i]*v[i]/vm > GAMMA) ii--;
	  }

	  if(ii<N_2 && gamma<0.) U = 0;

	  gg = (gp+gx)*soft;
	  uc = Xp*(gx+gg) - Xx*gI;              // u cos of rotation to PCF
	  us = Xx*(gp+gg) - Xp*gI;              // u sin of rotation to PCF

	  if(ii<N_1 && gamma!=0) { 
	    uc = Xp*(gc+gR)+Xx*gI;
	    us = Xx*(gc-gR)+Xp*gI;
	  }

	  vc = (gp*uc + gI*us);                 // (u*f+)/|u|^2 - 'cos' for v
	  vs = (gx*us + gI*uc);                 // (u*fx)/|u|^2 - 'sin' for v
	  um = rotx(Fp,uc,Fx,us,u);             // calculate u and return its norm
	  vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;     // calculate v and return its norm

// normalize u and v vectors

	  um = sqrt(um);
	  vm = sqrt(vm);

	  hh = gg = 0.;
	  for(i=0; i<nIFO; i++) {
	    u[i] = u[i]/um;
	    v[i] = v[i]/vm;
	    hh  += u[i]*am[i];                              // (u*X) - solution 
	    gg  += v[i]*am[i];                              // (v*X) - solution 
	  } 

// calculate energy disbalance vectors

	  co=si=0.;
	  for(i=0; i<nIFO; i++) {                           // disbalance vectors
	    cc = local ? am[i]/(am[i]*am[i]+2.) : 1.;
	    pp[i] = cc*(am[i]-hh*u[i])*u[i];
	    qq[i] = cc*((2.*hh*u[i]-am[i])*v[i] + u[i]*u[i]*gg);
	    co += pp[i]*pp[i] + qq[i]*qq[i];                // cos (version 1)
	    si += pp[i]*qq[i];                              // sin
	  }
	  cc = sqrt(si*si+co*co)+1.e-24;
	  co = co/cc;
	  si = si/cc;
          if(!eDisbalance) {co=1.;si=0.;}  

// corrected likelihood

	  hh=gg = 0.;
	  for(i=0; i<nIFO; i++) {                           // solution for h(t,f)
	    e[i] = u[i]*co + v[i]*si;                       // final projection vector
	    r[i] = v[i]*co - u[i]*si;                       // orthogonal v vector
	    hh += e[i]*am[i];                               // solution for hu(t,f)
	    gg += r[i]*am[i];                               // solution for hv(t,f)
	  }

// second iteration

	  co=si=0.;
	  for(i=0; i<nIFO; i++) {                           // disbalance vectors
	    cc = local ? am[i]/(am[i]*am[i]+2.) : 1.;
	    pp[i] = cc*(am[i]-hh*e[i])*e[i];
	    qq[i] = cc*((2.*hh*e[i]-am[i])*r[i] + e[i]*e[i]*gg);
	    co += pp[i]*pp[i] + qq[i]*qq[i];                // cos (version 1)
	    si += pp[i]*qq[i];                              // sin
	  }
	  cc = sqrt(si*si+co*co)+1.e-24;
	  co = co/cc;
	  si = si/cc;
          if(!eDisbalance) {co=1.;si=0.;}  
	  
	  if(type=='E') U = 0;
	  hh = 0.; 
	  for(i=0; i<nIFO; i++) {                           // solution for h(t,f)
	    e[i] = e[i]*co+r[i]*si;                         // final projection vector
	    hh  += e[i]*am[i]*U;                            // solution for h(t,f)
	  }

// fill in pix

	  Lp = hh*hh;                                       // likelihood

	  if(Et>Eo) {
	    Lo += (type=='E') ? Et-nIFO : Lp;               // calculate Lc for x-check
	    E += Et;
	  }

	  pix = this->wc_List[n].getPixel(id,pI[j]);
	  pix->likelihood = (type=='E') ? Et-nIFO : Lp;
	  pix->theta = nLikelihood.getTheta(l);
	  pix->phi   = nLikelihood.getPhi(l);
	  if(!core)  pix->core = Et<Eo ? false : true;
	  
	  for(i=0; i<nIFO; i++) {
	    pix->setdata(am[i],'S',i);                      // whitened data
	    pix->setdata(e[i]*hh/sqrt(nv[i][j]),'W',i);     // detector response  
	  }
	  
	  if(ID) {
	    this->pixeLHood.data[pix->time] = Lp;
	    this->pixeLNull.data[pix->time] = Et-Lp+1.;
	    if(pOUT) {
	      cout<<j<<"  SNR="<<Et<<" ";
	      for(i=0; i<nIFO; i++) {
		cout<<e[i]*e[i]<<":"<<am[i]*am[i]/Et<<" ";
	      }
	      cout<<endl;
	    }
	  }
	  count++;	  
	}

// fill in backward delay configuration

	vtof->clear();
        NETX (
	vtof->push_back(int(M/2)-int(m0[l])); ,
	vtof->push_back(int(M/2)-int(m1[l])); ,
	vtof->push_back(int(M/2)-int(m2[l])); ,
	vtof->push_back(int(M/2)-int(m3[l])); ,
	vtof->push_back(int(M/2)-int(m4[l])); ,
	vtof->push_back(int(M/2)-int(m5[l])); ,
	vtof->push_back(int(M/2)-int(m6[l])); ,
	vtof->push_back(int(M/2)-int(m7[l])); )

	skyProb *= Em/IIm;
	T = cTo.data[k]+getifo(0)->TFmap.start();          // trigger time
	if(iID<=0 && type!='E') getSkyArea(id,n,T);        // calculate error regions

	if(fabs((Lm-Lo)/Lo)>1.e-4) 
	  cout<<"likelihood: incorrect likelihood : "<<Lm<<" "<<Lo<<" "<<Em<<endl; 

	if(E>0 && ID) { 
	  cout<<"max value: "<<STAT<<" at (theta,phi) = ("<<nLikelihood.getTheta(l)
	      <<","<<nLikelihood.getPhi(l)<<")  Likelihood: loop: "<<Lm
	      <<", final: "<<Lo<<", eff: "<<Em<<endl;
	  break;
	}

	if(ID && !EFEC) {
	  this->nSensitivity.gps = T;
	  this->nAlignment.gps = T;	
	  this->nNetIndex.gps = T;		 
	  this->nDisbalance.gps = T;		 
	  this->nLikelihood.gps = T;		 
	  this->nNullEnergy.gps = T;		 
	  this->nCorrEnergy.gps = T;		 
	  this->nCorrelation.gps = T;
	  this->nSkyStat.gps = T;	 
	  this->nPenalty.gps = T;	 
	  this->nProbability.gps = T;	 
	}
     }    // end of loop over clusters
     if(ID) break;
  }       // end of loop over time shifts
  return count;
}


//*********************************************************************************
// calculate network likelihood for 2-NIFO detectors assuming elliptical polarisation.
// elliptical constraint is used in a combination with the energy disbalance constraint
// It is designed to replace likelihood functions for 2,3,4 and NIFO networks. 
// If the network has less then NIFO detectors, all arrays are allocated for NIFO 
// detectors anyway. The arrays are initialized so that the dummy detectors
// do not contribute into calculation of the coherent statistics. 
//*********************************************************************************

long network::likelihoodI(char type, double Ao, int iID, size_t lag, int ind, bool core)
{
  this->like('I'); 

  size_t jN = 500;
  size_t nIFO = this->ifoList.size();
  size_t ID = abs(iID);
  int   N_1 = nIFO>2 ? int(nIFO)-1 : 2;
  int   N_2 = nIFO>2 ? int(nIFO)-2 : 1;
  if(nIFO>NIFO || !this->filter.size() || this->filter[0].index.size()!=32) {
    cout<<"network::likelihoodE(): invalid number of detectors or delay filter is not set.\n";
    return false;
  }

// regulators hard <- soft <0> weak -> hard
//   gamma =    -1 <-       0       -> 1
//   delta =     0  ->  1

  double Eo    = nIFO*Ao*Ao;                         // energy threshold in the sky loop
  double soft  = delta>0 ? delta : 0.;
  double GAMMA = 1.-gamma*gamma;                     // network regulator
  int    LOCAL = local ? 1 : 0;                      // ED minimization case

  double rho  = this->netRHO*this->netRHO*nIFO;    // threshold on rho
  double one  = (type=='g' || type=='G') ? 0. : 1.;  // circular polarization
  double En   = (type!='i' && type!='I') ? 1.e9 : 1.;
  bool ISG    = false;
  bool isgr   = false;

  if(type=='I' || type=='S' || type=='G') ISG = true;
  if(type=='i' || type=='s' || type=='g') ISG = true;
  if(type=='i' || type=='s' || type=='g' || type=='r') isgr = true;

  if(!ID && ind>=0) ind = -1;
  this->tYPe = type;
  this->acor = Ao;

  int ii,II,IIm;
  double gr,gg,gc,gp,gx,cc,gP,gX,gR,gI,EE,Et,T,Em,LPm,HH; 
  double em,STAT,Lo,Lm,Lp,Co,E00,E90,L00,C00,AA,CO,SI;
  double co,si,stat,XP,XX,L90,C90,sI,cO,aa,bb,as,ac,ap,La;
  double xp,xx,hh,um,vm,uc,us,vc,vs,hp,hx,HP,HX,wp,wx,WP,WX;
  size_t i,j,k,l,m,n,V,U,K,id,j5;

  size_t I = this->ifoList[0]->TFmap.maxLayer()+1;
  size_t M = this->filter.size()/I;                  // total number of delays 
  size_t L = ind<0 ? this->index.size() : ind+1;     // total number of source locations 
  int    R = int(this->ifoList[0]->getTFmap()->rate()/I+0.5);

// pointers to data
  double* pdata[NIFO];
  double* qdata[NIFO];
  double* pq;
  for(n=0; n<nIFO; n++) {
    pdata[n] = getifo(n)->getTFmap()->data;
    qdata[n] = getifo(n)->getTFmap()->data;
  }

// buffer for wavelet layer delay filter
  std::vector<float>* F00;
  std::vector<short>* J00;
  std::vector<float>* F90;
  std::vector<short>* J90;
  
// get antenna patterns and index arrays
  double*  fp[NIFO];                     // f+ 
  double*  fx[NIFO];                     // fx
  double* ffp[NIFO];                     // f+f+ + fxfx 
  double* ffm[NIFO];                     // f+f+ - fxfx
  double* fpx[NIFO];                     // 2*f+fx
  wavearray<double> f00(L); f00 = 0.; // dummy zero array

  for(n=0; n<NIFO; n++) {
     fp[n] = n<nIFO ? getifo(n)->fp.data  : f00.data;
     fx[n] = n<nIFO ? getifo(n)->fx.data  : f00.data;
    ffp[n] = n<nIFO ? getifo(n)->ffp.data : f00.data;
    ffm[n] = n<nIFO ? getifo(n)->ffm.data : f00.data;
    fpx[n] = n<nIFO ? getifo(n)->fpx.data : f00.data;
  }
  short* mm = skyMask.data;
  NETX (
  short* m0 = nIFO>1 ? getifo(0)->index.data : getifo(0)->index.data; ,
  short* m1 = nIFO>1 ? getifo(1)->index.data : getifo(1)->index.data; ,
  short* m2 = nIFO>2 ? getifo(2)->index.data : getifo(1)->index.data; ,
  short* m3 = nIFO>3 ? getifo(3)->index.data : getifo(1)->index.data; ,
  short* m4 = nIFO>4 ? getifo(4)->index.data : getifo(1)->index.data; ,
  short* m5 = nIFO>5 ? getifo(5)->index.data : getifo(1)->index.data; ,
  short* m6 = nIFO>6 ? getifo(6)->index.data : getifo(1)->index.data; ,
  short* m7 = nIFO>7 ? getifo(7)->index.data : getifo(1)->index.data; )

// allocate buffers
  std::vector<size_t> pI;
  
  wavearray<double> LL(jN);  
  wavearray<double> GG(jN);  
  wavearray<short>  jU(jN);  
  wavearray<double> am00(jN*NIFO);     // buffers for 00 data
  wavearray<double> am90(jN*NIFO);     // buffers for 90 data
  wavearray<double> xi00(jN*NIFO);     // buffers for 00 response
  wavearray<double> xi90(jN*NIFO);     // buffers for 90 response
  wavearray<double> Fplus(jN*NIFO);    // buffers for F+
  wavearray<double> Fcros(jN*NIFO);    // buffers for Fx

//BM  wavearray<double> e00[NIFO];
//BM  wavearray<double> e90[NIFO];
  wavearray<double> a00[NIFO];
  wavearray<double> a90[NIFO];
  wavearray<double> NV[NIFO];          // noise variance
  wavearray<double> NR[NIFO];          // noise rms
  wavearray<double> cid;            // buffers for cluster ID
  wavearray<double> cTo;            // buffers for cluster time
  wavearray<double> vol;            // buffers for cluster volume

  double   u[NIFO];
  double   v[NIFO];
  double   e[NIFO];
  double  ee[NIFO];
  double  rr[NIFO];
  double  pp[NIFO];
  double  qq[NIFO];

  double* am;
  double* AM;
  double* xi;
  double* XI;
  double* Fp;
  double* Fx;
  double* nv[NIFO];
  double* nr[NIFO];
  double* pa00[NIFO];
  double* pa90[NIFO];

  for(n=0; n<NIFO; n++) {
     NV[n].resize(jN);    NV[n] = 0.; nv[n] = NV[n].data; 
     NR[n].resize(jN);    NR[n] = 0.; nr[n] = NR[n].data; 
    a00[n].resize(jN*M); a00[n] = 0.; 
//BM    e00[n].resize(jN*M); e00[n] = 0.;
    a90[n].resize(jN*M); a90[n] = 0.; 
//BM    e90[n].resize(jN*M); e90[n] = 0.;
    rr[n] = 1.;
  }

  netpixel* pix;
  std::vector<int>* vint;
  std::vector<int>* vtof;

  double  logbpp,inet;
  size_t  count = 0;
  ap = ac = as = 0.;

  if(ID) { 
    this->pixeLHood = getifo(0)->TFmap; 
    this->pixeLHood = 0.; 
    this->pixeLNull = getifo(0)->TFmap; 
    this->pixeLNull = 0.; 
    this->nSensitivity = 0.;
    this->nAlignment = 0.;	
    this->nNetIndex = 0.;		 
    this->nDisbalance = 0.;		 
    this->nLikelihood = 0.;		 
    this->nNullEnergy = 0.;		 
    this->nCorrEnergy = 0.;		 
    this->nCorrelation = 0.;
    this->nSkyStat = 0.;	 
    this->nEllipticity = 0.;		 
    this->nPolarisation = 0.;		 
    this->nProbability = 0.;		 
    this->nAntenaPrior = 0.;
  }

//+++++++++++++++++++++++++++++++++++++++
// liklihood calculation for clusters
//+++++++++++++++++++++++++++++++++++++++

  for(n=lag; n<nLag; n++) {                  // loop over time shifts 

     if(!this->wc_List[n].size()) continue;
     logbpp = -log(this->wc_List[n].getbpp());

     cid = this->wc_List[n].get((char*)"ID",0,'S',optim ? R : -R);   // get cluster ID
     cTo = this->wc_List[n].get((char*)"time",0,'L',optim ? R : -R); // get cluster time

     K = cid.size();

     for(k=0; k<K; k++) {      // loop over clusters 

	id = size_t(cid.data[k]+0.1);

	if(ID && id!=ID) continue;

	vint = &(this->wc_List[n].cList[id-1]);           // pixel list
	vtof = &(this->wc_List[n].nTofF[id-1]);           // TofFlight configurations

	V = vint->size();
	if(!V) continue;

	pI.clear();

	for(j=0; j<V; j++) {   // loop over pixels 

	  pix = this->wc_List[n].getPixel(id,j);

	  if(!pix) {
	    cout<<"network::likelihood() error: NULL pointer"<<endl;
	    exit(1);
	  }

	  if(R != int(pix->rate+0.5)) continue;           // check rate 
	  if(!pix->core && core) continue;                // check core flag

	  pI.push_back(j);                                // save pixel index

	}

	V = pI.size();
	if(!V) continue;
	
	if(NV[0].size() < V) {                            // reallocate arrays           
	  U = V+100;
	  for(i=0; i<NIFO; i++) {
             NV[i].resize(U);    NV[i] = 0.;  nv[i] = NV[i].data;
	     NR[i].resize(U);    NR[i] = 0.;  nr[i] = NR[i].data; 
//BM            e00[i].resize(U*M); e00[i] = 0.; 
            a00[i].resize(U*M); a00[i] = 0.; 
//BM            e90[i].resize(U*M); e90[i] = 0.; 
            a90[i].resize(U*M); a90[i] = 0.; 
	  }
	  jU.resize(U); LL.resize(U); GG.resize(U);
	  am00.resize(U*NIFO); am90.resize(U*NIFO);
	  xi00.resize(U*NIFO); xi90.resize(U*NIFO);
	  Fplus.resize(U*NIFO); Fcros.resize(U*NIFO);
	}

	for(j=0; j<V; j++) {   // loop over selected pixels 

	  pix = this->wc_List[n].getPixel(id,pI[j]);

	  cc = 0.;
	  for(i=0; i<nIFO; i++) {
	    ee[i] = 1./pix->data[i].noiserms;
	    cc += ee[i]*ee[i];                          // total inverse variance
	  }

	  GG.data[j] = sqrt(cc);
	  for(i=0; i<nIFO; i++) {
	    nv[i][j] = ee[i]*ee[i];                     // inverse variance
	    ee[i]   /= sqrt(cc);
	    nr[i][j] = ee[i];                           // normalized 1/rms
	    qdata[i] = pdata[i] + pix->data[i].index;   // pointer to data   
	  }

// apply delay filter
	   
	  for(i=0; i<nIFO; i++) {
	    pq = qdata[i];

	    for(m=0; m<M; m++){          // loop over delays                   
	      F00 = &(filter[pix->frequency*M+m].value); 
	      J00 = &(filter[pix->frequency*M+m].index);
	      F90 = &(filter90[pix->frequency*M+m].value); 
	      J90 = &(filter90[pix->frequency*M+m].index);
	      l = m*V + j; 

	      gg = dot32(F00,pq,J00);        // apply filter
//BM              e00[i].data[l] = gg*ee[i];
              a00[i].data[l] = gg;

	      gg = dot32(F90,pq,J90);        // apply filter
//BM              e90[i].data[l] = gg*ee[i];
              a90[i].data[l] = gg;
	    }
	  }
	}

        STAT = -1.e64; m=U=IIm=0; Em=Lm=LPm=AA=SI=CO=0.;

//============================================================
// weak constraint on 00 data
//============================================================

	l = ind<0 ? 0 : ind;                 // process selected index
	for(; l<L; l++) {	             // loop over sky locations
	  
	  skyProb.data[l] = 0.;

          if(skyMaskCC.size()==L) {
            // transform l in celestial coordinates cc_l and check the skyMaskCC
            skymap* sm = &(this->nSkyStat);
            double th = sm->getTheta(l);
            double ph = sm->getPhi(l);
            double gpsT = cTo.data[k]+getifo(0)->TFmap.start();          // trigger time
            double ra = sm->phi2RA(ph,  gpsT);
            int cc_l = this->getIndex(th,ra);
            if (!skyMaskCC.data[cc_l]) continue;
          }

	  if(!mm[l] && ind<0) continue;      // skip sky configurations

          NETX (	  
	  pa00[0] = a00[0].data + m0[l]*V; ,
	  pa00[1] = a00[1].data + m1[l]*V; ,
	  pa00[2] = a00[2].data + m2[l]*V; ,
	  pa00[3] = a00[3].data + m3[l]*V; ,
	  pa00[4] = a00[4].data + m4[l]*V; ,
	  pa00[5] = a00[5].data + m5[l]*V; ,
	  pa00[6] = a00[6].data + m6[l]*V; ,
	  pa00[7] = a00[7].data + m7[l]*V; )

	  EE=E00=L00=C00=Lp=0.; II=0;
	  
	  for(j=0; j<V; j++) {                          // 00 phase

	    j5 = j*NIFO;
	    am = am00.data+j5;
	    inix(pa00,j,am);
	    Et = dotx(am,am);	                        // total energy     	    
	    ii = int(Et>Eo);
	    E00+= Et*ii;

	    Fp = Fplus.data+j5;
	    Fx = Fcros.data+j5;
	    mulx(fp,l,nr,j,Fp);
	    mulx(fx,l,nr,j,Fx);

	    gp = dotx(Fp,Fp)+1.e-12;                    // fp^2
	    gx = dotx(Fx,Fx)+1.e-12;                    // fx^2
	    gI = dotx(Fp,Fx);                           // fp*fx

	    xi = xi00.data+j5;
	    xp = dotx(Fp,am);                           // (X*f+)
	    xx = dotx(Fx,am);                           // (X*fx)       
	    um = rotx(Fp,xp*gx-xx*gI,
		      Fx,xx*gp-xp*gI,xi);
	    hh = dotx(xi,am,e);
	    La = hh*hh/um;
	    Co = La - dotx(e,e)/um;
	    ii*= int(Co>0.);
	    EE+= Et*ii;
	    Lp+= La*Co*ii/(Et-La+1+Co);
	    II+= ii;
	    LL.data[j] = La*ii;
	    L00+= La*ii; 
	    C00+= Co*ii; 
	    mulx(xi,hh*ii/um);
	  }

	  cc = E00>0 ? C00/(E00-L00+fabs(C00)) : 0.;
	  if(cc<this->netCC || cc*C00<rho) continue;

	  E90=L90=C90=inet=0.;

          NETX (
	  pa90[0] = a90[0].data + m0[l]*V; ,
	  pa90[1] = a90[1].data + m1[l]*V; ,
	  pa90[2] = a90[2].data + m2[l]*V; ,
	  pa90[3] = a90[3].data + m3[l]*V; ,
	  pa90[4] = a90[4].data + m4[l]*V; ,
	  pa90[5] = a90[5].data + m5[l]*V; ,
	  pa90[6] = a90[6].data + m6[l]*V; ,
	  pa90[7] = a90[7].data + m7[l]*V; )

	  for(j=0; j<V; j++) {                          // 90 phase 

	    j5 = j*NIFO;
	    AM = am90.data+j5;
	    inix(pa90,j,AM);
	    Et = dotx(AM,AM);	                        // total energy     	    
	    ii = int(Et>Eo);
	    E90+= Et*ii;

	    Fp = Fplus.data+j5;
	    Fx = Fcros.data+j5;
	    gp = dotx(Fp,Fp)+1.e-12;                    // fp^2
	    gx = dotx(Fx,Fx)+1.e-12;                    // fx^2
	    gI = dotx(Fp,Fx);                           // fp*fx

	    XI = xi90.data+j5;
	    XP = dotx(Fp,AM);                           // (X(90)*f+)
	    XX = dotx(Fx,AM);                           // (X(90)*fx)       
	    vm = rotx(Fp,XP*gx-XX*gI,
		      Fx,XX*gp-XP*gI,XI);
	    HH = dotx(XI,AM,e);
	    La = HH*HH/vm;
	    Co = La - dotx(e,e)/vm;
	    ii*= int(Co>0.);
	    EE+= Et*ii;
	    Lp+= La*Co*ii/(Et-La+1+Co);
	    II+= ii;
	    LL.data[j] += La*ii;
	    L90+= La*ii; 
	    C90+= Co*ii; 
	    mulx(XI,HH*ii/vm);
	  }

	  cc = E90>0 ? C90/(E90-L90+fabs(C90)) : 0.;
	  if(cc<this->netCC || cc*C90<rho) continue;

	  Co = C00+C90;
	  Lo = L00+L90;

	  sI=cO=aa=bb=0.;

//++++++++++++++++elliptical case++++++++++++++++++++++
// find solution for polarisation angle and ellipticity
//++++++++++++++++elliptical case++++++++++++++++++++++

	  if(ISG) {

	    Lo=Lp=Co=0.;

	    for(j=0; j<V; j++) {               // loop over selected pixels 

	      j5 = j*NIFO;
	      Fp = Fplus.data+j5;
	      Fx = Fcros.data+j5;
	      gp = dotx(Fp,Fp)+1.e-12;         // fp^2
	      gx = dotx(Fx,Fx)+1.e-12;         // fx^2
	      gI = dotx(Fp,Fx);                // fp*fx
	      gg = gp+gx;

      	      wp = dotx(Fp,xi00.data+j5);      // +00 response
	      wx = dotx(Fx,xi00.data+j5);      // x00 response
	      WP = dotx(Fp,xi90.data+j5);      // +90 response
	      WX = dotx(Fx,xi90.data+j5);      // x90 response
	      
	      xp = (wp*wp + WP*WP);  
	      xx = (wx*wx + WX*WX);  
	      La = LL.data[j];

	      sI += 2*(wp*wx+WP*WX-La*gI)/gg;  // sin(4psi) 
	      cO += (xp-xx - La*(gp-gx))/gg;   // cos(4psi)
	      bb += La - (xp+xx)/gg;     
	      aa += 2*(wp*WX - wx*WP)/gg;
	    }

	    gg = sqrt(cO*cO+sI*sI);
	    cO = cO/gg; sI = sI/gg;
	    aa/= bb+gg+V*En;                   // ellipticity
	    if(aa> one) aa = 1.;
	    if(aa<-one) aa =-1.;

	    for(j=0; j<V; j++) {               // loop over selected pixels 

	      Fp = Fplus.data+j*NIFO;
	      Fx = Fcros.data+j*NIFO;
	      um = rotx(Fp,1+cO,Fx, sI,u);     // rotate by 2*polarization angle
	      vm = rotx(Fx,1+cO,Fp,-sI,v);     // rotate by 2*polarization angle
	      gp = dotx(u,u)+1.e-12;           // fp^2
	      gx = dotx(v,v)+1.e-12;           // fx^2
	      gI = dotx(u,v);                  // f+fx
	      gg = gp+aa*aa*gx;                // inverse network sensitivity
	      
	      am = am00.data+j*NIFO;
	      xi = xi00.data+j*NIFO;
	      um = dotx(xi,xi)+1.e-6;
	      hh = dotx(xi,am,pp);
	      xp = dotx(u,xi);
	      xx = dotx(v,xi)*aa;
	      bb = (xp*xp+xx*xx)/(um*gg*um);
	      La = hh*hh*bb;
	      cc = bb*dotx(pp,pp);

	      AM = am90.data+j*NIFO;
	      XI = xi90.data+j*NIFO;
	      vm = dotx(XI,XI)+1.e-6;
	      HH = dotx(XI,AM,qq);
	      XP = dotx(u,XI);
	      XX = dotx(v,XI)*aa;
	      bb = (XP*XP+XX*XX)/(vm*gg*vm);
	      La+= HH*HH*bb;
	      cc+= bb*dotx(qq,qq);
	      
	      bb = 2*(xp*XX-xx*XP)/(um*gg*vm);
	      La+= hh*HH*bb;
	      cc+= bb*dotx(pp,qq);
	      cc = La-cc;
	      Lo+= La;
	      Co+= cc;
	      Et = dotx(am,am) + dotx(AM,AM);
	      Lp+= La*cc/(Et-La+2+fabs(cc));
	    }
	  }

	  cc = EE>0 ? Co/(EE-Lo+II+fabs(Co)) : 0.;
	  EE = E00+E90;
	  if(!isgr) Lp = Lo*cc;
	  if(Lp>LPm) {LPm=Lp; Em=EE; IIm=II;}         // max likelihood

	  skyProb.data[l]  = Lp/EE;                   // probability map

	  cc = EE>0 ? Co/(EE-Lo+II+fabs(Co)) : 0.;
	  if(cc<this->netCC || cc*Co<rho*2) continue;

//=============================================================
// zero phase data  
//=============================================================

	  Lp=Lo=Co=gP=gX=0.;
	  
	  for(j=0; j<V; j++) {                   // loop over selected pixels 

	    xi = xi00.data+j*NIFO;	    

	    if(dotx(xi,xi)<=0.) continue;        // skip insignificant pixel
	    
	    am = am00.data+j*NIFO;
	    Et = dotx(am,am);
	    Fp = Fplus.data+j*NIFO;
	    Fx = Fcros.data+j*NIFO;
	    gp = dotx(Fp,Fp)+1.e-12;             // fp^2
	    gx = dotx(Fx,Fx)+1.e-12;             // fx^2
	    gI = dotx(Fp,Fx);                    // fp*fx
	    xp = dotx(Fp,am);                    // (X*f+)
	    xx = dotx(Fx,am);                    // (X*fx)     
	    uc = xp*gx - xx*gI;                  // u cos of rotation to PCF
	    us = xx*gp - xp*gI;                  // u sin of rotation to PCF
	    vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	    vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	    um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	    vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector

// regulator

	    ii = netx(u,um,v,vm,GAMMA); 
	    inet += ii*Et;
	    if(ii<N_2 && gamma<0.) {inix(xi,0); continue;}

	    gg  = (gp+gx)*soft; 
	    uc += xp*gg;                         // u cos of rotation to PCF
	    us += xx*gg;                         // u sin of rotation to PCF
	    
	    if(ii<N_1 && gamma!=0) { 
	      gR = (gp-gx)/2;
	      gc = sqrt(gR*gR+gI*gI);            // norm of complex antenna pattern
	      uc = xp*(gc+gR)+xx*gI;
	      us = xx*(gc-gR)+xp*gI;
	    }
	    
	    vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	    vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	    um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	    vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector

// energy disbalance vectors

            NETX (
	    rr[0] = LOCAL*am[0]/(am[0]*am[0]+2.)+1-LOCAL; ,
	    rr[1] = LOCAL*am[1]/(am[1]*am[1]+2.)+1-LOCAL; ,
	    rr[2] = LOCAL*am[2]/(am[2]*am[2]+2.)+1-LOCAL; ,
	    rr[3] = LOCAL*am[3]/(am[3]*am[3]+2.)+1-LOCAL; ,
	    rr[4] = LOCAL*am[4]/(am[4]*am[4]+2.)+1-LOCAL; ,
	    rr[5] = LOCAL*am[5]/(am[5]*am[5]+2.)+1-LOCAL; ,
	    rr[6] = LOCAL*am[6]/(am[6]*am[6]+2.)+1-LOCAL; ,
	    rr[7] = LOCAL*am[7]/(am[7]*am[7]+2.)+1-LOCAL; )

	    hh = dotx(u,am)/um;
            NETX (
	    pp[0] = rr[0]*(am[0]-hh*u[0])*u[0]; ,
	    pp[1] = rr[1]*(am[1]-hh*u[1])*u[1]; ,
	    pp[2] = rr[2]*(am[2]-hh*u[2])*u[2]; ,
	    pp[3] = rr[3]*(am[3]-hh*u[3])*u[3]; ,
	    pp[4] = rr[4]*(am[4]-hh*u[4])*u[4]; ,
	    pp[5] = rr[5]*(am[5]-hh*u[5])*u[5]; ,
	    pp[6] = rr[6]*(am[6]-hh*u[6])*u[6]; ,
	    pp[7] = rr[7]*(am[7]-hh*u[7])*u[7]; )
	    
	    gg = dotx(v,am)/um;
	    hh*= 2.;
            NETX (
	    qq[0] = rr[0]*((hh*u[0]-am[0])*v[0]+u[0]*u[0]*gg); ,
	    qq[1] = rr[1]*((hh*u[1]-am[1])*v[1]+u[1]*u[1]*gg); ,
	    qq[2] = rr[2]*((hh*u[2]-am[2])*v[2]+u[2]*u[2]*gg); ,
	    qq[3] = rr[3]*((hh*u[3]-am[3])*v[3]+u[3]*u[3]*gg); ,
	    qq[4] = rr[4]*((hh*u[4]-am[4])*v[4]+u[4]*u[4]*gg); ,
	    qq[5] = rr[5]*((hh*u[5]-am[5])*v[5]+u[5]*u[5]*gg); ,
	    qq[6] = rr[6]*((hh*u[6]-am[6])*v[6]+u[6]*u[6]*gg); ,
	    qq[7] = rr[7]*((hh*u[7]-am[7])*v[7]+u[7]*u[7]*gg); )
	    
	    co = dotx(qq,qq)/vm+dotx(pp,pp)/um+1.e-24;     // cos term
	    si = dotx(pp,qq);                              // sin term
            if(!eDisbalance) {co=1.;si=0.;} 
	    em = rotx(u,co,v,si/vm,e);                     // calculate rotated vector e

// second iteration
	      
	    vm = rotx(v,co,u,-si/um,v)+1.e-24;      // calculate rotated vector v
	    hh = dotx(e,am)/em;
	   
            NETX ( 
	    pp[0] = rr[0]*(am[0]-hh*e[0])*e[0]; ,
	    pp[1] = rr[1]*(am[1]-hh*e[1])*e[1]; ,
	    pp[2] = rr[2]*(am[2]-hh*e[2])*e[2]; ,
	    pp[3] = rr[3]*(am[3]-hh*e[3])*e[3]; ,
	    pp[4] = rr[4]*(am[4]-hh*e[4])*e[4]; ,
	    pp[5] = rr[5]*(am[5]-hh*e[5])*e[5]; ,
	    pp[6] = rr[6]*(am[6]-hh*e[6])*e[6]; ,
	    pp[7] = rr[7]*(am[7]-hh*e[7])*e[7]; )
	    
	    gg = dotx(v,am)/em;
	    hh*= 2.;
            NETX (
	    qq[0] = rr[0]*((hh*e[0]-am[0])*v[0]+e[0]*e[0]*gg); ,
	    qq[1] = rr[1]*((hh*e[1]-am[1])*v[1]+e[1]*e[1]*gg); ,
	    qq[2] = rr[2]*((hh*e[2]-am[2])*v[2]+e[2]*e[2]*gg); ,
	    qq[3] = rr[3]*((hh*e[3]-am[3])*v[3]+e[3]*e[3]*gg); ,
	    qq[4] = rr[4]*((hh*e[4]-am[4])*v[4]+e[4]*e[4]*gg); ,
	    qq[5] = rr[5]*((hh*e[5]-am[5])*v[5]+e[5]*e[5]*gg); ,
	    qq[6] = rr[6]*((hh*e[6]-am[6])*v[6]+e[6]*e[6]*gg); ,
	    qq[7] = rr[7]*((hh*e[7]-am[7])*v[7]+e[7]*e[7]*gg); )
	    
	    co = dotx(qq,qq)/vm+dotx(pp,pp)/em+1.e-24;       // cos term
	    si = dotx(pp,qq);                                // sin term
            if(!eDisbalance) {co=1.;si=0.;}  
	    em = rotx(e,co,v,si/vm,e);                       // calculate final vector
	    hh = dotx(e,am,ee);                              // GW amplitude
	    La = hh*hh/em;
	    cc = La-dotx(ee,ee)/em;
	    Lo+= La;
	    Co+= cc;
	    Lp+= La*cc*int(cc>0)/(Et-La+1+cc);
	    mulx(e,hh/em,xi);
	  }

//=============================================================
// 90 degrees phase energy disbalance  
//=============================================================

	  for(j=0; j<V; j++) {                   // select pixels and get dot ptoducts 
	    
	    XI = xi90.data+j*NIFO;

	    if(dotx(XI,XI)<=0) continue;         // skip insignificant pixel

	    AM = am90.data+j*NIFO;
	    Et = dotx(AM,AM);
	    Fp = Fplus.data+j*NIFO;
	    Fx = Fcros.data+j*NIFO;
	    gp = dotx(Fp,Fp)+1.e-12;             // fp^2
	    gx = dotx(Fx,Fx)+1.e-12;             // fx^2
	    gI = dotx(Fp,Fx);                    // fp*fx
	    XP = dotx(Fp,AM);                    // (X*f+)
	    XX = dotx(Fx,AM);                    // (X*fx)      
	    uc = XP*gx - XX*gI;                  // u cos of rotation to PCF
	    us = XX*gp - XP*gI;                  // u sin of rotation to PCF
	    vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	    vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	    um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	    vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector

// regulator

	    ii = netx(u,um,v,vm,GAMMA); 
	    inet += ii*Et;
	    if(ii<N_2 && gamma<0.) {inix(XI,0.); continue;}

	    gg  = (gp+gx)*soft; 
	    uc += XP*gg;                         // u cos of rotation to PCF
	    us += XX*gg;                         // u sin of rotation to PCF
	    
	    if(ii<N_1 && gamma!=0) { 
	      gR = (gp-gx)/2;
	      gc = sqrt(gR*gR+gI*gI);              // norm of complex antenna pattern
	      uc = XP*(gc+gR)+XX*gI;
	      us = XX*(gc-gR)+XP*gI;
	    }

	    vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	    vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	    um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	    vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector
 
// energy disbalance vectors

            NETX (
	    rr[0] = LOCAL*AM[0]/(AM[0]*AM[0]+2.)+1-LOCAL; ,
	    rr[1] = LOCAL*AM[1]/(AM[1]*AM[1]+2.)+1-LOCAL; ,
	    rr[2] = LOCAL*AM[2]/(AM[2]*AM[2]+2.)+1-LOCAL; ,
	    rr[3] = LOCAL*AM[3]/(AM[3]*AM[3]+2.)+1-LOCAL; ,
	    rr[4] = LOCAL*AM[4]/(AM[4]*AM[4]+2.)+1-LOCAL; ,
	    rr[5] = LOCAL*AM[5]/(AM[5]*AM[5]+2.)+1-LOCAL; ,
	    rr[6] = LOCAL*AM[6]/(AM[6]*AM[6]+2.)+1-LOCAL; ,
	    rr[7] = LOCAL*AM[7]/(AM[7]*AM[7]+2.)+1-LOCAL; )

	    hh = dotx(u,AM)/um;
            NETX (
	    pp[0] = rr[0]*(AM[0]-hh*u[0])*u[0]; ,
	    pp[1] = rr[1]*(AM[1]-hh*u[1])*u[1]; ,
	    pp[2] = rr[2]*(AM[2]-hh*u[2])*u[2]; ,
	    pp[3] = rr[3]*(AM[3]-hh*u[3])*u[3]; ,
	    pp[4] = rr[4]*(AM[4]-hh*u[4])*u[4]; ,
	    pp[5] = rr[5]*(AM[5]-hh*u[5])*u[5]; ,
	    pp[6] = rr[6]*(AM[6]-hh*u[6])*u[6]; ,
	    pp[7] = rr[7]*(AM[7]-hh*u[7])*u[7]; )
	    
	    gg = dotx(v,AM)/um;
	    hh*= 2.;
            NETX (
	    qq[0] = rr[0]*((hh*u[0]-AM[0])*v[0]+u[0]*u[0]*gg); ,
	    qq[1] = rr[1]*((hh*u[1]-AM[1])*v[1]+u[1]*u[1]*gg); ,
	    qq[2] = rr[2]*((hh*u[2]-AM[2])*v[2]+u[2]*u[2]*gg); ,
	    qq[3] = rr[3]*((hh*u[3]-AM[3])*v[3]+u[3]*u[3]*gg); ,
	    qq[4] = rr[4]*((hh*u[4]-AM[4])*v[4]+u[4]*u[4]*gg); ,
	    qq[5] = rr[5]*((hh*u[5]-AM[5])*v[5]+u[5]*u[5]*gg); ,
	    qq[6] = rr[6]*((hh*u[6]-AM[6])*v[6]+u[6]*u[6]*gg); ,
	    qq[7] = rr[7]*((hh*u[7]-AM[7])*v[7]+u[7]*u[7]*gg); )
	    
	    co = dotx(qq,qq)/vm+dotx(pp,pp)/um+1.e-24;     // cos term
	    si = dotx(pp,qq);                              // sin term
            if(!eDisbalance) {co=1.;si=0.;}  
	    em = rotx(u,co,v,si/vm,e);                     // calculate rotated vector e

// second iteration
	      
	    vm = rotx(v,co,u,-si/um,v)+1.e-24;      // calculate rotated vector v
	    hh = dotx(e,AM)/em;
	   
            NETX ( 
	    pp[0] = rr[0]*(AM[0]-hh*e[0])*e[0]; ,
	    pp[1] = rr[1]*(AM[1]-hh*e[1])*e[1]; ,
	    pp[2] = rr[2]*(AM[2]-hh*e[2])*e[2]; ,
	    pp[3] = rr[3]*(AM[3]-hh*e[3])*e[3]; ,
	    pp[4] = rr[4]*(AM[4]-hh*e[4])*e[4]; ,
	    pp[5] = rr[5]*(AM[5]-hh*e[5])*e[5]; ,
	    pp[6] = rr[6]*(AM[6]-hh*e[6])*e[6]; ,
	    pp[7] = rr[7]*(AM[7]-hh*e[7])*e[7]; )
	    
	    gg = dotx(v,AM)/em;
	    hh*= 2.;
            NETX (
	    qq[0] = rr[0]*((hh*e[0]-AM[0])*v[0]+e[0]*e[0]*gg); ,
	    qq[1] = rr[1]*((hh*e[1]-AM[1])*v[1]+e[1]*e[1]*gg); ,
	    qq[2] = rr[2]*((hh*e[2]-AM[2])*v[2]+e[2]*e[2]*gg); ,
	    qq[3] = rr[3]*((hh*e[3]-AM[3])*v[3]+e[3]*e[3]*gg); ,
	    qq[4] = rr[4]*((hh*e[4]-AM[4])*v[4]+e[4]*e[4]*gg); ,
	    qq[5] = rr[5]*((hh*e[5]-AM[5])*v[5]+e[5]*e[5]*gg); ,
	    qq[6] = rr[6]*((hh*e[6]-AM[6])*v[6]+e[6]*e[6]*gg); ,
	    qq[7] = rr[7]*((hh*e[7]-AM[7])*v[7]+e[7]*e[7]*gg); )
	    
	    co = dotx(qq,qq)/vm+dotx(pp,pp)/em+1.e-24;       // cos term
	    si = dotx(pp,qq);                                // sin term
            if(!eDisbalance) {co=1.;si=0.;}  
	    em = rotx(e,co,v,si/vm,e);                       // calculate final vector
	    hh = dotx(e,AM,ee);                              // GW amplitude
	    La = hh*hh/em;
	    cc = La-dotx(ee,ee)/em;
	    Lo+= La;
	    Co+= cc;
	    Lp+= La*cc*int(cc>0)/(Et-La+1+cc);
	    mulx(e,hh/em,XI);
	  }

//++++++++++++++++elliptical case++++++++++++++++++++++
// calculate elliptical likelihood and coherent energy
//++++++++++++++++elliptical case++++++++++++++++++++++
	  
	  if(ISG) {

	    Lp=Lo=Co=0.;
	  
	    for(j=0; j<V; j++) {               // loop over selected pixels 

	      Fp = Fplus.data+j*NIFO;
	      Fx = Fcros.data+j*NIFO;
	      um = rotx(Fp,1+cO,Fx, sI,u);     // rotate by 2*polarization angle
	      vm = rotx(Fx,1+cO,Fp,-sI,v);     // rotate by 2*polarization angle
	      gp = dotx(u,u)+1.e-12;           // fp^2
	      gx = dotx(v,v)+1.e-12;           // fx^2
	      gg = gp + aa*aa*gx;              // network sensitivity
	      
	      am = am00.data+j*NIFO;
	      xi = xi00.data+j*NIFO;
	      um = dotx(xi,xi)+1.e-6;
	      hh = dotx(xi,am,pp);
	      xp = dotx(u,xi);
	      xx = dotx(v,xi)*aa;
	      bb = (xp*xp+xx*xx)/(um*gg*um);
	      La = hh*hh*bb;
	      cc = bb*dotx(pp,pp);

	      AM = am90.data+j*NIFO;
	      XI = xi90.data+j*NIFO;
	      vm = dotx(XI,XI)+1.e-6;
	      HH = dotx(XI,AM,qq);
	      XP = dotx(u,XI);
	      XX = dotx(v,XI)*aa;
	      bb = (XP*XP+XX*XX)/(vm*gg*vm);
	      La+= HH*HH*bb;
	      cc+= bb*dotx(qq,qq);
	      
	      bb = 2*(xp*XX-xx*XP)/(um*gg*vm);
	      La+= hh*HH*bb;
	      cc+= bb*dotx(pp,qq);
	      cc = La-cc;
	      Co+= cc;
	      Lo+= La;
	      Et = dotx(am,am)+dotx(AM,AM);
	      Lp+= La*cc*int(cc>0)/(Et-La+2+cc);
	    }
	  }

// final detection statistics
	  	  
	  cc = EE>0 ? Co/(EE-Lo+II+fabs(Co)) : 0.;
	  stat = isgr ? Lp/EE : Lo*cc/EE;
	  inet/= EE;


	  if(stat>=STAT) { m=l; STAT=stat; AA=aa; CO=cO; SI=sI; Lm=Lo/2.; }
	  
	  if(ID) {                            // fill skymaps
	    for(j=0; j<V; j++) {               // loop over selected pixels 
	      Fp = Fplus.data+j*NIFO;
	      Fx = Fcros.data+j*NIFO;
	      Et = dotx(pa00,j,pa00,j);
	      gp = dotx(Fp,Fp)+1.e-12;             // fp^2
	      gx = dotx(Fx,Fx)+1.e-12;             // fx^2
	      gI = dotx(Fp,Fx);                    // fp*fx
	      gr = (gp+gx)/2;
	      gR = (gp-gx)/2;
	      gc = sqrt(gR*gR+gI*gI);              // norm of complex antenna pattern
	      gP+= (gr+gc);                        // average + sensitivity
	      gX+= (gr-gc);                        // average x sensitivity
	    }

	    this->nAntenaPrior.set(l, gP/V);
	    this->nSensitivity.set(l, gP/V);
	    this->nAlignment.set(l, gX/V);	
	    this->nLikelihood.set(l, Lo/2);		 
	    this->nNullEnergy.set(l, (EE-Lo)/2.);		 
	    this->nCorrEnergy.set(l, Co/2);		 
	    this->nCorrelation.set(l, cc);
	    this->nSkyStat.set(l, stat);		 
	    this->nProbability.set(l, skyProb.data[l]);		 
	    this->nDisbalance.set(l, fabs(L00-L90)/fabs(L00+L90));		 
      	    this->nEllipticity.set(l, aa);		 
	    this->nPolarisation.set(l, atan2(sI,cO));		 
	    this->nNetIndex.set(l, inet);
	  }	    
	}          

	if(STAT<0.001 && ind<0) {
	  this->wc_List[n].ignore(id);     // reject cluster
	  continue;
	}

//============================================================
// final calculation of likelihood for selected sky location
//============================================================

	l = ind<0 ? m : ind;

        NETX (
	pa00[0] = a00[0].data + m0[l]*V; ,
	pa00[1] = a00[1].data + m1[l]*V; ,
	pa00[2] = a00[2].data + m2[l]*V; ,
	pa00[3] = a00[3].data + m3[l]*V; ,
	pa00[4] = a00[4].data + m4[l]*V; ,
	pa00[5] = a00[5].data + m5[l]*V; ,
	pa00[6] = a00[6].data + m6[l]*V; ,
	pa00[7] = a00[7].data + m7[l]*V; )
        NETX (
	pa90[0] = a90[0].data + m0[l]*V; ,
	pa90[1] = a90[1].data + m1[l]*V; ,
	pa90[2] = a90[2].data + m2[l]*V; ,
	pa90[3] = a90[3].data + m3[l]*V; ,
	pa90[4] = a90[4].data + m4[l]*V; ,
	pa90[5] = a90[5].data + m5[l]*V; ,
	pa90[6] = a90[6].data + m6[l]*V; ,
	pa90[7] = a90[7].data + m7[l]*V; )
	
	Lo=0;

	for(j=0; j<V; j++) {            // select pixels and get dot ptoducts 
	  
	  pix = this->wc_List[n].getPixel(id,pI[j]);
	  pix->theta = nLikelihood.getTheta(l);
	  pix->phi   = nLikelihood.getPhi(l);
	  pix->ellipticity = ISG ? AA : 0.;
	  pix->polarisation = ISG ? atan2(SI,CO)/4. : 0.;

	  E00=E90=gp=gx=gI=xp=xx=XP=XX = 0.;
	  U = 1;

	  
	  Fp = Fplus.data+j*NIFO;
	  Fx = Fcros.data+j*NIFO;
	  am = am00.data+j*NIFO;
	  AM = am90.data+j*NIFO;
	  for(i=0; i<nIFO; i++) {
	    Fp[i] = fp[i][l]*nr[i][j];
	    Fx[i] = fx[i][l]*nr[i][j];
	    am[i] = pa00[i][j];
	    AM[i] = pa90[i][j];
	    E00 += am[i]*am[i];
	    E90 += AM[i]*AM[i];

	    gp  += Fp[i]*Fp[i]+1.e-12;
	    gx  += Fx[i]*Fx[i]+1.e-12;
	    gI  += Fp[i]*Fx[i];
	    xp  += Fp[i]*am[i];
	    xx  += Fx[i]*am[i];
	    XP  += Fp[i]*AM[i];
	    XX  += Fx[i]*AM[i];

	    pix->setdata(am[i],'S',i);  
	    pix->setdata(AM[i],'P',i);  
	    xi00.data[i] = 0.;
	    xi90.data[i] = 0.;

	  }

	  gr = (gp+gx)*soft;
	  gR = (gp-gx)/2.;
	  gc = sqrt(gR*gR+gI*gI);              // norm of complex antenna pattern

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// condition 00 phase
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// find weak vector

	  uc = (xp*gx - xx*gI);                // u cos of rotation to PCF
	  us = (xx*gp - xp*gI);                // u sin of rotation to PCF
	  vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	  vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	  um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	  vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector
	  hh = dotx(u,am,e);
	  cc = dotx(e,e);
	  
	  if((hh*hh-cc)/um<=0. || E00<Eo) U=0;
  
// regulator

	  ii = netx(u,um,v,vm,GAMMA); 
	  if(ii<N_2 && gamma<0.) U=0;       // superclean selection cut
	  
	  uc += xp*gr;                         // u cos of rotation to PCF
	  us += xx*gr;                         // u sin of rotation to PCF
	  
	  if(ii<N_1 && gamma!=0) { 
	    uc = xp*(gc+gR)+xx*gI;
	    us = xx*(gc-gR)+xp*gI;
	  }

	  vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	  vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	  um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	  vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector

// normalize u and v vectors

	  um = sqrt(um);
	  vm = sqrt(vm);

	  hh = gg = 0.;
	  for(i=0; i<nIFO; i++) {
	    u[i] = u[i]/um;
	    v[i] = v[i]/vm;
	    hh += u[i]*am[i];                               // (u*X) - solution 
	    gg += v[i]*am[i];                               // (v*X) - solution 
	  } 

// calculate energy disbalance vectors

	  co=si=0.;
	  for(i=0; i<nIFO; i++) {                           // disbalance vectors
	    cc = local ? am[i]/(am[i]*am[i]+2.) : 1.;
	    pp[i] = cc*(am[i]-hh*u[i])*u[i];
	    qq[i] = cc*((2.*hh*u[i]-am[i])*v[i] + u[i]*u[i]*gg);
	    co += pp[i]*pp[i] + qq[i]*qq[i];                // cos (version 1)
	    si += pp[i]*qq[i];                              // sin
	  }
	  cc = sqrt(si*si+co*co)+1.e-24;
	  co = co/cc;
	  si = si/cc;
          if(!eDisbalance) {co=1.;si=0.;}  

// corrected likelihood

	  hh = gg = 0.;
	  for(i=0; i<nIFO; i++) {                           // solution for h(t,f)
	    e[i] = u[i]*co + v[i]*si;                       // final projection vector
	    v[i] = v[i]*co - u[i]*si;                       // orthogonal v vector
	    u[i] = e[i];
	    hh += u[i]*am[i];                          // solution for hu(t,f)
	    gg += v[i]*am[i];                          // solution for hv(t,f)
	  }

// second iteration

	  co=si=0.;
	  for(i=0; i<nIFO; i++) {                           // disbalance vectors
	    cc = local ? am[i]/(am[i]*am[i]+2.) : 1.;
	    pp[i] = cc*(am[i]-hh*u[i])*u[i];
	    qq[i] = cc*((2.*hh*u[i]-am[i])*v[i] + u[i]*u[i]*gg);
	    co += pp[i]*pp[i] + qq[i]*qq[i];                // cos (version 1)
	    si += pp[i]*qq[i];                              // sin
	  }
	  cc = sqrt(si*si+co*co)+1.e-24;
	  co = co/cc;
	  si = si/cc;
          if(!eDisbalance) {co=1.;si=0.;}  

	  hh = 0.;
	  for(i=0; i<nIFO; i++) {              // corrected u vector for 00 amplitude 
	    u[i] = u[i]*co + v[i]*si;
	    e[i] = am[i]*u[i];
	    hh += e[i]*U;
	  } 

	  La = hh*hh; 
	  wp = wx = 0.;
	  for(i=0; i<nIFO; i++) {              // detector response for 00 data
	    pix->setdata(u[i]*hh/sqrt(nv[i][j]),'W',i);  
	    wp += Fp[i]*u[i]*hh; 
	    wx += Fx[i]*u[i]*hh; 
	  }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// condition 90 phase
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// find weak vector

	  U  = 1;
	  uc = XP*gx - XX*gI;                  // u cos of rotation to PCF
	  us = XX*gp - XP*gI;                  // u sin of rotation to PCF
	  vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	  vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	  um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	  vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector
	  hh = dotx(u,AM,e);
	  cc = dotx(e,e);
	  
	  if((hh*hh-cc)/um<=0. || E90<Eo) U=0;

// sky regulator
	    
	  ii = netx(u,um,v,vm,GAMMA); 
	  if(ii<N_2 && gamma<0.) U=0;       // superclean selection cut
	  
	  uc += XP*gr;                         // u cos of rotation to PCF
	  us += XX*gr;                         // u sin of rotation to PCF
	  
	  if(ii<N_1 && gamma!=0) { 
	    uc = XP*(gc+gR)+XX*gI;
	    us = XX*(gc-gR)+XP*gI;
	  }

	  vc = gp*uc + gI*us;                  // (u*f+)/|u|^2 - 'cos' for v
	  vs = gx*us + gI*uc;                  // (u*fx)/|u|^2 - 'sin' for v
	  um = rotx(Fp,uc,Fx,us,u);            // calculate new response vector
	  vm = rotx(Fx,-vc,Fp,vs,v)+1.e-24;    // calculate complementary vector

// normalize u and v vectors

	  um = sqrt(um);
	  vm = sqrt(vm);

	  hh = gg = 0.;
	  for(i=0; i<nIFO; i++) {
	    u[i] = u[i]/um;
	    v[i] = v[i]/vm;
	    hh += u[i]*AM[i];                               // (u*X) - solution 
	    gg += v[i]*AM[i];                               // (v*X) - solution 
	  } 

// calculate energy disbalance vectors

	  co=si=0.;
	  for(i=0; i<nIFO; i++) {                           // disbalance vectors
	    cc = local ? AM[i]/(AM[i]*AM[i]+2.) : 1.;
	    pp[i] = cc*(AM[i]-hh*u[i])*u[i];
	    qq[i] = cc*((2.*hh*u[i]-AM[i])*v[i] + u[i]*u[i]*gg);
	    co += pp[i]*pp[i] + qq[i]*qq[i];                // cos (version 1)
	    si += pp[i]*qq[i];                              // sin
	  }
	  cc = sqrt(si*si+co*co)+1.e-24;
	  co = co/cc;
	  si = si/cc;
          if(!eDisbalance) {co=1.;si=0.;}  

// corrected likelihood

	  hh = gg = 0.;
	  for(i=0; i<nIFO; i++) {                           // solution for h(t,f)
	    e[i] = u[i]*co + v[i]*si;                       // final projection vector
	    v[i] = v[i]*co - u[i]*si;                       // orthogonal v vector
	    u[i] = e[i];
	    hh += u[i]*AM[i];                               // solution for hu(t,f)
	    gg += v[i]*AM[i];                               // solution for hv(t,f)
	  }

// second iteration

	  co=si=0.;
	  for(i=0; i<nIFO; i++) {                           // disbalance vectors
	    cc = local ? AM[i]/(AM[i]*AM[i]+2.) : 1.;
	    pp[i] = cc*(AM[i]-hh*u[i])*u[i];
	    qq[i] = cc*((2.*hh*u[i]-AM[i])*v[i] + u[i]*u[i]*gg);
	    co += pp[i]*pp[i] + qq[i]*qq[i];                // cos (version 1)
	    si += pp[i]*qq[i];                              // sin
	  }
	  cc = sqrt(si*si+co*co)+1.e-24;
	  co = co/cc;
	  si = si/cc;
          if(!eDisbalance) {co=1.;si=0.;}  


	  hh = 0.;
	  for(i=0; i<nIFO; i++) {              // corrected u vector for 00 amplitude 
	    u[i] = u[i]*co + v[i]*si;
	    e[i] = AM[i]*u[i];
	    hh += e[i]*U;
	  } 

	  La+= hh*hh;                                        // 90 likelihood
	  WP = WX = 0.;
	  for(i=0; i<nIFO; i++) {                            // detector response for 90 data
	    pix->setdata(u[i]*hh/sqrt(nv[i][j]),'U',i);  
	    WP += Fp[i]*u[i]*hh; 
	    WX += Fx[i]*u[i]*hh; 
	  }

	  if(ISG) {
	    ap = (1+AA*AA);
	    cc = (1-AA*AA); 
	    as = SI*cc; 
	    ac = CO*cc;
	    gg = (gp+gx)*ap + (gp-gx)*ac + 2*gI*as;	    
	    hp = (wp*(ap+ac) + 2*AA*WX + wx*as)/gg;
	    hx = (wx*(ap-ac) - 2*AA*WP + wp*as)/gg;
	    HP = (WP*(ap+ac) - 2*AA*wx + WX*as)/gg;
	    HX = (WX*(ap-ac) + 2*AA*wp + WP*as)/gg;
	    La = rotx(Fp,hp,Fx,hx,e)+rotx(Fp,HP,Fx,HX,e);

	    for(i=0; i<nIFO; i++) {       
	      pix->setdata((hp*fp[i][l]+hx*fx[i][l])/GG.data[j],'W',i);  // 00 detector respose
	      pix->setdata((HP*fp[i][l]+HX*fx[i][l])/GG.data[j],'U',i);  // 90 detector respose
	    }
	  }

	  pix->likelihood = La/2.;
	  Lo += La/2;
	  	  
	  if(!core)  pix->core = (E00<Eo && E90<Eo) ? false : true;
	  if(ID) {
	    this->pixeLHood.data[pix->time] = La/2.;
	    this->pixeLNull.data[pix->time] = (E00+E90-La)/2+1.;
	  }

	  count++;
	}

// fill in backward delay configuration

	vtof->clear();
        NETX (
	vtof->push_back(int(M/2)-int(m0[l])); ,
	vtof->push_back(int(M/2)-int(m1[l])); ,
	vtof->push_back(int(M/2)-int(m2[l])); ,
	vtof->push_back(int(M/2)-int(m3[l])); ,
	vtof->push_back(int(M/2)-int(m4[l])); ,
	vtof->push_back(int(M/2)-int(m5[l])); ,
	vtof->push_back(int(M/2)-int(m6[l])); ,
	vtof->push_back(int(M/2)-int(m7[l])); )

// calculation of error regions

	skyProb *= Em/IIm;
	T = cTo.data[k]+getifo(0)->TFmap.start();          // trigger time
	if(iID<=0 && type!='E') getSkyArea(id,n,T);        // calculate error regions

	if(fabs(Lm-Lo)/Lo>1.e-4)
	  cout<<"likelihood warning: "<<Lm<<" "<<Lo<<endl;

	if(Em>0 && ID) { 
	  cout<<"max value: "<<STAT<<" at (theta,phi) = ("<<nLikelihood.getTheta(l)
	      <<","<<nLikelihood.getPhi(l)<<")  Likelihood: loop: "
	      <<Lm<<", final: "<<Lo<<", sky: "<<LPm/2<<", energy: "<<Em/2<<endl;
	  break;
	}

        if (mdcListSize() && n==0) {  // only for lag=0 && simulation mode
          if (this->getwave(id, 0, 'W')) {
          //if (this->getwave(id, 0, 'S')) {
            detector* pd;
            size_t M = this->ifoList.size();
            for(int i=0; i<(int)M; i++) {    // loop over detectors
              pd = this->getifo(i);
              pd->RWFID.push_back(id);  // save cluster ID

              // save reconstructed waveform
              
              double gps = this->getifo(0)->getTFmap()->start();
              double wfStart = gps + pd->waveForm.start();
              //double wfRate = pd->waveForm.rate();
              //double wfSize = pd->waveForm.size();
              
              WSeries<double>* wf = new WSeries<double>;
              *wf = pd->waveForm;
              wf->start(wfStart);
              pd->RWFP.push_back(wf);
            }
          }
        }

	if(ID && !EFEC) {
	  this->nSensitivity.gps = T;
	  this->nAlignment.gps = T;	
	  this->nDisbalance.gps = T;		 
	  this->nLikelihood.gps = T;		 
	  this->nNullEnergy.gps = T;		 
	  this->nCorrEnergy.gps = T;		 
	  this->nCorrelation.gps = T;
	  this->nSkyStat.gps = T;	 
	  this->nEllipticity.gps = T;		 
	  this->nPolarisation.gps = T;		 
	  this->nNetIndex.gps = T;
	}
     }    // end of loop over clusters
     if(ID) break;
  }       // end of loop over time shifts
  return count;
}

//**************************************************************************
//: initialize network data matrix NDM, works with likelihoodB()
//**************************************************************************

bool network::setndm(size_t ID, size_t lag, bool core, int type)
{
   int ii;
   size_t j,n,m,k,K,V,id;
   size_t N = this->ifoList.size(); // number of detectors
   int  N_1 = N>2 ? int(N)-1 : 2;
   int  N_2 = N>2 ? int(N)-2 : 1;
   if(!N) return false;

   wavearray<double> cid;        // cluster ID
   wavearray<double> rat;        // cluster rate
   wavearray<double> lik;        // likelihood
   vector<wavearray<double> > snr(N); // data stream snr
   vector<wavearray<double> > nul(N); // biased null stream
   netpixel* pix;
   wavecomplex gC;
   detector* pd;

   std::vector<int>* vi;
   std::vector<wavecomplex> A;   // antenna patterns
   vectorD esnr; esnr.resize(N); // SkSk snr
   vectorD xsnr; xsnr.resize(N); // XkSk snr
   vectorD am; am.resize(N);     // amplitude vector
   vectorD pp; pp.resize(N);     // energy disbalance vector
   vectorD qq; qq.resize(N);     // complementary disbalance vector
   vectorD ee; ee.resize(N);     // temporary
   vectorD Fp; Fp.resize(N);     // + antenna pattern
   vectorD Fx; Fx.resize(N);     // x antenna pattern
   vectorD u;   u.resize(N);     // PCF u vector
   vectorD v;   v.resize(N);     // PCF v vector
   vectorD e;   e.resize(N);     // PCF final vector
   vectorD r;   r.resize(N);     // PCF final vector

   double um,vm,cc,hh,gr,gc,gI,Et,E,Xp,Xx,gp,gx;
   double gg,co,si,uc,us,vc,vs,gR,a,nr;
   double S_NDM = 0.;
   double S_NUL = 0.;
   double S_NIL = 0.;
   double S_SNR = 0.;
   double x_SNR = 0.;
   double e_SNR = 0.;
   double bIAS  = 0.;
   double response = 0.;
   size_t count = 0;

// regulators hard <- soft <0> weak -> hard
//   gamma =    -1 <-       0       -> 1

   double soft  = delta>0. ? delta : 0.;
   double GAMMA = 1.-gamma*gamma;          // network regulator
   bool status  = false;

   this->gNET = this->aNET = this->eCOR = E = 0.;
   A.resize(N);   

   for(n=0; n<N; n++) {    
     nul[n] = this->wc_List[lag].get((char*)"null",n+1,'W',type);
     snr[n] = this->wc_List[lag].get((char*)"energy",n+1,'S',type);
     this->getifo(n)->sSNR = 0.;
     this->getifo(n)->xSNR = 0.;
     this->getifo(n)->ekXk = 0.;
     this->getifo(n)->null = 0.;
     for(m=0; m<NIFO; m++) { this->getifo(n)->ED[m] = 0.; }
     for(m=0; m<N; m++) { NDM[n][m] = 0.; }
     esnr[n] = xsnr[n] = 0.;
    }

   if(!this->wc_List[lag].size()) return status;
    
   cid = this->wc_List[lag].get((char*)"ID",0,'S',type);
   rat = this->wc_List[lag].get((char*)"rate",0,'S',type);
   lik = this->wc_List[lag].get((char*)"like",0,'S',type);
   K   = cid.size();
    
   for(k=0; k<K; k++) {      // loop over clusters 
      
     id = size_t(cid[k]+0.1);
     if(id != ID) continue;
      
     vi = &(this->wc_List[lag].cList[ID-1]);
     V = vi->size();
     if(!V) continue;

// normalization of antenna patterns
// calculation of the rotation angles
// calculation of the likelihood matrix

     for(j=0; j<V; j++) { 
	
       pix = this->wc_List[lag].getPixel(ID,j);
       if(!pix) {
	 cout<<"network::setndm() error: NULL pointer"<<endl;
	 exit(1);
       }
       if(!pix->core && core) continue;
       if(pix->rate != rat.data[k]) continue;

       count++;
       gr=Et=gg=Xp=Xx = 0.; 
       gC=0.;

       for(n=0; n<N; n++) {                 // calculate noise normalization
	 nr    = pix->getdata('N',n);       // noise rms
	 gg   += 1./(nr*nr);
       }
       gg = sqrt(gg);

       for(n=0; n<N; n++) {                 // calculate patterns
	 am[n] = pix->getdata('S',n);       // snr amplitude
	 nr    = pix->getdata('N',n)*gg;    // noise rms
	 Et   += am[n]*am[n];
	 pd    = this->getifo(n);
	 Fp[n] = pd->mFp.get(pix->theta,pix->phi)/nr;
	 Fx[n] = pd->mFx.get(pix->theta,pix->phi)/nr;

	 A[n].set(Fp[n],Fx[n]);

	 gr   += A[n].abs()/2.;
	 gC   += A[n]*A[n];
	 Xp   += am[n]*Fp[n];
	 Xx   += am[n]*Fx[n];
       }
       E += Et;
       gc = gC.mod()/2.;
       gR = gC.real()/2.;
       gI = gC.imag()/2.;
       gp = gr+gR+1.e-12; 
       gx = gr-gR+1.e-12;

       this->gNET += (gr+gc)*Et; 
       this->aNET += (gr-gc)*Et; 

// find weak vector

       uc = Xp*gx - Xx*gI;          // u cos of rotation to PCF
       us = Xx*gp - Xp*gI;          // u sin of rotation to PCF
       vc = gp*uc + gI*us;          // cos of rotation to PCF for v
       vs = gx*us + gI*uc;          // sin of rotation to PCF for v

       um = vm = hh = cc = 0.;
       for(n=0; n<N; n++) {
	 u[n] = Fp[n]*uc + Fx[n]*us;       // u - weak vector
	 v[n] = Fp[n]*vs - Fx[n]*vc;       // v - orthogonal to u
	 um += u[n]*u[n];
	 vm += v[n]*v[n];
	 hh += u[n]*am[n];
	 cc += u[n]*am[n]*u[n]*am[n];
       }
       vm += 1.e-24;                       // H1H2 regulator

       if((hh*hh-cc)/um <= 0.) continue;   // negative correlated energy

// regulator

       ii = 0;
       for(n=0; n<N; n++) {
	 if(u[n]*u[n]/um > 1-GAMMA) ii++;
	 if(u[n]*u[n]/um+v[n]*v[n]/vm > GAMMA) ii--;
       }
       this->iNET += ii*Et; 

       if(ii<N_2 && gamma<0.) continue;    // superclean set

       gg = (gp+gx)*soft;
       uc = Xp*(gx+gg) - Xx*gI;            // u cos of rotation to PCF
       us = Xx*(gp+gg) - Xp*gI;            // u sin of rotation to PCF

       if(ii<N_1 && gamma!=0) { 
	 uc = Xp*(gc+gR)+Xx*gI;
	 us = Xx*(gc-gR)+Xp*gI;
       }

       vc = gp*uc + gI*us;                 // (u*f+)/|u|^2 - 'cos' for v
       vs = gx*us + gI*uc;                 // (u*fx)/|u|^2 - 'sin' for v
       um = vm = 0.;
       for(n=0; n<N; n++) {
	 u[n] = Fp[n]*uc + Fx[n]*us;
	 v[n] = Fp[n]*vs - Fx[n]*vc;
	 um += u[n]*u[n];                  // calculate u and return its norm
	 vm += v[n]*v[n];                  // calculate u and return its norm
       }
       vm += 1.e-24;                       // H1H2 regulator

// calculate unity vectors in PCF

       hh = gg = 0.;
       for(n=0; n<N; n++) {
	 u[n] /=sqrt(um); 
	 v[n] /=sqrt(vm);                            // unity vectors in PCF
	 hh += u[n]*am[n];                           // (u*X) - solution 
	 gg += v[n]*am[n];                           // (v*X) - solution 
       } 
       
// calculate energy disbalance vectors
       
       co = si = 0.;
       for(n=0; n<N; n++) {                          // disbalance vectors
	 cc = local ? am[n]/(am[n]*am[n]+2.) : 1.;
	 qq[n] = cc*((2*hh*u[n]-am[n])*v[n]+u[n]*u[n]*gg);   // complementary energy disbalance
	 pp[n] = cc*(am[n]-hh*u[n])*u[n];                    // energy disbalance
	 co += pp[n]*pp[n] + qq[n]*qq[n];            // cos
	 si += pp[n]*qq[n];                          // sin
       }
       cc = atan2(si,co+1.e-24);                     // rotation angle
       co = cos(cc);                                 // cos(psi)
       si = sin(cc);                                 // sin(psi)
       if(!eDisbalance) {co=1.;si=0.;}  
       
       hh = gg = 0.;
       for(n=0; n<N; n++) {
	 e[n] = u[n]*co+v[n]*si;                     // final projection vector
	 r[n] = v[n]*co-u[n]*si;
	 hh  += e[n]*am[n];
	 gg  += r[n]*am[n];
       }

// second iteration
       
       co = si = 0.;
       for(n=0; n<N; n++) {                          // disbalance vectors
	 cc = local ? am[n]/(am[n]*am[n]+2.) : 1.;
	 pp[n] = cc*(am[n]-hh*e[n])*e[n];                    // energy disbalance
	 qq[n] = cc*((2*hh*e[n]-am[n])*r[n]+e[n]*e[n]*gg);   // complementary energy disbalance
	 co += pp[n]*pp[n] + qq[n]*qq[n];            // cos (version 1)
	 si += pp[n]*qq[n];                          // sin
       }
       cc = atan2(si,co+1.e-24);                     // rotation angle
       co = cos(cc);                                 // cos(psi)
       si = sin(cc);                                 // sin(psi)
       if(!eDisbalance) {co=1.;si=0.;}  
       
       for(n=0; n<N; n++) {
	 e[n] = e[n]*co+r[n]*si;                     // final projection vector
       }

// likelihood matrix

       for(n=0; n<N; n++) {                          // loop over NDM elements 
	 response = 0.;
	 this->getifo(n)->null += e[n]*e[n];         // bias

	 for(m=0; m<N; m++) {        
	   gg = e[n]*am[n]*e[m]*am[m];               // en*an * em*am
	   NDM[n][m] += gg;
	   response  += e[n]*e[m]*am[m];             // detector response
	   if(n!=m) this->eCOR += gg;                // correlated energy	   
	   status = true;
	 }

	 esnr[n] += response*response;               // reconstructed SNR: Sk*Sk
	 xsnr[n] += am[n]*response;                  // reconstructed SNR: Xk*Sk
	 e_SNR   += response*response;               // total reconstructed SNR: sum Sk*Sk
	 x_SNR   += am[n]*response;                  // total reconstructed SNR: sum Xk*Sk
	 this->getifo(n)->ED[3] += fabs(response*(am[n]-response));
	 this->getifo(n)->ED[4] += fabs(response*(am[n]-response));
       }
     }
     
// check normalization
     
     this->norm  = x_SNR/e_SNR;                      // norm factor
     if(fabs(this->norm-1) > 1.e-4)
       cout<<"network::setndm(): incorrect likelihood normalization: "<<this->norm<<endl;

     for(n=0; n<N; n++) {
       bIAS += this->getifo(n)->null;
       this->getifo(n)->null += nul[n].data[k];      // detector unbiased null stream
       this->getifo(n)->sSNR = esnr[n];              // s-energy of the detector response
       this->getifo(n)->xSNR = xsnr[n];              // x-energy of the detector response
       S_NUL += this->getifo(n)->null;               // total unbiased null stream
       S_NIL += nul[n].data[k];                      // biased null stream
       S_SNR += snr[n].data[k];                      // total energy 

       this->getifo(n)->ED[0] = (esnr[n]-xsnr[n]); 
       this->getifo(n)->ED[1] = fabs(esnr[n]-xsnr[n]);
       this->getifo(n)->ED[2] = fabs(esnr[n]-xsnr[n]);

       for(m=0; m<N; m++) S_NDM += NDM[n][m]; 

     }

     if(count) { this->gNET /= E; this->aNET /= E; this->iNET /= E;}

     a = S_NDM - lik.data[k]; 
     if(fabs(a)/S_SNR>1.e-6) 
       cout<<"ndm-likelihood mismatch:  "<<a<<" "<<S_NDM<<" "<<lik.data[k]<<" "<<norm<<endl;
     a = fabs(1 - (S_NDM+S_NIL)/S_SNR)/count;
     if(a>1.e-5) 
       cout<<"biased energy disbalance:  "<<a<<"  "<<S_SNR-S_NDM<<"  "<<S_NIL<<"  size="<<count<<endl;

     if(status) break;
   }
   return status;
}


//**************************************************************************
//: initialize network data matrix (NDM), works with likelihoodI
//**************************************************************************
bool network::SETNDM(size_t ID, size_t lag, bool core, int type)
{
   int ii;
   size_t j,n,m,k,K,V;
   size_t N = this->ifoList.size(); // number of detectors
   int  N_1 = N>2 ? int(N)-1 : 2;
   int  N_2 = N>2 ? int(N)-2 : 1;
   if(!N) return false;

   wavearray<double> cid;        // cluster ID
   wavearray<double> rat;        // cluster rate
   wavearray<double> lik;        // likelihood
   vector<wavearray<double> > snr(N); // data stream snr
   vector<wavearray<double> > nul(N); // biased null stream
   vector<wavearray<double> > SNR(N); // data stream snr
   vector<wavearray<double> > NUL(N); // biased null stream
   netpixel* pix;
   wavecomplex gC,Z;
   detector* pd;

   std::vector<int>* vint;
   std::vector<wavecomplex> A;   // patterns in DPF
   vectorD esnr; esnr.resize(N); // SkSk snr
   vectorD xsnr; xsnr.resize(N); // XkSk snr
   vectorD ssnr; ssnr.resize(N); // total energy of non shifted detector output
   vectorD SSNR; SSNR.resize(N); // total energy of phase shifted detector output
   vectorD h00;   h00.resize(N); // unmodeled 00 response vector
   vectorD h90;   h90.resize(N); // unmodeled 00 response vector
   vectorD u00;   u00.resize(N); // unmodeled 00 unit response vector
   vectorD u90;   u90.resize(N); // unmodeled 90 unit response vector
   vectorD am;     am.resize(N); // 00 phase response
   vectorD AM;     AM.resize(N); // 90 phase response
   vectorD qq;     qq.resize(N); // energy disbalance vector
   vectorD pp;     pp.resize(N); // energy disbalance vector
   vectorD ee;     pp.resize(N); // temporary
   vectorD Fp;     Fp.resize(N); // + antenna pattern
   vectorD Fx;     Fx.resize(N); // x antenna pattern
   vectorD  u;      u.resize(N); // unity vector
   vectorD  v;      v.resize(N); // unity vector
   vectorD  e;      e.resize(N); // unity vector

   double a,b,aa,psi,gg,gr,gc,gI,gR,E90,E00,E,fp,fx,vc,vs;
   double xx,xp,XX,XP,uc,us,co,hh,gp,gx,xi00,xi90,o00,o90;
   double hgw,HGW,wp,WP,wx,WX,HH,um,vm,si,cc;
   double S_NDM = 0.;
   double S_NUL = 0.;
   double s_snr = 0.;
   double S_SNR = 0.;
   size_t count = 0;

// regulator  soft <- weak -> hard
//   gamma =    -1 <-   0  -> 1

   double soft  = delta>0. ? delta : 0.;
   double GAMMA = 1.-gamma*gamma;                // network regulator
   double Eo = this->acor*this->acor*N;
   
   double nC = this->MRA ? 1. : 2.;		// NDM normalization Coefficient

   bool status  = false;
   bool ISG     = false;

   if(tYPe=='I' || tYPe=='S' || tYPe=='G') ISG = true;
   if(tYPe=='i' || tYPe=='s' || tYPe=='g') ISG = true;
   
   A.resize(N);   

   this->gNET = 0.;
   this->aNET = 0.;
   this->iNET = 0.;
   this->eCOR = 0.;
   E = 0.;

   for(n=0; n<N; n++) {    
     nul[n] = this->wc_List[lag].get((char*)"null",n+1,'W',type);
     NUL[n] = this->wc_List[lag].get((char*)"null",n+1,'U',type);
     snr[n] = this->wc_List[lag].get((char*)"energy",n+1,'S',type);
     SNR[n] = this->wc_List[lag].get((char*)"energy",n+1,'P',type);
     this->getifo(n)->sSNR = 0.;
     this->getifo(n)->xSNR = 0.;
     this->getifo(n)->ekXk = 0.;
     this->getifo(n)->null = 0.;
     for(m=0; m<5; m++) { this->getifo(n)->ED[m] = 0.; }
     for(m=0; m<N; m++) { NDM[n][m] = 0.; }
     esnr[n]=xsnr[n]=ssnr[n]=SSNR[n]=0.;
    }

   if(!this->wc_List[lag].size()) return false;
    
   cid = this->wc_List[lag].get((char*)"ID",0,'S',type);
   rat = this->wc_List[lag].get((char*)"rate",0,'S',type);
   lik = this->wc_List[lag].get((char*)"like",0,'S',type);
   K   = cid.size();
    
   for(k=0; k<K; k++) {      // loop over clusters 
      
     if(size_t(cid[k]+0.1) != ID) continue;
      
     vint = &(this->wc_List[lag].cList[ID-1]);
     V = vint->size();
     if(!V) continue;

     // normalization of antenna patterns
     // calculation of the rotation angles
     // calculation of the likelihood matrix

     for(j=0; j<V; j++) { 
	
       pix = this->wc_List[lag].getPixel(ID,j);
       if(!pix) {
	 cout<<"network::SETNDM() error: NULL pointer"<<endl;
	 exit(1);
       }
       if(!pix->core && core) continue;
       if((pix->rate != rat.data[k]) && type) continue;

    
       psi = 2*pix->polarisation;           // polarisation rotation angle
       Z.set(cos(psi),-sin(psi));            

       count++;
       gr=gg=xp=xx=XP=XX=E00=E90 = 0.;
       o00 = o90 = 1.;
       gC = 0.;

       for(n=0; n<N; n++) {
	 b    = pix->getdata('N',n);        // noise rms
	 gg  += 1./b/b;                     // noise normalization
       }
       gg = sqrt(gg);

       for(n=0; n<N; n++) {                 // calculate patterns
	 am[n] = pix->getdata('S',n);       // snr amplitude
	 AM[n] = pix->getdata('P',n);       // snr 90 degrees amplitude
	 E00  += am[n]*am[n];            
	 E90  += AM[n]*AM[n];            
	 b     = pix->getdata('N',n)*gg;    // noise rms
	 pd    = this->getifo(n);
	 fp    = pd->mFp.get(pix->theta,pix->phi);
	 fx    = pd->mFx.get(pix->theta,pix->phi);
	 A[n].set(fp/b,fx/b);
	 A[n] *= Z;                         // rotate patterns
	 Fp[n] = A[n].real();
	 Fx[n] = A[n].imag();
	 gr   += A[n].abs()/2.;
	 gC   += A[n]*A[n];
	 xp   += Fp[n]*am[n];
	 xx   += Fx[n]*am[n];
	 XP   += Fp[n]*AM[n];
	 XX   += Fx[n]*AM[n];
       }
       
       E += E00+E90;
       gc = gC.mod()/2.;
       gR = gC.real()/2.;
       gI = gC.imag()/2.;
       gp = gr+gR+1.e-12; 
       gx = gr-gR+1.e-12;
       aa = pix->ellipticity;

       this->norm = aa;                               // save to store in root file as norm
       this->gNET += (gr+gc)*(E00+E90); 
       this->aNET += (gr-gc)*(E00+E90); 

//====================================================
// calculate unity vectors in PCF for 00 degree phase
//====================================================

// find weak vector

       uc = xp*gx - xx*gI;          // u cos of rotation to PCF
       us = xx*gp - xp*gI;          // u sin of rotation to PCF
       vc = gp*uc + gI*us;          // cos of rotation to PCF for v
       vs = gx*us + gI*uc;          // sin of rotation to PCF for v

       um = vm = hh = cc = 0.;
       for(n=0; n<N; n++) {
	 u[n] = Fp[n]*uc + Fx[n]*us;
	 v[n] = Fp[n]*vs - Fx[n]*vc;
	 um += u[n]*u[n];
	 vm += v[n]*v[n];
	 hh += u[n]*am[n];
	 cc += u[n]*am[n]*u[n]*am[n];
       }
       vm += 1.e-24;                       // H1H2 regulator

       if((hh*hh-cc)/um<=0. || E00<Eo) o00=0.;

// sky regulator

       ii = 0;
       for(n=0; n<N; n++) {
	 if(u[n]*u[n]/um > 1-GAMMA) ii++;
	 if(u[n]*u[n]/um+v[n]*v[n]/vm > GAMMA) ii--;
       }
       this->iNET += ii*E00; 
       if(ii<N_2 && gamma<0.) o00=0.;      // superclean selection cut

       gg = (gp+gx)*soft; 
       uc = xp*(gx+gg) - xx*gI;            // u cos of rotation to PCF
       us = xx*(gp+gg) - xp*gI;            // u sin of rotation to PCF

       if(ii<N_1 && gamma!=0) { 
	 uc = xp*(gc+gR)+xx*gI;
	 us = xx*(gc-gR)+xp*gI;
       }

       vc = gp*uc + gI*us;                 // (u*f+)/|u|^2 - 'cos' for v
       vs = gx*us + gI*uc;                 // (u*fx)/|u|^2 - 'sin' for v
       um = vm = 0.;
       for(n=0; n<N; n++) {
	 u[n] = Fp[n]*uc + Fx[n]*us;
	 v[n] = Fp[n]*vs - Fx[n]*vc;
	 um += u[n]*u[n];                  // calculate u and return its norm
	 vm += v[n]*v[n];                  // calculate u and return its norm
       }
       vm += 1.e-24;                       // H1H2 regulator

// calculate unity vectors in PCF

       hh = gg = 0.;
       for(n=0; n<N; n++) {
	 u[n] /=sqrt(um); 
	 v[n] /=sqrt(vm);                            // unity vectors in PCF
	 hh += u[n]*am[n];                           // (u*X) - solution 
	 gg += v[n]*am[n];                           // (v*X) - solution 
       }        

// calculate energy disbalance vectors

       co=si=0.;
       for(n=0; n<N; n++) {                              // disbalance vectors
	 cc = local ? am[n]/(am[n]*am[n]+2.) : 1.;
	 pp[n] = cc*(am[n]-hh*u[n])*u[n];
	 qq[n] = cc*((2.*hh*u[n]-am[n])*v[n] + u[n]*u[n]*gg);
	 co += pp[n]*pp[n] + qq[n]*qq[n];                // cos (version 1)
	 si += pp[n]*qq[n];                              // sin
       }
       cc = sqrt(si*si+co*co)+1.e-24;
       co = co/cc;
       si = si/cc;
       if(!eDisbalance) {co=1.;si=0.;}  

// corrected likelihood

       hh = gg = 0.;
       for(n=0; n<N; n++) {                              // solution for h(t,f)
	 e[n] = u[n]*co + v[n]*si;                       // final projection vector
	 v[n] = v[n]*co - u[n]*si;                       // orthogonal v vector
	 u[n] = e[n];
	 hh += u[n]*am[n];                               // solution for hu(t,f)
	 gg += v[n]*am[n];                               // solution for hv(t,f)
       }

// second iteration

       co=si=0.;
       for(n=0; n<N; n++) {                              // disbalance vectors
	 cc = local ? am[n]/(am[n]*am[n]+2.) : 1.;
	 pp[n] = cc*(am[n]-hh*u[n])*u[n];
	 qq[n] = cc*((2.*hh*u[n]-am[n])*v[n] + u[n]*u[n]*gg);
	 co += pp[n]*pp[n] + qq[n]*qq[n];                // cos (version 1)
	 si += pp[n]*qq[n];                              // sin
       }
       cc = sqrt(si*si+co*co)+1.e-24;
       co = co/cc;
       si = si/cc;
       if(!eDisbalance) {co=1.;si=0.;}  

       hh = wp = wx = 0.;
       for(n=0; n<N; n++) {
	 u[n] = u[n]*co+v[n]*si;                         // final projection vector
	 hh  += u[n]*am[n];
	 wp  += Fp[n]*u[n];
	 wx  += Fx[n]*u[n]*aa;
       }
       for(n=0; n<N; n++) {
	 h00[n] = u[n]*am[n]*o00;
	 u00[n] = u[n]*o00;
	  am[n] = hh*u[n]*o00;                           // 90 detector response
       }

//==============================================
// calculate unity vectors in PCF for 90 phase
//==============================================

// find weak vector

       uc = XP*gx - XX*gI;          // u cos of rotation to PCF
       us = XX*gp - XP*gI;          // u sin of rotation to PCF
       vc = gp*uc + gI*us;          // cos of rotation to PCF for v
       vs = gx*us + gI*uc;          // sin of rotation to PCF for v

       um = vm = hh = cc = 0.;
       for(n=0; n<N; n++) {
	 u[n] = Fp[n]*uc + Fx[n]*us;
	 v[n] = Fp[n]*vs - Fx[n]*vc;
	 um += u[n]*u[n];
	 vm += v[n]*v[n];
	 hh += u[n]*AM[n];
	 cc += u[n]*AM[n]*u[n]*AM[n];
       }
       vm += 1.e-24;                       // H1H2 regulator

       if((hh*hh-cc)/um<=0. || E90<Eo) o90=0.;

// sky regulator

       ii = 0;
       for(n=0; n<N; n++) {
	 if(u[n]*u[n]/um > 1-GAMMA) ii++;
	 if(u[n]*u[n]/um+v[n]*v[n]/vm > GAMMA) ii--;
       }
       this->iNET += ii*E90; 
       if(ii<N_2 && gamma<0.) o90=0.;      // superclean selection cut

       gg = (gp+gx)*soft; 
       uc = XP*(gx+gg) - XX*gI;            // u cos of rotation to PCF
       us = XX*(gp+gg) - XP*gI;            // u sin of rotation to PCF

       if(ii<N_1 && gamma!=0) { 
	 uc = XP*(gc+gR)+XX*gI;
	 us = XX*(gc-gR)+XP*gI;
       }

       vc = gp*uc + gI*us;                 // (u*f+)/|u|^2 - 'cos' for v
       vs = gx*us + gI*uc;                 // (u*fx)/|u|^2 - 'sin' for v
       um = vm = 0.;
       for(n=0; n<N; n++) {
	 u[n] = Fp[n]*uc + Fx[n]*us;
	 v[n] = Fp[n]*vs - Fx[n]*vc;
	 um += u[n]*u[n];                  // calculate u and return its norm
	 vm += v[n]*v[n];                  // calculate u and return its norm
       }
       vm += 1.e-24;                       // H1H2 regulator

// calculate unity vectors in PCF

       hh = gg = 0.;
       for(n=0; n<N; n++) {
	 u[n] /=sqrt(um); 
	 v[n] /=sqrt(vm);                            // unity vectors in PCF
	 hh += u[n]*AM[n];                           // (u*X) - solution 
	 gg += v[n]*AM[n];                           // (v*X) - solution 
       }        

// calculate energy disbalance vectors

       co=si=0.;
       for(n=0; n<N; n++) {                              // disbalance vectors
	 cc = local ? AM[n]/(AM[n]*AM[n]+2.) : 1.;
	 pp[n] = cc*(AM[n]-hh*u[n])*u[n];
	 qq[n] = cc*((2.*hh*u[n]-AM[n])*v[n] + u[n]*u[n]*gg);
	 co += pp[n]*pp[n] + qq[n]*qq[n];                // cos (version 1)
	 si += pp[n]*qq[n];                              // sin
       }
       cc = sqrt(si*si+co*co)+1.e-24;
       co = co/cc;
       si = si/cc;
       if(!eDisbalance) {co=1.;si=0.;}  

// corrected likelihood

       hh=gg = 0.;
       for(n=0; n<N; n++) {                              // solution for h(t,f)
	 e[n] = u[n]*co + v[n]*si;                       // final projection vector
	 v[n] = v[n]*co - u[n]*si;                       // orthogonal v vector
	 u[n] = e[n];
	 hh += u[n]*AM[n];                               // solution for hu(t,f)
	 gg += v[n]*AM[n];                               // solution for hv(t,f)
       }

// second iteration

       co=si=0.;
       for(n=0; n<N; n++) {                              // disbalance vectors
	 cc = local ? AM[n]/(AM[n]*AM[n]+2.) : 1.;
	 pp[n] = cc*(AM[n]-hh*u[n])*u[n];
	 qq[n] = cc*((2.*hh*u[n]-AM[n])*v[n] + u[n]*u[n]*gg);
	 co += pp[n]*pp[n] + qq[n]*qq[n];                // cos (version 1)
	 si += pp[n]*qq[n];                              // sin
       }
       cc = sqrt(si*si+co*co)+1.e-24;
       co = co/cc;
       si = si/cc;
       if(!eDisbalance) {co=1.;si=0.;}  

       HH = WP = WX = 0.;
       for(n=0; n<N; n++) {
	 u[n] = u[n]*co+v[n]*si;                         // final projection vector
	 HH  += u[n]*AM[n];
	 WP  += Fp[n]*u[n];
	 WX  += Fx[n]*u[n]*aa;
       }
       for(n=0; n<N; n++) {
	 h90[n] = u[n]*AM[n]*o90;
	 u90[n] = u[n]*o90;
	  AM[n] = HH*u[n]*o90;                        // 90 detector response
       }

       gg = gp + aa*aa*gx;                            // inverse network sensitivity
       cc = ISG ? (wp*WX-wx*WP)/gg    : 0.0;          // cross term
       hh = ISG ? (wp*wp+wx*wx)/gg/nC : 1./nC;        // 00 L term
       HH = ISG ? (WP*WP+WX*WX)/gg/nC : 1./nC;        // 90 L term

//==============================================
// likelihood matrix
//==============================================

       hgw = HGW = 0.;

       for(n=0; n<N; n++) {
	 hgw += (am[n]*Fp[n] + aa*AM[n]*Fx[n])/gg;    // h(0 deg)
	 HGW += (AM[n]*Fp[n] - aa*am[n]*Fx[n])/gg;    // h(90 deg)
       }

       for(n=0; n<N; n++) {                           // loop over NDM elements 
	 xi00 = 0.;
	 xi90 = 0.;

	 for(m=0; m<N; m++) {            

	   a = h00[n]*h00[m]*hh
	     + h90[n]*h90[m]*HH
 	     + h00[n]*h90[m]*cc;

	   xi00 += u00[n]*h00[m];                     // unmodeled 00 response
	   xi90 += u90[n]*h90[m];                     // unmodeled 90 response

	   NDM[n][m] += a;
	   if(n!=m) this->eCOR += a;                  // correlated energy	   	 

	   a = h00[n]*h00[m] - h90[n]*h90[m];         // unmodeled diff	   
	   this->getifo(n)->ED[3] += a;
	   status = true;
	 }

	 a = u00[n]*u00[n]*hh + u90[n]*u90[n]*HH;     // bias
	 this->getifo(n)->null += a;

	 xx = pix->getdata('S',n);                    // 0-phase
	 if(ISG) xi00 = hgw*Fp[n]-aa*HGW*Fx[n];       // 0-phase response
	 esnr[n] += xi00*xi00;                        // reconstructed SNR: Sk*Sk
	 xsnr[n] += xx*xi00;                          // reconstructed SNR: Xk*Sk
	 ssnr[n] += xx*xx;                            // total energy of non shifted output
	 this->getifo(n)->ED[1] += xi00*(xx-xi00);
	 this->getifo(n)->ED[4] += fabs(xi00*(xx-xi00));

	 XX = pix->getdata('P',n);                    // 90-phase
	 if(ISG) xi90 = HGW*Fp[n]+aa*hgw*Fx[n];       // 90-phase response
	 esnr[n] += xi90*xi90;                        // reconstructed SNR: Sk*Sk
	 xsnr[n] += XX*xi90;                          // reconstructed SNR: Xk*Sk
	 SSNR[n] += XX*XX;                            // total energy of phase shifted output
	 this->getifo(n)->ED[2] += xi90*(XX-xi90);
	 this->getifo(n)->ED[4] += fabs(xi90*(XX-xi90));

       }
     }

// take into account norm-factor 
     
     for(n=0; n<N; n++) {   
                         
       b = (SSNR[n]+ssnr[n] - esnr[n])/nC;
       this->getifo(n)->null += b;                    // detector biased null stream
       this->getifo(n)->sSNR = esnr[n]/nC;            // s-energy of the detector response
       this->getifo(n)->xSNR = xsnr[n]/nC;            // x-energy of the detector response
       S_NUL += b;                                    // total biased null stream
       s_snr += ssnr[n];                              // total energy 
       S_SNR += SSNR[n];                              // total energy of phase shifted stream 

       for(m=0; m<N; m++) S_NDM += NDM[n][m]; 
 
       if(fabs(ssnr[n]-snr[n].data[k])/ssnr[n] > 1.e-6 || fabs(SSNR[n]-SNR[n].data[k])/SSNR[n]>1.e-6)
	  cout<<"SETNDM()-likelihoodI() SNR mismatch: "<<n<<" "<<ID<<" "
	     <<ssnr[n]<<":"<<snr[n].data[k]<<" "<<SSNR[n]<<":"<<SNR[n].data[k]<<endl;
     }

     if(count) { this->gNET /= E; this->aNET /= E; this->iNET /= E; }

     a = S_NDM - lik.data[k]; 
     if(fabs(a)/S_NDM>1.e-6) 
       cout<<"NDM-likelihood mismatch:  "<<a<<" "<<S_NDM<<" "<<lik.data[k]<<endl;

     a = fabs(1 - nC*(S_NDM+S_NUL)/(s_snr+S_SNR))/count;
     if(a>1.e-5) 
       cout<<"biased energy disbalance:   "<<a<<"  "<<S_NDM+S_NUL<<" "<<(s_snr+S_SNR)/nC<<endl;

     if(status) break;
   }
   return status;
}


//**************************************************************************
//: set parameters for time shift analysis 
//**************************************************************************
int network::setTimeShifts(size_t lagSize, double lagStep, 
			   size_t lagOff, size_t lagMax, 
			   const char* fname, const char* fmode, size_t* lagSite) {
  netcluster wc;
  size_t nIFO = this->ifoList.size();
  this->wc_List.clear(); this->livTime.clear();

  if(lagStep<=0.) {
    cout << "network::setTimeShifts : lagStep must be positive" << endl;
    exit(1);
  }

  if(lagSize<1) lagSize=1;

  if(strcmp(fmode,"r") && strcmp(fmode,"w") && strcmp(fmode,"s")) {
    cout << "network::setTimeShifts : bad fmode : must be r/w/s" << endl;
    exit(1);
  }

  if(fname) { if(strlen(fname)<1) fname = NULL; }  // check file name

  TRandom3 rnd;
  size_t n,m,k;
  size_t nList = 0;
  size_t maxList;
  size_t lagIDS = lagOff;
  int*   lagList[NIFO];
  int    lagL[NIFO];
  int    lagH[NIFO];
  int    N[NIFO];
  int    id[NIFO];  
  int    ID[NIFO];  
  int    maxIter = 10000000;  
  detector* pd = NULL;  

  for(n=0;n<NIFO;n++) {
    lagL[n] = kMinInt;
    lagH[n] = kMaxInt;
    N[n]    = 0;
    id[n]   = 0;  
    ID[n]   = 0;  
  }

// default lag list

  if(lagMax==0) {

    lagIDS += int(getifo(0)->sHIFt/lagStep);
    maxList = lagSize+lagIDS;

    for(n=0; n<nIFO; n++) lagList[n] = new int[maxList];
    for(m=0; m<maxList; m++) {
      for(n=0; n<nIFO; n++) {
	pd = this->getifo(n);
	lagList[n][m] = n==0 ? m : int(pd->sHIFt/lagStep);
      }
    }
    nList=maxList;
    goto final;
  }

// read list of lags from file fname fmode="r" or from string fname fmode="s"

  if(fname && (!strcmp(fmode,"r") || !strcmp(fmode,"s"))) {
    if(!strcmp(fmode,"r")) {	// read from file

      ifstream in; 
      in.open(fname, ios::in);
      if(!in.good()) {
        cout << "network::setTimeShifts : Error Opening File : " << fname << endl;
        exit(1);
      }

      char str[1024];
      int fpos=0;
      maxList=0;
      while(true) {
        in.getline(str,1024);
        if (!in.good()) break;
        if(str[0] != '#') maxList++;
      }

      for(n=0; n<nIFO; n++) lagList[n] = new int[maxList];
      in.clear(ios::goodbit);
      in.seekg(0, ios::beg);
      while(true) {
        fpos=in.tellg();
        in.getline(str,1024);
        if(str[0] == '#') continue;
        in.seekg(fpos, ios::beg);
        in >> m;
        for(n=0; n<nIFO; n++) in >> lagList[n][m];
        if (!in.good()) break;
        fpos=in.tellg();
        in.seekg(fpos+1, ios::beg);
      }

      in.close();
    }

    if(!strcmp(fmode,"s")) {	// read from string

      stringstream in;
      in << fname;		// when fmode='s' then fname contains the lag list

      char str[1024];
      int fpos=0;
      maxList=0;
      while(true) {
        in.getline(str,1024);
        if (!in.good()) break;
        if(str[0] != '#') maxList++;
      }

      for(n=0; n<nIFO; n++) lagList[n] = new int[maxList];
      in.clear(ios::goodbit);
      in.seekg(0, ios::beg);
      while(true) {
        fpos=in.tellg();
        in.getline(str,1024);
        if(str[0] == '#') continue;
        in.seekg(fpos, ios::beg);
        in >> m;
        for(n=0; n<nIFO; n++) in >> lagList[n][m];
        if (!in.good()) break;
        fpos=in.tellg();
        in.seekg(fpos+1, ios::beg);
      }
    }

// check boundaries

    int lagP=0;
    for (n=0; n<nIFO; n++) {lagL[n]=0;lagH[n]=lagMax;}
    for(m=0; m<maxList; m++){
      bool check=true;
      for (n=0; n<nIFO; n++) id[n]=lagList[n][m];

// Lags must be in the range 0:lagMax

      for (n=0; n<nIFO; n++) if(id[n]<0||id[n]>int(lagMax)) check=false;

// Difference between 2 lags belonging to different detectors must be <= lagMax

      for (int i=nIFO-1;i>=0;i--) {
        for (int j=i-1;j>=0;j--) {
          if (!(((id[i]-id[j])>=(lagL[i]-lagH[j]))&&
                ((id[i]-id[j])<=(lagH[i]-lagL[j])))) check=false;
        }
      }
      if (check) lagP++;
    }

    if(lagP==0) {
      cout << "network::setTimeShifts : no lags in the list" << endl;
      cout << "lagP : " << lagP << " " << lagSize << endl;
      exit(1);
    }
    if(lagP!=int(maxList)) {
      cout << "network::setTimeShifts : lags out of lagMax" << endl;
      cout << "lagP : " << lagP << " " << lagSize << endl;
      exit(1);
    }
    nList=maxList;
    goto final;
  }

// extended lags list

  if(lagSite!=NULL) for(n=0; n<nIFO; n++) {
    if(lagSite[n] >= nIFO) {
      cout << "network::setTimeShifts : Error lagSite - value out of range " << endl;
      exit(-1);
    }
  } 

  for(n=1; n<nIFO; n++) N[n]=lagMax;
  for(n=0; n<nIFO; n++) {lagL[n]=0;lagH[n]=lagMax;}

  maxList=lagOff+lagSize;  
  for(n=0; n<nIFO; n++) lagList[n] = new int[maxList];
  for(n=0; n<nIFO; n++) lagList[n][nList]=0; 
  nList++;

  rnd.SetSeed(13);
  for (int k=0;k<maxIter;k++) {
    for(n=0; n<nIFO; n++) ID[n] = TMath::Nint(rnd.Uniform(-(N[n]+0.5),N[n]+0.5));
    for(n=0; n<nIFO; n++) id[n] = (lagSite==NULL) ? ID[n] : ID[lagSite[n]];
    bool check=true;
    for(int i=nIFO-1;i>=0;i--) {
      for(int j=i-1;j>=0;j--) {
        if(!(((id[i]-id[j])>=(lagL[i]-lagH[j]))&&
             ((id[i]-id[j])<=(lagH[i]-lagL[j])))) check=false;
        if(lagSite!=NULL) {
          if(lagSite[i]!=lagSite[j] && id[i]==id[j]) check=false;
        } else {
          if(id[i]==id[j]) check=false;
        }
      }
    }
//  check if lag is already in the list
    if(check) {
      for(m=0;m<nList;m++) {
        bool pass=true;
        for(n=0; n<nIFO; n++) if(lagList[n][m]!=id[n]) pass=false;
        if(pass) check=false;
      }
    }
    if(check) {
      if(NETX(id[0]||,id[1]||,id[2]||,id[3]||,id[4]||,id[5]||,id[6]||,id[7]||) false) { // skip zero lag
        for(n=0; n<nIFO; n++) lagList[n][nList]=id[n];
        nList++;
      }
    }
    if (nList>=maxList) break;
  }

// shift lags with respect to the first detector
// negative lags are converted into positive 

final:                          // extract selected lags from the extended lag list

  for(m=0; m<nList; m++) {
    int lagMin = kMaxInt;
    for(n=0; n<nIFO; n++) if (lagList[n][m]<lagMin) lagMin=lagList[n][m];
    for(n=0; n<nIFO; n++) lagList[n][m]-=lagMin;
  }

  if(lagIDS+lagSize>nList) {
    cout << "network::setTimeShifts : lagOff+lagSize > nList of lags : " << nList << endl;
    exit(1);
  }

  for(n=0; n<nIFO; n++){
    pd = this->getifo(n);
    m  = pd->lagShift.size();
    if(m!=lagSize) pd->lagShift.resize(lagSize);
    pd->lagShift = 0.;
  } 

// write in the final list those lags which are inside the segment boundaries
// compute segment lenght 

  double R = this->getifo(0)->getTFmap()->rate();
  double segLen = this->getifo(0)->getTFmap()->size(); 
  double edge = this->Edge;
  size_t selSize=0;
  size_t lagMaxSeg=0;
  double zero = 0.;

// check boundaries

  segLen = (segLen/R-2*edge)/lagStep;
  lagMaxSeg = int(segLen)-1;

  for(n=0; n<nIFO; n++) {
    lagL[n] = 0;
    lagH[n] = lagMaxSeg;
  }

  for(m=0; m<lagSize; m++) { 
    bool check = true;
    for (n=0; n<nIFO; n++) id[n]=lagList[n][m+lagIDS];  

// Lags must be in the range 0:lagMax
    for(n=0; n<nIFO; n++) if(id[n]<0||id[n]>int(lagMaxSeg)) check=false;

// Difference between 2 lags belonging to diffent detectors must be <= lagMax
    for(int i=nIFO-1; i>=0; i--) {
      for(int j=i-1; j>=0; j--) {
        if (!(((id[i]-id[j])>=(lagL[i]-lagH[j]))&&
              ((id[i]-id[j])<=(lagH[i]-lagL[j])))) check=false;
      }
    }

// lag is within the boundaries -> store in lagShift

    if (check) {
      if(lagMax) {                                     // extended lags
	for(n=0; n<nIFO; n++) {
	  k = lagList[n][m+lagIDS]; 
	  if(k) check = false;                         // check if zero lag is present
	  this->getifo(n)->lagShift.data[selSize] = k*lagStep;
	}
      }
      else {
	k = lagList[0][m+lagIDS]; 
	this->getifo(0)->lagShift.data[selSize] = k*lagStep;
	zero = 0;
	for(n=1; n<nIFO; n++) {
	  pd = this->getifo(n);
	  zero += fabs(pd->sHIFt-k*lagStep);
	  pd->lagShift.data[selSize] = pd->sHIFt;
	}
	if(zero>0.1) check = false;                    // check if zero lag is present
      }
      wc.shift = check ? 0 : m+lagOff; 
      wc_List.push_back(wc); 
      livTime.push_back(0.); 
      selSize++;
    }
  }

  if(selSize==0) {
    cout << "network::setTimeShifts error: no lag was selected" << endl;
    exit(0);
  }

  for(n=0; n<nIFO; n++) {
    m = this->getifo(n)->lagShift.size();
    if(m!=selSize) this->getifo(n)->lagShift.resize(selSize);
  }

// dump lags list

  if(fname && !strcmp(fmode,"w") && lagMax) {

    FILE *fP=NULL;
    if((fP = fopen(fname, "w")) == NULL) {
      cout << "network::setTimeShifts error: cannot open file " << fname << endl;
      exit(1);
    }

    // write header 
    fprintf(fP,"#");for (n=0;n<=nIFO;n++) fprintf(fP,"--------------");fprintf(fP,"\n");
    fprintf(fP,"#total %10d lags \n",int(nList));
    fprintf(fP,"#");for (n=0;n<=nIFO;n++) fprintf(fP,"--------------");fprintf(fP,"\n");
    fprintf(fP,"#%13s%14s%14s\n","   nIFO","lagStep"," lagMax");
    fprintf(fP,"#%13d%14.3f%14d\n",int(nIFO),lagStep,int(lagMax));
    fprintf(fP,"#");for (n=0;n<=nIFO;n++) fprintf(fP,"--------------");fprintf(fP,"\n");
    fprintf(fP,"#%13s","lagId");
    for(n=0; n<nIFO; n++) fprintf(fP,"%12s-%1d","lagShift",int(n));
    fprintf(fP,"\n");
    fprintf(fP,"#");for (n=0;n<=nIFO;n++) fprintf(fP,"--------------");fprintf(fP,"\n");

    // write lags
    for(m=0; m<nList; m++){ 
      fprintf(fP,"%14d", int(m));
      for (n=0; n<nIFO; n++) fprintf(fP,"%14d",lagList[n][m]);  
      fprintf(fP,"\n");
    }

    if(fP!=NULL) fclose(fP);
  }

// free memory

  for(n=0; n<nIFO; n++) delete [] lagList[n];

// print selected lags

  printf("%8s ","lag");
  for(n=0; n<nIFO; n++) printf("%12.12s%2s","ifo",getifo(n)->Name);
  printf("\n");
  for(m=0; m<selSize; m++){ 
    printf("%8d ",(int)wc_List[m].shift);
    for(n=0; n<nIFO; n++) printf("%14.5f",this->getifo(n)->lagShift.data[m]);
    printf("\n");
  }

  nLag=selSize; Step=lagStep;
  return selSize;
}

//***************************************************************
// set delay filter for a network:  
// time delay convention: + - shift TS right
//                        - - shift TS left
///***************************************************************
size_t network::setFilter(detector* d)
{
  this->filter.clear();
  std::vector<delayFilter>().swap(this->filter);   // release memory

  if(!d) d = this->getifo(0);                 // reference detector
  double rate = d->getTFmap()->rate();        // data rate

  if(ifoList.size()<2 || !d || rate==0.) {
    cout<<"network::setFilter() error: incomplete network initialization"<<endl;
    return 0;
  }

  double T = this->getDelay((char*)"MAX")+0.002;     // maximum delay
  delayFilter v = d->filter[0];               // delay filter

  int i,j,k,n,m;
  int M = int(d->nDFL);                       // number of wavelet layers
  int K = int(v.index.size());                // delay filter length
  int N = int(d->nDFS);                       // number of filter delays
  int J = int((fabs(T)*rate*N)/M+0.5);        // total delay in samples

  if(N<M) { cout<<"network::setFilter() error"<<endl; return 0; }
  if(!K) return 0;

  this->getifo(0)->nDFS = N;              // store number of filter delays in ref detector
  this->filter.reserve((2*J-1)*M);        // allocate memory for filter

  for(i=0; i<M; i++) {                    
    for(j=-(J-1); j<J; j++) {             // loop over delays
       m = j>0 ? (j+N/2-1)/N : (j-N/2)/N; // delay in wavelet pixels
       n = j - m*N;                       // n - delay in samples
       if(n <= 0) n = -n;                 // filter index for negative delays
       else       n = N-n;                // filter index for positive delays
       v  = d->filter[n*M+i];
       for(k=0; k<K; k++) v.index[k] -= m*M;
       this->filter.push_back(v);
    }
  }

  return 2*J-1;
}

//***************************************************************
// set 0-phase delay filter for a network from detector:  
// used in cWB script with wat-4.7.0 and earlier
// time delay convention: + - shift TS right
//                        - - shift TS left
///***************************************************************
void network::setDelayFilters(detector* d)
{
  size_t N = this->ifoList.size();
  if(N < 2) return;
  if(d) this->getifo(0)->setFilter(*d);
  this->setFilter(this->getifo(0));
  this->getifo(0)->clearFilter();
  return;
}

//***************************************************************
// set delay filters for a network from detector filter files
// time delay convention: + - shift TS right
//                        - - shift TS left
///***************************************************************
void network::setDelayFilters(char* fname, char* gname)
{
  size_t N = this->ifoList.size();

  if(N < 2) return;
  if(gname) {
    this->getifo(0)->readFilter(gname);
    this->setFilter(this->getifo(0));
    this->filter90.clear();
    std::vector<delayFilter>().swap(this->filter90);   // release memory
    this->filter90 = this->filter;
  }
  this->getifo(0)->readFilter(fname);
  this->setFilter(this->getifo(0));
  this->getifo(0)->clearFilter();
  return;
}

//***************************************************************
// set delay filters for a network from a network filter file 
// gname defines the phase shifted filter 
///***************************************************************
void network::setFilter(char* fname, char* gname)
{
  size_t N = this->ifoList.size();
  if(N < 2) return;
  if(gname) {
    this->readFilter(gname);
    this->filter90 = this->filter;
  }
  this->readFilter(fname);
  return;
}

//***************************************************************
//  Dumps network filter to file *fname in binary format.
//***************************************************************
void network::writeFilter(const char *fname)
{
  size_t i,j,k;
  FILE *fp;

  if ( (fp=fopen(fname, "wb")) == NULL ) {
     cout << " network::writeFilter() error : cannot open file " << fname <<". \n";
     return ;
  }

  size_t M = size_t(getifo(0)->TFmap.maxLayer()+1);   // number of wavelet layers
  size_t N = size_t(filter.size()/M);                 // number of delays
  size_t K = size_t(filter[0].index.size());          // delay filter length
  size_t n = K * sizeof(float);
  size_t m = K * sizeof(short);

  wavearray<float> value(K);
  wavearray<short> index(K);

  fwrite(&K, sizeof(size_t), 1, fp);  // write filter length
  fwrite(&M, sizeof(size_t), 1, fp);  // number of layers
  fwrite(&N, sizeof(size_t), 1, fp);  // number of delays
  
  for(i=0; i<M; i++) {         // loop over wavelet layers
    for(j=0; j<N; j++) {       // loop over delays
       for(k=0; k<K; k++) {    // loop over filter coefficients
	  value.data[k] = filter[i*N+j].value[k];
	  index.data[k] = filter[i*N+j].index[k];
       }
       fwrite(value.data, n, 1, fp);
       fwrite(index.data, m, 1, fp);
    }
  }
  fclose(fp);
}

//***************************************************************
//  Read network filter from file *fname.
//***************************************************************
void network::readFilter(const char *fname)
{
  size_t i,j,k;
  FILE *fp;

  if ( (fp=fopen(fname, "rb")) == NULL ) {
     cout << " network::readFilter() error : cannot open file " << fname <<". \n";
     exit(1);
  }

  size_t M;           // number of wavelet layers
  size_t N;           // number of delays
  size_t K;           // delay filter length

  fread(&K, sizeof(size_t), 1, fp);  // read filter length
  fread(&M, sizeof(size_t), 1, fp);  // read number of layers
  fread(&N, sizeof(size_t), 1, fp);  // read number of delays
  
  size_t n = K * sizeof(float);
  size_t m = K * sizeof(short);
  wavearray<float> value(K);
  wavearray<short> index(K);
  delayFilter v;

  v.value.clear(); v.value.reserve(K);
  v.index.clear(); v.index.reserve(K);
  filter.clear(); filter.reserve(N*M);

  for(k=0; k<K; k++) {    // loop over filter coefficients
     v.value.push_back(0.);
     v.index.push_back(0);
  }

  for(i=0; i<M; i++) {         // loop over wavelet layers
    for(j=0; j<N; j++) {       // loop over delays
       fread(value.data, n, 1, fp);
       fread(index.data, m, 1, fp);
       for(k=0; k<K; k++) {    // loop over filter coefficients
	  v.value[k] = value.data[k];
	  v.index[k] = index.data[k];
       }
       filter.push_back(v);
    }
  }
  fclose(fp);
}

// extract accurate timr delay amplitudes for a given sky location
void network::updateTDamp(int l,  float** v00, float** v90) { 
// parameter 1 - sky location index
// parameter 2 - 0-phase array for time-delayed amplitudes 
// parameter 3 - 90-phase array for time-delayed amplitudes 
// Algorithm: for a given pixel set extract TF data from the sparse maps.
// Obtain time domain data for each resolution, time-shift
// detectors for a given sky location l. Reproduce time-shifted TF maps and
// extract time-shifted amplitudes. Update time delay arrays - only one sky
// location is updated     

   int nres = this->wdmList.size();              // number of resolutions
   int nIFO = this->ifoList.size();              // number of detectors
   int V    = int(this->pList.size());           // number of pixels
   int layers;

   WSeries<double> WW;
   wavearray<double> x,y;
   netpixel* pix;
   detector* pd;

   for(int i=0; i<nres; i++) {
      for(int k=0; k<nIFO; k++) {
	 pd = this->getifo(k);
	 pd->vSS[i].Expand(false);                      // expand sparse map
 
	 WW = pd->vSS[i];                               // copy TF map
	 pd->vSS[i].Inverse();                          // 0-phase TS
	 WW.Inverse(-2);                                // 90-phase TS
	 pd->vSS[i].getLayer(x,0);                      // get I time series
	 WW.getLayer(y,0);                              // get Q time series
	 x+=y; x*=0.5;                                  // prepare time series
	 x.delay(pd->index[l]/this->rTDF);              // time shift data in x
	 pd->vSS[i].Forward(x);                         // prepare TF map
	 layers = pd->vSS[i].maxLayer()+1;              // number of WDM layers
      
	 for(int j=0; j<V; j++) {                       // loop over pixels
	    pix = this->pList[j];
	    if(pix->layers != layers) continue;         // skip wrong resolution
	    int ind = int(pix->data[k].index);          // index in TF array
	    v00[k][j] = pd->vSS[i].GetMap00(ind);       // update 00 amplitude
	    v90[k][j] = pd->vSS[i].GetMap90(ind);       // update 00 amplitude
	 }

	 pd->vSS[i].Shrink();                           // shrink TF map
      }	 
   }
}

//***************************************************************
// delay detectors in the network with respect to reference  
// to match sky location theta and phi
// index array should be setup
///***************************************************************
void network::delay(double theta, double phi)
{
  size_t m;
  size_t N = this->ifoList.size();          // number of detectors
  size_t k = this->getIndex(theta,phi);     // sky index
  detector* d;

  for(size_t n=1; n<N; n++){
    d = this->getifo(n);
    m = d->index.data[k];                   // delay index
    this->delay(d,m); 
  }
  return;
}

//***************************************************************
// delay detector in a network:  
// m - is the delay index
///***************************************************************
void network::delay(detector* d, size_t m)
{
  double R  = d->getTFmap()->rate();
  
  size_t i,j,k;
  size_t N = d->getTFmap()->size();
  size_t I = d->TFmap.maxLayer()+1;
  size_t M = this->filter.size()/I;         // total number of delays 
  size_t K = this->filter[0].index.size();  // filter length
  size_t jB = size_t(this->Edge*R/I)*I;     // number of samples in the edges
  size_t jS;

  slice S;
  delayFilter* pv;

// buffer for wavelet layer delay filter
  double* F = (double*)malloc(K*sizeof(double));
  int*    J =    (int*)malloc(K*sizeof(int));
  
  N -= jB;                                   // correction for left boundary

  WSeries<double> temp = d->TFmap;
  d->TFmap=0.;

//  cout<<"m="<<m<<" N="<<N<<" R="<<R<<" I="<<I<<" M="<<M<<" K="<<K<<endl;
  
  double* p1 = temp.data;
  double* b0 = temp.data;
  
  for(i=0; i<I; i++) {                       // loop over wavelet layers

// set filter array for this layer and delay index
    pv = &(filter[i*M+m]);
    for(k=0; k<K; k++){             
      F[k] = double(pv->value[k]);
      J[k] = int(pv->index[k]);
    }

    S  = d->getTFmap()->getSlice(i);
    jS = S.start()+jB;

    for(j=jS; j<N; j+=I) {              // loop over samples in the layer
      p1=b0+j;
      d->TFmap.data[j] = dot32(F,p1,J); // apply delay filter
    }
  }
  free(F); 
  free(J);
}

//***************************************************************
//:set index array for delayed amplitudes
// used with wavelet delay filters
// time delay convention: t+tau - arrival time at the center of Earth
// ta1-tau0 - how much det1 should be delayed to be sinchronized with det0
///***************************************************************
void network::setDelayIndex(int mode)
{
  double t;
  int i,ii;
  size_t n,m,l,k;
  size_t N = ifoList.size();           // number of detectors

  double tt[NIFO][NIFO];                  
  double TT[NIFO];                  
  int    mm[NIFO][NIFO];                  

  if(N<2) {
    cout<<"network::setDelayIndex(): invalid network\n";
    return;
  }

  detector* dr[NIFO];
  for(n=0; n<N; n++) dr[n] = ifoList[n];

  size_t I = dr[0]->nDFL;                     // number of wavelet layers
  size_t K = this->filter.size()/I;           // number of delays
  size_t L = dr[0]->tau.size();               // skymap size  

  double rate = dr[0]->getTFmap()->rate();    // data rate
  rate *= dr[0]->nDFS/I;                      // up-sample rate

  if(pOUT) cout<<"filter size="<<this->filter.size()
	       <<" layers="<<I<<" delays="<<K<<" samples="<<dr[0]->nDFS<<endl;

  if(!(K&1) || rate == 0.) {
    cout<<"network::setDelayIndex(): invalid network\n";
    return;
  }

  for(n=0; n<N; n++) {
    if(dr[n]->index.size() != L) {
       dr[n]->index.resize(L);
    }
  }

// calculate time interval the di detector is delayed to be 
// sinchronized with dr
// time delay > 0 - shift di right (future) 
// time delay < 0 - shift di left  (past)

  this->nPenalty = dr[0]->tau;

  for(l=0; l<L; l++){

// calculate time delay matrix
//  0 d01 d02  
// d10  0 d12  
// d20 d21 0

    for(n=0; n<N; n++) {
      for(m=0; m<N; m++) {
	t = dr[n]->tau.get(l)-dr[m]->tau.get(l);
	i = int(t*rate+2*K+0.5) - 2*K;
	mm[n][m] = i;
	tt[n][m] = t*rate;
      }
    }

    for(n=0; n<N; n++) {
      TT[n] = 0.;                           // max delay for n-th configuration
      for(m=0; m<N; m++) {
	for(k=0; k<N; k++) {
	  t = fabs(mm[n][k]-mm[n][m]-tt[m][k]);
	  if(TT[n] < t) TT[n] = t; 
	}
      }
    }      

    t = 20.; i = N;
    for(m=0; m<N; m++) {
      if(t>TT[m]) { t = TT[m]; k = m; }     // first best configuration
    }
    this->nPenalty.set(l,double(t));

    t = dr[k]->tau.get(l);
    i = mIFO<9 ? mm[k][this->mIFO] : int(t*rate+2*K+0.5)-2*K;
        
//  0 d01 d02      0  d01  d02 
// d10  0 d12  ->  0 d'01 d'02 
// d20 d21 0       0 d"01 d"02

    for(m=0; m<N; m++) {
      ii = (K/2+mm[k][m])-i;                // convert to time delay with respect to master IFO
      dr[m]->index.data[l] = ii; 
      if(ii < 0) cout<<"network::setDelayIndex error: sky index<0: "<<k<<endl;
    }
  }
  return;
}

//***************************************************************
//:set theta, phi index array 
//:will not work on 32 bit with the option other than 0,2,4
//***************************************************************
size_t network::setIndexMode(size_t mode)
{
  detector* dr = ifoList[0];

  if(ifoList.size()<2 || !dr->tau.size()) {
    cout<<"network::setIndex() - invalid network"<<endl;
    return 0;
  } 

  size_t i,j,n,m;
  size_t L = dr->tau.size();
  size_t N = ifoList.size(); 
  size_t I = mode ? dr->nDFL : 1;             // number of wavelet layers
                                              // TO BE FIXED !!! works only for 1G 
                                              // for 2G only mode=0 works !!! 
  size_t K = this->filter.size()/I;           // number of delays
  size_t J = 0;                               // counter for rejected locations
  size_t M = mIFO<9 ? mIFO : 0;               // reference detector
  long long ll;

  if(this->index.size()!=L) this->index.resize(L);
  if(this->skyMask.size()!=L) this->skyMask.resize(L);
  if(this->skyHole.size()!=L) { this->skyHole.resize(L); this->skyHole = 1.; }
  for(j=0; j<L; j++) { 
    index.data[j] = j; 
    skyMask.data[j] = size_t(skyHole.data[j]+0.1); 
  }
  if(!mode) return 0;

  if(mode==2 || mode==4) {
    dr->tau.downsample(skyMask,mode);
    return 0;
  }

  wavearray<long long> delay(L);
  long long **pp = (long long**)malloc(L*sizeof(long long*));
  skymap* sm = &nSkyStat;

  for(n=0; n<N; n++) {
    if(!this->getifo(n)->index.size()) {
      cout<<"network::setIndex() - invalid network"<<endl;
      return 0;
    }
  }

  for(i=0; i<L; i++){
    delay.data[i] = 0;
    pp[i] = delay.data+i;
    m = 0;
    for(n=0; n<N; n++) {
      if(n == M) continue;
      ll = this->getifo(n)->index.data[i];
      if(this->mIFO==99) ll += K/2 - this->getifo(0)->index.data[i];
      delay.data[i] += ll<<(m*12);
      m++;
    }
  }

  delay.waveSort(pp,0,L-1);
  ll = *(pp[0]);
  for(i=1; i<L; i++) { 
    j = pp[i] - delay.data;
   if(ll == *(pp[i])) {
      skyMask.data[j] = 0;           // remove duplicate delay configurations
      J++;
      if(pOUT) cout<<" "<<j<<"|"<<sm->getTheta(j)<<"|"<<sm->getPhi(j);
  }
    else {
      ll = *(pp[i]);
      if(pOUT) cout<<"\n ll="<<ll<<" "<<j<<"|"<<sm->getTheta(j)<<"|"<<sm->getPhi(j);
    }
  }

  free(pp);
  return J;
}

void
network::print() {

  // print detector's info
  int nIFO = ifoListSize();
  for(int n=0; n<nIFO; n++) getifo(n)->print();

  // print MDC log infos

  cout << "----------------------------------------------" << endl;
  cout << " INJECTIONS : " << this->mdcListSize() << endl;
  cout << "----------------------------------------------" << endl;
  for(size_t k=0;k<this->mdcListSize();k++) {
     string str(this->getmdcList(k));
     cout << endl << str.c_str() << endl;
  }

  return;
}

