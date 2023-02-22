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


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// S. Klimenko, University of Florida, Gainesville, FL
// G.Vedovato,  INFN,  Sezione  di  Padova, Italy
// WAT network class
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define NETWORK_CC
#include <time.h>
#include <iostream>
#include <stdexcept>
#include <xmmintrin.h>
#include <immintrin.h> 
#include "TRandom3.h"
#include "TMath.h" 
#include <fstream>
//#include "Meyer.hh"
//#include "injection.hh"
#include "network.hh"
#include "TComplex.h"

using namespace std;

ClassImp(network)

// constructors

network::network() : 
   nRun(0), nLag(1), nSky(0), mIFO(0), rTDF(0), Step(0.), Edge(0.), gNET(0.), aNET(0.), iNET(0), 
   eCOR(0.), norm(1.), e2or(0.), acor(sqrt(2.)), pOUT(false), EFEC(true), local(true), optim(true), 
   delta(0.), gamma(0.), precision(0.02), pSigma(4.), penalty(1.), netCC(-1.), netRHO(0.), 
   wfsave(false), pattern(1), _WDM(false) 
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
	 network::getSkyArea(id,lag,T,var);       // calculate error regions
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

      //cout<<"subnetcut "<<V<<" "<<id<<" "<<wdmMRA.clusterCC.size()<<" "<<wdmMRA.sizeCC.size()<<endl;
      
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

      if(fmin(suball,submra)>subnet && rHo>fabs(this->netRHO) && Lt>subnorm*Lo) {
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


      this->pList.clear();
      pRate.clear();
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
	  m,(int)M,p,med,iGamma(N*m,0.2),v10,iGamma(N*m,p10),val,iGamma(N*m,p),N*log(m),fff);
   return (iGamma(N*m,p)+val)*0.3+N*log(m);
}


void network::printwc(size_t n) { 
   netcluster* p = this->getwc(n);
   int iTYPE = 1; 
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

//  if(!M) { free(pp); return; }
//
//  double dT = 1.e13;
//  double injTime = 1.e12;
//  int injID = -1;
//  int mdcID = -1;
//  injection INJ(this->ifoList.size());
//
//  for(m=0; m<M; m++) {
//    mdcID = this->getmdc__ID(m);
//    dT = fabs(To - this->getmdcTime(mdcID));
//    if(dT<injTime && INJ.fill_in(this,mdcID)) {
//      injTime = dT;
//      injID = mdcID;
//      if(pOUT) printf("getSkyArea: %4d %12.4f %7.3f %f \n",int(m),To,dT,s);
//    }
//  }
//
//  if(INJ.fill_in(this,injID)) {
//
//    th = INJ.theta[0];
//    ph = INJ.phi[0];
//    i  = this->getIndex(th,ph);
//
//    vI->push_back(int(i));
//    vP->push_back(float(p[i]));
//
//    vol = sum = 0.;
//    for(l=L-1; l>Lm; l--){
//      vol += s;
//      sum += *pp[l];
//      if(pp[l]-p == int(i)) break;
//    }
//    (*vf)[0]  = sqrt(vol);
//    (*vf)[10] = sum;
//    j = pp[L-1]-p;                                    // reference sky index at max
//
//    if(pOUT) {
//      printf("getSkyArea: %5d %12.4f %6.1f %6.1f %6.1f %6.1f %6.2f %6.2f %6.2f %7.5f, %e %d \n",
//	     int(id),INJ.time[0]-this->getifo(0)->TFmap.start(),INJ.theta[0],INJ.phi[0],
//	     sm->getTheta(j),sm->getPhi(j),(*vf)[0],(*vf)[5],(*vf)[9],(*vf)[10],p[i],int(i));
//    }
//  }
  
  free(pp);
  return;
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
  
  //cout<<"b: "<<nList<<" "<<lagSize<<" "<<maxList<<endl;

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

//cout<<"c: "<<lagIDS<<" "<<lagSize<<" "<<maxList<<endl;
 
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

  // print selected lags (SK: turned it of 2/22/22) 
/*
  printf("%8s ","lag");
  for(n=0; n<nIFO; n++) printf("%12.12s%2s","ifo",getifo(n)->Name);
  printf("\n");
  for(m=0; m<selSize; m++){ 
    printf("%8d ",(int)wc_List[m].shift);
    for(n=0; n<nIFO; n++) printf("%14.5f",this->getifo(n)->lagShift.data[m]);
    printf("\n");
  }
*/
  nLag=selSize; Step=lagStep;
  return selSize;
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

//***************************************************************
//:set theta, phi index array 
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
  size_t K = size_t(getDelay((char*)"MAX")*rTDF)+1;           // number of delays
  size_t J = 0;                               // counter for rejected locations
  size_t M = mIFO<9 ? mIFO : 0;               // reference detector
  long long ll;

  //cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  //cout<<L<<" "<<N<<" "<<K<<endl;
  
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

void network::print() {

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

