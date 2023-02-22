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


//**************************************************************
// Wavelet Analysis Tool  
// Sergey Klimenko, University of Florida
// class for coherent network analysis used with DMT and ROOT
//**************************************************************

#ifndef NETWORK_HH
#define NETWORK_HH

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "TH2F.h"
#include "wavearray.hh"
#include "wseries.hh"
#include "detector.hh"
#include "skymap.hh"
#include "netcluster.hh"
#include "wat.hh"
#include "monster.hh"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TVectorD.h"
#ifndef __CINT__
#include "watsse.hh"
#include "watsse4.hh"
#endif

using namespace std;

typedef std::vector<double> vectorD;

struct waveSegment {
  int    index;
  double start;
  double stop;
};

class network : public TNamed
{
  public:
      
      // constructors
      
      //: Default constructor
      network();

      //: Copy constructor
      //!param: value - object to copy from 
      network(const network&);
      
      //: destructor
      virtual ~network();
    
      // operators

      network& operator= (const network&);

      // accessors

      //: add detector to the network  
      //!param: detector structure
      //!return number of detectors in the network 
      size_t add(detector*);

      //:do wdm wavelet transformation for all detectors
      inline void Forward(); 
      inline void Forward(WDM<double>* pwdm);
      inline void Inverse(int k=-1); 

      // set time shifts
      //!param number of time lags
      //!param time shift step in seconds
      //!param first lag ID
      //!param maximum lag ID
      //!param file name for lag configurations     
      //!param r/w/s - read/write/string mode     
      int setTimeShifts(size_t=1, double=1., size_t=0, size_t=0, 
			const char* = NULL, const char* = "w", size_t* = NULL);

      // print wc_List
      //param - time lag
      void printwc(size_t);

      //: initialize cluster for selected TF area, put it in wc_List[0]; 
      //!param: cluster start time relative to segment start
      //!param: cluster duration 
      //!return cluster list size
      // time shifts between detectors are controlled by sHIFt parameters in the detector objects
      // cluster bandwidth is controlled by TFmap.low(), TFmap.high() parameters.
      virtual size_t initwc(double, double);

      //: initialize network sky maps: fill antenna pattern maps
      //!param: healpix order
      void setSkyMaps(int);  

      // set veto array from the input list of DQ segments
      //!param: time window around injections
      double setVeto(double=5.);

      //: read injections
      //!param: MDC log file
      //!param: approximate gps time 
      //        (injections in a window +-10000 sec around gps are selected)
      //!param: position of time field
      //!param: position of type field
      // returns number of injections 
      size_t readMDClog(char*,double=0.,int=11,int=12);

      //: read segment list
      //!param: segment list file
      //!param: start time collumn number
      // returns number of segments 
      size_t readSEGlist(char*,int=1);

      //:get sky index for given theta, phi
      //!param: theta [deg]
      //!param: phi [deg]
      inline int getIndex(double theta, double phi) {
	return getifo(0)->tau.getSkyIndex(theta,phi);
      }

      //:set antenna pattern buffers 
      //!param: detector (use theta, phi index array)
      void setAntenna(detector*);
      // set antenna patterns for all detectors
      void setAntenna();

      // 2G analysis algorithm for selection of significant network pixles
      // LAG - time shift lag defining how detectors are shifted wrt each other.
      // Eo  - pixel energy threshold
      // DD  - dummy
      // hist- pointer to a diagnostic histogram showing network pixel energy. 
      long getNetworkPixels(int LAG, double Eo, double DD=1., TH1F* hist=NULL);

      // apply subnetwork cut
      // subnet:  sub network threshold 
      // subcut:  sub network threshold in the skyloop
      // subnorm: norm (Lo/Lt) sub network threshold 
      // lag: lag index
      // return number of processed pixels
      // works only with TD amplitudes. 
      // Includes x regulator for background rejection
      long subNetCut(int lag, float subnet=0.6, float subcut=0.33, float subnorm=0.0, TH2F* hist=NULL);      

      // 2G likelihood with multi-resolution cluster analysis
      // mode: analysis mode
      //  lag: lag index
      //   ID: cluster ID, if ID=0 - process all clusters
      //       fill nLikelihood skymap if ID<=0 - need for CED
      // hist: chirp histogram: If not needed, TGraphErrors* hist=NULL
      // shold be used as input
      //return number of processed pixels
      long likelihood2G(char mode, int lag, int ID, TH2F* hist=NULL);
      long likelihoodWP(char mode, int lag, int ID, TH2F* hist=NULL, char* Search=const_cast<char*>(""));
      
      // read the wdm xtalk catalog
      // param: catalog filename
      inline void setMRAcatalog(char* fn){ wdmMRA.read(fn);}

      
      //:reconstruct time clusters 
      //!param: time gap in pixels
      //!return: number of reconstructed clusters
      inline size_t cluster(int kt, int kf);

      //:return number of events 
      inline size_t events();

      //:return number of events with specified type, lag and cID
      // type : sCuts
      // lag  : lag<0 -> select all lags  
      inline size_t events(int type, int lag=-1);

      // set noise rms fields for pixels in the network netcluster structure
      inline void setRMS();

      // delink superclusters in wc_List
      inline void delink(){
	for(size_t i=0; i<nLag; i++) wc_List[i].delink();
      }

      // get MRA waveforms of type atype in time domain given lag nomber and cluster ID
      // if tof = true, apply time-of-flight corrections
      // fill in waveform arrays in the detector class
      // atype = 'W' - get whitened detector Wavelet PC
      // atype = 'w' - get detector Wavelet PC
      // atype = 'S' - get whitened reconstructed response (Signal)
      // atype = 's' - get reconstructed response (Signal)
      // mode: -1/0/1 - return 90/mra/0 phase
      // tof:  false/true - time-of-flight correction
      bool getMRAwave(size_t ID, size_t lag, char atype='S', int mode=0, bool tof=false);

      //: calculation of sky error regions
      //!param: cluster id
      //!param: lag
      //!param: cluster time      
      //void getSkyArea(size_t id, size_t lag, double T);
      //!param: noise rms per DoF
      virtual void getSkyArea(size_t id, size_t lag, double T, double rms);

      // read earth skyMask coordinates from file (1G)
      // parameter - fraction of skymap to be rejected 
      // parameter - file name
      size_t setSkyMask(double f, char* fname);

      // read celestial/earth skyMask coordinates from file 
      // parameter - file name
      // parameter - sky coordinates : 'e'=earth, 'c'=celestial
      size_t setSkyMask(char* fname, char skycoord);

      // read celestial/earth skyMask coordinates from skymap
      // parameter - skymap
      // parameter - sky coordinates : 'e'=earth, 'c'=celestial
      size_t setSkyMask(skymap sm, char skycoord);

      // set threshold on amplitude of core pixels
      // set threshold on subnetwork energy
      inline void setAcore(double a) {
         this->acor = a;
         a = 1-Gamma(ifoList.size(),a*a*ifoList.size());    // probability
         this->e2or = iGamma(ifoList.size()-1,a);           // subnetwork energy threshold
      }

      //: get size of mdcList
      inline size_t mdcListSize() { return mdcList.size(); }
      //: get size of mdcType
      inline size_t mdcTypeSize() { return mdcType.size(); }
      //: get size of mdcTime
      inline size_t mdcTimeSize() { return mdcTime.size(); }
      //: get size of mdc__ID
      inline size_t mdc__IDSize() { return mdc__ID.size(); }
      //: get size of livTime
      inline size_t livTimeSize() { return livTime.size(); }
      //: get element of mdcList
      inline string getmdcList(size_t n) { return mdcListSize()>n ? mdcList[n] : "\n"; }
      //: get element of mdcType
      inline string getmdcType(size_t n) { return mdcTypeSize()>n ? mdcType[n] : "\n"; }
      //: get pointer to mdcTime
      inline std::vector<double>* getmdcTime() { return &mdcTime; }
      //: get element of mdcTime
      inline double getmdcTime(size_t n) { return mdcTimeSize()>n ? mdcTime[n] : 0.; }
      //: get element of mdc__ID
      inline size_t getmdc__ID(size_t n) { return mdc__IDSize()>n ? mdc__ID[n] : 0; }
      //: get element of livTime
      inline double getliveTime(size_t n) { return livTimeSize()>n ? livTime[n] : 0.; }
  
      //: get size of ifoList
      inline size_t ifoListSize() { return ifoList.size(); }
      //: get size of wc_List
      inline size_t wc_ListSize() { return wc_List.size(); }
      //:return pointer to a detector n
      inline detector* getifo(size_t n) { return ifoListSize()>n ? ifoList[n] : NULL; }
      //:return pointer to a netcluster in wc_List
      inline netcluster* getwc(size_t n) { return wc_ListSize()>n ? &wc_List[n] : NULL; }

      //: add wdm transform. return number of wdm tronsforms in the list
      size_t add(WDM<double>* wdm) {if(wdm) wdmList.push_back(wdm); return wdmList.size();}
      //: get size of wdmList
      inline size_t wdmListSize() { return wdmList.size(); }
      //get wdm transform for input number of wdm layers
      inline WDM<double>* getwdm(size_t M) { 
         for(size_t n=0;n<wdmListSize();n++) if(M==(wdmList[n]->maxLayer()+1)) return wdmList[n]; 
        return NULL; 
      }

      // set run number
      inline void setRunID(size_t n) { this->nRun=n; return; }     
      // set wavelet boundary offset
      inline void setOffset(double t) { this->Edge = t; return; }
      
      // set threshold for double OR energy
      //!param: threshold
      inline void set2or(double p) { this->e2or = p; }

      // calculate WaveBurst threshold as a function of resolution
      //!param: selected fraction of LTF pixels assuming Gaussian noise
      double THRESHOLD(double bpp);

      // calculate WaveBurst threshold for patterns
      //!param: selected fraction of LTF pixels assuming Gaussian noise
      //!param: Gamma distribution shape parameter
      double THRESHOLD(double bpp, double shape);

      // calculate maximum delay between detectors
      // "max" - maximum delay
      // "min" - minimum delay
      // "MAX" - max(abs(max),abs(min)
      //  def  - max-min
      double getDelay(const char* c="");

      //:recalculate time delay skymaps from Earth center to
      //:'L1' - Livingston
      //:'H1' - Hanford
      //:'V1' - Cashina
      //:'G1' - Hannover
      //:'T1' - Tama
      //:'A1' - Australia
      void setDelay(const char* = "L1");

      // set optimal delay index
      // the delay index is symmetric - can be negative.
      // index=0 corresponds to zero delay
      // index>0 corresponds to positive delay (shift right) 
      // index<0 corresponds to negative delay (shift left) 
      // rate: effective data rate (1/delay_step)
      virtual void setDelayIndex(double rate);

      //:set sky index array
      //: mode: 0 - initialize index arrays
      //:       1 - + exclude duplicate delay configurations
      virtual size_t setIndexMode(size_t=0);                         //!!!!! single detector.

      // extract accurate time delay amplitudes for a given sky location
      // parameter 1 - sky location index
      // parameter 2 - array for 0-phase delayed amplitudes 
      // parameter 3 - array for 0-phase delayed amplitudes 
      void updateTDamp(int, float**, float**);

      // set constraint parameters
      inline void constraint(double d=1., double g=0.0001);

      bool wdm()  {return _WDM;}                  // true/false - WDM/wavescan
      void wdm(bool _WDM)   {this->_WDM =_WDM; }  // set wdm used/unused	

      // print network parameters
      void print();             // *MENU*
      virtual void Browse(TBrowser*) {print();}

      static inline   void pnt_(float**, float**, short**, int, int);
      static inline   void cpp_(float*&, float**);
      static inline   void cpf_(float*& a, double** p);
      static inline   void cpf_(float*& a, double** p, size_t); 

             inline    int _sse_MRA_ps(float*, float*, float, int);
             inline    int _sse_mra_ps(float*, float*, float, int);
             inline wavearray<float> _avx_norm_ps(float**, float**, std::vector<float*> &, int);
             inline wavearray<float> _avx_norm_ps(float**, float**, float*, int);
             inline    void _avx_saveGW_ps(float**, float**, int);

      void test_sse(int, int);

// data members

      size_t nRun;              // run number
      size_t nLag;              // number of time lags
      long   nSky;              // number of pixels for sky probability area
      size_t mIFO;              // master IFO
      double rTDF;              // effective rate of time-delay filter
      double Step;              // time shift step
      double Edge;              // time offset at the boundaries
      double gNET;              // network sensitivity
      double aNET;              // network alignment
      double iNET;              // network index
      double eCOR;              // correlation energy
      double norm;              // norm factor
      double e2or;              // threshold on 2D OR energy
      double acor;              // threshold on coherent pixel energy
      bool   pOUT;              // true/false printout flag
      bool   EFEC;              // true/false - EFEC/selestial coordinate system
      char   tYPe;              // search type
      bool   local;             // true/false - local/global normalization
      bool   optim;             // true/false - process optimal/all resolutions
      double delta;             // weak constraint parameter:
      double gamma;             // hard constraint parameter:
      double precision;         // precision of energy calculation
      double pSigma;            // integration limit in sigmas for probability
      double penalty;           // penalty factor:
      double netCC;             // threshold on netcc:
      double netRHO;            // threshold on rho:
      bool   wfsave;		// true/false - if false only simulated wf are saved (2G)
      int    pattern;           // clustering pattern

      WSeries<double> whp;      // + polarization
      WSeries<double> whx;      // x polarization

      std::vector<detector*>    ifoList;   // detectors
      std::vector<char*>        ifoName;   // detector's names
      std::vector<netcluster>   wc_List;   // netcluster structures for time shifts
      std::vector<double>       livTime;   // live time for time shifts
      std::vector<std::string>  mdcList;   // list of injections
      std::vector<std::string>  mdcType;   // list of injection types
      std::vector<double>       mdcTime;   // gps time of selected injections
      std::vector<size_t>       mdc__ID;   // ID of selected injections
      std::vector<waveSegment>  segList;   // DQ segment list
      std::vector<WDM<double>*> wdmList;   //! list of wdm tranformations

      skymap nSensitivity;           // network sensitivity
      skymap nAlignment;             // network alignment factor
      skymap nCorrelation;           // network correlation coefficient
      skymap nLikelihood;            // network likelihood
      skymap nNullEnergy;            // network null energy
      skymap nPenalty;               // signal * noise penalty factor
      skymap nCorrEnergy;            // reduced correlated energy
      skymap nNetIndex;              // network index
      skymap nDisbalance;            // energy disbalance
      skymap nSkyStat;               // sky optimization statistic
      skymap nEllipticity;           // waveform ellipticity
      skymap nPolarisation;          // polarisation angle
      skymap nProbability;           // probability skymap
      skymap nAntenaPrior;           // network sensitivtiy used as a prior for skyloc

      WSeries<double> pixeLHood;     // pixel likelihood statistic
      WSeries<double> pixeLNull;     // pixel null statistic

      wavearray<int>    index;       // theta, phi mask index array
      wavearray<short>  skyMask;     // index array for setting sky mask
      wavearray<double> skyMaskCC;   // index array for setting sky mask Celestial Coordinates
      wavearray<double> skyHole;     // static sky mask describing "holes"
      wavearray<short>  veto;        // veto array for pixel selection
      wavearray<double> skyProb;     // sky probability
      wavearray<double> skyENRG;     // energy skymap

// data arrays for MRA and likelihood analysis

      std::vector<netpixel*>  pList; //! list of pixel pointers for MRA
      monster        wdmMRA;         //! wdm multi-resolution analysis
      wavearray<float> a_00;         //! buffer for cluster sky 00 amplitude
      wavearray<float> a_90;         //! buffer for cluster sky 90 amplitudes
      wavearray<float> rNRG;         //! buffers for cluster residual energy 
      wavearray<float> pNRG;         //! buffers for cluster MRA energy 

// data arrays for polar coordinates storage : [0,1] = [radius,angle] 

      wavearray<double>  p00_POL[2]; //! buffer for projection on network plane 00 ampl
      wavearray<double>  p90_POL[2]; //! buffer for projection on network plane 90 ampl
      wavearray<double>  r00_POL[2]; //! buffer for standard response 00 ampl  
      wavearray<double>  r90_POL[2]; //! buffer for standard response 90 ampl

private:
   
      bool   _WDM;		     // true/false - used/not-used WDM

   ClassDef(network,6)

}; // class network

//:do wdm wavelet transformation for all detectors
inline void network::Forward() {
   for(size_t i=0; i<ifoList.size(); i++)
      ifoList[i]->TFmap.Forward();
}
inline void network::Forward(WDM<double>* pwdm) {
   for(size_t i=0; i<ifoList.size(); i++)
      ifoList[i]->TFmap.Forward(*(ifoList[i]->getHoT()), *pwdm);
}
inline void network::Inverse(int k) {
   for(size_t i=0; i<ifoList.size(); i++)
      ifoList[i]->TFmap.Inverse(k);
}

// set noise rms fields for pixels in the network netcluster structure
inline void network::setRMS() {
  size_t n = ifoList.size();
  if(!ifoList.size() || wc_List.size()!=nLag) return;
  for(size_t i=0; i<n; i++) {
    for(size_t j=0; j<nLag; j++) {
      if(!getwc(j)->size()) continue;
      if(!getifo(i)->setrms(getwc(j),i)) {
	cout<<"network::setRMS() error\n";
        exit(1);
      }
    } 
  }
  return;
}      

inline void network::pnt_(float** q, float** p, short** m, int l, int n) {
// point 0-7 float pointers to first network pixel
   NETX(q[0] = (p[0] + m[0][l]*n);,
        q[1] = (p[1] + m[1][l]*n);,
        q[2] = (p[2] + m[2][l]*n);,
        q[3] = (p[3] + m[3][l]*n);,
        q[4] = (p[4] + m[4][l]*n);,
        q[5] = (p[5] + m[5][l]*n);,
        q[6] = (p[6] + m[6][l]*n);,
        q[7] = (p[7] + m[7][l]*n);)
      return;
}   

inline void network::cpf_(float*& a, double** p) {
// copy to a data defined by array of pointers p and increment target pointer
   for(int i=0;i<XIFO;i++) *(a++) = *p[i];
   a+=NIFO-XIFO;
   return;
}

inline void network::cpf_(float*& a, double** p, size_t i) {
// copy to a data defined by array of pointers p and increment target pointer
   for(int k=0;k<XIFO;k++) *(a++) = p[k][i];
   a+=NIFO-XIFO;
   return;
}

inline void network::cpp_(float*& a, float** p) {
// copy to a data defined by array of pointers p and increment pointers
   for(int i=0;i<XIFO;i++) *(a++) = *p[i]++;
   a+=NIFO-XIFO;
   return;
}

inline size_t network::cluster(int kt, int kf) {
   if(!wc_List.size()) return 0;
   size_t m = 0;
   for(size_t n=0; n<nLag; n++) m += wc_List[n].cluster(kt,kf); 
   return m;
}      

//:return number of events 
inline size_t network::events() {
   if(!wc_List.size()) return 0;
   size_t m = 0;
   for(size_t n=0; n<nLag; n++) {
      m += wc_List[n].esize();
   } 
   return m;
}      

//:return number of events with specified type, lag and cID
inline size_t network::events(int type, int lag) {
   if(!wc_List.size()) return 0;
   if(lag>=(int)nLag) lag=nLag-1;
   size_t m = 0;
   if(lag>=0) m += wc_List[lag].esize(type);
   else for(size_t n=0; n<nLag; n++) m += wc_List[n].esize(type);
   return m;
}

inline wavearray<float> network::_avx_norm_ps(float** p, float** q,
					      std::vector<float*> &pAVX, int I) {
   wavearray<float> norm(NIFO+1);     // output array for packet norms
   float* g = norm.data+1; norm=0.;

#ifndef __CINT__

// return packet norm for each detector      
   int i,j,k,n,m;
   int  M = abs(I);
   int II = abs(I*2);
   float o = 1.e-12;
   float* mk = pAVX[1];                 // pixel energy mask                     
   float* rn = pAVX[22];                // halo noise                     
   wavearray<float> tmp(NIFO);          // array to store data
   float* t = tmp.data; tmp=0.;
   float e,u,v;

   float am[4*8] _ALIGNED; __m128* _am = (__m128*)am;
   float  x[4*8] _ALIGNED; __m128*  _x = (__m128*)x;
   float  h[4*8] _ALIGNED; __m128*  _h = (__m128*)h;  // halo

   float* an = pAVX[17];                // M*4*NIFO array                     
   NETX(__m128* _a0 = (__m128*)(an+4*M*0);, __m128* _a1 = (__m128*)(an+4*M*1);, 
	__m128* _a2 = (__m128*)(an+4*M*2);, __m128* _a3 = (__m128*)(an+4*M*3);, 
	__m128* _a4 = (__m128*)(an+4*M*4);, __m128* _a5 = (__m128*)(an+4*M*5);, 
	__m128* _a6 = (__m128*)(an+4*M*6);, __m128* _a7 = (__m128*)(an+4*M*7);) 

   for(m=0; m<M; m++) {
      if(I>0) rn[m] = 0.;
      NETX(_a0[m] = _mm_set_ps(q[0][m],q[0][m],p[0][m],p[0][m]); q[0][M+m]=0; , 
           _a1[m] = _mm_set_ps(q[1][m],q[1][m],p[1][m],p[1][m]); q[1][M+m]=0; , 
           _a2[m] = _mm_set_ps(q[2][m],q[2][m],p[2][m],p[2][m]); q[2][M+m]=0; , 
           _a3[m] = _mm_set_ps(q[3][m],q[3][m],p[3][m],p[3][m]); q[3][M+m]=0; , 
           _a4[m] = _mm_set_ps(q[4][m],q[4][m],p[4][m],p[4][m]); q[4][M+m]=0; , 
           _a5[m] = _mm_set_ps(q[5][m],q[5][m],p[5][m],p[5][m]); q[5][M+m]=0; , 
           _a6[m] = _mm_set_ps(q[6][m],q[6][m],p[6][m],p[6][m]); q[6][M+m]=0; , 
           _a7[m] = _mm_set_ps(q[7][m],q[7][m],p[7][m],p[7][m]); q[7][M+m]=0; ) 
   }

   for(m=0; m<M; m++) {
      if(mk[m]<=0.) continue;
                                            
      int      J = wdmMRA.size(m)*2;
      float   cc = 0;
      float*   c = wdmMRA.getXTalk(m);
      __m128* _c = (__m128*)(c+4);

      NETX(u=p[0][m]; v=q[0][m]; _am[0]=_mm_set_ps(v,u,v,u); _x[0]=_mm_setzero_ps(); ,
	   u=p[1][m]; v=q[1][m]; _am[1]=_mm_set_ps(v,u,v,u); _x[1]=_mm_setzero_ps(); ,
	   u=p[2][m]; v=q[2][m]; _am[2]=_mm_set_ps(v,u,v,u); _x[2]=_mm_setzero_ps(); ,
	   u=p[3][m]; v=q[3][m]; _am[3]=_mm_set_ps(v,u,v,u); _x[3]=_mm_setzero_ps(); ,
	   u=p[4][m]; v=q[4][m]; _am[4]=_mm_set_ps(v,u,v,u); _x[4]=_mm_setzero_ps(); ,
	   u=p[5][m]; v=q[5][m]; _am[5]=_mm_set_ps(v,u,v,u); _x[5]=_mm_setzero_ps(); ,
	   u=p[6][m]; v=q[6][m]; _am[6]=_mm_set_ps(v,u,v,u); _x[6]=_mm_setzero_ps(); ,
	   u=p[7][m]; v=q[7][m]; _am[7]=_mm_set_ps(v,u,v,u); _x[7]=_mm_setzero_ps(); ) 

      for(j=0; j<J; j+=2) {      
         n = int(c[j*4]);
	 NETX(_x[0]=_mm_add_ps(_x[0],_mm_mul_ps(_c[j],_a0[n]));,
	      _x[1]=_mm_add_ps(_x[1],_mm_mul_ps(_c[j],_a1[n]));,
	      _x[2]=_mm_add_ps(_x[2],_mm_mul_ps(_c[j],_a2[n]));,
	      _x[3]=_mm_add_ps(_x[3],_mm_mul_ps(_c[j],_a3[n]));,
	      _x[4]=_mm_add_ps(_x[4],_mm_mul_ps(_c[j],_a4[n]));,
	      _x[5]=_mm_add_ps(_x[5],_mm_mul_ps(_c[j],_a5[n]));,
	      _x[6]=_mm_add_ps(_x[6],_mm_mul_ps(_c[j],_a6[n]));,
	      _x[7]=_mm_add_ps(_x[7],_mm_mul_ps(_c[j],_a7[n]));)
      }

      NETX(_h[0]=_mm_mul_ps(_x[0],_am[0]);,
	   _h[1]=_mm_mul_ps(_x[1],_am[1]);,
	   _h[2]=_mm_mul_ps(_x[2],_am[2]);,
	   _h[3]=_mm_mul_ps(_x[3],_am[3]);,
	   _h[4]=_mm_mul_ps(_x[4],_am[4]);,
	   _h[5]=_mm_mul_ps(_x[5],_am[5]);,
	   _h[6]=_mm_mul_ps(_x[6],_am[6]);,
	   _h[7]=_mm_mul_ps(_x[7],_am[7]);)

      NETX(t[0]=h[ 0]+h[ 1]+h[ 2]+h[ 3]; t[0]=t[0]>0?t[0]:0; g[0]+=t[0]; ,
	   t[1]=h[ 4]+h[ 5]+h[ 6]+h[ 7]; t[1]=t[1]>0?t[1]:0; g[1]+=t[1]; ,
	   t[2]=h[ 8]+h[ 9]+h[10]+h[11]; t[2]=t[2]>0?t[2]:0; g[2]+=t[2]; ,
	   t[3]=h[12]+h[13]+h[14]+h[15]; t[3]=t[3]>0?t[3]:0; g[3]+=t[3]; ,
	   t[4]=h[16]+h[17]+h[18]+h[19]; t[4]=t[4]>0?t[4]:0; g[4]+=t[4]; ,
	   t[5]=h[20]+h[21]+h[22]+h[23]; t[5]=t[5]>0?t[5]:0; g[5]+=t[5]; ,
	   t[6]=h[24]+h[25]+h[26]+h[27]; t[6]=t[6]>0?t[6]:0; g[6]+=t[6]; ,
	   t[7]=h[28]+h[29]+h[30]+h[31]; t[7]=t[7]>0?t[7]:0; g[7]+=t[7]; )
	     
      if(I<0) continue;

      NETX(u=p[0][m]; v=q[0][m]; e=(u*u+v*v)/(t[0]+o); q[0][M+m]=(e>=1)?0:e; ,
	   u=p[1][m]; v=q[1][m]; e=(u*u+v*v)/(t[1]+o); q[1][M+m]=(e>=1)?0:e; ,
	   u=p[2][m]; v=q[2][m]; e=(u*u+v*v)/(t[2]+o); q[2][M+m]=(e>=1)?0:e; ,
	   u=p[3][m]; v=q[3][m]; e=(u*u+v*v)/(t[3]+o); q[3][M+m]=(e>=1)?0:e; ,
	   u=p[4][m]; v=q[4][m]; e=(u*u+v*v)/(t[4]+o); q[4][M+m]=(e>=1)?0:e; ,
	   u=p[5][m]; v=q[5][m]; e=(u*u+v*v)/(t[5]+o); q[5][M+m]=(e>=1)?0:e; ,
	   u=p[6][m]; v=q[6][m]; e=(u*u+v*v)/(t[6]+o); q[6][M+m]=(e>=1)?0:e; ,
	   u=p[7][m]; v=q[7][m]; e=(u*u+v*v)/(t[7]+o); q[7][M+m]=(e>=1)?0:e; )

      NETX(u=x[ 0]+x[ 2]; v=x[ 1]+x[ 3]; rn[m]+=u*u+v*v; ,
      	   u=x[ 4]+x[ 6]; v=x[ 5]+x[ 7]; rn[m]+=u*u+v*v; ,
      	   u=x[ 8]+x[10]; v=x[ 9]+x[11]; rn[m]+=u*u+v*v; ,
      	   u=x[12]+x[14]; v=x[13]+x[15]; rn[m]+=u*u+v*v; ,
      	   u=x[16]+x[18]; v=x[17]+x[19]; rn[m]+=u*u+v*v; ,
      	   u=x[20]+x[22]; v=x[21]+x[23]; rn[m]+=u*u+v*v; ,
      	   u=x[24]+x[26]; v=x[25]+x[27]; rn[m]+=u*u+v*v; ,
      	   u=x[28]+x[30]; v=x[29]+x[31]; rn[m]+=u*u+v*v; )

   }

   for(n=1; n<=XIFO; n++) {                        // save norms 
      if(I>0) { 
	 e = q[n-1][II+4]*q[n-1][II+4];            // TF-Domain SNR 
	 if(norm.data[n]<2.) norm.data[n]=2; 
	 q[n-1][II+5] = norm.data[n];              // save norms
	 norm.data[n] = e/norm.data[n];            // detector {1:NIFO} SNR 
      }
      norm.data[0] += norm.data[n];                // total SNR
   }
#endif
   return norm;                                                                             
} 

inline wavearray<float> network::_avx_norm_ps(float** p, float** q, float* ec, int I) {
// use GW norm from data packet      
// p - GW array
// q - Data array
   float e;
   int II = abs(I*2);
   wavearray<float> norm(NIFO+1);          // array for packet norms
   float* nn = norm.data;                  // array for packet norms
   norm = 0;
   for(int n=1; n<=XIFO; n++) {            // save norms 
      nn[n] = q[n-1][II+5];                // get data norms
      p[n-1][II+5] = nn[n];                // save norms
      e = p[n-1][II+4]*p[n-1][II+4];       // TF-Domain SNR 
      nn[n] = e/nn[n];                     // detector {1:NIFO} SNR 
      nn[0] += nn[n];                      // total SNR
      for(int i=0; i<I; i++) {             // save signal norms 
	 p[n-1][I+i] = ec[i]>0 ? q[n-1][I+i] : 0.;        // set signal norms 
      }
   }
   return norm;
}

inline void network::_avx_saveGW_ps(float** p, float** q, int I) {
// save GW strain amplitudes into a_00, a_90 arrays
// p,q  - input - GW warray
// I    - number of GW pixels
// in likelihoodWP these arrays should be stored exactly in the same order.

   for(int i=0; i<I; i++) {
     for(int n=0; n<NIFO; n++) {
        a_00[i*NIFO+n]=p[n][i];
        a_90[i*NIFO+n]=q[n][i];
     }
   }
   return;
}

inline int network::_sse_MRA_ps(float* amp, float* AMP, float Eo, int K) {
// fast multi-resolution analysis inside sky loop                         
// select max E pixel and either scale or skip it based on the value of residual 
// pointer to 00 phase amplitude of monster pixels                               
// pointer to 90 phase amplitude of monster pixels                               
// Eo - energy threshold                                                         
//  K - number of principle components to extract                                
// returns number of MRA pixels                                                  

#ifndef __CINT__
   int j,n,mm;                                                                   
   int k = 0;                                                                    
   int m = 0;                                                                    
   int f = NIFO/4;                                                               
   int V = (int)this->rNRG.size();                                               
   float*  ee = this->rNRG.data;                            // residual energy   
   float*  pp = this->pNRG.data;                            // residual energy   
   float   EE = 0.;                                         // extracted energy  
   float   E;                                                                    
   float mam[NIFO] _ALIGNED;
   float mAM[NIFO] _ALIGNED;
   this->pNRG=-1;                                                                
   for(j=0; j<V; ++j) if(ee[j]>Eo) pp[j]=0;                                      

   __m128* _m00 = (__m128*) mam;
   __m128* _m90 = (__m128*) mAM;
   __m128* _amp = (__m128*) amp;
   __m128* _AMP = (__m128*) AMP;
   __m128* _a00 = (__m128*) a_00.data;
   __m128* _a90 = (__m128*) a_90.data;

   while(k<K){

      for(j=0; j<V; ++j) if(ee[j]>ee[m]) m=j;               // find max pixel
      if(ee[m]<=Eo) break;  mm = m*f;                                        

             E = _sse_abs_ps(_a00+mm,_a90+mm); EE += E;     // get PC energy
      int    J = wdmMRA.size(m);                              
      float* c = wdmMRA.getXTalk(m);                        // c1*c2+c3*c4=c1*c3+c2*c4=0

      if(E/EE < 0.01) break;                                // ignore small PC

      _sse_cpf_ps(mam,_a00+mm);                             // store a00 for max pixel
      _sse_cpf_ps(mAM,_a90+mm);                             // store a90 for max pixel
      _sse_add_ps(_amp+mm,_m00);                            // update 00 PC           
      _sse_add_ps(_AMP+mm,_m90);                            // update 90 PC           

      for(j=0; j<J; j++) {
         n = int(c[0]+0.1);
         if(ee[n]>Eo) {    
            ee[n] = _sse_rotsub_ps(_m00,c[4],_m90,c[5],_a00+n*f);    // subtract PC from a00
            ee[n]+= _sse_rotsub_ps(_m00,c[6],_m90,c[7],_a90+n*f);    // subtract PC from a90
         }                                                                                  
         c += 8;                                                                            
      }                                                                                     
      pp[m] = _sse_abs_ps(_amp+mm,_AMP+mm);    // store PC energy                           
      k++;                                                                                  
   }
   return k;                                                                             
#else
   return 0;                                                                             
#endif
}                                                                                        

inline int network::_sse_mra_ps(float* amp, float* AMP, float Eo, int K) {
// fast multi-resolution analysis inside sky loop                         
// select max E pixel and either scale or skip it based on the value of residual 
// pointer to 00 phase amplitude of monster pixels                               
// pointer to 90 phase amplitude of monster pixels                               
// Eo - energy threshold                                                         
//  K - max number of principle components to extract                                
// returns number of MRA pixels                                                  

#ifndef __CINT__
   int j,n,mm,J;                                                                   
   int k = 0;                                                                    
   int m = 0;                                                                    
   int f = NIFO/4;                                                               
   int V = (int)this->rNRG.size();                                               
   float*  ee = this->rNRG.data;                            // residual energy   
   float*  pp = this->pNRG.data;                            // PC  energy   
   float*  c  = NULL;
   float   E2 = Eo/2;                                       // threshold  
   float   E;                                                                    
   float mam[NIFO] _ALIGNED;                                                              
   float mAM[NIFO] _ALIGNED;                                                              

   __m128* _m00 = (__m128*) mam;
   __m128* _m90 = (__m128*) mAM;
   __m128* _amp = (__m128*) amp;
   __m128* _AMP = (__m128*) AMP;
   __m128* _a00 = (__m128*) a_00.data;
   __m128* _a90 = (__m128*) a_90.data;

   this->pNRG = 0;

   while(k<K){

      for(j=0; j<V; ++j) if(ee[j]>ee[m]) m=j;               // find max pixel
      if(ee[m]<=Eo) break;  mm = m*f;                                        

      //cout<<k<<" "<<" V= "<<V<<" m="<<m<<" ee[m]="<<ee[m];

      float cc = 0.; 

      _sse_zero_ps(_m00);
      _sse_zero_ps(_m90);

      J = wdmMRA.size(m);                              
      c = wdmMRA.getXTalk(m);                              // c1*c2+c3*c4=c1*c3+c2*c4=0
      for(j=0; j<J; j++) {
         n = int(c[0]+0.1);
         if(ee[n]>Eo) {    
            _sse_rotadd_ps(_a00+n*f,c[4],_a90+n*f,c[6],_m00);    // construct 00 vector
            _sse_rotadd_ps(_a00+n*f,c[5],_a90+n*f,c[7],_m90);    // construct 90 vector
	 }
	 if(ee[n]>0) cc += c[1];
	 c += 8;                                                                           
      }                                                                                     
      _sse_mul_ps(_m00,cc>1?1./cc:0.7); 
      _sse_mul_ps(_m90,cc>1?1./cc:0.7);
      E = _sse_abs_ps(_m00,_m90);                           // get PC energy

      if(E > ee[m]) {                                       // correct overtuning PC
       	 _sse_cpf_ps(mam,_a00+mm);                          // store a00 for max pixel
     	 _sse_cpf_ps(mAM,_a90+mm);                          // store a90 for max pixel
      } 
      _sse_add_ps(_amp+mm,_m00);                            // update 00 PC           
      _sse_add_ps(_AMP+mm,_m90);                            // update 90 PC           

      J = wdmMRA.size(m);                              
      c = wdmMRA.getXTalk(m);                               // c1*c2+c3*c4=c1*c3+c2*c4=0
      for(j=0; j<J; j++) {
	 n = int(c[0]+0.1);
	 if(E<E2 && n!=m) {c+=8; continue;}
	 if(ee[n]>Eo) {    
	    ee[n] = _sse_rotsub_ps(_m00,c[4],_m90,c[5],_a00+n*f);    // subtract PC from a00
	    ee[n]+= _sse_rotsub_ps(_m00,c[6],_m90,c[7],_a90+n*f);    // subtract PC from a90
	    ee[n]+= 1.e-6;
	 }                                                                                  
	 c += 8;                                                                            
      }                                
      pp[m] = _sse_abs_ps(_amp+mm,_AMP+mm)+E2/2;             // store PC energy                           
      //cout<<" "<<ee[m]<<" "<<k<<" "<<E<<" "<<EE<<" "<<endl;                               
      //cout<<" "<<m<<" "<<cc<<" "<<ee[m]<<" "<<E<<" "<<pp[m]<<endl;                               
      k++;                                                                                  
   }
   return k;                                                                             
#else
   return 0;                                                                             
#endif
}

// set constraint parameters
inline void network::constraint(double d, double g) { 
   this->delta = d==0. ? 0.00001 : d; 
   this->gamma = g; 
}


#endif // NETWORK_HH
