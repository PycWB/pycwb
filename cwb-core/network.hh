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
#include "watplot.hh"               // remove
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TVectorD.h"
#include "watsse.hh"
#include "watavx.hh"

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

      //:do forward/inverse wavelet transformation by k steps
      //!param: number of steps
      // do not use with WDM transform !!!
      void Forward(size_t k) 
      { for(size_t i=0; i<ifoList.size(); i++) ifoList[i]->TFmap.Forward(k); }
      void Inverse(size_t k) 
      { for(size_t i=0; i<ifoList.size(); i++) ifoList[i]->TFmap.Inverse(k); }

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

      //: initialize network sky maps: fill netvork sensitivity and alignment factor
      //!param: sky map granularity step, degrees
      //!param: theta begin, degrees
      //!param: theta end,   degrees
      //!param: phi begin,   degrees
      //!param: phi end,     degrees
      void setSkyMaps(double,double=0.,double=180.,double=0.,double=360.);

      //: initialize network sky maps: fill netvork sensitivity and alignment factor
      //!param: healpix order
      void setSkyMaps(int);  

      //:calculate network data matrix (NDM)
      //!param: cluster ID
      //!param: lag index
      //!param: statistic identificator
      //!param: resolution idenificator
      //!return: status
      bool setndm(size_t, size_t, bool=true, int=1);
      bool SETNDM(size_t, size_t, bool=true, int=1);	 // used with likelihoodI

      //:read NDM element defined by detectors i,j
      //!param: first detector
      //!param: second detector
      inline double getNDM(size_t i, size_t j) { return NDM[i][j]; }

      //:set delay filters for a network
      //:(-t,t)_i=0, (-t,t)_i=1, .....
      //!param: detector
      // return number of delays for each layer. 
      size_t setFilter(detector* = NULL);                // from detector object

      //:set delay filters and index arrays for a network
      void setDelayFilters(detector* = NULL);            // from detector
      void setDelayFilters(char*, char* = NULL);         // from detector filter files
      void setFilter(char*, char* = NULL);               // from network filter files

      //: Dumps network filter to file *fname in binary format.
      void writeFilter(const char *fname);

      //: reads network filter from file 
      void readFilter(const char*);

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

      //: read DQ segments
      //!param: MDC log file
      // returns number of injections 
      //size_t readSegments(char*,int=11,int=12);

      // set optimal delay index
      // new version which works with WDM delay filters
      // the delay index is symmetric - can be negative.
      // index=0 corresponds to zero delay
      // index>0 corresponds to positive delay (shift right) 
      // index<0 corresponds to negative delay (shift left) 
      // rate: effective data rate (1/delay_step)
      void setDelayIndex(double rate);

      // set optimal delay index
      // work with dumb wavelet 1G delay filters
      //!param: dummy
      void setDelayIndex(int=0);

      //:set sky index array
      //: mode: 0 - now downselection
      //:       1 - exclude duplicate delay configurations
      //:       2 - downselect by 2
      //:       4 - downselect by 4
      size_t setIndexMode(size_t=0);

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

      // 1G: delay detectors in the network with respect to reference  
      // to match sky location theta and phi
      // index array should be setup
      void delay(double theta, double phi);
      // 1G delay detector in the network:  
      // time delay convention: + - shift TS right
      //                        - - shift TS left
      //!param: detector pointer
      //!param: delay filter index
      void delay(detector*, size_t);

      // 1G analysis algorithm for selection of significant network pixles
      //:select TF pixels by setting a threshold on the pixel network likelihood.
      //:integrated time shifts for estimation of the background
      //:input parameters define threshold on TF pixel parameters 
      //!param: threshold on lognormal pixel energy (in units of noise rms)
      //!param: threshold on total pixel energy (in units of noise rms)
      //!param: threshold on likelihood (in units of amplitude SNR per detector) 
      //!return: number of selected samples.
      long  coherence(double, double=0., double=0.);

      // 2G analysis algorithm for selection of significant network pixles
      // LAG - time shift lag defining how detectors are shifted wrt each other.
      // Eo  - pixel energy threshold
      // DD  - dummy
      // hist- pointer to a diagnostic histogram showing network pixel energy. 
      long getNetworkPixels(int LAG, double Eo, double DD=1., TH1F* hist=NULL);

      //:selection of clusters based on: 
      //  'C' - network correlation coefficient
      //  'l' - likelihood (biased)
      //  'L' - likelihood (unbiased)
      //  'A' - snr - null assymetry (double OR)
      //  'E' - snr - null energy    (double OR)
      //  'a' - snr - null assymetry (strict)
      //  'e' - snr - null energy    (strict)
      //!param: threshold
      //!param: minimum cluster size processed by the corrcut
      //!param: cluster type
      //!return: number of rejected pixels.
      size_t netcut(double, char='L', size_t=0, int=1);

      // apply subnetwork cut
      // subnet:  sub network threshold 
      // subcut:  sub network threshold in the skyloop
      // subnorm: norm (Lo/Lt) sub network threshold 
      // lag: lag index
      // return number of processed pixels
      // works only with TD amplitudes. 
      // Includes x regulator for background rejection
      long subNetCut(int lag, float subnet=0.6, float subcut=0.33, float subnorm=0.0, TH2F* hist=NULL);      

      // calculate network likelihood for reconstructed clusters when
      // independent h+ and hx polarisations are reconstructed 
      // implementation for 2-5 detector network.
      //!param: maximized statistic: 
      //!param: threshold to define core pixels (in units of noise rms)
      //        effective only if the second parameter is false
      //!param: cluster ID, if ID=0 - fill likelihood field for all clusters
      //        otherwise, calculate likelihood only for specified cluster.
      //        fill nLikelihood skymap if ID<=0 - need for CED
      //!param: lag index
      //!param: sky index
      //!param: true - for core pixels, false - for core & halo pixels 
      //!return number of processed pixels
      // Options for sky statistics
      //       'e'/'E' - power 
      //       'b'/'B' - un-modeled search with 0 phase data
      long likelihoodB(char='E', double=sqrt(2.), int=0, size_t=0, int=-1, bool=false);      

      // implementation for 2-5 detector network and elliptical constraint
      //       'p'/'P' - un-modeled search with 0 and 90deg phase data
      //       'i'/'I' - (inspiral) elliptical constraint
      //       's'/'S' - (supernova) linear constraint
      //       'g'/'G' - (short GRB) circular constraint
      long likelihoodI(char='P', double=sqrt(2.), int=0, size_t=0, int=-1, bool=false);      

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

      // combines likelihood0() and likelihoodI() and likelihoodB()
      long likelihood(char='E', double=sqrt(2.), int=0, size_t=0, int=-1, bool=false);      
      
      // set pixel ranks in the network netcluster structure 
      // param: time window around pixel for rank calculation
      // param: frequency window around pixel for rank calculation (not used)
      size_t setRank(double, double=0.);

      
      // read the wdm xtalk catalog
      // param: catalog filename
      inline void setMRAcatalog(char* fn){ wdmMRA.read(fn);}

      
      //:reconstruct time clusters 
      //!param: time gap in pixels
      //!return: number of reconstructed clusters
      size_t cluster(int kt, int kf) {
	 if(!wc_List.size()) return 0;
	 size_t m = 0;
	 for(size_t n=0; n<nLag; n++) m += wc_List[n].cluster(kt,kf); 
	 return m;
      }      

      //:return number of events 
      size_t events() {
	if(!wc_List.size()) return 0;
	size_t m = 0;
	for(size_t n=0; n<nLag; n++) {
           m += wc_List[n].esize();
        } 
	return m;
      }      

      //:return number of events with specified type, lag and cID
      // type : sCuts
      // lag  : lag<0 -> select all lags  
      size_t events(int type, int lag=-1) {
	if(!wc_List.size()) return 0;
        if(lag>=(int)nLag) lag=nLag-1;
	size_t m = 0;
        if(lag>=0) m += wc_List[lag].esize(type);
	else for(size_t n=0; n<nLag; n++) m += wc_List[n].esize(type);
	return m;
      }      

      // set noise rms fields for pixels in the network netcluster structure
      inline void setRMS();

      // delink superclusters in wc_List
      inline void delink(){
	for(size_t i=0; i<nLag; i++) wc_List[i].delink();
      }

      //: extract time series for detector responses
      //!param: cluster ID
      //!param: delay index
      //!param: time series type
      //!return: true if time series are extracted
      bool getwave(size_t, size_t, char='W');

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
      void getSkyArea(size_t id, size_t lag, double T);
      //!param: noise rms per DoF
      void getSkyArea(size_t id, size_t lag, double T, double rms);

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
      //!param: detector index
      inline detector* getifo(size_t n) { return ifoListSize()>n ? ifoList[n] : NULL; }
      //:return pointer to a netcluster in wc_List
      //!param: delay index
      inline netcluster* getwc(size_t n) { return wc_ListSize()>n ? &wc_List[n] : NULL; }

      //: add wdm
      //!param: pointer to wdm
      //!return number of wdm tronsforms in the list
      size_t add(WDM<double>* wdm) {if(wdm) wdmList.push_back(wdm); return wdmList.size();}
      //: get size of wdmList
      inline size_t wdmListSize() { return wdmList.size(); }
      //!param: number of wdm layers
      inline WDM<double>* getwdm(size_t M) { 
         for(size_t n=0;n<wdmListSize();n++) if(M==(wdmList[n]->maxLayer()+1)) return wdmList[n]; 
        return NULL; 
      }

      // set run number
      //!param: run
      inline void setRunID(size_t n) { this->nRun=n; return; }
      
      // set wavelet boundary offset
      //!param: run
      inline void setOffset(double t) { this->Edge = t; return; }
      
      // set constraint parameter
      //!param: constraint parameter, p=0 - no constraint
      inline void constraint(double d=1., double g=0.0001) { 
         this->delta = d==0. ? 0.00001 : d; 
         this->gamma = g; 
         //this->gamma=g>0. ? g : 0.00001; 
      }

      // set threshold for double OR energy
      //!param: threshold
      inline void set2or(double p) { this->e2or = p; }

      // calculate WaveBurst threshold as a function of resolution
      //!param: selected fraction of LTF pixels assuming Gaussian noise 
      //!param: maximum time delay between detectors
      double threshold(double, double);

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

      // extract accurate time delay amplitudes for a given sky location
      // parameter 1 - sky location index
      // parameter 2 - array for 0-phase delayed amplitudes 
      // parameter 3 - array for 0-phase delayed amplitudes 
      void updateTDamp(int, float**, float**); 

      // return WDM used/unused
      bool wdm()  {return _WDM;}	
      // get likelihood type, X=likelihoodX, M=likelihoodM, ''=others
      char like() {return _LIKE;}	

      // print network parameters
      void print();             // *MENU*
      virtual void Browse(TBrowser*) {print();}

      static inline double sumx(double*);
      static inline double dotx(double*, double*);
      static inline double dotx(float*, float*);
      static inline double dot4(double*, double*);
      static inline double dotx(double*, double*, double*);
      static inline double dotx(float*, float*, float*);
      static inline double dot4(double*, double*, double*);
      static inline double dotx(double*, double**, size_t);
      static inline double dotx(double**, size_t, double*);
      static inline double dotx(double**, size_t, double**, size_t);
      static inline double dotx(double**, size_t, double**, size_t, double*);
      static inline double dotx(double*, double*, double);
      static inline double dotx(float*, float*, float);
      static inline   void addx(double*, double*, double*);
      static inline   void addx(double*, double**, size_t, double*);
      static inline   void addx(double**, size_t, double**, size_t, double*);
      static inline double dotx(double*, double**, size_t, double*);
      static inline double dot32(std::vector<float>*, double*, std::vector<short>*);
      static inline double dot32(double*, double*, int*);
      static inline double divx(double*, double*);    
      static inline double rotx(double*, double, double*, double, double*);    
      static inline double rotx(float*, float, float*, float, float*);    
      static inline double rot4(double*, double, double*, double, double*);    
      static inline  float rots(float*, float, float*, float, float*);    
      static inline   void mulx(double**, size_t, double**, size_t, double*);
      static inline   void mulx(double*, double, double*);
      static inline   void mulx(float*, float, float*);
      static inline   void mulx(double*, double);
      static inline   void mulx(float*, float);
      static inline   void inix(double**, size_t, double*);
      static inline   void inix(double*, double);
      static inline   void inix(float*, float);
      static inline    int netx(double*, double, double*, double, double);
      static inline    int netx(float*, float, float*, float, float);
      static inline   void pnt_(float**, float**, short**, int, int);
      static inline   void cpp_(float*&, float**);
      static inline   void cpf_(float*& a, double** p);
      static inline   void cpf_(float*& a, double** p, size_t); 

      static inline   void dpfx(float* fp, float* fx);
      static inline   void pnpx(float* fp, float* fx, float* am, float* AM, float* u, float* v);
      static inline   void dspx(float* u, float* v, float* am, float* AM);
      static inline   void dspx(float* fp, float* fx, float* am, float* AM, float* u, float* v);

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
      double e2or;              // threshold on double OR energy
      double acor;              // threshold on coherent pixel energy
      bool   pOUT;              // true/false printout flag
      bool   EFEC;              // true/false - EFEC/selestial coordinate system
      char   tYPe;              // likelihood type
      bool   local;             // true/false - local/global normalization
      bool   optim;             // true/false - process optimal/all resolutions
      double delta;             // weak constraint parameter:
      double gamma;             // hard constraint parameter:
      double precision;         // precision of energy calculation
      double pSigma;            // integration limit in sigmas for probability
      double penalty;           // penalty factor:
      double netCC;             // threshold on netcc:
      double netRHO;            // threshold on rho:
      bool   eDisbalance;       // true/false - enable/disable energy disbalance ECED
      bool   MRA;		// true/false - used/not-used likelihoodMRA
      bool   wfsave;		// true/false - if false only simulated wf are saved (2G)
      int    pattern;           // clustering pattern

      std::vector<vectorD> NDM; // network data matrix

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

      std::vector<delayFilter> filter;    // delay filter (1G)
      std::vector<delayFilter> filter90;  // phase shifted delay filter (1G)

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

      void    like(char _LIKE) {this->_LIKE=_LIKE;}  // set likelihood type	
      void    wdm(bool _WDM)   {this->_WDM =_WDM; }  // set wdm used/unused	
      bool   _WDM;		     // true/false - used/not-used WDM
      char   _LIKE;		     // X=likelihoodX, M=likelihoodM, ''=others

      ClassDef(network,4)

}; // class network

//inline void network::setCatalogFile(char* fn)
//{  wdmOvlp.read(fn);}

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

// special functions
inline double network::sumx(double* a) {
  double d=0.;
  NETX(d+= a[0]; ,
       d+= a[1]; ,
       d+= a[2]; ,
       d+= a[3]; ,
       d+= a[4]; ,
       d+= a[5]; ,
       d+= a[6]; ,
       d+= a[7]; )
     return d;
}

inline double network::dotx(double* a, double* b) {
  double d=0.;
  NETX(d+= a[0]*b[0]; ,
       d+= a[1]*b[1]; ,
       d+= a[2]*b[2]; ,
       d+= a[3]*b[3]; ,
       d+= a[4]*b[4]; ,
       d+= a[5]*b[5]; ,
       d+= a[6]*b[6]; ,
       d+= a[7]*b[7]; )
     return d;
}

inline double network::dotx(float* a, float* b) {
  float d=0.;
  NETX(d+= a[0]*b[0]; ,
       d+= a[1]*b[1]; ,
       d+= a[2]*b[2]; ,
       d+= a[3]*b[3]; ,
       d+= a[4]*b[4]; ,
       d+= a[5]*b[5]; ,
       d+= a[6]*b[6]; ,
       d+= a[7]*b[7]; )
     return d;
}

inline double network::dot4(double* a, double* b) {
  double d=0.;
  d+= a[0]*b[0];
  d+= a[1]*b[1];
  d+= a[2]*b[2];
  d+= a[3]*b[3];
  return d;
}

inline double network::dotx(double* a, double* b, double* c) {
  double d=0.;
  NETX(c[0] = a[0]*b[0]; d+=c[0]; ,
       c[1] = a[1]*b[1]; d+=c[1]; ,
       c[2] = a[2]*b[2]; d+=c[2]; ,
       c[3] = a[3]*b[3]; d+=c[3]; ,
       c[4] = a[4]*b[4]; d+=c[4]; ,
       c[5] = a[5]*b[5]; d+=c[5]; ,
       c[6] = a[6]*b[6]; d+=c[6]; ,
       c[7] = a[7]*b[7]; d+=c[7]; )
     return d;
}

inline double network::dotx(float* a, float* b, float* c) {
  float d=0.;
  NETX(c[0] = a[0]*b[0]; d+=c[0]; ,
       c[1] = a[1]*b[1]; d+=c[1]; ,
       c[2] = a[2]*b[2]; d+=c[2]; ,
       c[3] = a[3]*b[3]; d+=c[3]; ,
       c[4] = a[4]*b[4]; d+=c[4]; ,
       c[5] = a[5]*b[5]; d+=c[5]; ,
       c[6] = a[6]*b[6]; d+=c[6]; ,
       c[7] = a[7]*b[7]; d+=c[7]; )
     return d;
}

inline double network::dot4(double* a, double* b, double* c) {
  double d=0.;
  c[0] = a[0]*b[0]; d+=c[0];
  c[1] = a[1]*b[1]; d+=c[1];
  c[2] = a[2]*b[2]; d+=c[2];
  c[3] = a[3]*b[3]; d+=c[3];
  return d;
}

inline double network::dotx(double* a, double** b, size_t j) {
  double d=0.;
  NETX(d+= a[0]*b[0][j]; ,
       d+= a[1]*b[1][j]; ,
       d+= a[2]*b[2][j]; ,
       d+= a[3]*b[3][j]; ,
       d+= a[4]*b[4][j]; ,
       d+= a[5]*b[5][j]; ,
       d+= a[6]*b[6][j]; ,
       d+= a[7]*b[7][j]; )
     return d;
}

inline double network::dotx(double** a, size_t i, double* b) {
  double d=0.;
  NETX(d+= a[0][i]*b[0]; ,
       d+= a[1][i]*b[1]; ,
       d+= a[2][i]*b[2]; ,
       d+= a[3][i]*b[3]; ,
       d+= a[4][i]*b[4]; ,
       d+= a[5][i]*b[5]; ,
       d+= a[6][i]*b[6]; ,
       d+= a[7][i]*b[7]; )
     return d;
}

inline double network::dotx(double** a, size_t i, double** b, size_t j) {
  double d=0.;
  NETX(d+= a[0][i]*b[0][j]; ,
       d+= a[1][i]*b[1][j]; ,
       d+= a[2][i]*b[2][j]; ,
       d+= a[3][i]*b[3][j]; ,
       d+= a[4][i]*b[4][j]; ,
       d+= a[5][i]*b[5][j]; ,
       d+= a[6][i]*b[6][j]; ,
       d+= a[7][i]*b[7][j]; )
     return d;
}

inline double network::divx(double* a, double* b) {
  double d=0.;
  NETX(d+= a[0]/b[0]; ,
       d+= a[1]/b[1]; ,
       d+= a[2]/b[2]; ,
       d+= a[3]/b[3]; ,
       d+= a[4]/b[4]; ,
       d+= a[5]/b[5]; ,
       d+= a[6]/b[6]; ,
       d+= a[7]/b[7]; )
     return d;
}

inline int network::netx(double* u, double um, double* v, double vm, double g) {
  double d=0.;
  double q = (1.-g)*um;
  NETX(d+= int(u[0]*u[0]>q) - int((u[0]*u[0]/um+v[0]*v[0]/vm)>g); ,
       d+= int(u[1]*u[1]>q) - int((u[1]*u[1]/um+v[1]*v[1]/vm)>g); ,
       d+= int(u[2]*u[2]>q) - int((u[2]*u[2]/um+v[2]*v[2]/vm)>g); ,
       d+= int(u[3]*u[3]>q) - int((u[3]*u[3]/um+v[3]*v[3]/vm)>g); ,
       d+= int(u[4]*u[4]>q) - int((u[4]*u[4]/um+v[4]*v[4]/vm)>g); ,
       d+= int(u[5]*u[5]>q) - int((u[5]*u[5]/um+v[5]*v[5]/vm)>g); ,
       d+= int(u[6]*u[6]>q) - int((u[6]*u[6]/um+v[6]*v[6]/vm)>g); ,
       d+= int(u[7]*u[7]>q) - int((u[7]*u[7]/um+v[7]*v[7]/vm)>g); )
     return d;
}

inline int network::netx(float* u, float um, float* v, float vm, float g) {
  float d=0.;
  float q = (1.-g)*um;
  NETX(d+= int(u[0]*u[0]>q) - int((u[0]*u[0]/um+v[0]*v[0]/vm)>g); ,
       d+= int(u[1]*u[1]>q) - int((u[1]*u[1]/um+v[1]*v[1]/vm)>g); ,
       d+= int(u[2]*u[2]>q) - int((u[2]*u[2]/um+v[2]*v[2]/vm)>g); ,
       d+= int(u[3]*u[3]>q) - int((u[3]*u[3]/um+v[3]*v[3]/vm)>g); ,
       d+= int(u[4]*u[4]>q) - int((u[4]*u[4]/um+v[4]*v[4]/vm)>g); ,
       d+= int(u[5]*u[5]>q) - int((u[5]*u[5]/um+v[5]*v[5]/vm)>g); ,
       d+= int(u[6]*u[6]>q) - int((u[6]*u[6]/um+v[6]*v[6]/vm)>g); ,
       d+= int(u[7]*u[7]>q) - int((u[7]*u[7]/um+v[7]*v[7]/vm)>g); )
     return d;
}

inline double network::dotx(double** a, size_t i, double** b, size_t j, double* p) {
  double d=0.;
  NETX(p[0] = a[0][i]*b[0][j];d+=p[0]; ,
       p[1] = a[1][i]*b[1][j];d+=p[1]; ,
       p[2] = a[2][i]*b[2][j];d+=p[2]; ,
       p[3] = a[3][i]*b[3][j];d+=p[3]; ,
       p[4] = a[4][i]*b[4][j];d+=p[4]; ,
       p[5] = a[5][i]*b[5][j];d+=p[5]; ,
       p[6] = a[6][i]*b[6][j];d+=p[6]; ,
       p[7] = a[7][i]*b[7][j];d+=p[7]; )
     return d;
}

inline double network::dotx(double* a, double* b, double c) {
  double d=0.;
  NETX(d+= a[0]*b[0]; ,
       d+= a[1]*b[1]; ,
       d+= a[2]*b[2]; ,
       d+= a[3]*b[3]; ,
       d+= a[4]*b[4]; ,
       d+= a[5]*b[5]; ,
       d+= a[6]*b[6]; ,
       d+= a[7]*b[7]; )
     return c*d;
}

inline double network::dotx(float* a, float* b, float c) {
  float d=0.;
  NETX(d+= a[0]*b[0]; ,
       d+= a[1]*b[1]; ,
       d+= a[2]*b[2]; ,
       d+= a[3]*b[3]; ,
       d+= a[4]*b[4]; ,
       d+= a[5]*b[5]; ,
       d+= a[6]*b[6]; ,
       d+= a[7]*b[7]; )
     return c*d;
}

inline double network::dotx(double* a, double** b, size_t j, double* p) {
  double d=0.;
  NETX(p[0] = a[0]*b[0][j];d+=p[0]; ,
       p[1] = a[1]*b[1][j];d+=p[1]; ,
       p[2] = a[2]*b[2][j];d+=p[2]; ,
       p[3] = a[3]*b[3][j];d+=p[3]; ,
       p[4] = a[4]*b[4][j];d+=p[4]; ,
       p[5] = a[5]*b[5][j];d+=p[5]; ,
       p[6] = a[6]*b[6][j];d+=p[6]; ,
       p[7] = a[7]*b[7][j];d+=p[7]; )
     return d;
}

inline void network::addx(double* a, double* b, double* p) {
  NETX(p[0] += a[0]*b[0]; ,
       p[1] += a[1]*b[1]; ,
       p[2] += a[2]*b[2]; ,
       p[3] += a[3]*b[3]; ,
       p[4] += a[4]*b[4]; ,
       p[5] += a[5]*b[5]; ,
       p[6] += a[6]*b[6]; ,
       p[7] += a[7]*b[7]; )
     return;
}

inline void network::addx(double* a, double** b, size_t j, double* p) {
  NETX(p[0] += a[0]*b[0][j]; ,
       p[1] += a[1]*b[1][j]; ,
       p[2] += a[2]*b[2][j]; ,
       p[3] += a[3]*b[3][j]; ,
       p[4] += a[4]*b[4][j]; ,
       p[5] += a[5]*b[5][j]; ,
       p[6] += a[6]*b[6][j]; ,
       p[7] += a[7]*b[7][j]; )
     return;
}

inline void network::addx(double** a, size_t i, double** b, size_t j, double* p) {
  NETX(p[0] += a[0][i]*b[0][j]; ,
       p[1] += a[1][i]*b[1][j]; ,
       p[2] += a[2][i]*b[2][j]; ,
       p[3] += a[3][i]*b[3][j]; ,
       p[4] += a[4][i]*b[4][j]; ,
       p[5] += a[5][i]*b[5][j]; ,
       p[6] += a[6][i]*b[6][j]; ,
       p[7] += a[7][i]*b[7][j]; )
     return;
}

inline void network::mulx(double** a, size_t i, double** b, size_t j, double* p) {
  NETX(p[0] = a[0][i]*b[0][j]; ,
       p[1] = a[1][i]*b[1][j]; ,
       p[2] = a[2][i]*b[2][j]; ,
       p[3] = a[3][i]*b[3][j]; ,
       p[4] = a[4][i]*b[4][j]; ,
       p[5] = a[5][i]*b[5][j]; ,
       p[6] = a[6][i]*b[6][j]; ,
       p[7] = a[7][i]*b[7][j]; )
  return;
}

inline void network::mulx(double* a, double b, double* p) {
  NETX(p[0] = a[0]*b; ,
       p[1] = a[1]*b; ,
       p[2] = a[2]*b; ,
       p[3] = a[3]*b; ,
       p[4] = a[4]*b; ,
       p[5] = a[5]*b; ,
       p[6] = a[6]*b; ,
       p[7] = a[7]*b; )
     return;
}

inline void network::mulx(float* a, float b, float* p) {
  NETX(p[0] = a[0]*b; ,
       p[1] = a[1]*b; ,
       p[2] = a[2]*b; ,
       p[3] = a[3]*b; ,
       p[4] = a[4]*b; ,
       p[5] = a[5]*b; ,
       p[6] = a[6]*b; ,
       p[7] = a[7]*b; )
     return;
}

inline void network::mulx(double* a, double b) {
  NETX(a[0] *= b; ,
       a[1] *= b; ,
       a[2] *= b; ,
       a[3] *= b; ,
       a[4] *= b; ,
       a[5] *= b; ,
       a[6] *= b; ,
       a[7] *= b; )
     return;
}

inline void network::mulx(float* a, float b) {
  NETX(a[0] *= b; ,
       a[1] *= b; ,
       a[2] *= b; ,
       a[3] *= b; ,
       a[4] *= b; ,
       a[5] *= b; ,
       a[6] *= b; ,
       a[7] *= b; )
     return;
}

inline void network::inix(double** a, size_t j, double* p) {
  NETX(p[0] = a[0][j]; ,
       p[1] = a[1][j]; ,
       p[2] = a[2][j]; ,
       p[3] = a[3][j]; ,
       p[4] = a[4][j]; ,
       p[5] = a[5][j]; ,
       p[6] = a[6][j]; ,
       p[7] = a[7][j]; )
     return;
}

inline void network::inix(double* p, double a) {
  NETX(p[0] = a; , 
       p[1] = a; , 
       p[2] = a; , 
       p[3] = a; , 
       p[4] = a; , 
       p[5] = a; , 
       p[6] = a; , 
       p[7] = a; ) 
     return;
}

inline void network::inix(float* p, float a) {
  NETX(p[0] = a; , 
       p[1] = a; , 
       p[2] = a; , 
       p[3] = a; , 
       p[4] = a; , 
       p[5] = a; , 
       p[6] = a; , 
       p[7] = a; ) 
     return;
}

inline double network::rotx(double* u, double c, double* v, double s, double* e) {
  double d=0.;
  NETX(e[0] = u[0]*c + v[0]*s;d+=e[0]*e[0]; ,
       e[1] = u[1]*c + v[1]*s;d+=e[1]*e[1]; ,
       e[2] = u[2]*c + v[2]*s;d+=e[2]*e[2]; ,
       e[3] = u[3]*c + v[3]*s;d+=e[3]*e[3]; ,
       e[4] = u[4]*c + v[4]*s;d+=e[4]*e[4]; ,
       e[5] = u[5]*c + v[5]*s;d+=e[5]*e[5]; ,
       e[6] = u[6]*c + v[6]*s;d+=e[6]*e[6]; ,
       e[7] = u[7]*c + v[7]*s;d+=e[7]*e[7]; )
  return d;
}    

inline double network::rotx(float* u, float c, float* v, float s, float* e) {
  double d=0.;
  NETX(e[0] = u[0]*c + v[0]*s;d+=e[0]*e[0]; ,
       e[1] = u[1]*c + v[1]*s;d+=e[1]*e[1]; ,
       e[2] = u[2]*c + v[2]*s;d+=e[2]*e[2]; ,
       e[3] = u[3]*c + v[3]*s;d+=e[3]*e[3]; ,
       e[4] = u[4]*c + v[4]*s;d+=e[4]*e[4]; ,
       e[5] = u[5]*c + v[5]*s;d+=e[5]*e[5]; ,
       e[6] = u[6]*c + v[6]*s;d+=e[6]*e[6]; ,
       e[7] = u[7]*c + v[7]*s;d+=e[7]*e[7]; )
  return d;
}    

inline double network::rot4(double* u, double c, double* v, double s, double* e) {
  double d=0.;
  e[0] = u[0]*c + v[0]*s;d+=e[0]*e[0];
  e[1] = u[1]*c + v[1]*s;d+=e[1]*e[1];
  e[2] = u[2]*c + v[2]*s;d+=e[2]*e[2];
  e[3] = u[3]*c + v[3]*s;d+=e[3]*e[3];
  return d;
}    

inline float network::rots(float* u, float c, float* v, float s, float* e) {
  float d=0.;
  NETX(e[0] -= u[0]*c + v[0]*s; d+=e[0]*e[0]; ,
       e[1] -= u[1]*c + v[1]*s; d+=e[1]*e[1]; ,
       e[2] -= u[2]*c + v[2]*s; d+=e[2]*e[2]; ,
       e[3] -= u[3]*c + v[3]*s; d+=e[3]*e[3]; ,
       e[4] -= u[4]*c + v[4]*s; d+=e[4]*e[4]; ,
       e[5] -= u[5]*c + v[5]*s; d+=e[5]*e[5]; ,
       e[6] -= u[6]*c + v[6]*s; d+=e[6]*e[6]; ,
       e[7] -= u[7]*c + v[7]*s; d+=e[7]*e[7]; )
  return d/2.;
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

inline double network::dot32(std::vector<float>* F, double* p, std::vector<short>* J) {
  return  (*F)[0] *p[(*J)[0]]  + (*F)[1] *p[(*J)[1]]  
	+ (*F)[2] *p[(*J)[2]]  + (*F)[3] *p[(*J)[3]]   
        + (*F)[4] *p[(*J)[4]]  + (*F)[5] *p[(*J)[5]]  
        + (*F)[6] *p[(*J)[6]]  + (*F)[7] *p[(*J)[7]]   
	+ (*F)[8] *p[(*J)[8]]  + (*F)[9] *p[(*J)[9]]  
	+ (*F)[10]*p[(*J)[10]] + (*F)[11]*p[(*J)[11]] 
	+ (*F)[12]*p[(*J)[12]] + (*F)[13]*p[(*J)[13]] 
	+ (*F)[14]*p[(*J)[14]] + (*F)[15]*p[(*J)[15]]
	+ (*F)[16]*p[(*J)[16]] + (*F)[17]*p[(*J)[17]] 
	+ (*F)[18]*p[(*J)[18]] + (*F)[19]*p[(*J)[19]]
        + (*F)[20]*p[(*J)[20]] + (*F)[21]*p[(*J)[21]] 
	+ (*F)[22]*p[(*J)[22]] + (*F)[23]*p[(*J)[23]]
	+ (*F)[24]*p[(*J)[24]] + (*F)[25]*p[(*J)[25]] 
	+ (*F)[26]*p[(*J)[26]] + (*F)[27]*p[(*J)[27]]
	+ (*F)[28]*p[(*J)[28]] + (*F)[29]*p[(*J)[29]] 
	+ (*F)[30]*p[(*J)[30]] + (*F)[31]*p[(*J)[31]];
}

inline double network::dot32(double* F, double* p, int* J) {
 return F[0] *p[J[0]]  + F[1] *p[J[1]]  + F[2] *p[J[2]]  + F[3] *p[J[3]]   
      + F[4] *p[J[4]]  + F[5] *p[J[5]]  + F[6] *p[J[6]]  + F[7] *p[J[7]]   
      + F[8] *p[J[8]]  + F[9] *p[J[9]]  + F[10]*p[J[10]] + F[11]*p[J[11]] 
      + F[12]*p[J[12]] + F[13]*p[J[13]] + F[14]*p[J[14]] + F[15]*p[J[15]]
      + F[16]*p[J[16]] + F[17]*p[J[17]] + F[18]*p[J[18]] + F[19]*p[J[19]]
      + F[20]*p[J[20]] + F[21]*p[J[21]] + F[22]*p[J[22]] + F[23]*p[J[23]]
      + F[24]*p[J[24]] + F[25]*p[J[25]] + F[26]*p[J[26]] + F[27]*p[J[27]]
      + F[28]*p[J[28]] + F[29]*p[J[29]] + F[30]*p[J[30]] + F[31]*p[J[31]];
}

inline void network::dpfx(float* fp, float* fx) {
// transformation to DPF for 4 consecutive pixels.
// rotate vectors fp and fx into DPF

  float t[32],T[32];

  __m128* _t  = (__m128*) t;  __m128* _T  = (__m128*) T;
  __m128* _fp = (__m128*) fp; __m128* _fx = (__m128*) fx;

  _sse_dpf4_ps(_fp,_fx,_t,_T);
  _sse_cpf4_ps(_fp,_t);
  _sse_cpf4_ps(_fx,_T);
}

inline void network::pnpx(float* fp, float* fx, float* am, float* AM, float* u, float* v) {
// projection to network plane (pnp)
// project vectors am and AM on the network plane fp,fx

   __m128* _fp = (__m128*) fp; __m128* _fx = (__m128*) fx;
   __m128* _am = (__m128*) am; __m128* _AM = (__m128*) AM;
   __m128* _u  = (__m128*) u;  __m128* _v  = (__m128*) v;

  _sse_pnp4_ps(_fp,_fx,_am,_AM,_u,_v);
}

inline void network::dspx(float* u, float* v, float* am, float* AM) {
// dual stream phase (dsp) transformation
// take projection vectors u and v,
// make them orthogonal,
// apply phase transformation boths to data a,A and projections u,v

  float U[32],V[32];

   __m128* _am = (__m128*) am; __m128* _AM = (__m128*) AM;
   __m128* _u  = (__m128*) u;  __m128* _v  = (__m128*) v;
   __m128* _U  = (__m128*) U;  __m128* _V  = (__m128*) V;

  _sse_dsp4_ps(_u,_v,_am,_AM,_U,_V);
  _sse_cpf4_ps(_u,_U);
  _sse_cpf4_ps(_v,_V);
}

inline void network::dspx(float* fp, float* fx, float* am, float* AM, float* u, float* v) {
// DPF + DSP transformations
// applied dpfx+pnpx+dspx

  dpfx(fp, fx);
  pnpx(fp, fx, am, AM, u, v);
  dspx(u, v, am, AM);
}

inline int network::_sse_MRA_ps(float* amp, float* AMP, float Eo, int K) {
// fast multi-resolution analysis inside sky loop                         
// select max E pixel and either scale or skip it based on the value of residual 
// pointer to 00 phase amplitude of monster pixels                               
// pointer to 90 phase amplitude of monster pixels                               
// Eo - energy threshold                                                         
//  K - number of principle components to extract                                
// returns number of MRA pixels                                                  

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
}                                                                                        

inline int network::_sse_mra_ps(float* amp, float* AMP, float Eo, int K) {
// fast multi-resolution analysis inside sky loop                         
// select max E pixel and either scale or skip it based on the value of residual 
// pointer to 00 phase amplitude of monster pixels                               
// pointer to 90 phase amplitude of monster pixels                               
// Eo - energy threshold                                                         
//  K - max number of principle components to extract                                
// returns number of MRA pixels                                                  

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
}                                                                                        

inline wavearray<float> network::_avx_norm_ps(float** p, float** q,
					      std::vector<float*> &pAVX, int I) {
   wavearray<float> norm(NIFO+1);     // output array for packet norms
   float* g = norm.data+1; norm=0.;

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

#endif // NETWORK_HH
