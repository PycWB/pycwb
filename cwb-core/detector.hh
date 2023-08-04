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


//**********************************************************
// Wavelet Analysis Tool  
// Sergey Klimenko, University of Florida
// universal data container for x-correlation analysis
// used with DMT and ROOT
//**********************************************************

#ifndef DETECTOR_HH
#define DETECTOR_HH

#include <iostream>
#include <vector>
#include "complex"
#include "wseries.hh"
#include "netcluster.hh"
#include "skymap.hh"
#include "wavecomplex.hh"
#include "skycoord.hh"
#include "TString.h"
#include "TNamed.h"
#include "sseries.hh"

typedef wavecomplex d_complex;
typedef std::vector<int> vector_int;

struct delayFilter {
  std::vector<short> index;   // relative wavelet array index
  std::vector<float> value;   // amplitude
};

struct detectorParams {
  char   name[32];
  double latitude;
  double longitude;
  double elevation;
  double AltX;       // elevation of the x arm
  double AzX;        // azimut of the x arm  (angle-deg from nord)
  double AltY;       // elevation of the y arm
  double AzY;        // azimut of the y arm (angle-deg from nord)
};

enum POLARIZATION {
  TENSOR  = 0,
  SCALAR  = 1
};

typedef WSeries<double> WSeriesD;

class detector : public TNamed
{
  public:
      
      // constructors
      
      //: Default constructor
      detector();

      //: detector name constructor
      detector(char*, double=0.);

      //: user defined detector constructor
      detector(detectorParams, double=0.);
    
      //: Copy constructor
      //!param: value - object to copy from 
      detector(const detector&);                       // use with caution
      
      //: destructor
      virtual ~detector();
    
      // operators

      detector& operator= (const detector&);           // use with caution
      detector& operator= (const WSeries<double> &);   // use with caution
      detector& operator<< (detector&);                // copy 'from' inj/rec stuff
      detector& operator>> (detector&);                // copy 'to'   inj/rec stuff
 
      // accessors

      //: return antenna pattern  
      //!param: source theta,phi, polarization angle psi in degrees
      wavecomplex antenna(double, double, double=0.);

      // returns the detector parameters
      detectorParams getDetectorParams();

      //: return detector delay  
      // time delay convention: t_detector-tau - arrival time at the center of Earth
      //!param: source theta,phi angles in degrees
      double getTau(double, double);

      //: set detector radius vector  
      //!param: Rx,Ry,Rz in ECEF frame
//      void setRV(double, double, double);
      //: set detector x-arm unit vector  
      //!param: Ex,Ey,Ez in ECEF frame
//      void setEx(double, double, double);
      //: set detector y-arm unit vector  
      //!param: Ex,Ey,Ez in ECEF frame
//      void setEy(double, double, double);

      //: initialize detector tensor  
      //!param: no parameters
      void init();

      //: initialize delay filter for non-heterodine wavelet
      //!param: filter length
      //!param: filter phase delay in degrees
      //!param: number of up-sample layers
      //!return filter size
      size_t setFilter(size_t, double=0., size_t=0);

      //: initialize delay filter from another detector 
      //!param: detector
      //!return filter size
      size_t setFilter(detector&); 

      //: write/read delay filter to file in the format:
      //: first string: wavelet filter length, number of wavelet layers  
      //!param: file name
      void writeFilter(const char*);
      void readFilter(const char*);

      // clear filter and release memory (need for vector class)
      inline void clearFilter() {
	filter.clear();
	std::vector<delayFilter>().swap(filter);   // release memory
      }

      //: apply sample delay to timeseries stored in TFmap to adjast
      //  timing for a given theta and phy sky location.
      //!param: theta
      //!param: phi
      void delay(double, double);
 
      //: apply sample delay to input timeseries to adjast
      //  timing for a given theta and phy sky location. Do not store in the TFmap
      //!param: input TS
      //!param: theta
      //!param: phi
     void delay(wavearray<double> &, double, double);

      //: apply delay T to input timeseries.
      //!param: input TS
      //!param: time delay T
     void delay(wavearray<double> &, double);

      //: apply non-heterodine delay filter to input WSeries and put result in TFmap  
      // time delay convention: + - shift TS right
      //                        - - shift TS left
      //!param: time delay in seconds
      //!param: WSeries which should be delayed
      void delay(double, WSeries<double> &);

      //: get pointer to time series data   
      //!param: no parameters
      inline wavearray<double>* getHoT() { return &HoT; }

      //: get pointer to TF data
      //!param: no parameters
      inline WSeries<double>* getTFmap() { return &TFmap; }

      //: get sparse map index in vSS vector for input resolution (wavelet rate r)
      inline int getSTFind(double r) {          
         for(size_t i=0; i<vSS.size(); i++) if(int(vSS[i].wrate()-r)==0) return i;
         return vSS.size()+1; 
      }
      //: get pointer to sparse TF data for input index n
      inline SSeries<double>* getSTFmap(size_t n) { return n<vSS.size() ? &vSS[n] : NULL; } 

      // operations with sparse maps
      inline size_t ssize() { return vSS.size(); } // get size of sparse TF vector (number of resolutions)
      inline void sclear() { vSS.clear(); }        // clear
      inline void addSTFmap(netcluster* pwc, double mTau=0.042);

      //: get size of sparse TF vector (number of resolutions)
      //!param: no parameters

      //: reconstruct wavelet series for a cluster, put it in waveForm  
      //! param: input parameter is the cluster ID in the netcluster structure
      //! param: input netcluster structure
      //! param: amplitude type
      //! param: amplitude index
      //! return: cluster average noise rms
      double getwave(int, netcluster&, char, size_t);

      //: set tau array
      // time delay convention: t_detector-tau - arrival time at the center of Earth
      //!param - step on phi and theta
      //!param - theta begin
      //!param - theta end
      //!param - phi begin
      //!param - phi end
      void setTau(double,double=0.,double=180.,double=0.,double=360.);

      //: set tau array
      // time delay convention: t_detector-tau - arrival time at the center of Earth
      //!param - healpix order
      void setTau(int); 

      //: set antenna patterns
      //!param - step on phi and theta
      //!param - theta begin
      //!param - theta end
      //!param - phi begin
      //!param - phi end
      void setFpFx(double,double=0.,double=180.,double=0.,double=360.);

      //: set antenna patterns
      //!param - healpix order
      void setFpFx(int); 

      //: return noise variance for selected wavelet layer or pixel
      //: used in coherence() and Rank() functions
      //!param: wavelet layer index (frequency)
      //!param: wavelet time index
      //!if param 2 is specified - return noise rms for specified pixel location 
      //!if param 2 is not specified - return rms averaged over the layer  
      double getNoise(size_t,int=-1);

      /* calculate and save average noise rms for each pixel in netcluster wc 
       * @memo save pixel noise rms 
       * @param input netcluster 
       * @param detector index in tne netcluster
       */
      bool setrms(netcluster*, size_t=0);

      //: return pointer to noise array
//      WSeries<double>* getNoise();

      //: whiten data in TFmap and save noise RMS in nRMS
      // param 1 - time window dT. if = 0 - dT=T, where T is wavearray duration
      // param 2 - 0 - no whitening, 1 - single whitening, >1 - double whitening
      // param 3 - boundary offset 
      // param 4 - noise sampling interval (window stride)  
      //           the number of measurements is k=int((T-2*offset)/stride)
      //           if stride=0, then stride is set to dT
      // output: save array of noise RMS in nRMS
      //!what it does: see algorithm description in wseries.hh
      void white(double dT=0, int wtype=1,double offset=0.,double stride=0.) {
         nRMS = TFmap.white(dT, wtype, offset, stride); return;
      }

      // set constant time shift
      inline void shift(double s) { 
	int I = TFmap.maxLayer()+1;
	double R = TFmap.wavearray<double>::rate();
	if(TFmap.size()<2 || R<=0.) this->sHIFt = s; 
	else                        this->sHIFt = int(s*R/I+0.1)*I/R;
      }
      // get constant time shift
      inline double shift() { return this->sHIFt; }

      // rotate arms in the detector plane by angle a in degrees
      void rotate(double);

      // apply band pass filter with cut-offs specified by parameters
      // f1>=0 && f2>=0 : band pass 
      // f1<0 && f2<0   : band cut 
      // f1<0 && f2>=0  : low  pass 
      // f1>=0 && f2<0  : high pass 
      // f1==0          : f1 = TFmap.getlow()
      // f2==0          : f2 = TFmap.gethigh()
      void bandPass1G(double f1=0.,double f2=0.);	// used by 1G
      void bandPass(double f1,double f2, double a=0.){
	 this->TFmap.bandpass(f1,f2,a);
      }; 
      void  bandCut(double f1,double f2) {bandPass(-fabs(f1),-fabs(f2));}
      void  lowPass(double f1)           {bandPass(-fabs(f1),0);}
      void highPass(double f2)           {bandPass(0,-fabs(f2));}

      // calculate hrss of injected responses
      // param 1 - injection wavelet series
      // param 2 - array with injection time
      // param 3 - integration window in seconds
      // param 4 - save input waveform  
      size_t setsim(WSeries<double> &, std::vector<double>*, double=5., double=8., bool saveWF=false);

      // modify input signals (wi) at times pT according the factor pF
      size_t setsnr(wavearray<double> &, std::vector<double>*, std::vector<double>*, double=5., double=8.);  

      // print detector parameters
      void print();		// *MENU*
      virtual void Browse(TBrowser *b) {print();}

      bool isBuiltin() {return TString(dP.name).Sizeof()>1 ? false : true;}

      void setPolarization(POLARIZATION polarization=TENSOR) {this->polarization=polarization;}
      POLARIZATION getPolarization() {return this->polarization;}

     inline double get_SS(){double rms=waveForm.rms(); return rms*rms*waveForm.size();} 
     inline double get_XX(){double rms=waveBand.rms(); return rms*rms*waveBand.size();} 
     inline double get_NN(){double rms=waveNull.rms(); return rms*rms*waveNull.size();} 
     inline double get_XS(){double rmx=waveBand.rms(); double rms=waveForm.rms(); return rmx*rms*waveBand.size();} 
     double getWFfreq(char atype='S');
     double getWFtime(char atype='S');

     int    wfsave()           {return this->wfSAVE;}
     void   wfsave(int wfSAVE) {this->wfSAVE = (wfSAVE>=0)&&(wfSAVE<=3) ? wfSAVE : 0;}
                               // 0 : save detector definitions
                               // 1 : save injected waveforms
                               // 2 : save reconstructed waveforms
                               // 3 : save injected & reconstructed waveforms

// data members

//: position in ECEF frame

      char Name[16];      // detector name
      size_t ifoID;       // detector ID in the network - set up by network method
      detectorParams dP;  // user detector parameters
      double Rv[3];       // radius vector to beam splitter
      double Ex[3];       // vector along x-arm     
      double Ey[3];       // vector along y-arm     
      double DT[9];       // detector tenzor
      double ED[5];       // network energy disbalance  
      double sHIFt;       // time shifts for background analysis
      double null;        // unbiased null stream
      double enrg;        // total energy of PC components   
      double sSNR;        // reconstructed response s-SNR  
      double xSNR;        // reconstructed response x-SNR  
      double ekXk;        // mean of reconstructed detector response  
      double rate;        // original data rate (before downsampling)  
      size_t nDFS;        // number of Delay Filter Samples
      size_t nDFL;        // number of Delay Filter Layers
      int    wfSAVE; 	  // used in streamer method to save waveforms stuff

      skymap tau;         // detector delay with respect to ECEF
      skymap mFp;         // F+ skymap
      skymap mFx;         // Fx skymap

      wavearray<double> HoT;        // detector time series

      std::vector<SSeries<double> > vSS; // sparse TFmap

      WSeries<double> TFmap;        // wavelet data
      WSeries<double> waveForm;     // buffer for a waveform
      WSeries<double> waveBand;     // buffer for a bandlimited waveform
      WSeries<double> waveNull;     // buffer for noise = data - signal
      WSeries<double> nRMS;         // noise RMS
      WSeries<float>  nVAR;         // noise variability

      std::vector<delayFilter> filter;  // delay filter 

      wavearray<double> fp;         // sorted F+ pattern
      wavearray<double> fx;         // sorted Fx pattern
      wavearray<double> ffp;        // sorted F+ * F+ + Fx * Fx
      wavearray<double> ffm;        // sorted F+ * F+ - Fx * Fx
      wavearray<double> fpx;        // sorted F+ * Fx * 2
      wavearray<short>  index;      // index array for delayed amplitude (network)
      wavearray<double> lagShift;   // time shifts for background analysis

      wavearray<double>  HRSS;      // hrss of injected signals
      wavearray<double>  ISNR;      // injected SNR
      wavearray<double>  FREQ;      // frequency of injected signals
      wavearray<double>  BAND;      // bandwith of injected signals
      wavearray<double>  TIME;      // central time of injected signals
      wavearray<double>  TDUR;      // duration of injected signals

      std::vector<int>  IWFID;                  // injected waveforms ID
      std::vector<wavearray<double>*>  IWFP;    // injected waveforms pointers 

      std::vector<int>  RWFID;                  // reconstructed waveforms ID
      std::vector<wavearray<double>*>  RWFP;    // reconstructed waveforms pointers 

      POLARIZATION polarization;    // gw polarization states : TENSOR (fp,fx) SCALAR (fo)

      ClassDef(detector,4)   

}; // class detector
	

inline void detector::addSTFmap(netcluster* pwc, double mTau){
   SSeries<double> SS; vSS.push_back(SS);
   int j = vSS.size()-1;
   vSS[j].SetMap(&TFmap);
   vSS[j].SetHalo(mTau);
   vSS[j].AddCore(ifoID,pwc);
   vSS[j].UpdateSparseTable();
}

inline size_t minDFindex(delayFilter &F){
   size_t n = F.value.size();
   if(!n) return 0;
   size_t j = 0;
   double x = 1.e13;
   for(size_t i=0; i<n; i++){
      if(x>fabs(F.value[i])) { x = fabs(F.value[i]); j=i; }
   }
   return j+1;
}

#endif // DETECTOR_HH

