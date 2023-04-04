/*
# Copyright (C) 2019 Sergey Klimenko, Valentin Necula
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


// Wavelet Analysis Tool: 
// Sergey Klimenko, University of Florida
//--------------------------------------------------------------------
// Implementation of
// Wilson-Daubechies transform
// References:
//--------------------------------------------------------------------


#ifndef WDM_HH
#define WDM_HH

#include "SymmArray.hh"
#include "SymmArraySSE.hh"
#include "SymmObjArray.hh"
#include "WaveDWT.hh"
#include "wavearray.hh"

#define MAXBETA 8

template<class DataType_t> class SSeries; 

template<class DataType_t> 
class WDM : public WaveDWT<DataType_t>
{
public:
   // dummy constructor
   WDM();

   // constructor
   WDM(int, int, int, int);

   // WDMK constructor
   WDM(int);

   //: copy constructors
   WDM(const WDM &);
   
   //: destructor
   virtual ~WDM();
   
   //: Duplicate on heap
   virtual WDM* Clone() const;

   //: light-weight duplicate
   virtual WDM* Init() const;
   
   // get maximum possible level of decomposition
   int getMaxLevel();
   
   void forward(int, int);
   void inverse(int, int);
   
   // get the time domain representation of the basis function corresponding to pixel (m,n)  
   // param1: frequency index
   // param2: time index
   // param3: where to store it
   // returns translation constant (time shift not included in w)
   int getBaseWave(int m, int n, SymmArray<double>& w);
   
   // same as above but for the quadrature
   int getBaseWaveQ(int m, int n, SymmArray<double>& w);
   
   // get the time domain representation of the basis function corresponding to pixel j  
   // param1: pixel index
   // param2: where to store it
   // param3: Quadrature flag (true for quadrature)
   // returns translation constant (sampling steps; time shift not included in w)
   int getBaseWave(int j, wavearray<double>& w, bool Quad=false);
   
   
   //: get array index of the first sample for (level,layer)
   int getOffset(int, int);
   
   //: forward transfom
   //: param: -1 - orthonormal, 0 - power map, >0 - upsampled map
   void t2w(int);
   
   //: inverse transform (flag == -2 means do inverse of Quadrature)
   void w2t(int flag);
   void w2tQ(int);
   // returns pixel amplitude
   // param1: frequency index
   // param2: time index
   // param3: delay index
   // param4: 00 phase - true, 90 phase - false
   double getPixelAmplitude(int, int, int, bool=false);
   double getPixelAmplitudeSSEOld(int, int, int, bool=false);
   float getPixelAmplitudeSSE(int m, int n, int dT, bool Quad);
   void getPixelAmplitudeSSE(int m, int n, int t1, int t2, float* r, bool Quad);
   
   
   double TimeShiftTest(int dt);
   double TimeShiftTestSSE(int dt);

   // override getTDamp from WaveDWT
   // return time-delayed amplitudes for sample n and TD index m:
   float getTDamp(int n, int m, char c='p');
   
   
   // override getTDvec from WaveDWT
   // return array of time-delayed amplitudes in the format:
   // [-n*dt , 0,  n*dt,  -n*dt,  0,  n*dt] - total 2(2n+1) amplitudes
   // where n = k*LWDM (k - second parameter) 
   // param - index in TF map
   // param - range of h(t) sample delays k
   // param - 'a','A' - delayed amplitudes, 'p','P' - delayed power.
   wavearray<float> getTDvec(int j, int K, char c='p');
   
   wavearray<float> getTDvecSSE(int j, int K, char c, SSeries<double>* pss);
   //wavearray<float> getTDvecSSE(int j, int K, char c);
   
   // similar to getTDvec
   void getTFvec(int j, wavearray<float>& r);
   
   //  void transform2();
   //  double pixel(int f0, int t0);
   
   static inline double *Cos[MAXBETA], *Cos2[MAXBETA], *SinCos[MAXBETA];
   static inline double CosSize[MAXBETA], Cos2Size[MAXBETA], SinCosSize[MAXBETA];
   static inline int objCounter;

   void initFourier();

   // getFilter(): returns 3 different  WDM filters
   // param1: filter length
 
   wavearray<double> getFilter(int n);
  // set symmetric time-delay filter arrays used to calculate TD filter 
   void setTDFilter(int nCoeffs, int L=1); //L determines fractional increment tau/L
   wavearray<double> getTDFilter2(int n, int L);
   wavearray<double> getTDFilter1(int n, int L); 

   
   // override getSlice() from WaveDWT
   std::slice getSlice(double n);

   void SetTFMap();   
   //double PrintTDFilters(int m, int dt, int nC);
   // return size of global TD array
   inline size_t Last(int n=0) { return T0.Last(); }
   // return half size of TD filter
   virtual size_t getTDFsize() { return T0.Last() ? T0[0].Last() : 0; }


   int BetaOrder;               // beta function order for Meyer 
   int precision;               // wavelet precision
   int KWDM;                    // K - parameter
   int LWDM;                    // unit time delay is tau/LWDM where tau is 1/hot_rate 
   wavearray<double> wdmFilter; // WDM filter
   
   SymmObjArray<SymmArraySSE<float> > T0;  // time-delay filters
   SymmObjArray<SymmArraySSE<float> > Tx;  // time-delay filters
   wavearray<float> sinTD, cosTD, sinTDx; 
      
   DataType_t** TFMap00;        //! pointer to 0-phase data, by default not initialized
   DataType_t** TFMap90;        //! pointer to 90-phase data, by default not initialized
   void (*SSE_TDF)();           //!
   float* td_buffer;		//!
   float* td_data;		//!
   SymmArraySSE<float> td_halo[6];
   
protected:
   void initSSEPointers();
   
   ClassDef(WDM,2)

}; // class WDM

#endif // WDM_HH

