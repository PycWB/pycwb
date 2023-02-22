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


// Wavelet Analysis Tool
//$Id: WaveDWT.hh,v 1.3 2001/12/15 03:27:29 jzweizig Exp $
#ifndef WAVEDWT_HH
#define WAVEDWT_HH

#include <valarray>
#include "Wavelet.hh"
#include "wavearray.hh"
#include "TNamed.h"
#include "TBuffer.h"

//namespace datacondAPI {
//namespace wat {

template<class DataType_t> class SSeries; 

template<class DataType_t> 
class WaveDWT : public Wavelet 
{
   public: 

      //: constructor
      WaveDWT(int mH=1, int mL=1, int tree=0, enum BORDER border=B_CYCLE);

      //: construct from the base class
      WaveDWT(const Wavelet &);

      //: copy constructor
      WaveDWT(const WaveDWT<DataType_t> &);

      //: Destructor
      virtual ~WaveDWT();
    
      //: duplicate on heap - uses copy constructor
      virtual WaveDWT<DataType_t>* Clone() const;

      //: light-weight duplicate 
      virtual WaveDWT<DataType_t>* Init() const {return this->Clone();}

      //: get maximum possible level of wavelet decompostion
      virtual int getMaxLevel();
      virtual int getMaxLevel (int i) {return Wavelet::getMaxLevel(i);}

      //: make slice for layer with specified index
      virtual std::slice getSlice(const double);

      //: make slice for (level,layer)
      virtual std::slice getSlice(const int, const int);

      virtual float getTDamp(int j, int k, char c='p') {
         // return time-delayed amplitude for delay index k
         // should be implemented in the transformation class (see WDM.hh)
         float x;
         if(j>=0 && j<(int)nWWS) x = c=='a' || c=='A' ? pWWS[j] : pWWS[j]*pWWS[j];
         return x;
      }

      virtual wavearray<float> getTDvec(int j, int k, char c='p') {
         //: return array of time-delayed amplitudes between -k : k
         //: should be implemented in the transformation class (see WDM.hh)
         wavearray<float> x(1);
         if(j>=0 && j<(int)nWWS) x.data[0] = c=='a' || c=='A' ? pWWS[j] : pWWS[j]*pWWS[j];
         return x;
      }

      virtual wavearray<float> getTDvecSSE(int j, int k, char c, SSeries<double>* pss){
         //: return array of time-delayed amplitudes between -k : k
         //: should be implemented in the transformation class (see WDM.hh)
         wavearray<float> x(1);
         cout<<"I am getTDvecSSE in WaveDWT\n";
         return x;
      }

      //: fills r with amplitudes needed to compute time-delays (for both quadratures)
      virtual void getTFvec(int j, wavearray<float>& r) {}

      //: returns size of time-delay filter
      virtual size_t getTDFsize() { return 0; }

      //: Allocate data (set pWWS)
      bool allocate(size_t, DataType_t *);

      //: return allocate status (true if allocated)
      bool allocate();

      //: Release data
      void release();

      //: forward wavelet transform
      virtual void t2w(int=1);
      //: inverse wavelet transform
      virtual void w2t(int=1);

      //: makes one FWT decomposition step
      virtual void forwardFWT(int, int,
			   const double*,
			   const double*);
      //: makes one FWT reconstruction step
      virtual void inverseFWT(int, int,
			   const double*,
			   const double*);

      //: makes one prediction step for Lifting Wavelet Transform
      virtual void predict(int,int,const double*);
      //: makes one update step for Lifting Wavelet Transform
      virtual void update(int,int,const double*);

      //: virtual functions for derived wavelet classes

      //: makes one FWT decomposition step
      virtual void forward(int,int){}
      //: makes one FWT reconstruction step
      virtual void inverse(int,int){}

      DataType_t *pWWS;     //! pointer to wavelet work space      
      unsigned long nWWS;   // size of the wavelet work space
      unsigned long nSTS;   // size of the original time series

      ClassDef(WaveDWT,1)

}; // class WaveDWT


//}; // namespace wat
//}; // namespace datacondAPI

#endif // WAVEDWT_HH

