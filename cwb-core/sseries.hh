/*
# Copyright (C) 2019 Sergey Klimenko, Gabriele Vedovato, Valentin Necula
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


/**********************************************************
 * Package:      Sparse WSeries Class 
 * File name:    sseries.hh
 * Author:       Gabriele Vedovato (vedovato@lnl.infn.it)
 **********************************************************/

#ifndef SSERIES_HH
#define SSERIES_HH

#ifndef WSERIES_HH
#include "wseries.hh"
#endif
#ifndef NETCLUSTER_HH
#include "netcluster.hh"
#endif
#include "TBits.h"

template<class DataType_t> 
class SSeries : public WSeries<DataType_t>
{
public:

  // Default constructor
  SSeries();

  // Construct SSeries for specific wavelet type
  // default constructor
  SSeries(const Wavelet &w);

  // Construct from wavearray
  // param: value - data to initialize the SSeries object
  SSeries(const wavearray<DataType_t>& value, const Wavelet &w);

  // Copy constructor - SK: this is not a copy constructor
  // param: value - object to copy from
  SSeries(const WSeries<DataType_t>& value);

  //SK: Copy constructor
  //SK: SSeries(const SSeries<DataType_t>& value);
  //SK: also need an assignment operator

  //: destructor
  virtual ~SSeries();

  //SK: we do not need Forward methods - just use those, which are in parent class
  //SK: we need , though, the Inverse() method to reconstruct x(t) from sparse map
  /*
  void Forward(int n = -1)
       {WSeries<DataType_t>::Forward(n);this->Init();}
  void Forward(wavearray<DataType_t> &w, int n = -1)
       {WSeries<DataType_t>::Forward(w,n);this->Init();}
  void Forward(wavearray<DataType_t> &w, Wavelet &s, int n = -1)
       {WSeries<DataType_t>::Forward(w,s,n);this->Init();}
  */

  // set an external TF map used to fill Map00/90 sparse tables
  void SetMap(WSeries<DataType_t>* pws) {Init(pws);}

  // add core pixels index to sparse map table
  void AddCore(size_t ifoID, netcluster* pwc, int ID=0);

  // set the halo parameters 
  void SetHalo(double maxTau=0.042, int lHalo=1, int tHalo=-1);

  // get the halo slice parameter
  // if eslice=false the extraHalo is returned othewise the time_Halo is returned
  int    GetHaloSlice(bool eslice=false) {return eslice ? extraHalo : time_Halo;}
  // get the time halo parameter 
  int    GetHaloLayer() {return layerHalo;}
  // get the halo delay time parameter
  double GetHaloDelay() {return net_Delay;}

  // Get Sparse Index Size
  int GetSparseSize(bool bcore=true); 

  // Get TF pixels : core+halo with index 
  // param: index - pixel index
  // param: pS    - pointer to where pixels are stored 
  bool GetSTFdata(int index, SymmArraySSE<float>* pS);

  // GetSparseIndex
  // Get core index list 
  wavearray<int> GetSparseIndex(bool bcore=true);

  // Reset sparse table
  void ResetSparseTable(); 

  // Add cluster=core+halo pixels to sparse tables
  void UpdateSparseTable();

  // rebuild wseries from sparse table
  void Expand(bool bcore=true);

  // set to 0 all TF map pixels which do not belong to core+halo
  void Clean();

  // resize to 0 the TF map : leave only the sparse map tables
  void Shrink() {CheckWaveletType("Shrink");this->wavearray<DataType_t>::resize(0);}

  // get slice from index
  inline int GetSlice(int index) {int nLayer=this->maxLayer()+1;return (index-index%nLayer)/nLayer;}

  // get layer from index
  inline int GetLayer(int index) {return index%(this->maxLayer()+1);}

   //private:

  void Init(WSeries<DataType_t>* pws=NULL, bool reset=true);

  // get number of WDM layers
  inline int GetLayers() {return this->maxLayer()+1;}          
  // get number of samples in wavelet layer
  inline int GetSlices() {return this->sizeZero();}            

  // get frequency resolution (Hz)
  inline float GetFreqResolution() {return (float)wdm_rate/this->getLevel()/2;}
  // get time resolution (sec)
  inline float GetTimeResolution() {return this->getLevel()/(float)wdm_rate;}

  // get map00/90 value from index
  inline float GetMap00(int index) {return float(this->pWavelet->pWWS[index]);}
  inline float GetMap90(int index) {return float(this->pWavelet->pWWS[index+this->maxIndex()+1]);}

  // get map00/90 value from slice and layer
  inline float GetMap00(int slice, int layer) {return GetMap00(slice*(this->maxLayer()+1)+layer);}
  inline float GetMap90(int slice, int layer) {return GetMap90(slice*(this->maxLayer()+1)+layer);}
 
  // set map00/90 value using index
  inline void SetMap00(int index, DataType_t value) {this->pWavelet->pWWS[index] = value;}
  inline void SetMap90(int index, DataType_t value) {this->pWavelet->pWWS[index+this->maxIndex()+1]=value;}
 
  // set map00/90 value using slice and layer
  inline void SetMap00(int slice, int layer, DataType_t value) {SetMap00(slice*(this->maxLayer()+1)+layer,value);}
  inline void SetMap90(int slice, int layer, DataType_t value) {SetMap90(slice*(this->maxLayer()+1)+layer,value);}

  // get core type using slice and layer
  inline short Core(int slice, int layer) 
   {return core.TestBitNumber(slice*(this->maxLayer()+1)+layer);} 

  // get core type using index
  inline short Core(int index) 
         {return core.TestBitNumber(index);} 

  // search methods to search elements in the sparse table 
  int binarySearch(int array[], int start, int end, int key);
  int binarySearch(int array[], int size, int key);

  // check if Wavelet type is WDM
  inline void CheckWaveletType(TString method) {
           if(this->pWavelet->m_WaveType!=WDMT) {
             cout << "SSeries<DataType_t>::"<<method.Data()
                  <<" : wavelet type not enabled " << endl; exit(1);}
         }

  // data members

  TBits  core;        			// core pixel array 1/0 : core/not-core

  // sparse table 
  wavearray<int>    sparseLookup;	// store the index pointer to the layers 
  TBits  	    sparseType;		// store pixel type 1/0  core/halo 
  wavearray<int>    sparseIndex;	// store pixel index
  wavearray<float>  sparseMap00;	// store pixel 00 amp
  wavearray<float>  sparseMap90;	// store pixel 90 amp;

  int    layerHalo;	// number of sparse layers associated to a pixel [+/- layerHalo]
  int    time_Halo;     // typically half length of time-delay filter
  int    extraHalo;     // number of extra sparse slices associated to a pixel [+/- (time_Halo+extraHalo)]
  double net_Delay;     // delay time (sec) used to compute extraHalo = net_Delay*this->wrate()

  // WDM params	(used to Expand class after Shrink or after loading from file)
  //SK: all parameters below are stored in pWavelet
  int wdm_BetaOrder;
  int wdm_m_Layer;
  int wdm_KWDM;
  int wdm_precision;
  int wdm_rate;
  int wdm_start;
  int wdm_nSTS;

  ClassDef(SSeries,2)
};

#endif
