/*
# Copyright (C) 2019 Sergey Klimenko
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
// Sergey Klimenko, University of Florida
// universal data container for cluster analysis
// used with DMT and ROOT
//

#ifndef WAVECLUSTER_HH
#define WAVECLUSTER_HH

#include <iostream>
#include "wavearray.hh"
#include <vector>
#include <list>
#include "WaveDWT.hh"
#include "wseries.hh"

typedef std::vector<int> vector_int;


class wavepixel {
  public:
  // need all this crap to use stl::vector<wavepixel> class
  wavepixel() {
    time = 0;  
    frequency = 0;
    rate = 1.;
    clusterID = 0;
    variability = 1.;
    noiserms = 1.;
    index = 0;
    neighbors.clear(); 
    amplitude.clear();
  }
  virtual ~wavepixel(){}
  bool operator == (const wavepixel &) const {return true;}
  bool operator <  (const wavepixel &) const {return true;}
  
  size_t clusterID;                // cluster ID
  size_t time;                     // time index
  size_t frequency;                // frequency index (layer)
  size_t index;                    // position in list
  float  rate;                     // wavelet layer rate
  float  variability;              // average noise variability
  double noiserms;                 // average noise rms
  bool   core;                     // pixel type: true - core , false - halo
  std::vector<int>  neighbors;     // vector of links to neighbors
  std::vector<double> amplitude;   // vector of pixel's amplitudes
};


class wavecluster
{
  public:
      
      // constructors
      
      //: Default constructor
      wavecluster();
    
      //: initialize from binary WSeries<double>
      //!param: input WSeries
      //!param: true if halo, false if no halo included 
      wavecluster(WSeries<double>&, bool=false);

      //: Copy constructor
      //!param: value - object to copy from 
      wavecluster(const wavecluster&);
      
      //: destructor
      virtual ~wavecluster();
    
      // operators

      wavecluster& operator= (const wavecluster&);

      // accessors

      //: initialize wavecluster class from binary WSeries<double>; 
      //!param: false - core only, true - core + halo 
      //!return cluster list size
      virtual size_t init(WSeries<double>&, bool=false);

      //: initialize wavecluster class from binary WSeries<double>; 
      //!param: max number of pixels in clusters to be cleaned (<4);
      //!param: false - core only, true - core + halo 
      //!return pixel occupancy 
      virtual double setMask(WSeries<double>&, int=1, bool=false);

      //: Get the pixel list size
      inline size_t size() { return pList.size(); }

      //: set black pixel probability
      inline void setbpp(double P) { bpp = P; return; }
      //: get black pixel probability
      inline double getbpp() { return bpp; }

      //: set low frequency for variability correction
      inline void setlow(double f) { low = f; return; }
      //: get low frequency
      inline double getlow() { return low; }

      //: set high frequency for variability correction
      inline void sethigh(double f) { high = f; return; }
      //: get low frequency
      inline double gethigh() { return high; }

      //: set noise array
      inline void set(WSeries<double> &n) { nRMS = n; return; }
      //: set variability array
      inline void set(wavearray<float> &v) { nVAR = v; return; }

      /* set noise rms in amplitude array
       * @memo save pixel noise rms 
       * @param rms array (WSeries<double>)
       * @param low frequency for lowpass filter correction
       * @param not used
       */
      void setrms(WSeries<double> &, double=-1., double=-1.);

      /* set noise variabilty in amplitude array
       * @memo save pixel noise variabilty 
       * @param variability array (wavearray<float>)
       * @param low frequency for variability correction
       * @param high frequency for variability correction
       */
      void setvar(wavearray<float> &, double=-1., double=-1.);

      //: Get the the minimum size of the pixel's amplitude vector in the list
      //  return - (minimum_size) if the vector size is different 
      //  for different pixels
      inline int asize();

      //: set selection cuts vector used in mask(), occupancy(), getCluster()
      //!param: cluster ID number
      //!return void
      inline void ignore(size_t i=0) { 
	if(i>0 && i<=sCuts.size()) sCuts[i-1] = true;
	else if(i==0) { for(i=0; i<sCuts.size(); i++) sCuts[i] = false; }
      }

      //: remove halo pixels from pixel list
      //!param: if true - de-cluster pixels
      //!return size of the list 
      virtual size_t cleanhalo(bool=false);

      //: push amplitudes from input WSeries to this pList.amplitude vector
      //!param: this and WSeries objects should have the same tree type  
      //!       and the approximation level size
      //!param: start time offset: start-a.start, illegal if negative
      //!return size of amplitude vector 
      virtual size_t apush(WSeries<double> &a, double=0.);

      //: append input cluster list
      //!param: input cluster list
      //!return size of appended list 
      virtual size_t append(wavecluster &);

      //: merge clusters in the list
      //!param: non
      //!return size of merged list 
      virtual size_t merge(double=0.);

      //: time coincidence between cluster lists 
      //!param: input cluster list
      //!return size of the coincidence list 
      virtual size_t coincidence(wavecluster &, double=1.);

      //: set clusterID field for pixels in pList vector 
      //: create cList structure - list of references to cluster's pixels 
      //!return number of clusters
      virtual size_t cluster();

      //: recursively calculate clusterID pixels in pList vector
      //!param: pixel index in pList vector
      //!return cluster volume (total number of pixels)
      virtual size_t cluster(wavepixel*);

      //: access function to get cluster parameters passed selection cuts
      //!param: string with parameter name
      //!param: amplitude field index
      //!param: rate index, if 0 ignore rate for calculation of cluster parameters
      //!return wavearray object with parameter values for clusters
      wavearray<float> get(char*, int=0, size_t=0);

      //: return noise RMS for selected pixel in pMask
      //!param: pixel time, sec
      //!param: pixel low frequency 
      //!param: pixel high frequency 
      double getNoiseRMS(double, double, double);

// data members

      double start;    // interval start GPS time
      double stop;     // interval stop GPS time 
      double low;      // low frequency boubdary
      double high;     // high frequency boundary
      double bpp;      // black pixel probability
      double shift;    // time shift
      int    ifo;      // detector index: 1/2/3 - L1/H1/H2
      int    run;      // run ID

      //: pixel list
      std::vector<wavepixel> pList;
      //: selection cuts
      std::vector<bool> sCuts;
      //: cluster list created by cluster() with pixel reference to pList
      std::list<vector_int> cList;
      //: cluster type defined by rate
      std::vector<vector_int> cRate;
      //: calibrated noise RMS
      WSeries<double> nRMS;
      //: noise variability
      wavearray<float> nVAR;

}; // class wavecluster

//: compare function to sort pixel objects
int compare_pix(const void*, const void*);
	
inline int wavecluster::asize(){ 
  size_t M = pList.size();
  size_t n = 10000;
  size_t N = 0;
  size_t m,k;
  if(!M) return 0;
  for(k=0; k<M; k++){ 
    m = (&(pList[k]))->amplitude.size();
    if(m<n) n = m;
    if(m>N) N = m;
  }
  if(N-n) { 
     printf("wavecluster::asize(): invalid size of amplitude vector: %d %d\n",(int)N,(int)n);
  }
  return n; 
}


#endif // WAVECLUSTER_HH


















