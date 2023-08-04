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


// Wavelet Analysis Tool  
// Sergey Klimenko, University of Florida
// universal data container for network cluster analysis
// used with DMT and ROOT
//

#ifndef NETCLUSTER_HH
#define NETCLUSTER_HH

#include <iostream>
#include "wavearray.hh"
#include <vector>
#include <list>
#include "TH2F.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "WaveDWT.hh"
#include "wseries.hh"
#include "netpixel.hh"
#include "wat.hh"
#include "TNamed.h"
#include "TFile.h"
#include "TTree.h"
#include "WDM.hh"

typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;

struct upixel {                 // micro pixel
  double time;                  // time
  double dt;
  double frequency;             // frequency
  double df;
  double likelihood;            // likelihood
};

class network;

class clusterdata : public TNamed {
public:
   clusterdata(){}
   ~clusterdata(){}
   float energy;                   // total cluster energy
   float enrgsky;                  // cluster energy in all resolutions
   float likenet;                  // signal energy
   float netecor;                  // network coherent energy
   float normcor;                  // normalized coherent energy
   float netnull;                  // null energy in the sky loop with Gauss correction
   float netED;                    // energy disbalance
   float Gnoise;                   // estimated contribution of Gaussian noise
   float likesky;                  // likelihood at all resolutions
   float skycc;                    // network cc from the sky loop (all resolutions)
   float netcc;                    // network cc for MRA or SRA analysis
   float skyChi2;                  // chi2 stat from the sky loop (all resolutions)
   float subnet;                   // first subNetCut statistic
   float SUBNET;                   // second subNetCut statistic
   float skyStat;                  // localization statistic
   float netRHO;                   // coherent SNR per detector
   float netrho;                   // reduced coherent SNR per detector
   float theta;                    // source angle theta index
   float phi;                      // source angle phi index
   float iota;                     // inclination angle 
   float psi;                      // polarisation angle
   float ellipticity;              // waveform ellipticity
   float cTime;                    // supercluster central time
   float cFreq;                    // supercluster central frequency
   float gNET;                     // network acceptance
   float aNET;                     // network alignment
   float iNET;                     // network index
   float norm;                     // packet norm
   float nDoF;                     // cluster degrees of freedom
   float tmrgr;                    // merger time
   float tmrgrerr;                 // merger time error
   float mchirp;                   // chirp mass
   float mchirperr;                // chirp mass error
   float chi2chirp;                // chi2 over NDF
   float chirpEfrac;               // chirp energy fraction
   float chirpPfrac;               // chirp pixel fraction
   float chirpEllip;               // chirp ellipticity
   int skySize;                    // number of sky pixels
   int skyIndex;                   // index in the skymap
   TF1 fit;                        //! chirp fit parameters (don't remove ! fix crash when exit from CINT)
   TGraphErrors chirp;             // chirp graph 
   wavearray<float> mchpdf;        // chirp mass PDF
   ClassDef(clusterdata,3)
};

ClassImp(clusterdata)


class netcluster : public TNamed
{
public:
      
   // constructors
      
   //: Default constructor
   netcluster();
   
   //: Copy constructor
   //!param: value - object to copy from 
   netcluster(const netcluster&);
   
   //: destructor
   virtual ~netcluster();
   
   // operators
   
   netcluster& operator= (const netcluster&);
   
   // copy function (used in = operator as x.cpf(y,false)
   //: copy content of y into x. 
   // If no clusters are reconstructed - copy all pixels.
   // If clusters are reconstructed - copy selected clusters.
   // param: netcluster object y
   // param: condition to select clusters: 
   //        true  - copy selected clusters with core pixels 
   //        false - copy all selected clusters
   // param: min size of BIG clusters
   //        =0 - copy all clusters regardles of their size
   //        >0 - ignore BIG clusters 
   //        <0 - copy BIG clusters 
   size_t cpf(const netcluster &, bool=false, int=0); 
   
   
   // accessors
   
   //: clear lists and release memory; 
   inline void clear();
   //: clean time delay data 
   inline void clean(int cID=0);
   //: clean input wseries ws removing rejected clusters
   //: output number of remaining pixels 
   int clean(WSeries<double>& ws);
   
   //: Get pixel list size
   inline size_t size() { return pList.size(); }
   //: Get pixel list capacity
   inline size_t capacity() { return pList.capacity(); }
   //: Get cluster list size
   inline size_t csize() { return cList.size(); }
   //: Get number of clusters with specified type
   inline size_t esize(int k=2) {
      size_t n=0;
      for(size_t i=0; i<sCuts.size(); i++) {
	 if(k<2 && sCuts[i]==k) n++;
	 if(k>1 && sCuts[i] <1) n++;
      }
      return n;
   }

   //: Get number of selected pixels
   inline size_t psize(int k=2) {
      size_t n=0;
      for(size_t i=0; i<sCuts.size(); i++) {
	 if(k>1 && sCuts[i] <1) n+=cList[i].size();
	 if(k<2 && sCuts[i]==k) n+=cList[i].size();
      }
      return n;
   }
   
   //: Get pixel pointer
   //:parameter n -  cluster ID
   //:parameter i -  pixel index in cList[n]
   inline netpixel* getPixel(size_t n, size_t i);
   
   //: set black pixel probability
   inline void setbpp(double P) { bpp = P; return; }
   //: get black pixel probability
   inline double getbpp() { return bpp; }
   
   //: set pixel core fields to be equal to the input parameter
   //  core: true/false
   //  id  : set core for selected cluster id
   size_t setcore(bool core,int id=0);

   //: set selection cuts vector used in mask(), occupancy(), getCluster()
   //!param: cluster ID number
   //!return void
   inline void ignore(size_t n) {
      if(n>0 && n<=sCuts.size()) sCuts[n-1] = 1;
   }

   //: set sCuts vector excluding rejected clusters
   //!param: sCuts flag
   inline void setcuts(int n=0) { 
      for(size_t i=0; i<sCuts.size(); i++) 
	 if(sCuts[i]!=1) sCuts[i] = n; 
   }
   
   //: remove halo pixels from pixel list
   //!param: if false - de-cluster pixels
   //!return size of the list 
   virtual size_t cleanhalo(bool=false);
   
   //: add halo pixels matching the paccket pattern to the pixel list
   //!param: packet pattern mode
   //!return size of the list 
   size_t addhalo(int=0);
   
   //: append pixel list from input cluster list
   //: cluster metadata is lost
   //!param: input netcluster
   //!return size of appended pixel list 
   virtual size_t append(netcluster& wc);
   
   //: add netpixel object to the list
   inline void append(netpixel& p) { pList.push_back(p); }
   
   // destroy supercluster neighbors links (delink superclusters)
   // preserve links for pixels with the same wavelet resolution
   virtual size_t delink();
   
   // select clusters 
   // name - parameter name
   // thr  - parameter threshold
   virtual wavearray<double> select(char*, double);
   
   //:reconstruct clusters at several TF resolutions (superclusters)
   //!param: statistic:  E - excess power, L - likelihood
   //!param: selection threshold T
   //!       for likelihood clusters, T defines a threshold on clusters
   //!       in a superclusters.
   //!param: true - use only core pixels, false - use core & halo pixels
   //!return size of pixel list of selected superclusters.
   virtual size_t supercluster(char atype, double S, bool core);  // used in 1G pipeline
   virtual size_t supercluster(char atype, double S, double gap, bool core, TH1F* = NULL);

   //:merge clusters if they are close to each other
   //! T - maximum time gap in seconds
   //! F - maximum frequency gap in Hz
   virtual size_t defragment(double T, double F, TH2F* = NULL);

   void PlotClusters();
   
   //: set clusterID field for pixels in pList vector 
   //: create cList structure - list of references to cluster's pixels 
   //!return number of clusters
   virtual size_t cluster();
   
   //: recursively calculate clusterID pixels in pList vector
   //!param: pixel pointer in pList vector
   //!return cluster volume (total number of pixels)
   virtual size_t cluster(netpixel* p);
   
   //:produce time-frequency clusters 
   //:sort pixels on index, form pixel groups, 
   //:set neighbor links for each group and cluster them  
   //!param: time gap between pixels in units of pixels 
   //!param: frequenct gap between pixels in units of pixels 
   //!return pixel number of time clusters
   virtual size_t cluster(int kt,int kf);
   
   //: access function to get cluster parameters passed selection cuts
   //!param: string with parameter name
   //!param: index in the amplitude array, which define detector
   //!param: character identifier for amplitude vector: 
   //        'W'-wavelet, 'S'-snr, 'R'-rank
   //!param: rate index, if 0 ignore rate for calculation of cluster parameters
   //        if negative - extract clusters with rate = abs(rate)
   //!return wavearray object with parameter values for clusters
   wavearray<double> get(char* name, size_t index=0, char atype='R', int type=1, bool=true);
   
   //: extract WSeries for defined cluster ID
   //!param: cluster ID
   //!param: WSeries where to put the waveform 
   //!return: noise rms
   double getwave(int, WSeries<double>&, char='W', size_t=0);

   // extract MRA waveforms for network net, cluster ID and detector index n            
   // works only with WDM. Create WSeries<> objects for each resolution,                             
   // find principle components, fill in WSeries<>, Inverse                                               
   // construct waveform from MRA pixels at different resolutions 
   // atype = 'W' - get whitened detector output (Wavelet data)
   // atype = 'w' - get detector output (Wavelet data)
   // atype = 'S' - get whitened reconstructed response (Signal)
   // atype = 's' - get reconstructed response (Signal)
   // mode: -1/0/1 - return 90/mra/0 phase
   wavearray<double> getMRAwave(network* net, int ID, size_t n, char atype='S', int mode=0); 
   
   // write pixel structure into binary file
   // only metadata and pixels are written
   // first parameter is file name
   // second parameter is the append mode 0/1 - wb/ab - new/append
   size_t write(const char* file, int app=0);

   // write pixel structure with TD vectors attached into a file
   // only some metadata and pixels are written, no cluster metadata is stored
   // fp - file pointer
   // app = 0 - store metadata
   // app = 1 - store pixels with the TD vectors attached by setTDAmp()
   size_t write(FILE* fp, int app=0);
  
   // write pixel structure with TD vectors attached into a root file
   // froot   - root file pointer
   // tdir    - internal root directory where the tree is located
   // tname   - name of tree containing the cluster
   // app = 0 - store light netcluster
   // app = 1 - store pixels with the TD vectors attached by setTDAmp()
   // cycle   - sim -> it is the factor id : prod -> it is the lag number
   // irate   - wavelet layer irate
   //           if irate is negative the value 'irate' is used to build the tree name
   //           this permits to create a tree for each irate
   // cID     - cluster id (cID=0 -> write all clusters)
   size_t write(TFile *froot, TString tdir, TString tname, int app=0, int cycle=0, int irate=0, int cID=0);
 
   // read entire pixel structure from binary file
   // returns number of pixels
   size_t read(const char *);
   
   // read metadata and pixels stored in a file on cluster by cluster basis
   // clusters should be contiguous in the file 
   // maxPix = 0 - read metadata
   // maxPix > 0 - do not store TD vectors for clusters with size > maxPix
   // maxPix < 0 - do not store TD vectors for clusters with size < maxPix
   // clusters with no TD vectors attached are marked rejected.
   // returns # of pixel in the cluster
   size_t read(FILE* file, int maxPix);

   // read metadata and pixels stored in a file on cluster by cluster basis
   // clusters should be contiguous in the file (written by write(FILE*))
   // froot    - root file pointer
   // tdir     - internal root directory where the tree is located
   // tname    - name of tree containing the cluster
   // nmax = 0 - read metadata
   // nmax > 0 - read no more than maxPix pixels from a cluster
   // nmax < 0 - read all heavy pixels from a cluster
   // nmax  -2 - as for (nmax<0) & skip heavy instructions to speedup read
   // cycle    - sim -> it is factor id : prod -> it is the lag number
   // rate     - wavelet layer rate
   //            if rate is negative the value 'rate' is used to build the tree name
   // cID      - cluster ID (=0 : read all clusters)
   std::vector<int> read(TFile* froot, TString tdir, TString tname, int nmax=0, int cycle=0, int rate=0, int cID=0);
   
   inline void setlow(double f) { flow=f; return; }
   inline void sethigh(double f) { fhigh=f; return; }
   inline double getlow() { return flow; }
   inline double gethigh() { return fhigh; }
   
   // set arrays for time-delayed amplitudes in collected coherent pixels 
   // returns number of pixels to process: if count=0 - nothing to process
   // param: input network 
   // param: 'a','A' - delayed amplitudes, 'e','E' - delayed power
   // param: number of pixels (per resolution) to setup
   size_t loadTDamp(network& net, char c, size_t BATCH=10000, size_t LOUD=0);
   // fast version which uses sparse TF maps in detector class
   size_t loadTDampSSE(network& net, char c, size_t BATCH=10000, size_t LOUD=0);

   // param 1 - cluster ID
   // param 2 - chi2 threshold for selection of chirp pixels
   // param 3 - tmerger cut: exclude pixels with time > tmerger+param3 - special use
   // param 4 - threshold for pixel selection - special use
   // returns reconstructed chirp energy
   double mchirp(int ID, double=2.5, double=1.e20, double=0);
  
   // get micropixels
   std::vector<upixel> getupixels(int ID, double& mindt, double& mindf);
   // draw micropixels
   void drawupixels(int ID, std::vector<upixel> vupix, TString ofile);

   // mchirp micropixel version
   double mchirp_upix(int ID, double seed);
 
   // draw chirp cluster
   void chirpDraw(int id);
   
   // print detector parameters
   void print();             // *MENU*
   virtual void Browse(TBrowser *b) {print();}

   // data members
      
   double rate;     // original Time series rate 
   double start;    // interval start GPS time
   double stop;     // interval stop  GPS time 
   double bpp;      // black pixel probability
   double shift;    // time shift
   double flow;     // low frequency boundary
   double fhigh;    // high frequency boundary
   size_t nPIX;     // minimum number of pixels at all resolutions
   int    run;      // run ID
   bool   pair;     // true - 2 resolutions, false - 1 resolution
   double nSUB;	    // subnetwork threshold for a single network pixel 
   
   std::vector<netpixel> pList;     // pixel list
   std::vector<clusterdata> cData;  // cluster metadata
   std::vector<int> sCuts;          /* cluster selection flags (cuts)
                                       1 - rejected
                                       0 - not processed / accepted
                                      -1 - not complete
                                      -2 - ready for processing */
   std::vector<vector_int> cList;   // cluster list defined by vector of pList references
   std::vector<vector_int> cRate;   // cluster type defined by rate
   std::vector<float> cTime;        // supercluster central time
   std::vector<float> cFreq;        // supercluster central frequency
   std::vector<vector_float> sArea; // sky error regions
   std::vector<vector_float> p_Map; // sky pixel map
   std::vector<vector_int> p_Ind;   // sky pixel index
   std::vector<vector_int> nTofF;   // sky time delay configuration for waveform backward correction

   ClassDef(netcluster,4)
 
}; // class netcluster

//: compare function to sort pixel objects on time
int compare_PIX(const void*, const void*);
	
inline netpixel* netcluster::getPixel(size_t n, size_t i) { 
  if(!n) return &pList[i];
  if(cList.size()<n) { 
    printf("getPixel(): size=%d, ID=%d\n",(int)cList.size(),(int)n); 
    return NULL; 
  }
  if(cList[n-1].size()<=i) { 
    printf("getPixel(): size=%d, index=%d\n",(int)cList[n-1].size(),(int)i); 
    return NULL; 
  }
  return &pList[cList[n-1][i]]; 
}

// release vector memory
inline void netcluster::clear() {
   while(!pList.empty()) pList.pop_back();
   pList.clear(); std::vector<netpixel>().swap(pList);
   while(!cList.empty()) cList.pop_back();
   cList.clear(); std::vector<vector_int>().swap(cList);
   while(!cData.empty()) cData.pop_back();
   cData.clear(); std::vector<clusterdata>().swap(cData);
   while(!sCuts.empty()) sCuts.pop_back();
   sCuts.clear(); std::vector<int>().swap(sCuts);
   while(!cRate.empty()) cRate.pop_back();
   cRate.clear(); std::vector<vector_int>().swap(cRate); 
   while(!cTime.empty()) cTime.pop_back();
   cTime.clear(); std::vector<float>().swap(cTime);
   while(!cFreq.empty()) cFreq.pop_back();
   cFreq.clear(); std::vector<float>().swap(cFreq);
   while(!sArea.empty()) sArea.pop_back();
   sArea.clear(); std::vector<vector_float>().swap(sArea);
   while(!p_Map.empty()) p_Map.pop_back();
   p_Map.clear(); std::vector<vector_float>().swap(p_Map);
   while(!p_Ind.empty()) p_Ind.pop_back();
   p_Ind.clear(); std::vector<vector_int>().swap(p_Ind);
}

// clean time delay data 
inline void netcluster::clean(int cID) {
  for(int i=0; i<(int)this->cList.size(); ++i) {   // loop over clusters 
    const vector_int& vint = cList[i];
    if((cID!=0)&&((int)pList[vint[0]].clusterID!=cID)) continue;
    for(int j=0; j<(int)vint.size(); j++) {        // loop over pixels
      if(pList[vint[j]].tdAmp.size()) pList[vint[j]].clean();
    }
  }
}

#endif // NETCLUSTER_HH


















