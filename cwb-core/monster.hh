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


#ifndef MONSTER_HH
#define MONSTER_HH

#include "WDM.hh"
#include "netcluster.hh"
#include "wavearray.hh"
#include <algorithm>
#include "stdint.h"

using namespace std;

struct xtalk{int index;  float CC[4];};   // AA, AQ, QA, QQ

struct xtalkArray{struct xtalk* data; int size;};

typedef vector<xtalk> vector_XT;

//template <class T> 
class monster{
public:
   
   // dummy constructor
   monster();
   
   // constructor; computes catalog
   //! param: array of pointers to WDM transforms
   //! param: number of WDM transforms in the array
   monster(WDM<double>** wdm, int nRes);
   
   // constructor; reads catalog from file
   //! param: catalog file name 
   monster(char* filename);
   
   // copy constructor
   //! param: object to be copied
   monster(const monster& x);
   
   // destructor
   virtual ~monster();
   
   // release memory used by the catalog
   void deallocate();
   
   // write the catalog to file
   //! param: file name
   void write(char* filename);
   
   // read catalog from file
   //! param: file name 
   void read(char* filename);
   
   
   // returns the overlap values for two pixels
   //! param: numbers of layers identifying the resolution of the first pixel 
   //! param: TF map index of the first pixel 
   //! param: numbers of layers identifying the resolution of the second pixel
   //! param: TF map index of the second pixel
   //! return: xtalk structure, use the CC values (coupling coefficients)
   xtalk getXTalk(int nLay1, size_t indx1, int nLay2, size_t indx2);              
   
   // same as above but now the quadratures are specified, too (quad1, quad2)
   // returns the corresponding overlap (coupling coefficient)
   float getXTalk(int nLay1, int quad1, size_t indx1, int nLay2, int quad2, size_t indx2);  
   
   // FILL cluster overlap amplitudes
   //! param: pointer to netcluster structure
   //! param: which cluster to process
   //! param: check TD vectors or not 
   //! return list of pixel IDs
   std::vector<int> getXTalk(netcluster* pwc, int id, bool check=true);

   // get xtalk size - number of neighbour pixel including the main one
   inline int size() {
      if(clusterCC.size()!=sizeCC.size()) exit(1);
      return clusterCC.size();
   }

   // get xtalk vector size for pixel m
   inline int size(int m) {
      return sizeCC[m];
   }

   // get pointer to xtalk vector for pixel m
   inline float* getXTalk(int m) {
      return clusterCC[m];
   }

   // returns 4 x-talk coefficients between pixels p1 and p2
   inline xtalk getXTalk(netpixel* p1, netpixel* p2) {  
      return getXTalk(p1->layers, p1->time, p2->layers, p2->time);
   }

       
   //void PrintSums();
     
//protected:
   xtalkArray (***catalog)[2];          // stores overlap values [r1][r2][r1_freq][parity] ; r2<=r1   
   int tag;                             // catalog tag number
   int nRes;                            // number of resolutions
   int BetaOrder;                       // beta function order for Meyer 
   int precision;                       // wavelet precision
   int KWDM;                            // WDM K - parameter K/M
   int* layers;                         //! M for each resolution
   std::vector<float*>  clusterCC;      // cluster coupling coefficients
   std::vector<int>     sizeCC;         // size of the vector for cluster coupling coefficients
   
   // used by THtml doc
   ClassDef(monster,3)			
};


#endif
