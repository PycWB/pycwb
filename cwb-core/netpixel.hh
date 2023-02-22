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
// universal pixel data container for network cluster analysis
// used with DMT and ROOT
// 

#ifndef NETPIXEL_HH
#define NETPIXEL_HH

#include <iostream>
#include "wavearray.hh"
#include "WaveDWT.hh"
#include "wseries.hh"
#include "TNamed.h"

struct pixdata {
  double noiserms;                // average noise rms
  double wave;                    // vector of 00 pixel's wavelet amplitudes
  double w_90;                    // vector of 90 pixel's wavelet amplitudes
  double asnr;                    // vector of 00 pixel's whitened amplitudes
  double a_90;                    // vector of 90 pixel's whitened amplitudes
  float  rank;                    // vector of pixel's rank amplitudes
  int    index;                   // index in wavearray
};

class netpixel : public TNamed {
  public:

  netpixel();
  netpixel(size_t); 

  virtual ~netpixel(){ this->clear(); }
  bool operator == (const netpixel &) const {return true;}
  bool operator <  (const netpixel &) const {return true;}

  //  operator =
  netpixel& operator= (const netpixel&);

  // set pixel data
  inline bool setdata(double a, char type='R', size_t n=0){
    if(n<this->size()) {
           if(type == 'N' || type == 'n') this->data[n].noiserms = a;
      else if(type == 'I' || type == 'i') this->data[n].index = int(a+0.1);
      else if(type == 'W' || type == 'w') this->data[n].wave = a;
      else if(type == 'U' || type == 'u') this->data[n].w_90 = a;
      else if(type == 'S' || type == 's') this->data[n].asnr = a;
      else if(type == 'P' || type == 'p') this->data[n].a_90 = a;
      else if(type == 'R' || type == 'r') this->data[n].rank = a;
      else                                this->data[n].asnr = a;
	   return true;
    }
    else { return false; }
  }

  // get pixel data
  inline double getdata(char type='R', size_t n=0){
    if(n<this->size()) {
           if(type == 'N' || type == 'n') return double(this->data[n].noiserms);
      else if(type == 'I' || type == 'i') return double(this->data[n].index);
      else if(type == 'W' || type == 'w') return double(this->data[n].wave);
      else if(type == 'U' || type == 'u') return double(this->data[n].w_90);
      else if(type == 'S' || type == 's') return double(this->data[n].asnr);
      else if(type == 'P' || type == 'p') return double(this->data[n].a_90);
      else if(type == 'R' || type == 'r') return double(this->data[n].rank);
      else                                return double(this->data[n].asnr);
    }
    return 0;
  }

  // get size of pixel arrays
  inline size_t size(){ return this->data.size(); }
  // get capacity of pixel object
  inline size_t capacity(){ return data.capacity(); }
  // clear pixel
  inline void clear(){ 
     data.clear(); std::vector<pixdata>().swap(data);
     tdAmp.clear(); std::vector<wavearray<float> >().swap(tdAmp); 
     neighbors.clear(); std::vector<int>().swap(neighbors);
  }
  // clear pixel
  inline void clean(){ 
     tdAmp.clear(); std::vector<wavearray<float> >().swap(tdAmp); 
  }
  // add link to neighbors
  inline void append(int n){ neighbors.push_back(n); }
  // write pixel to file
  bool write(const FILE *);
  // read pixel from file
  bool read(const FILE *);
  
  size_t clusterID;                // cluster ID
  size_t time;                     // time index for master detector
  size_t frequency;                // frequency index (layer)
  size_t layers;                   // number of frequency layers
  float  rate;                     // wavelet layer rate
  float  likelihood;               // likelihood
  float  null;                     // null
  float  theta;                    // source angle theta index
  float  phi;                      // source angle phi index
  float  ellipticity;              // waveform ellipticity
  float  polarisation;             // waveform polarisation
  bool   core;                     // pixel type: true - core , false - halo

  std::vector<pixdata> data;                     
  std::vector<wavearray<float> > tdAmp;                     
  std::vector<int> neighbors;      // vector of links to neighbors

  ClassDef(netpixel,2)
};

#endif // NETPIXEL_HH


















