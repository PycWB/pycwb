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


//---------------------------------------------------
// WAT pixel class for network analysis
// S. Klimenko, University of Florida
//---------------------------------------------------

#define NETPIXEL_CC
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "netpixel.hh"

using namespace std;

ClassImp(netpixel)

// constructors

netpixel::netpixel() {
  theta = phi = -1.;
  clusterID = time = frequency = 0;
  rate = 1.; likelihood = 0.; core = false;
  ellipticity = 0.; polarisation = 0.;
  data.clear();
  neighbors.clear();
}

netpixel::netpixel(size_t n) {
  theta = phi = -1.;
  clusterID = time = frequency = 0;
  rate = 1.; likelihood = 0.; core = false;
  ellipticity = 0.; polarisation = 0.;
  pixdata pix;

  data.clear();
  neighbors.clear();
  if(!n) return;

  pix.noiserms    = 0;   // average noise rms				
  pix.wave        = 0;	 // vector of pixel's wavelet amplitudes	
  pix.w_90        = 0;	 // vector of pixel's phase shifted wavelet amplitudes	
  pix.asnr        = 0;	 // vector of pixel's whitened amplitudes	
  pix.a_90        = 0;	 // vector of pixel's phase shifted amplitudes	
  pix.rank        = 0;	 // vector of pixel's rank amplitudes		
  pix.index       = 0;	 // index in the wavearray for other detectors

  data.push_back(pix);
  data.reserve(n);
  for(size_t i=1; i<n; i++) data.push_back(pix);
}

//  netpixel::operator =
netpixel& netpixel::operator=(const netpixel& value)
{
  this->clusterID   = value.clusterID;   // cluster ID
  this->time        = value.time;        // time index for zero detector (accounts for time shift)
  this->frequency   = value.frequency;   // frequency index (layer)
  this->layers      = value.layers;      // number of frequency layers
  this->rate        = value.rate;        // wavelet layer rate
  this->likelihood  = value.likelihood;  // likelihood
  this->ellipticity = value.ellipticity; // likelihood
  this->polarisation= value.polarisation;// likelihood
  this->theta       = value.theta;       // source angle theta index
  this->phi         = value.phi;         // source angle phi index
  this->core        = value.core;        // pixel type: true - core , false - halo
  this->data        = value.data;        // copy data
  this->neighbors   = value.neighbors;   // copy neighbors
  this->tdAmp       = value.tdAmp;       // copy TD vectors

  return *this;
}

// write pixel in file
bool netpixel::write(const FILE *fp)
{
   size_t i,j;
  size_t n = this->neighbors.size();
  size_t m = this->data.size();
  size_t k = this->tdAmp.size();
  size_t I = n ? n : 1;

  k = (k==m) ? this->tdAmp[0].size() : 0;

  double db[14];
  wavearray<double> wb(I+m*(7+k));

  // write metadata

  db[0] = (double)m;               // number of detectors
  db[1] = (double)I;               // number of  neighbors
  db[2] = (double)this->clusterID; // cluster ID
  db[3] = this->time;              // time index for zero detector (accounts for time shift)
  db[4] = this->frequency;         // frequency index (layer)
  db[5] = this->core ? 1. : 0.;    // pixel type: true - core , false - halo
  db[6] = this->rate;              // wavelet layer rate
  db[7] = this->likelihood;        // likelihood
  db[8] = this->theta;             // source angle theta index
  db[9] = this->phi;               // source angle phi index
  db[10]= this->ellipticity;       // waveform ellipticity
  db[11]= this->polarisation;      // waveform polarisation
  db[12]= this->layers;            // number of layers
  db[13] = (double)k;              // number of TD amplitudes per detector

  if(fwrite(db, 14*sizeof(double), 1, (FILE*)fp)!=1) return false;

  // write neighbors
  for(i=0; i<I; i++) wb.data[i] = n ? neighbors[i] : 0.;

  // write amplitudes
  for(i=0; i<m; i++) {
    wb.data[I++] = this->data[i].noiserms;       // average noise rms
    wb.data[I++] = this->data[i].wave;           // vector of pixel's wavelet amplitudes
    wb.data[I++] = this->data[i].w_90;           // vector of pixel's wavelet amplitudes
    wb.data[I++] = this->data[i].asnr;           // vector of pixel's whitened amplitudes
    wb.data[I++] = this->data[i].a_90;           // vector of pixel's phase shifted amplitudes
    wb.data[I++] = (double)this->data[i].rank;   // vector of pixel's rank amplitudes
    wb.data[I++] = (double)this->data[i].index;  // index in wavearray
  }

  // write TD amplitudes
  for(i=0; i<m; i++)
     for(j=0; j<k; j++)
        wb.data[I++] = (double)this->tdAmp[i].data[j];

  if(fwrite(wb.data, I*sizeof(double), 1, (FILE*)fp)!=1) return false;
  return true;
}

// read pixel from file
bool netpixel::read(const FILE *fp)
{
  size_t i;
  int    j;
  pixdata pix;
  double db[14];

  this->clear();

  // read metadata
  if(fread(db, 14*sizeof(double), 1, (FILE*)fp)!=1) return false;

  this->clusterID = size_t(db[2]+0.1);    // cluster ID
  this->time = size_t(db[3]+0.1);         // time index for zero detector 
  this->frequency = size_t(db[4]+0.1);    // frequency index (layer)
  this->core = db[5]>0. ? true : false;   // pixel type: true - core , false - halo
  this->rate = (float)db[6];              // wavelet layer rate
  this->likelihood = (float)db[7];        // likelihood
  this->theta = (float)db[8];             // source angle theta index
  this->phi = (float)db[9];               // source angle phi index
  this->ellipticity = (float)db[10];      // waveform ellipticity
  this->polarisation = (float)db[11];     // waveform polarisation
  this->layers = size_t(db[12]+0.1);      // number of layers

  size_t m = size_t(db[0]+0.1);           // number of detectors
  size_t I = size_t(db[1]+0.1);           // neighbor size
  size_t k = size_t(db[13]+0.1);          // number TD amplitudes per detector
  size_t n = I + m*(7+k);                 // buffer size
  wavearray<double> wb(n);
  wavearray<float> tda(k);

  if(fread(wb.data, n*sizeof(double), 1, (FILE*)fp)!=1) return false;

  // get neighbors
  for(i=0; i<I; i++) {
    j = wb.data[i]<0. ? int(wb.data[i]-0.1) : int(wb.data[i]+0.1);
    if(j) this->neighbors.push_back(j);
  }

  // get data
  for(i=0; i<m; i++) {
    pix.noiserms = wb.data[I++];         // average noise rms
    pix.wave = wb.data[I++];             // vector of pixel's wavelet amplitudes
    pix.w_90 = wb.data[I++];             // vector of pixel's wavelet amplitudes
    pix.asnr = wb.data[I++];             // vector of pixel's whitened amplitudes
    pix.a_90 = wb.data[I++];             // vector of pixel's phase shifted amplitudes
    pix.rank = float(wb.data[I++]);      // vector of pixel's rank amplitudes
    pix.index = int(wb.data[I++]+0.1);   // index in wavearray
    this->data.push_back(pix);
  }

  // get TD amplitudes
  for(i=0; i<m; i++) {
     for(j=0; j<(int)k; j++) {
        tda.data[j] = float(wb.data[I++]);
     }
     if(k) this->tdAmp.push_back(tda);
  }

  return true;
}














