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


/*-------------------------------------------------------
 * Package: 	Wavelet Analysis Tool
 * File name: 	waverdc.hh
 *-------------------------------------------------------
*/

#ifndef WAVERDC_HH
#define WAVERDC_HH

#include <iostream>
#include "wavearray.hh"
using namespace std;

typedef wavearray<short> waveShort; 
typedef wavearray<float> waveFloat; 
typedef wavearray<double> waveDouble; 

class WaveRDC : public wavearray<unsigned int>
{
public:

  int nSample;		// number of samples in comressed data array
  int nLayer;		// number of layers in comressed data array
  int optz;		// current layer compression options

  WaveRDC();

  virtual ~WaveRDC();

  WaveRDC& operator =(const WaveRDC &);         // assign
  WaveRDC& operator+=(const WaveRDC &);         // concatenate
//  WaveRDC& operator<<(const WaveRDC &);         // copy

  WaveRDC& operator-=(const WaveRDC &x){return *this;};      // no operation!
  WaveRDC& operator*=(const WaveRDC &x){return *this;};      // no operation!
  WaveRDC& operator =(const unsigned int x){return *this;};  // no operation!
  WaveRDC& operator+=(const unsigned int x){return *this;};  // no operation!
  WaveRDC& operator-=(const unsigned int x){return *this;};  // no operation!
  WaveRDC& operator*=(const unsigned int x){return *this;};  // no operation!

  virtual int DumpRDC(const char*, int = 0);

  int Compress(const waveShort  &);
//  int Compress(const waveShort  &, double);
//  int Compress(const waveFloat  &, double);
  int Compress(const waveDouble &, double);

  int unCompress(waveFloat &, int level = 1);
  int unCompress(waveDouble &, int level = 1);
  int unCompress(wavearray<int> &, int level = 1);

  void Dir(int v = 1);

  float getScale(const waveDouble &, double);
  void  getShort(const waveDouble &, waveShort &);
  void  getSign(const waveDouble &, waveShort &);

//private:

    int freebits;	// free bits in the last word of current block
    int kLong;          // encoding bit length for large integers
    int kShort;         // encoding bit length of short word
    int kBSW;           // length of the Block Service Word
  short Bias;           // constant bias subtracted from the data
  short Zero;           // number that encodes 0;      
  float Scale;          // coefficient to scale data
  float rmsLimit;       // limit on the data rms

   int   Push(short *, int, unsigned int *, int &, int, int);
  void   Push(unsigned int &, unsigned int *, int &, int);

   int   Pop(int *, int, int &, int, int);
  void   Pop(unsigned int &, int &, int);


  inline int  getOPTZ() {return optz;};
  inline int   wabs(int i)    {return i>0 ? i : -i;} 
  inline short wabs(short i)  {return i>0 ? i : -i;} 
  inline int   wint(double a) {return int(2*a)-int(a);} 

  inline size_t getLSW(size_t opt){    // get length of Layer Service Word (size_t)
     size_t n = 2;
     if(opt & 0x2 ) n++;
     if((opt & 0x8) && (opt & 0x3)) n++;
     if(opt & 0x10) n++;
     return n;
  }

  ClassDef(WaveRDC,1)

};


#endif








