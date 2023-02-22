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


/**********************************************************
 * Package: 	Wavelet Analysis Tool
 * this class defines complex numbers
 * File name: 	wavecomplex.h
 **********************************************************/

#ifndef WAVECOMPLEX_HH
#define WAVECOMPLEX_HH

#include <stdio.h>
#include <math.h>
#include "Rtypes.h"	// used by THtml doc

class wavecomplex
{

  public:

  wavecomplex(double,double);                      // Constructor

  wavecomplex();                                   // Default constructor

  wavecomplex(const wavecomplex&);                       // copy Constructor

  virtual ~wavecomplex();                          // Destructor

// operators

          wavecomplex& operator= (const wavecomplex &);
  virtual wavecomplex& operator+=(const wavecomplex &);
  virtual wavecomplex& operator-=(const wavecomplex &);
  virtual wavecomplex& operator*=(const wavecomplex &);
  virtual wavecomplex& operator/=(const wavecomplex &);
  virtual wavecomplex  operator+ (const wavecomplex &);
  virtual wavecomplex  operator- (const wavecomplex &);
  virtual wavecomplex  operator* (const wavecomplex &);
  virtual wavecomplex  operator/ (const wavecomplex &);

          wavecomplex& operator= (const double);
  virtual wavecomplex& operator+=(const double);
  virtual wavecomplex& operator-=(const double);
  virtual wavecomplex& operator*=(const double);
  virtual wavecomplex& operator/=(const double);
  virtual wavecomplex  operator+ (const double);
  virtual wavecomplex  operator- (const double);
  virtual wavecomplex  operator* (const double);
  virtual wavecomplex  operator/ (const double);

// member functions

  inline double real() const { return re; }
  inline double imag() const { return im; }
  inline double  arg() const { return atan2(im,re); }
  inline double  abs() const { return re*re+im*im; }
  inline double  mod() const { return sqrt(re*re+im*im); }
  inline void    set(double x, double y) { re=x; im=y; return; }
  inline wavecomplex conj() { wavecomplex z(re,-im); return z; }

//   private:

  double re;		        // real
  double im;		        // imagenary

  // used by THtml doc
  ClassDef(wavecomplex,1)	
};

#endif // WAVECOMPLEX_HH
