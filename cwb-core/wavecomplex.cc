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
 * complex class 
 * File name: 	wavecomplex.cc
 *-------------------------------------------------------
*/


#include <time.h>
#include <iostream>
#include "wavecomplex.hh"

ClassImp(wavecomplex)	 // used by THtml doc

//: constructors
wavecomplex::wavecomplex() 
{ re = im = 0.; }
wavecomplex::wavecomplex(double a, double b)
{ re = a; im = b; }

//: copy constructor 
wavecomplex::wavecomplex(const wavecomplex& a)
{ *this = a; }

//: destructor
wavecomplex::~wavecomplex() {} 

//: operators
wavecomplex& wavecomplex::operator=(const wavecomplex& a)
{ re=a.real(); im=a.imag(); return *this; }

wavecomplex& wavecomplex::operator+=(const wavecomplex &a)
{ re+=a.real(); im+=a.imag(); return *this; }
wavecomplex& wavecomplex::operator-=(const wavecomplex &a)
{ re-=a.real(); im-=a.imag(); return *this; }
wavecomplex& wavecomplex::operator*=(const wavecomplex &a)
{
  double x = re*a.real()-im*a.imag();
  im = re*a.imag()+im*a.real(); 
  re = x;
  return *this;
}
wavecomplex& wavecomplex::operator/=(const wavecomplex &a)
{
  double x = a.abs();
  double y = (re*a.real()+im*a.imag())/x;
  im = (im*a.real()-re*a.imag())/x; 
  re = y;
  return *this;
}

wavecomplex wavecomplex::operator+(const wavecomplex &a)
{ wavecomplex z = *this; z+=a; return z; }
wavecomplex wavecomplex::operator-(const wavecomplex &a)
{ wavecomplex z = *this; z-=a; return z; }
wavecomplex wavecomplex::operator*(const wavecomplex &a)
{ wavecomplex z = *this; z*=a; return z; }
wavecomplex wavecomplex::operator/(const wavecomplex &a)
{ wavecomplex z = *this; z/=a; return z; }


wavecomplex& wavecomplex::operator=(const double c)
{ re=c; im=0.; return *this; }

wavecomplex& wavecomplex::operator+=(const double c)
{ re+=c; return *this; }
wavecomplex& wavecomplex::operator-=(const double c)
{ re-=c; return *this;}
wavecomplex& wavecomplex::operator*=(const double c)
{ re*=c; im*=c; return *this;}
wavecomplex& wavecomplex::operator/=(const double c)
{ re/=c; im/=c; return *this;}

wavecomplex wavecomplex::operator+(const double c)
{ wavecomplex z = *this; z+=c; return z;}
wavecomplex wavecomplex::operator-(const double c)
{ wavecomplex z = *this; z-=c; return z;}
wavecomplex wavecomplex::operator*(const double c)
{ wavecomplex z = *this; z*=c; return z;}
wavecomplex wavecomplex::operator/(const double c)
{ wavecomplex z = *this; z/=c; return z;}



