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
 * Package:     Wavelet Analysis Tool
 * File name:   lossy.h
 *-------------------------------------------------------
*/ 
#include "wavearray.hh"
#include "waverdc.hh"

int Compress(wavearray<double> &, int* &, int, int,
                    double, double, int, int);

int Compress(float [], int, int* &, int, int,
                    double, double, int, int);

int Compress(short [], int, int* &, int, int,
                    double, double, int, int);

int unCompress(int*, wavearray<float> &);
int unCompress(int*, float* &);
int unCompress(int*, short* &);

/*
int Compress(wavearray<double> &, int* &, int=0, int=0,
                    double=1., double=1., int=10, int=10);

int Compress(float [], int, int* &, int=0, int=0,
                    double=1., double=1., int=10, int=10);

int Compress(short [], int, int* &, int=0, int=0,
                    double=1., double=1., int=10, int=10);

int unCompress(int*, wavearray<float> &);
int unCompress(int*, float* &);
int unCompress(int*, short* &);
*/





