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


/********************************************************/
/* Wavelet Analysis Tool                                */
/* file readfrfile.hh                                    */
/********************************************************/

//#ifndef _READFRFILE_H
//  #define _READFRFILE_H

#ifndef _STRING_H
  #include <string.h>
#endif
#include "waverdc.hh"
#include "lossy.hh"
#include "FrameL.h"

bool 
ReadFrFile(wavearray<double> &out, 
	   double tlen, 
	   double tskip, 
	   char *cname, 
	   char *fname,
	   bool seek=true, 
	   char *channel_type="adc");

wavearray<float>* 
ReadFrFile(double tlen, 
	   double tskip, 
	   char *cname, 
	   char *fname,
	   bool seek=true, 
	   char *channel_type="adc");

//#endif // _READFRFILE_H
