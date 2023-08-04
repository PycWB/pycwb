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
/* file readframe.hh                                    */
/********************************************************/

//#ifndef _READFRAME_H
//  #define _READFRAME_H

#ifndef _STRING_H
  #include <string>
#endif
#include "waverdc.hh"
#include "wseries.hh"
#include "lossy.hh"
#include "FrameL.h"

WSeries<float>* 
ReadFrame(double t, char *cname, char *fname, bool seek=true);

//#endif // _READFRAME_H
