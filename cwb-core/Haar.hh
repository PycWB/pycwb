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
//--------------------------------------------------------------------
// Implementation of 
// the Haar wavelet transform using lifting scheme 
// References:
//   A.Cohen, I.Daubechies, J.Feauveau Bases of compactly supported wavelets
//   Comm. Pure. Appl. Math. 45, 485-560, 1992
//--------------------------------------------------------------------

//$Id: Haar.hh,v 0.2 2001/08/06 19:37:00 klimenko Exp $
#ifndef HAAR_HH
#define HAAR_HH

#include "WaveDWT.hh"

//namespace datacondAPI {
//namespace wat {

template<class DataType_t>
class Haar : public WaveDWT<DataType_t>
{
   public:

      //: construct from wavelet parameters
      Haar(int tree=0);
      
      //: construct from the base class
      Haar(const Wavelet &);

      //: copy constructors
      Haar(const Haar<DataType_t> &);

      //: destructor

      virtual ~Haar();

      //: Duplicate on heap
      virtual Haar* Clone() const;

      //: decomposition method
      void forward(int level, int layer);
      //: reconstruction method      
      void inverse(int level, int layer);

      ClassDef(Haar,1) 

}; // class Haar

//}; // namespace wat
//}; // namespace datacondAPI

#endif // HAAR_HH












