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
// Bi-orthogonal wavelet transforms using lifting scheme 
// References:
//   A.Cohen, I.Daubechies, J.Feauveau Bases of compactly supported wavelets
//   Comm. Pure. Appl. Math. 45, 485-560, 1992
//--------------------------------------------------------------------

//$Id: Biorthogonal.hh,v 0.2 2001/08/06 19:37:00 klimenko Exp $
#ifndef BIORTHOGONAL_HH
#define BIORTHOGONAL_HH

#include "WaveDWT.hh"

//namespace datacondAPI {
//namespace wat {

template<class DataType_t>
class Biorthogonal : public WaveDWT<DataType_t>
{
//   private:
   public:

      //: pointer to array of forward predict filter coefficients.
      double *PForward;	//!
      //: pointer to array of inverse predict filter coefficients.
      double *PInverse;	//!
      //: pointer to array of forward update filter coefficients.
      double *UForward;	//!
      //: pointer to array of inverse update filter coefficients.
      double *UInverse;	//!

      void setFilter();

      
      //: construct from base class
      Biorthogonal(const Wavelet &);

      //: copy constructors
      Biorthogonal(const Biorthogonal<DataType_t> &);

      //: construct from wavelet parameters
      Biorthogonal(int order=4, int tree=0, enum BORDER border=B_POLYNOM);

      //: destructor
      virtual ~Biorthogonal();

      //: Duplicate on heap
      virtual Biorthogonal* Clone() const;

      //: decomposition method
      void forward(int level, int layer);
      //: reconstruction method      
      void inverse(int level, int layer);

      ClassDef(Biorthogonal,1)

}; // class Biorthogonal

//}; // namespace wat
//}; // namespace datacondAPI

#endif // BIORTHOGONAL_HH












