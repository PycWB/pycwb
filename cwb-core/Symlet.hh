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
// Symlet wavelets using Fast Wavelet Transform 
// References:
//   I.Daubechies, Ten lectures on wavelets
//   ISBN 0-89871-274-2, 1992
//--------------------------------------------------------------------

//$Id: Symlet.hh,v 0.2 2001/08/06 19:37:00 klimenko Exp $
#ifndef SYMLET_HH
#define SYMLET_HH

#include "WaveDWT.hh"

//namespace datacondAPI {
//namespace wat {

template<class DataType_t>
class Symlet : public WaveDWT<DataType_t>
{
   private:

      //: forward LP filter coefficients.
      double *pLForward;	//!
      //: inverse LP filter coefficients.
      double *pLInverse;	//!
      //: forward LP filter coefficients.
      double *pHForward;	//!
      //: inverse LP filter coefficients.
      double *pHInverse;	//!

      void setFilter();

   public:
      
      //: construct from base class
      Symlet(const Wavelet &);

      //: copy constructors
      Symlet(const Symlet<DataType_t> &);

      //: construct from wavelet parameters
      Symlet(int order=4, int tree=0, enum BORDER border=B_CYCLE);

      //: destructor
      virtual ~Symlet();

      //: Duplicate on heap
      virtual Symlet* Clone() const;

      //: decomposition method
      virtual void forward(int level, int layer);
      //: reconstruction method      
      virtual void inverse(int level, int layer);

      ClassDef(Symlet,1)

}; // class Symlet

//}; // namespace wat
//}; // namespace datacondAPI

#endif // SYMLET_HH












