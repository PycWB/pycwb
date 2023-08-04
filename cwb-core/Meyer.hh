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
// Meyer wavelets using Fast Wavelet Transform
// References:
//--------------------------------------------------------------------

//$Id: Meyer.hh,v 0.2 2001/08/06 19:37:00 klimenko Exp $
#ifndef MEYER_HH
#define MEYER_HH

#include "WaveDWT.hh"

//namespace datacondAPI {
//namespace wat {

template<class DataType_t>
class Meyer : public WaveDWT<DataType_t>
{
   private:

      //: forward LP filter coefficients.
      double *pLForward;  //!
      //: inverse LP filter coefficients.
      double *pLInverse;  //!
      //: forward LP filter coefficients.
      double *pHForward;  //!
      //: inverse LP filter coefficients.
      double *pHInverse;  //!

      void setFilter();

   public:

      //: default construct 
      Meyer();

      //: construct from base class
      Meyer(const Wavelet &);

      //: copy constructors
      Meyer(const Meyer<DataType_t> &);

      //: construct from wavelet parameters
      Meyer(int m, int tree=0, enum BORDER border=B_CYCLE);

      //: destructor
      virtual ~Meyer();

      //: Duplicate on heap
      virtual Meyer* Clone() const;

      //: calculate wavelet filter
      //!param: taper function order n
      //!param: beta(n,n) - value of Euler's beta function
      //!param: integration step
      double filter(int, double, double=1.e-6);

      // get maximum possible level of decomposition
      int getMaxLevel();

      //: decomposition method
      virtual void forward(int level, int layer);
      //: reconstruction method
      virtual void inverse(int level, int layer);

      ClassDef(Meyer,1);

}; // class Meyer

//}; // namespace wat
//}; // namespace datacondAPI

#endif // MEYER_HH

