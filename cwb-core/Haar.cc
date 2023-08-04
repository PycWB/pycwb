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
//   W. Sweldens - Building your own wavelets at home
//--------------------------------------------------------------------
//$Id: Haar.cc,v 1.3 2004/07/12 07:05:20 jzweizig Exp $

#define BIORTHOGONAL_CC

#include "Haar.hh"

//namespace datacondAPI {
//namespace wat {

ClassImp(Haar<double>)

// constructors

template<class DataType_t> Haar<DataType_t>::
Haar(const Wavelet &w) : 
WaveDWT<DataType_t>(w) 
{ 
   this->m_WaveType = HAAR;
}

template<class DataType_t> Haar<DataType_t>::
Haar(const Haar<DataType_t> &w) : 
WaveDWT<DataType_t>(w) 
{ 
   this->m_WaveType = HAAR;
}

template<class DataType_t> Haar<DataType_t>::
Haar(int tree) :
WaveDWT<DataType_t>(1,1,tree,B_CYCLE) 
{
  this->m_WaveType = HAAR;
}

// destructor
template<class DataType_t>
Haar<DataType_t>::~Haar()
{  }

// clone
template<class DataType_t>
Haar<DataType_t>* Haar<DataType_t>::Clone() const
{
  return new Haar<DataType_t>(*this);
}

// decompose function does one step of forward transformation.
// <level> input parameter is the level to be transformed
// <layer> input parameter is the layer to be transformed.
template<class DataType_t>
void Haar<DataType_t>::forward(int level,int layer)
{
   level++;                           // increment level (next level now)
   int stride = 1<<level;    // stride parameter

   unsigned int i;
   double sq2 = sqrt(2.);
   
   DataType_t *dataA;
   DataType_t *dataD; 

   dataA=this->pWWS+this->getOffset(level,layer<<1);     // pointer to approximation layer
   dataD=this->pWWS+this->getOffset(level,(layer<<1)+1); // pointer to detail layer

// predict
  for(i=0; i<this->nWWS; i+=stride) {
    *(dataD+i) -= *(dataA+i);
  }

// update
  for(i=0; i<this->nWWS; i+=stride) {
    *(dataA+i) += *(dataD+i) * 0.5;
  }

// normalization
  for(i=0; i<this->nWWS; i+=stride) {
    *(dataA+i) *= sq2;
    *(dataD+i) /= sq2;
  }
  
}

// reconstruct function does one step of inverse transformation.
// <level> input parameter is the level to be reconstructed
// <layer> input parameter is the layer to be reconstructed.
template<class DataType_t>
void Haar<DataType_t>::inverse(int level,int layer)
{
   level++;                             // increment level (next level now)
   int stride = 1<<level;      // stride parameter

   unsigned int i;
   double sq2 = sqrt(2.);

   DataType_t *dataA;
   DataType_t *dataD; 

   dataA=this->pWWS+this->getOffset(level,layer<<1);     // pointer to approximation layer
   dataD=this->pWWS+this->getOffset(level,(layer<<1)+1); // pointer to detail layer

// undo normalization
  for(i=0; i<this->nWWS; i+=stride) {
    *(dataA+i) /= sq2;
    *(dataD+i) *= sq2;
  }

// undo update
  for(i=0; i<this->nWWS; i+=stride) {
    *(dataA+i) -= *(dataD+i) * 0.5;
  }

// undo predict
  for(i=0; i<this->nWWS; i+=stride) {
    *(dataD+i) += *(dataA+i);
  }

}

// instantiations

#define CLASS_INSTANTIATION(class_) template class Haar< class_ >;

CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)
//CLASS_INSTANTIATION(std::complex<float>)
//CLASS_INSTANTIATION(std::complex<double>)

#undef CLASS_INSTANTIATION

//template Haar<float>::
//Haar(const Haar<float> &);
//template Haar<double>::
//Haar(const Haar<double> &);

//}  // end namespace wat
//}  // end namespace datacondAPI






