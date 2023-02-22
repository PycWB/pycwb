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


// $Id: Wavelet.cc,v 1.4 2002/01/01 23:18:08 sigg Exp $

#define WAVEDWT_CC

#include "Wavelet.hh"

ClassImp(Wavelet)

//namespace datacondAPI {
//namespace wat {

// constructors

Wavelet::Wavelet(int mH, int mL, int tree, enum BORDER border) : 
m_WaveType(HAAR), m_Heterodine(false), m_Parity(true)
{ 
   m_H = mH;
   m_L = mL;
   m_Border = border;
   m_Level = 0;
   m_Layer = 0;
   m_TreeType = tree;
}

Wavelet::Wavelet(const Wavelet &w) 
{ 
   m_H = w.m_H;
   m_L = w.m_L;
   m_Border = w.m_Border;
   m_Level = w.m_Level;
   m_Layer = w.m_Layer;
   m_TreeType = w.m_TreeType;
   m_WaveType = w.m_WaveType;
   m_Heterodine = w.m_Heterodine;
   m_Parity = w.m_Parity;
}

// destructor
Wavelet::~Wavelet()
{ }

Wavelet* Wavelet::Clone() const
{
   return new Wavelet(*this);
}

//*******************************
//*  wavedata acess functions   *
//*******************************

int Wavelet::getOffset(int level, int layer)
{       
   if(m_Heterodine) return layer;
   int n=0;

   for(int i=0; i<level; i++)
      if((layer>>i)&1) n += 1<<(level-1-i);

   return n;
}

int Wavelet::convertL2F(int level, int layer)
{       
   if(m_Heterodine) return layer;
   int n = layer;
   int j;
   for(int i=1; i<level; i++) {
      j = (1<<i) & (n);
      if(j) n = ((1<<i)-1) ^ (n);
   }
   
   return n;
}

int Wavelet::convertF2L(int level, int index)
{       
   if(m_Heterodine) return index;
   int n = index;
   int j;
   for(int i=level-1; i>=1; i--) {
      j = ((1<<i) & (n));
      if(j) n = ((1<<i)-1) ^ (n);
   }
   return n;
}

int Wavelet::convertO2F(int level, int index)
{       
   return convertL2F(level,getOffset(level, index));
}

int Wavelet::convertF2O(int level, int index)
{       
   return getOffset(level,convertF2L(level,index));
}

void Wavelet::print()
{       
   cout << endl << endl;
   cout.precision(14);
   cout << "TreeType\t= "    << this->m_TreeType  << endl;
   cout << "Level\t\t= "     << this->m_Level  << endl;
   cout << "Layer\t\t= "     << this->m_Layer << endl;
   cout << "H\t\t= "         << this->m_H  << endl;
   cout << "L\t\t= "         << this->m_L << endl;
   cout << "Heterodine\t= "  << this->m_Heterodine << endl;
   cout << "Parity\t\t= "    << this->m_Parity << endl;
   cout << endl;

   return;
}

//} // namespace wat 
//} // namespace datacondAPI 

