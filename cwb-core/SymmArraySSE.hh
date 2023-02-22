/*
# Copyright (C) 2019 Sergey Klimenko, Valentin Necula
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


#ifndef SYMMARRAYSSE_HH
#define SYMMARRAYSSE_HH

#include "stdio.h"
#include "TNamed.h"

// meant to be used with int32, int64, float, double

// a[-n], a[-n+1] a[-n+2] ... a[0] a[1] .... a[n]

template <class Record>
class SymmArraySSE : public TNamed{
public:
   SymmArraySSE(unsigned int n=0);
   SymmArraySSE(const SymmArraySSE&);       //copy constructor
   virtual ~SymmArraySSE();
   SymmArraySSE& operator=(const SymmArraySSE& other);
   void Init(Record x);
   void Resize(int nn); // new n
   void Write(FILE* f);
   void Read(FILE* f);
   Record& operator[](int i){ return zero[i];}
   Record* SSEPointer(){   return rec;}  
   int SSESize(){ return SizeSSE;}    
   int Last() {return last;}
   void ZeroExtraElements();
   
protected:
   void allocateSSE(); // aligned allocation; uses SizeSSE, last; sets rec, zero
   int last, SizeSSE;   // SizeSSE in bytes (multiple of 8)
   Record* rec;      //!
   Record* zero;     //! always in the middle of the allocated space
   int recSize;
   
   ClassDef(SymmArraySSE,1)
};



#endif 
