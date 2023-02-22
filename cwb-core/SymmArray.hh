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


#ifndef SYMMARRAY_HH
#define SYMMARRAY_HH

#include "stdio.h"
#include "TNamed.h"

// guaranteed to work only with (struct of) atomic types

// a[-n], a[-n+1] a[-n+2] ... a[0] a[1] .... a[n]

template <class Record>
class SymmArray : public TNamed {
public:
   SymmArray(unsigned int n=0);
   SymmArray(const SymmArray&);       //copy constructor
   virtual ~SymmArray();
   SymmArray& operator=(const SymmArray& other);
   void Init(Record x);
   void Resize(int sz);
   void Write(FILE* f);
   void Read(FILE* f);
   Record& operator[](int i){ return zero[i];}
   int Last() {return Size/2;}
   
   
protected:
   void Resize0(int sz);
   int Size;
   Record* rec;		//!
   Record* zero;	//!
   int recSize;

   ClassDef(SymmArray,1)
};

#endif 
