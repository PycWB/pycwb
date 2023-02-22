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


#ifndef SYMMOBJARRAY_HH
#define SYMMOBJARRAY_HH

#include "stdio.h"
#include "TNamed.h"

// guaranteed to work only with classes that implement the "persistent" interface (Read/Write)

// a[-n], a[-n+1] a[-n+2] ... a[0] a[1] .... a[n]

template <class T>
class SymmObjArray : public TNamed {
public:
   explicit SymmObjArray(unsigned int n=0);
   explicit SymmObjArray(const SymmObjArray&);       //copy constructor
   virtual ~SymmObjArray();
   SymmObjArray& operator=(const SymmObjArray& other);
   void Resize(unsigned int sz);                      // data is lost
   void Write(FILE* f);
   void Read(FILE* f);
   T& operator[](int i){ return zero[i];}
   unsigned int Last() {return Size/2;}
   
protected:
   void Resize0(unsigned int sz);
   int Size;
   T* rec;	//!
   T* zero;	//!

   ClassDef(SymmObjArray,1)
};


#endif 
