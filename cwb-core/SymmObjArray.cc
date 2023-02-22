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


#include "SymmObjArray.hh"

#include "SymmArray.hh"
#include "SymmArraySSE.hh"

ClassImp(SymmObjArray<SymmArraySSE<float> >)

template <class T>
SymmObjArray<T>::SymmObjArray(unsigned int n)
{  Size = 2*n+1; //mind the plus
   rec = new T[Size];
   zero = rec+n;
   if(rec == 0 )printf("SymmObjArray::SymmObjArray : memory not allocated\n");
}

template <class T>
SymmObjArray<T>::SymmObjArray(const SymmObjArray& a)
{  zero = rec = 0;
   *this=a;
}

template <class T>
SymmObjArray<T>& SymmObjArray<T>::operator=(const SymmObjArray& other)
{  Resize0(other.Size);
   for(int i=0; i<Size; ++i) rec[i]= other.rec[i];
   zero = rec + Size/2;
   return *this;
}

template <class T>
SymmObjArray<T>::~SymmObjArray()
{  delete [] rec;
}

template <class T>
void SymmObjArray<T>::Resize(unsigned int sz)
{  Resize0(sz = 2*sz+1);
}

template <class T>
void SymmObjArray<T>::Resize0(unsigned int sz)
{  delete [] rec;
   Size = sz;
   rec = new T[Size];
   zero = rec + Size/2;
}

template <class T>
void SymmObjArray<T>::Write(FILE* f)
{  fwrite(&Size, sizeof(int), 1, f);
   for(int i=0; i<Size; ++i)rec[i].Write(f);
}

template <class T>
void SymmObjArray<T>::Read(FILE* f)
{  int newSize;
   fread(&newSize, sizeof(int), 1, f);
   if(Size)delete [] rec;
   Size = newSize;
   rec = new T[Size];
   zero = rec + Size/2;
   for(int i=0; i<Size; ++i)rec[i].Read(f);
}

template class SymmObjArray< SymmArray<int> >;
template class SymmObjArray< SymmArray<float> >;
template class SymmObjArray< SymmArray<double> >;

template class SymmObjArray< SymmArraySSE<int> >;
template class SymmObjArray< SymmArraySSE<float> >;
template class SymmObjArray< SymmArraySSE<double> >;
