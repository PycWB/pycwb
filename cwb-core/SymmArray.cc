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


#include "SymmArray.hh"


ClassImp(SymmArray<double>)


template <class Record>
SymmArray<Record>::SymmArray(unsigned int n)
{  //printf("Hello SymmArray::SymmArray, n=%d\n", n);
   Size = 2*n+1; //mind the plus
   rec = new Record[Size];
   zero = rec+n;
   recSize = sizeof(Record);
   if(rec ==0 )printf("SymmArray::SymmArray : memory not allocated\n");
}

template <class Record>
SymmArray<Record>::SymmArray(const SymmArray& a)
{  zero = rec = 0;
   *this=a;
}

template <class Record>
SymmArray<Record>& SymmArray<Record>::operator=(const SymmArray& other)
{  recSize = other.recSize;
   Resize0(other.Size);
   for(int i=0; i<Size; ++i)rec[i] = other.rec[i];
   zero = rec + Size/2;
   return *this;
}

template <class Record>
SymmArray<Record>::~SymmArray()
{  delete [] rec;
}

template <class Record>
void SymmArray<Record>::Resize(int sz)
{  Resize0(sz = 2*sz+1);
}

template <class Record>
void SymmArray<Record>::Resize0(int sz)
{  if(sz==Size)return;
   delete [] rec;
   Size = sz;
   rec = new Record[Size];
   zero = rec + Size/2;
}

template <class Record>
void SymmArray<Record>::Write(FILE* f)
{  fwrite(&Size, sizeof(int), 1, f);
   fwrite(&recSize, sizeof(int), 1, f);
   if(Size)fwrite(rec, recSize, Size, f);
}

template <class Record>
void SymmArray<Record>::Read(FILE* f)
{  int newSize, newRecSz;
   fread(&newSize, sizeof(int), 1, f);
   fread(&newRecSz, sizeof(int), 1, f);
   if(newRecSz!=recSize){
     printf("Array::Read abort b/c different record size %d %d\n", newRecSz,
     recSize);
     return;
   } 
   if(newSize != Size){
      if(Size)delete [] rec;
      Size = newSize;
      rec = new Record[Size];
      zero = rec + Size/2;
   }
   fread(rec, recSize, Size, f);
   //printf("cool\n");
}

template <class Record>
void SymmArray<Record>::Init(Record x)
{  for(int i=0; i<Size; ++i)rec[i] = x;
}

template class SymmArray<int>;
template class SymmArray<float>;
template class SymmArray<double>;
