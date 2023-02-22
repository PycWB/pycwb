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


#include "SymmArraySSE.hh"

#include <stdlib.h>

ClassImp(SymmArraySSE<float>)

template <class Record>
SymmArraySSE<Record>::SymmArraySSE(unsigned int n)
{  recSize = sizeof(Record);
   SizeSSE = ((2*n+1)*recSize/16+1)*16; // in bytes
   last = n;
   allocateSSE();  
}

template <class Record>
SymmArraySSE<Record>::SymmArraySSE(const SymmArraySSE& a)
{  zero = rec = 0;
   *this=a;
}

template <class Record>
SymmArraySSE<Record>& SymmArraySSE<Record>::operator=(const SymmArraySSE& other)
{  recSize = other.recSize;
   Resize(other.last);
   for(int i=0; i<SizeSSE/recSize; ++i)rec[i] = other.rec[i];
   return *this;
}

template <class Record>
SymmArraySSE<Record>::~SymmArraySSE()
{  free(rec);
}

template <class Record>
void SymmArraySSE<Record>::Resize(int nn)
{  
   int newSizeSSE = ((2*nn+1)*recSize/16+1)*16; // in bytes
   if(newSizeSSE==SizeSSE)return; 
   free(rec);
   last = nn;
   SizeSSE = newSizeSSE;
   allocateSSE();  
}

template <class Record>
void SymmArraySSE<Record>::allocateSSE()
{  rec = 0;
   if(posix_memalign((void**)&rec, 16, SizeSSE))
      printf("SymmArraySSE::SymmArraySSE : memory not allocated\n");
   zero = rec + SizeSSE/(2*recSize); 
   //if(rec)printf("SSE %d allocated\n", SizeSSE);
   //else printf("SSE %d not allocated\n", SizeSSE);
}
   
template <class Record>
void SymmArraySSE<Record>::ZeroExtraElements()
{  int n = SizeSSE/recSize;
   for(int i=-n/2; i<-last; ++i)zero[i] = 0;
   for(int i=last+1; i<n/2; ++i)zero[i] = 0;
}


template <class Record>
void SymmArraySSE<Record>::Write(FILE* f)
{  fwrite(&last, sizeof(int), 1, f);
   fwrite(&recSize, sizeof(int), 1, f);
   fwrite(rec, recSize, 2*last+1, f);
}

template <class Record>
void SymmArraySSE<Record>::Read(FILE* f)
{  int n, newRecSz;
   fread(&n, sizeof(int), 1, f);
   fread(&newRecSz, sizeof(int), 1, f);
   
   if(newRecSz!=recSize){
     printf("Array::Read abort b/c different record size: %d vs %d\n", newRecSz,recSize);
     return;
   } 
   
   int newSizeSSE = ((2*n+1)*recSize/16+1)*16; // in bytes
   last = n;
   
   if(newSizeSSE != SizeSSE){
      if(SizeSSE)free(rec);
      SizeSSE = newSizeSSE;
      allocateSSE();
   }
   fread(rec, recSize, 2*n+1, f);
}

template <class Record>
void SymmArraySSE<Record>::Init(Record x)
{  for(int i=0; i<2*last+1; ++i)rec[i] = x;
}

template class SymmArraySSE<int>;
template class SymmArraySSE<float>;
template class SymmArraySSE<double>;

