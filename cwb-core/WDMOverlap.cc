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


#include "WDMOverlap.hh"

#include "SymmArray.hh"

template <class T>
WDMOverlap<T>::WDMOverlap()
{  nRes = 0;
   layers = 0;
   catalog = 0;
}


template <class T>
WDMOverlap<T>::WDMOverlap(WDM<T>** wdm0, int nRes, double minOvlp)
{  // constructor using nRes pointers to WDM objects for which it creates
   // a catalog containing all the basis function overlap values
   // above a threshold (minOvlp)  
   int i,j, k;
   this->nRes = nRes;
   
   layers = new int [nRes];
   WDM<T>* wdm[nRes];
   
   for(i=0;i<nRes; ++i){
      layers[i] = wdm0[i]->m_Layer;
      wdm[i] = new WDM<T>(layers[i], wdm0[i]->KWDM, wdm0[i]->BetaOrder, 12);
   }
   
   for(i=0;i<nRes-1; ++i)if(layers[i]>layers[i+1]){
      printf("WDMOverlap::WDMOverlap : layers not ordered properly, exit\n");
      return;
   }
     
   struct overlaps tmp[10000];
   SymmArray<double> td1A, td1Q, td2A_even, td2Q_even, td2A_odd, td2Q_odd;
   
   int totOvlps = 0;
   catalog = new   (struct ovlArray (**[nRes])[2] );
   for(i=0; i<nRes; ++i) catalog[i] = new (struct ovlArray (*[i+1])[2]);
   for(i=0; i<nRes; ++i)for(j=0; j<=i; ++j){
      catalog[i][j] = new struct ovlArray [layers[i]+1][2];
      
      // get step and filter "length" for both resolutions
      int step1 = wdm[i]->getTDFunction(1, 1, td1A);
      int last1 = td1A.Last();
      
      int step2 = wdm[j]->getTDFunction(1, 1, td2A_odd);
      int last2 = td2A_odd.Last();
            
      int maxN = (last1 + last2 + step1 + 1)/step2 + 1;
      for(k=0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){    // k freq index (r1)
         wdm[i]->getTDFunction(k ,l, td1A);
         wdm[i]->getTDFunctionQ(k ,l, td1Q);
         
         int nOvlp = 0;
         for(int m = 0; m<=layers[j]; ++m){     // loop onver frequency index (r2)
            wdm[j]->getTDFunction(m, 0, td2A_even);  
            wdm[j]->getTDFunction(m, 1, td2A_odd);
            wdm[j]->getTDFunctionQ(m, 0, td2Q_even);
            wdm[j]->getTDFunctionQ(m, 1, td2Q_odd);
                       
            for(int n = -maxN; n<=maxN; ++n){   // loop over time index
               
               int shift = n*step2 - l*step1;
               int left = -last1;
               if(shift - last2> left) left = shift - last2;
               int right = last1;
               if(shift + last2<right)right = shift + last2;
               if(left>right)continue;
               
               SymmArray<double> *ptd2A, *ptd2Q; 
               if(n&1){
                  ptd2A = &td2A_odd;
                  ptd2Q = &td2Q_odd;
               }
               else{
                  ptd2A = &td2A_even;
                  ptd2Q = &td2Q_even;
               }
               
               T ovlpAA = 0, ovlpAQ = 0, ovlpQA = 0, ovlpQQ = 0;
               
               for(int q=left; q<=right; ++q){
                  ovlpAA += td1A[q]*(*ptd2A)[q-shift];
                  ovlpQA += td1Q[q]*(*ptd2A)[q-shift];
                  ovlpAQ += td1A[q]*(*ptd2Q)[q-shift];
                  ovlpQQ += td1Q[q]*(*ptd2Q)[q-shift];
               }
               /*
               if(m==0)
                  if(n&1) ovlpAA = ovlpQA = 0;
                  else ovlpAQ = ovlpQQ = 0;
               if(m==layers[j])
                  if((m+n)&1)ovlpAA = ovlpQA = 0;
                  else ovlpAQ = ovlpQQ = 0;
               */
               
               //if(i==j)ovlpAA = ovlpQQ = 0;
                
               if( fabs(ovlpAA)> minOvlp || fabs(ovlpAQ)> minOvlp ||
                   fabs(ovlpQA)> minOvlp || fabs(ovlpQQ)> minOvlp ){
                  tmp[nOvlp].ovlpAA = ovlpAA;
                  //if(k==0 && l==0 && fabs(ovlpAA)>minOvlp)
                  //   printf("m = %d , n = %d , ovlp = %lf\n", m,n,ovlpAA);
                  tmp[nOvlp].ovlpAQ = ovlpAQ;
                  tmp[nOvlp].ovlpQA = ovlpQA;
                  tmp[nOvlp].ovlpQQ = ovlpQQ;
                  tmp[nOvlp].index = n*(layers[j]+1) + m;
                  ++nOvlp;
               }
            }  // end loop over time index (res 2)
         }     // end loop over freq index (res 2)
         if(nOvlp>10000)printf("ERROR, tmp array too small\n");
         
         catalog[i][j][k][l].data = new struct overlaps[nOvlp];
         for(int n = 0; n<nOvlp; ++n) catalog[i][j][k][l].data[n] = tmp[n];
         catalog[i][j][k][l].size = nOvlp;
         
         totOvlps += nOvlp;
      }        // end double loop over freq index (res 1) and parity
   }           // end double loop over resolution pairs
   printf("total stored overlaps = %d\n", totOvlps);
   for(int i=0; i<nRes; ++i)delete wdm[i];
}

template <class T>
WDMOverlap<T>::WDMOverlap(char* fn)
{  // constructor which reads the catalog from a file
   read(fn);
}

template <class T>
WDMOverlap<T>::WDMOverlap(const WDMOverlap<T>& x)
{  // copy constructor
   nRes = x.nRes;
   layers = new int[nRes];
   for(int i=0; i<nRes; ++i)layers[i] = x.layers[i];
   catalog = new   (struct ovlArray (**[nRes])[2] );
   for(int i=0; i<nRes; ++i){
      catalog[i] = new (struct ovlArray (*[i+1])[2]);
      for(int j=0; j<=i; ++j){
         catalog[i][j] = new struct ovlArray [layers[i]+1][2];
         for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
            ovlArray& oa = catalog[i][j][k][l];
            ovlArray& xoa = x.catalog[i][j][k][l];
            oa.size = xoa.size;
            oa.data = new struct overlaps[oa.size];
            for(int kk = 0; kk< oa.size ; ++kk)oa.data[kk] = xoa.data[kk];
         }
      }
   }
}

template <class T>
WDMOverlap<T>::~WDMOverlap()
{  // destructor
   deallocate();
}

template <class T>
void WDMOverlap<T>::read(char* fn)
{  // read the catalog from a file
   if(nRes)deallocate();
   FILE*f = fopen(fn, "r");
   float tmp;
   fread(&tmp, sizeof(float), 1, f);
   nRes = (int)tmp;
   layers = new int[nRes];
   for(int i=0; i<nRes; ++i){
      fread(&tmp, sizeof(float), 1, f);
      printf("layers[%d] = %d\n", i, layers[i] = (int)tmp);
      
   }
   
   catalog = new   (struct ovlArray (**[nRes])[2] );
   for(int i=0; i<nRes; ++i){
      catalog[i] = new (struct ovlArray (*[i+1])[2]);
      for(int j=0; j<=i; ++j){
         catalog[i][j] = new struct ovlArray [layers[i]+1][2];
         for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
            ovlArray& oa = catalog[i][j][k][l];
            fread(&tmp, sizeof(float), 1, f);
            oa.size = (int)tmp;
            oa.data = new struct overlaps[oa.size];
            fread(oa.data, sizeof(struct overlaps), oa.size, f);
         }
      }
   }
   fclose(f);
}

template <class T>
void WDMOverlap<T>::write(char* fn)
{  // write the catalog to a file
   FILE* f = fopen(fn, "w");
   float aux = nRes;
   fwrite(&aux, sizeof(float), 1, f);
   for(int i=0; i<nRes; ++i){
      aux = layers[i];
      fwrite(&aux, sizeof(float), 1, f);
   }
   
   for(int i=0; i<nRes; ++i)for(int j=0; j<=i; ++j)
      for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
         ovlArray& oa = catalog[i][j][k][l];
         aux = oa.size;
         fwrite(&aux, sizeof(float), 1, f);
         fwrite(oa.data, sizeof(struct overlaps), oa.size, f);
      }
   fclose(f);
}

template <class T>
void WDMOverlap<T>::deallocate()
{  // release allocated memory
   if(nRes==0)return; 
   for(int i=0; i<nRes; ++i){
      for(int j=0; j<=i; ++j){
         for(int k=0; k<=layers[i]; ++k){
            delete [] catalog[i][j][k][0].data;
            delete [] catalog[i][j][k][1].data;
         }
         delete [] catalog[i][j];
      }
      delete [] catalog[i];
   }
   delete [] catalog;
   delete [] layers;
   nRes = 0;
   layers = 0;
}



template <class T>
struct overlaps WDMOverlap<T>::getOverlap(int nLayer1, size_t indx1, int nLayer2, size_t indx2)
{  // access function that returns all 4 overlap values between two pixels

   struct overlaps ret={1, 3,3,3,3};
   int r1, r2;
   for(r1 = 0; r1<nRes; ++r1)if(nLayer1 == layers[r1]+1)break;
   for(r2 = 0; r2<nRes; ++r2)if(nLayer2 == layers[r2]+1)break;
   if(r1==nRes || r2 == nRes)printf("WDMOverlap::getOverlap : resolution not found\n");
   bool swap = false;
   if(r1<r2){
      int aux = r1;
      r1 = r2;
      r2 = aux;
      size_t aux2 = indx1;
      indx1 = indx2;
      indx2 = aux2;
      swap = true;
   }
   int time1 = indx1/(layers[r1]+1);
   int freq1 = indx1%(layers[r1]+1);
   //int time2 = indx2/(layers[r2]+1);
   //int freq2 = indx2%(layers[r2]+1);
     
   int odd = time1&1;
   int32_t index = (int32_t)indx2;
   index -= (time1-odd)*layers[r1]/layers[r2]*(layers[r2]+1);
   
   struct ovlArray& vector = catalog[r1][r2][freq1][odd];
   for(int i=0; i<vector.size; ++i)if(index == vector.data[i].index){
      ret = vector.data[i];
      if(swap){
         float tmp = ret.ovlpAQ;
         ret.ovlpAQ = ret.ovlpQA;
         ret.ovlpQA = tmp;
      }
      break;
   }
   //printf("getOverlap: [%d, %d] -> [%d, %d] ; index = %d {%d}:\n", 
      //freq1, time1, freq2, time2, index, vector.size);
   return ret;
}

template <class T>
float WDMOverlap<T>::getOverlap(int nLayer1, int quad1, size_t indx1, int nLayer2, int quad2, size_t indx2)
{  // access function that returns one overlap value between two pixels

   struct overlaps res = getOverlap(nLayer1, indx1, nLayer2, indx2);
   if(res.ovlpAA>2)return 0;
   if(quad1){
      if(quad2)return res.ovlpQQ;
      return res.ovlpQA;
   }
   if(quad2)return res.ovlpAQ;
   return res.ovlpAA;
}

template <class T>
void WDMOverlap<T>::getClusterOverlaps(netcluster* pwc, int clIndex, int V, void* q)
{  // FILL cluster overlap amplitudes

   vector<int>& pIndex = pwc->cList[clIndex];
   vector<vector<struct overlaps> >* qq = (vector<vector<struct overlaps> >*)q;
   for(int j=0; j<V; ++j){
      netpixel& pix = pwc->pList[pIndex[j]];
      size_t indx1 = pix.time;
      int nLay1 = pix.layers;
         
      std::vector<struct overlaps> tmp;
      for(int k = 0; k<V; ++k){
         netpixel& pix2 = pwc->pList[pIndex[k]];
         struct overlaps tmpOvlps = getOverlap(nLay1, indx1, (int)pix2.layers, pix2.time);
         if(tmpOvlps.ovlpAA>2)continue;
         tmpOvlps.index = k;
         tmp.push_back(tmpOvlps);
      }    
      qq->push_back(tmp);
   }
}
    
/*    
template <class T>
void WDMOverlap<T>::PrintSums()
{  for(int i=0; i<nRes; ++i)for(int j=0; j<=i; ++j)
      for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
         ovlArray& oa = catalog[i][j][k][l];
         if(oa.size==0)continue;
         double res = 0;
         int cntr=0;
         
         for(int m=0; m<oa.size; ++m)res += oa.data[m].ovlpAA*oa.data[m].ovlpAA;
         printf("%3d x %3d - %3d [%d] : AA = %lf", i,j,k,l,res);
         
         res = 0;
         for(int m=0; m<oa.size; ++m)if(fabs(oa.data[m].ovlpAQ)>0.9999e-2){
            res += oa.data[m].ovlpAQ*oa.data[m].ovlpAQ;
            if(i==j && l == 0 && oa.data[m].index == k)printf("%f\n", oa.data[m].ovlpAQ);
            ++cntr;
         }
         printf("  AQ = %lf (%d) ", res, cntr);
         
         res = 0;
         for(int m=0; m<oa.size; ++m)res += oa.data[m].ovlpQA*oa.data[m].ovlpQA;
         printf("  QA = %lf", res);
         
         res = 0;
         for(int m=0; m<oa.size; ++m)res += oa.data[m].ovlpQQ*oa.data[m].ovlpQQ;
         printf("  QQ = %lf (nPix = %d) \n", res, oa.size);
         
      }
}
*/

//template class WDMOverlap<float> ;
//template class WDMOverlap<double> ;


#define CLASS_INSTANTIATION(class_) template class WDMOverlap< class_ >;

CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)

#undef CLASS_INSTANTIATION
