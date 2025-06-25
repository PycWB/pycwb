#include "monster.hh"
#include <xmmintrin.h>
#include "SymmArray.hh"

ClassImp(monster)	 // used by THtml doc

/* Begin_Html
<center><h2>monster class</h2></center>

<p>In the CWB analysis we use a number of different time-frequency resolutions
at the same time, and it is necessary to know the "cross-talk" or "overlap"
coefficients between pixels belonging to different resolutions (and even
different quadratures)</p>

<p>More concretely, for any pixel in given TF map we want to know
its representation in other TF maps of interest.</p>

<p>The monster class computes these coefficients given a set of WDM
transforms defining the resolutions of interests. They are stored in a
multidimensional array called 'catalog', and the catalog is usually saved 
in a file for later use. Few such catalogs are stored in the directory 
$HOME_WAT_FILTERS/wdmXTalks 
</p>

<p>For practical reasons we store only coefficients with absolute value above a
threshold, typically 0.01 and as such the catalog provides a good but approximate
representation of a pixel at different resolutions.
</p>

<h3>Basic test of the catalog</h3>

<p>We used white noise and produced two TF maps. Then, using the catalog, we can
subtract one from the other (one pixel at a time). Ideally, we should remove the
signal after the subtraction, but given that we do not store small coefficients
in the catalog, some residual signal remains.</p>

<p>The script <a href="tutorials/reviewMonster.C"> implements this basic test.
</p>

End_Html */

monster::monster()
{  tag = 1;                             // catalog tag number
   nRes = 0;                            // number of resolutions
   BetaOrder = 2;                       // beta function order for Meyer 
   precision = 12;                      // wavelet precision
   KWDM = 1;                            // WDM K - parameter K/M
   layers = 0;
   catalog = 0;
   clusterCC.clear();
}



monster::monster(WDM<double>** wdm0, int nRes)
{  // 
   // computes catalog for nRes different resolutions
   // specified by wdm0 (vector of pointers to WDM transforms)
   
   int i,j, k;
   this->tag = 1;                                // current catalog tag
   this->nRes = nRes;                            // number of resolutions
   this->BetaOrder=wdm0[0]->BetaOrder;           // beta function order for Meyer 
   this->precision=wdm0[0]->precision;           // wavelet precision
   this->KWDM = wdm0[0]->KWDM/wdm0[0]->m_Layer;  // WDM K - parameter K/M

   clusterCC.clear();

   if(nRes>NRES_MAX) {
      printf("monster::monster : number of resolutions gt NRES_MAX=%d, exit\n",NRES_MAX);
      exit(1);
   }
   
   layers = new int [nRes];
   WDM<double>* wdm[nRes];
   
   for(i=0;i<nRes; ++i){
      layers[i] = wdm0[i]->m_Layer;
      if(layers[i]*this->KWDM != wdm0[i]->KWDM) {
	 printf("monster::monster : mixed WDM set: KWDM=%d, exit\n",this->KWDM);
	 exit(1);
      }
      wdm[i] = new WDM<double>(layers[i], wdm0[i]->KWDM, this->BetaOrder, this->precision);
   }
   
   for(i=0;i<nRes-1; ++i)if(layers[i]>layers[i+1]){
      printf("monster::monster : layers not ordered properly, exit\n");
      exit(1);
   }
     
   struct xtalk tmp[10000];
   SymmArray<double> td1A, td1Q, td2A_even, td2Q_even, td2A_odd, td2Q_odd;
   double minOvlp = 1e-2;
   
   int totOvlps = 0;
   catalog = new   (struct xtalkArray (**[NRES_MAX])[2] );
   for(i=0; i<nRes; ++i) catalog[i] = new (struct xtalkArray (*[NRES_MAX])[2]);
   for(i=0; i<nRes; ++i)for(j=0; j<=i; ++j){
      catalog[i][j] = new struct xtalkArray [layers[i]+1][2];
      printf("Processing resolution pair [%d x %d]...\n", layers[i], layers[j]);
      
      // get step and filter "length" for both resolutions
      int step1 = wdm[i]->getBaseWave(1, 1, td1A);
      int last1 = td1A.Last();
      
      int step2 = wdm[j]->getBaseWave(1, 1, td2A_odd);
      int last2 = td2A_odd.Last();
            
      int maxN = (last1 + last2 + step1 + 1)/step2 + 1;
      double invli = 1./layers[i];
      for(k=0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){    // k freq index (r1)
         wdm[i]->getBaseWave(k ,l, td1A);
         wdm[i]->getBaseWaveQ(k ,l, td1Q);
         
         int nOvlp = 0;
         double invlj = 1./layers[j];
         double kfreqmin = (k-1)*invli;
         double kfreqmax = (k+1)*invli;
         for(int m = 0; m<=layers[j]; ++m){     // loop onver frequency index (r2)
            //if((m+1)*invlj < kfreqmin)continue;
            //if((m-1)*invlj > kfreqmax) break;
            
            
            wdm[j]->getBaseWave(m, 0, td2A_even);  
            wdm[j]->getBaseWave(m, 1, td2A_odd);
            wdm[j]->getBaseWaveQ(m, 0, td2Q_even);
            wdm[j]->getBaseWaveQ(m, 1, td2Q_odd);
                       
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
               
               float CC[4]; CC[0] = 0, CC[1] = 0, CC[2] = 0, CC[3] = 0;
               
               for(int q=left; q<=right; ++q){
                  CC[0] += td1A[q]*(*ptd2A)[q-shift];
                  CC[2] += td1Q[q]*(*ptd2A)[q-shift];
                  CC[1] += td1A[q]*(*ptd2Q)[q-shift];
                  CC[3] += td1Q[q]*(*ptd2Q)[q-shift];
               }

               /*
               if(m==0)
                  if(n&1) CC[0] = CC[2] = 0;
                  else CC[1] = CC[3] = 0;
               if(m==layers[j])
                  if((m+n)&1)CC[0] = CC[2] = 0;
                  else CC[1] = CC[3] = 0;
               */
               
               //if(i==j)CC[0] = CC[3] = 0;
                
               if( fabs(CC[0])> minOvlp || fabs(CC[1])> minOvlp ||
                   fabs(CC[2])> minOvlp || fabs(CC[3])> minOvlp ){
                  tmp[nOvlp].CC[0] = CC[0];
                  //if(k==0 && l==0 && fabs(CC[0])>minOvlp)
                  //   printf("m = %d , n = %d , ovlp = %lf\n", m,n,CC[0]);
                  tmp[nOvlp].CC[1] = CC[1];
                  tmp[nOvlp].CC[2] = CC[2];
                  tmp[nOvlp].CC[3] = CC[3];
                  tmp[nOvlp].index = n*(layers[j]+1) + m;
                  ++nOvlp;
               }
            }  // end loop over time index (res 2)
         }     // end loop over freq index (res 2)
         if(nOvlp>10000)printf("ERROR, tmp array too small\n");
         
         catalog[i][j][k][l].data = new struct xtalk[nOvlp];
         for(int n = 0; n<nOvlp; ++n) catalog[i][j][k][l].data[n] = tmp[n];
         catalog[i][j][k][l].size = nOvlp;
         
         totOvlps += nOvlp;
      }        // end double loop over freq index (res 1) and parity
   }           // end double loop over resolution pairs
   printf("total stored xtalk = %d\n", totOvlps);
   for(int i=0; i<nRes; ++i) delete wdm[i];
}


monster::monster(char* fn)
{  //
   // constructor, reads catalog from file 
  
   this->nRes = 0;
   read(fn);
   clusterCC.clear();
}


monster::monster(const monster& x)
{  //
   // copy constructor

   if(x.nRes>NRES_MAX) {
      printf("monster::monster : number of resolutions gt NRES_MAX=%d, exit\n",NRES_MAX);
      exit(1);
   }

   this->tag = x.tag;                     // current catalog tag
   this->nRes = x.nRes;                   // number of resolutions
   this->BetaOrder=x.BetaOrder;           // beta function order for Meyer 
   this->precision=x.precision;           // wavelet precision
   this->KWDM = x.KWDM;                   // WDM K - parameter K/M
   layers = new int[this->nRes];
   for(int i=0; i<this->nRes; ++i)layers[i] = x.layers[i];
   catalog = new   (struct xtalkArray (**[NRES_MAX])[2] );
   for(int i=0; i<this->nRes; ++i){
      catalog[i] = new (struct xtalkArray (*[NRES_MAX])[2]);
      for(int j=0; j<=i; ++j){
         catalog[i][j] = new struct xtalkArray [layers[i]+1][2];
         for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
            xtalkArray& oa = catalog[i][j][k][l];
            xtalkArray& xoa = x.catalog[i][j][k][l];
            oa.size = xoa.size;
            oa.data = new struct xtalk[oa.size];
            for(int kk = 0; kk< oa.size ; ++kk)oa.data[kk] = xoa.data[kk];
         }
      }
   }
}


monster::~monster()
{  
   deallocate();
}


void monster::read(char* fn)
{  //
   // reads catalog from file 

   if(this->nRes>NRES_MAX) {
      printf("monster::read : number of resolutions gt NRES_MAX=%d, exit\n",NRES_MAX);
      exit(1);
   }
   
   if(this->nRes) deallocate();
   FILE*f = fopen(fn, "r");
   float tmp;
   fread(&tmp, sizeof(float), 1, f);
   this->nRes = (int)tmp;
   if(this->nRes<0) {
      this->nRes = -(this->nRes);
      fread(&tmp, sizeof(float), 1, f);
      this->tag = (int)tmp;
      fread(&tmp, sizeof(float), 1, f);
      this->BetaOrder = (int)tmp;
      fread(&tmp, sizeof(float), 1, f);
      this->precision = (int)tmp;
      fread(&tmp, sizeof(float), 1, f);
      this->KWDM = (int)tmp;
   }
   else {this->tag = 0;}
   layers = new int[this->nRes];
   for(int i=0; i<this->nRes; ++i){
      fread(&tmp, sizeof(float), 1, f);
      layers[i] = (int)tmp;
//      printf("layers[%d] = %d\n", i, layers[i] = (int)tmp);
      
   }
   
   catalog = new   (struct xtalkArray (**[NRES_MAX])[2] );
   for(int i=0; i<this->nRes; ++i){
      catalog[i] = new (struct xtalkArray (*[NRES_MAX])[2]);
      for(int j=0; j<=i; ++j){
         catalog[i][j] = new struct xtalkArray [layers[i]+1][2];
         for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
            xtalkArray& oa = catalog[i][j][k][l];
            fread(&tmp, sizeof(float), 1, f);
            oa.size = (int)tmp;
            oa.data = new struct xtalk[oa.size];
            fread(oa.data, sizeof(struct xtalk), oa.size, f);
         }
      }
   }
   fclose(f);
}


void monster::write(char* fn)
{  //
   // writes catalog to file 
   FILE* f = fopen(fn, "w");
   float aux = this->tag ? -nRes : nRes;        // positive/negative - old/new catalog format 
   fwrite(&aux, sizeof(float), 1, f);
   aux = this->tag;
   fwrite(&aux, sizeof(float), 1, f);
   aux = this->BetaOrder;
   fwrite(&aux, sizeof(float), 1, f);
   aux = this->precision;
   fwrite(&aux, sizeof(float), 1, f);
   aux = this->KWDM;
   fwrite(&aux, sizeof(float), 1, f);
   for(int i=0; i<this->nRes; ++i){
      aux = layers[i];
      fwrite(&aux, sizeof(float), 1, f);
   }
   
   for(int i=0; i<this->nRes; ++i)for(int j=0; j<=i; ++j)
      for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
         xtalkArray& oa = catalog[i][j][k][l];
         aux = oa.size;
         fwrite(&aux, sizeof(float), 1, f);
         fwrite(oa.data, sizeof(struct xtalk), oa.size, f);
      }
   fclose(f);
}


void monster::deallocate()
{  //
   // deallocates memory 
   int J = this->clusterCC.size();
   for(int j=0; j<J; j++) _mm_free(this->clusterCC[j]);
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




struct xtalk monster::getXTalk(int nLayer1, size_t indx1, int nLayer2, size_t indx2)
{  //
   // returns the overlap ("cross-talk") values for two pixels, 
   // and for all possible quadrature pairs. 
   // nLayer identifies the time-frequency map (resolution) of the pixel, 
   // indx specifies the location of the pixel on the TF map 

   struct xtalk ret={1, {3.,3.,3.,3.}};
   
   int r1, r2;
   for(r1 = 0; r1<nRes; ++r1)if(nLayer1 == layers[r1]+1)break;
   for(r2 = 0; r2<nRes; ++r2)if(nLayer2 == layers[r2]+1)break;
   if(r1==nRes || r2 == nRes){
      printf("monster::getXTalk : resolution not found %d %d %d %d %d %d %d\n",nRes,nLayer1,nLayer2,r1,r2,layers[0],layers[6]);
      exit(1);
   }
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
   
   index -= (time1-odd)*(layers[r1]/layers[r2])*(layers[r2]+1);
   
   struct xtalkArray& vector = catalog[r1][r2][freq1][odd];
   int i=0;
   for(; i<vector.size; ++i)if(index == vector.data[i].index){
      ret = vector.data[i];
      if(swap){
         float tmp = ret.CC[1];
         ret.CC[1] = ret.CC[2];
         ret.CC[2] = tmp;
      }
      break;
   }
   //printf("getXTalk: [%d, %d] -> [%d, %d] ; index = %d {%d}:\n", 
      //freq1, time1, freq2, time2, index, vector.size);
   return ret;
}


float monster::getXTalk(int nLayer1, int quad1, size_t indx1, int nLayer2, int quad2, size_t indx2)
{  //
   // returns the overlap value for two pixels when the quadratures are known ( uses getXTalk function above)

   struct xtalk res = getXTalk(nLayer1, indx1, nLayer2, indx2);
   if(res.CC[0]>2)return 0;
   if(quad1){
      if(quad2)return res.CC[3];
      return res.CC[2];
   }
   if(quad2)return res.CC[1];
   return res.CC[0];
}


std::vector<int> monster::getXTalk(netcluster* pwc, int id, bool check)
{  //
   // fill in cluster coupling coefficients into q
   // pwc - pointer to netcluster object
   // id - cluster ID
   // check - true/false - check / do not check TD vectors
   // returns size of TD array
   
   vector<int>& pIndex = pwc->cList[id-1];
   vector<int> pI;

   int J = this->clusterCC.size();
   for(int j = 0; j<J; j++) _mm_free(this->clusterCC[j]);
   this->clusterCC.clear();
   std::vector<float*>().swap(clusterCC);
   this->sizeCC.clear();
   std::vector<int>().swap(sizeCC);

   netpixel* pixi;
   netpixel* pixj;

   int V = (int)pIndex.size();
   for(int i=0; i<V; i++){
      pixi = pwc->getPixel(id,i);
      if(!pixi) {
         cout<<"monster::netcluster error: NULL pointer"<<endl;
         exit(1);
      }   
      if(check && !pixi->tdAmp.size()) continue;         // check loaded pixels
      pI.push_back(i);

      int N = 0;
      int M = 0;
      int K = 0;

//      cout<<id<<" "<<i<<" "<<pixi->layers<<" ";
         
      wavearray<float> tmp(8*V);
      for(int j = 0; j<V; j++){
         pixj = pwc->getPixel(id,j);
         if(!pixj) {
            cout<<"monster::netcluster error: NULL pointer"<<endl;
            exit(1);
         }   
         if(check && !pixj->tdAmp.size()) continue;         // check loaded pixels
//         if(pixi->layers==pixj->layers && pixi->time!=pixj->time) continue;
         M++;
         struct xtalk tmpOvlp = getXTalk(pixi->layers, pixi->time, pixj->layers, pixj->time);
         if(tmpOvlp.CC[0]>2)continue;

	 N = (i==j) ? 0 : ++K;

         tmp.data[N*8+0] = float(M-1);
         tmp.data[N*8+1] = tmpOvlp.CC[0]*tmpOvlp.CC[0]+tmpOvlp.CC[1]*tmpOvlp.CC[1];
         tmp.data[N*8+2] = tmpOvlp.CC[2]*tmpOvlp.CC[2]+tmpOvlp.CC[3]*tmpOvlp.CC[3];
         tmp.data[N*8+3] = tmp.data[N*8+1]+tmp.data[N*8+2];
         tmp.data[N*8+4] = tmpOvlp.CC[0];
         tmp.data[N*8+5] = tmpOvlp.CC[2];
         tmp.data[N*8+6] = tmpOvlp.CC[1];
         tmp.data[N*8+7] = tmpOvlp.CC[3];
	 //cout<<"i="<<i<<" j="<<j<<" M="<<M-1<<" N="<<N<<" "<<tmp.data[N*8+3]<<endl;
      } 
      N = K+1;
      float* p8 = (float*)_mm_malloc(N*8*sizeof(float),32);    // N x 8 floats aligned array
      for(int n = 0; n<N*8; n++) p8[n] = tmp.data[n];
      sizeCC.push_back(N);
      clusterCC.push_back(p8);
   }
   return pI;
}

    
/*    

void monster::PrintSums()
{  for(int i=0; i<nRes; ++i)for(int j=0; j<=i; ++j)
      for(int k = 0; k<=layers[i]; ++k)for(int l=0; l<2; ++l){
         xtalkArray& oa = catalog[i][j][k][l];
         if(oa.size==0)continue;
         double res = 0;
         int cntr=0;
         
         for(int m=0; m<oa.size; ++m)res += oa.data[m].CC[0]*oa.data[m].CC[0];
         printf("%3d x %3d - %3d [%d] : AA = %lf", i,j,k,l,res);
         
         res = 0;
         for(int m=0; m<oa.size; ++m)if(fabs(oa.data[m].CC[1])>0.9999e-2){
            res += oa.data[m].CC[1]*oa.data[m].CC[1];
            if(i==j && l == 0 && oa.data[m].index == k)printf("%f\n", oa.data[m].CC[1]);
            ++cntr;
         }
         printf("  AQ = %lf (%d) ", res, cntr);
         
         res = 0;
         for(int m=0; m<oa.size; ++m)res += oa.data[m].CC[2]*oa.data[m].CC[2];
         printf("  QA = %lf", res);
         
         res = 0;
         for(int m=0; m<oa.size; ++m)res += oa.data[m].CC[3]*oa.data[m].CC[3];
         printf("  QQ = %lf (nPix = %d) \n", res, oa.size);
         
      }
}
*/

//template class monster<float> ;
//template class monster<double> ;

 /*
#define CLASS_INSTANTIATION(class_) template class monster< class_ >;

CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)

#undef CLASS_INSTANTIATION
 */
