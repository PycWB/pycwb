/*
# Copyright (C) 2019 Sergey Klimenko, Gabriele Vedovato
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


/*-------------------------------------------------------
 * Package: 	Wavelet Analysis Tool
 * generic data container to use with DMT and ROOT
 * File name: 	wavearray.cc
 *-------------------------------------------------------
*/
// wavearray class is the base class for wavelet data amd methods . 

#include <time.h>
#include <iostream>
//#include <sstream>
//#include <strstream>

//#include "PConfig.h"
#ifdef __GNU_STDC_OLD
#include <gnusstream.h>
#else
#include <sstream>
#endif

#include "wavearray.hh"
#include "wavefft.hh"
#include "TComplex.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TFile.h"

extern "C" {
   typedef int (*qsort_func) (const void*, const void*);
}

ClassImp(wavearray<double>)

using namespace std;

//: Default constructor
template<class DataType_t>
wavearray<DataType_t>::wavearray() : 
   data(NULL),Size(0),Rate(1.),Start(0.),Stop(0.),Edge(0.),fftw(NULL), ifftw(NULL) 
{   
   Slice = std::slice(0,0,0); 
}

//: allocates a data array with N=n elements.
template<class DataType_t>
wavearray<DataType_t>::wavearray(int n) : 
Rate(1.),Start(0.),Edge(0.),fftw(NULL),ifftw(NULL)
{ 
  if (n <= 0 ) n = 1;
  data = (DataType_t *)malloc(n*sizeof(DataType_t));
  Size = n;
  Stop = double(n);
  Slice = std::slice(0,n,1);
  *this = 0;
}

//: copy constructor 
template<class DataType_t>
wavearray<DataType_t>::wavearray(const wavearray<DataType_t>& a) :
   data(NULL),Size(0),Rate(1.),Start(0.),Stop(0.),Edge(0.),fftw(NULL), ifftw(NULL)
{ *this = a; }

// explicit construction from array
template<class DataType_t> template<class T> 
wavearray<DataType_t>::
wavearray(const T *p, unsigned int n, double r) : 
   data(NULL),Size(0),Rate(1.),Start(0.),Edge(0.),fftw(NULL), ifftw(NULL)
{ 
   unsigned int i;
   if(n != 0 && p != NULL){
      data = (DataType_t *)malloc(n*sizeof(DataType_t));
      for (i=0; i < n; i++) data[i] = p[i];
      Size = n;
      Rate = r;
      Stop = n/r;
   }
   Slice = std::slice(0,n,1);
}

//: destructor
template<class DataType_t>
wavearray<DataType_t>::~wavearray()
{ 
   if(data)  free(data);
   resetFFTW();
}

//: operators =

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator=(const wavearray<DataType_t>& a)
{
   if (this==&a) return *this;

   unsigned int i;
   unsigned int N = a.Slice.size();
   unsigned int m = a.Slice.stride();

   this->Edge = a.Edge;
   if (N>0 && a.data) {
      DataType_t *pm  = a.data + a.Slice.start();
      wavearray<DataType_t>::resize(N);

      for (i=0; i < N; i++) { data[i] = *pm; pm+=m; }

      if(a.rate()>0.) {
	 start(a.start() + a.Slice.start()/a.rate());
         stop(start()+N*m/a.rate());
      }
      else {         
	 start(a.start());
         stop(a.stop());

      }

      rate(a.rate()/m);
      Slice = std::slice(0,  size(),1);
      const_cast<wavearray<DataType_t>&>(a).Slice = std::slice(0,a.size(),1);
   }
   else if(data) {
      free(data);
      data = NULL;
      Size = 0;
      Start = a.start();
      Stop = a.stop();
      Rate = a.rate();
      Slice = std::slice(0,0,0);
   }
   this->fftw = NULL;
   this->ifftw = NULL;

   return *this;
}

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator<<(wavearray<DataType_t>& a)
{
   unsigned int i;
   unsigned int N = limit(a);
   unsigned int n = Slice.stride();
   unsigned int m = a.Slice.stride();
   DataType_t *p = a.data + a.Slice.start();

   if(size())
      for (i=Slice.start(); i<N; i+=n){ data[i]  = *p; p += m; }
   
     Slice = std::slice(0,  size(),1);
   a.Slice = std::slice(0,a.size(),1);
   return *this;
}

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator=(const DataType_t c)
{
   unsigned int i;
   unsigned int n = Slice.stride();
   unsigned int N = limit();

   if(size())
      for (i=Slice.start(); i < N; i+=n) data[i]  = c;

   Slice = std::slice(0,  size(),1);
   return *this;
}

//: operators +=

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator+=(wavearray<DataType_t> &a)
{
   unsigned int i;
   unsigned int N = limit(a);
   unsigned int n = Slice.stride();
   unsigned int m = a.Slice.stride();
   DataType_t *p = a.data + a.Slice.start();

   if(size())
      for (i=Slice.start(); i<N; i+=n){ data[i]  += *p; p += m; }
   
     Slice = std::slice(0,  size(),1);
   a.Slice = std::slice(0,a.size(),1);
   return *this;
}

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator+=(const DataType_t c)
{
   unsigned int i;
   unsigned int n = Slice.stride();
   unsigned int N = limit();

   if(size())
      for (i=Slice.start(); i < N; i+=n) data[i]  += c;

   Slice = std::slice(0,  size(),1);
   return *this;
}

//: operators -=

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator-=(wavearray<DataType_t> &a)
{
   unsigned int i;
   unsigned int N = limit(a);
   unsigned int n = Slice.stride();
   unsigned int m = a.Slice.stride();
   DataType_t *p = a.data + a.Slice.start();

   if(size())
      for (i=Slice.start(); i<N; i+=n){ data[i]  -= *p; p += m; }
   
     Slice = std::slice(0,  size(),1);
   a.Slice = std::slice(0,a.size(),1);
   return *this;
}

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator-=(const DataType_t c)
{
   unsigned int i;
   unsigned int n = Slice.stride();
   unsigned int N = limit();

   if(size())
      for (i=Slice.start(); i < N; i+=n) data[i]  -= c;

   Slice = std::slice(0,  size(),1);
   return *this;
}

//: multiply all elements of data array by constant
template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator*=(const DataType_t c)
{
   unsigned int i;
   unsigned int n = Slice.stride();
   unsigned int N = limit();

   if(size())
      for (i=Slice.start(); i < N; i+=n) data[i]  *= c;

   Slice = std::slice(0,  size(),1);
   return *this;
}

// scalar production
template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator*=(wavearray<DataType_t>& a)
{
   unsigned int i;
   unsigned int N = limit(a);
   unsigned int n = Slice.stride();
   unsigned int m = a.Slice.stride();
   DataType_t *p = a.data + a.Slice.start();

   if(size())
      for (i=Slice.start(); i<N; i+=n){ data[i]  *= *p; p += m; }
   
     Slice = std::slice(0,  size(),1);
   a.Slice = std::slice(0,a.size(),1);
   return *this;
}

// operator[](const std::slice &)
template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator[](const std::slice& s)
{
   Slice = s;
   if(limit() > size()){
      cout << "wavearray::operator[slice]: Illegal argument "<<limit()<<" "<<size()<<"\n";
      Slice = std::slice(0,size(),1);
   }
   return *this;
}

// operator[](const std::slice &)
template<class DataType_t>
DataType_t & wavearray<DataType_t>::
operator[](const unsigned int n)
{
   if(n >= size()){
      cout << "wavearray::operator[int]: Illegal argument\n";
      return data[0];
   }
   return data[n];
}

//: Dumps data array to file *fname in ASCII format.
template<class DataType_t>
void wavearray<DataType_t>::Dump(const char *fname, int app)
{
  int n=size();
  char mode[3] = "w";
  if (app == 1) strcpy (mode, "a");

  FILE *fp;

  if ( (fp = fopen(fname, mode)) == NULL ) {
     cout << " Dump() error: cannot open file " << fname <<". \n";
     return;
  };

  if(app == 0) {
    fprintf( fp,"# start time: -start %lf \n", this->Start );
    fprintf( fp,"# sampling rate: -rate %lf \n", this->Rate );
    fprintf( fp,"# number of samples: -size %d \n", (int)this->Size );
  }

  if(app == 2) {
    double dt = 1./this->rate();
    for (int i = 0; i < n; i++) {
      double time = this->start()+i*dt;
      fprintf( fp,"%14f %e\n", time, data[i]);
    }
  } else {
    for (int i = 0; i < n; i++) fprintf( fp,"%e ", (float)data[i]);
  }

  fclose(fp); 
}
  
//: Dumps data array to file *fname in binary format.
template<class DataType_t>
void wavearray<DataType_t>::DumpBinary(const char *fname, int app)
{
  int n = size() * sizeof(DataType_t);
  char mode[5];
  strcpy (mode, "w");

  if (app == 1) strcpy (mode, "a");

  FILE *fp;

  if ( (fp=fopen(fname, mode)) == NULL ) {
     cout << " DumpBinary() error : cannot open file " << fname <<" \n";
     return ;
  }

  fwrite(data, n, 1, fp);
  fclose(fp);
}

//: Dumps data array to file *fname in binary format as "short".
template<class DataType_t> 
void wavearray<DataType_t>::DumpShort(const char *fname, int app)
{
  int n = size();
  char mode[5] = "w";
  if (app == 1) strcpy (mode, "a");

  FILE *fp;
  if ( (fp = fopen(fname, mode)) == NULL ) {
    cout << " DumpShort() error : cannot open file " << fname <<". \n";
    return;
  }

  short *dtemp;
  dtemp=new short[n];

  for ( int i=0; i<n; i++ ) dtemp[i]=short(data[i]) ;

  n = n * sizeof(short);

  fwrite(dtemp, n, 1, fp);
  fclose(fp);
  delete [] dtemp;
}

//: Dumps object wavearray to file *fname in root format.
template<class DataType_t>
void wavearray<DataType_t>::DumpObject(const char *fname)
{
  TFile* rfile = new TFile(fname, "RECREATE");
  this->Write("wavearray"); // write this object
  rfile->Close();
  delete rfile;
}

//: Read data from file in binary format.
template<class DataType_t> 
void wavearray<DataType_t>::ReadBinary(const char *fname, int N)
{
   int step = sizeof(DataType_t);
   FILE *fp;
   double d;

   if ( (fp=fopen(fname,"r")) == NULL ) {
      cout << " ReadBinary() error : cannot open file " << fname <<". \n";
      exit(1);
   }


   if(N == 0){              // find the data length
      while(!feof(fp)){ 
	 fread(&d,step,1,fp);
	 N++;
      }
      N--;
      rewind(fp);
   }	

   if(int(size()) != N) resize(N);
   int n = size() * sizeof(DataType_t);

   fread(data, n, 1, fp);    // Reading binary record
   fclose(fp);
}

//: Read data from file as short.
template<class DataType_t> 
void wavearray<DataType_t>::ReadShort(const char *fname)
{
  short *dtmp;
  dtmp = new short[size()];
  int n = size() * sizeof(short);
  FILE *fp;

  if ( (fp=fopen(fname,"r")) == NULL ) {
     cout << " ReadShort() error : cannot open file " << fname <<". \n";
     return;
  }

  cout << " Reading binary record, size="<< n <<"\n";

  fread(dtmp, n, 1, fp);
  for (unsigned int i = 0; i < size(); i++) 
     data[i] = DataType_t(dtmp[i]);
  fclose(fp);
  delete [] dtmp;
}

//: resizes data array to a new length n.
template<class DataType_t> 
void wavearray<DataType_t>::resize(unsigned int n)
{
   DataType_t *p = NULL;
   if(n==0){
     if(data) free(data);
     data = NULL;
     Size = 0;
     Stop = Start;
     Slice = std::slice(0,0,0); 
     return;
   }

   p = (DataType_t *)realloc(data,n*sizeof(DataType_t));

   if(p){ 
      data = p;
      Size = n;
      Stop = Start + n/Rate;
      Slice = std::slice(0,n,1);
   }
   else {
      cout<<"wavearray::resize(): memory allocation failed.\n";
      exit(0);
   }

   resetFFTW();
}

/************************************************************************
 * Creates new data set by resampling the original data from "a"        *
 * with new sample frequency "f". Uses polynomial interpolation scheme  *
 * (Lagrange formula) with np-points, "np" must be even number, by      *
 * default np=6 (corresponds to 5-th order polynomial interpolation).   *
 * This function calls wavearray::resize() function to adjust            *
 * current data array if it's size does not exactly match new number    *
 * of points.                                                           *
 ************************************************************************/

template<class DataType_t> 
void wavearray<DataType_t>::
resample(wavearray<DataType_t> const &a, double f, int nF)
{
   int nP = nF;
   if(nP<=1) nP = 6;
   if(int(a.size())<nP) nP = a.size();
   nP = (nP>>1)<<1;

   int N;
   int nP2 = nP/2;

   const DataType_t *p = a.data;

   int i;
   int iL;
   double x;
   double *temp=new double[nF];
   
   rate(f);
   double ratio = a.rate() / rate();
   N = int(a.size()/ratio + 0.5); 
   
   if ( (int)size() != N )  resize(N); 
   
// left border
   
   int nL = int(nP2/ratio);

   for (i = 0; i < nL; i++)
      data[i] = (DataType_t)Nevill(i*ratio, nP, p, temp);

// in the middle of array

   int nM = int((a.size()-nP2)/ratio);  
   if(nM < nL) nM = nL;

   if(nM&1 && nM>nL) { 
      x = nL*ratio;
      iL = int(x) - nP2 + 1 ;
      data[i] = (DataType_t)Nevill(x-iL, nP, p+iL, temp);
      nL++;
   }
   for (i = nL; i < nM; i+=2) { 
      x = i*ratio;
      iL = int(x) - nP2 + 1 ;
      data[i] = (DataType_t)Nevill(x-iL, nP, p+iL, temp);
      x += ratio;
      iL = int(x) - nP2 + 1 ;
      data[i+1] = (DataType_t)Nevill(x-iL, nP, p+iL, temp);
   }

// right border

   int nR = a.size() - nP;
   p += nR;
   for (i = nM; i < N; i++)
      data[i] = (DataType_t)Nevill(i*ratio-nR, nP, p, temp);

   delete [] temp;
}

template<class DataType_t> 
void wavearray<DataType_t>::resample(double f, int nF)
{
   wavearray<DataType_t> a;
   a = *this;
   resample(a,f,nF);
}

template<class DataType_t> 
void wavearray<DataType_t>::Resample(double f)
{
   if(f<=0) {
      cout<<"wavearray::Resample(): error - resample frequency must be >0\n";
      exit(1);
   }

   double rsize = f/this->rate()*this->size();	// new size

   if(fmod(rsize,2)!=0) {
      cout<<"wavearray::Resample(): error - resample frequency (" << f << ") not allowed\n";
      exit(1);
   }

   int size = this->size();			// old size

   this->FFTW(1);
   this->resize((int)rsize);
   for(int i=size;i<rsize;i++) this->data[i]=0;
   this->FFTW(-1);
   this->rate(f);
}

template<class DataType_t> 
void wavearray<DataType_t>::delay(double T)
{
   if(T==0) return;

   // search begin,end of non zero data
   int ibeg=0; int iend=0;
   for(int i=0;i<(int)this->size();i++) {
     if(this->data[i]!=0 && ibeg==0) ibeg=i;
     if(this->data[i]!=0) iend=i;
   }
   int ilen=iend-ibeg+1;
   // create temporary array for FFTW & add scratch buffer + T
   int ishift = fabs(T)*this->rate();
   int isize = 2*ilen+2*ishift;
   isize = isize + (isize%4 ? 4 - isize%4 : 0); // force to be multiple of 4
   wavearray<DataType_t> w(isize);
   w.rate(this->rate()); w=0;
   // copy this->data !=0 in the middle of w array & set this->data=0
   for(int i=0;i<ilen;i++) {w[i+ishift+ilen/2]=this->data[ibeg+i];this->data[ibeg+i]=0;}

   double pi = TMath::Pi();
   // apply time shift to waveform vector
   w.FFTW(1);
   TComplex C;
   double df = w.rate()/w.size();
   //cout << "T : " << T << endl;
   for (int ii=0;ii<(int)w.size()/2;ii++) {
     TComplex X(w[2*ii],w[2*ii+1]);
     X=X*C.Exp(TComplex(0.,-2*pi*ii*df*T));  // Time Shift
     w[2*ii]=X.Re();
     w[2*ii+1]=X.Im();
   }
   w.FFTW(-1);

   // copy shifted data to input this->data array
   for(int i=0;i<(int)w.size();i++) {
     int j=ibeg-(ishift+ilen/2)+i;
     if((j>=0)&&(j<(int)this->size())) this->data[j]=w[i];
   }
}


template<class DataType_t> 
void wavearray<DataType_t>::
Resample(wavearray<DataType_t> const &a, double f, int np)
{
  int i1, N, n1, n2, n, np2 = np/2;
  double s, x, *c, *v;
  c=new double[np];
  v=new double[np];

  rate(f);
  double ratio = a.rate() / rate();
  n = a.size();
  N = int(n/ratio  + 0.5); 

  if ( (int)size() != N )  resize(N); 

// calculate constant filter part c(k) = -(-1)^k/k!/(np-k-1)!
  for (int i = 0; i < np; i++) {
    int m = 1;
    for (int j = 0; j < np; j++)  if (j != i) m *= (i - j);
    c[i] = 1./double(m);
  }

  for (int i = 0; i < N; i++)
  { 
    x = i*ratio;        

    i1 = int(x);
    x = x - i1 + np2 - 1;

// to treat data boundaries we need to calculate critical numbers n1 and n2
    n1 = i1 - np2 + 1;
    n2 = i1 + np2 + 1 - n;

// here we calculate the sum of products like h(k)*a(k), k=0..np-1,
// where h(k) are filter coefficients, a(k) are signal samples
// h(k) = c(k)*prod (x-i), i != k, c(k) = -(-1)^k/k!/(np-k-1)! 

// get signal part multiplied by constant filter coefficients
    if ( n1 >= 0 )
      if ( n2 <= 0 ) 
// regular case - far from boundaries
        for (int j = 0; j < np; j++) v[j] = c[j]*a.data[i1 + j - np2 + 1];
      
      else {
// right border case
        x = x + n2;
        for (int j = 0; j < np; j++) v[j] = c[j]*a.data[n + j - np];
      }  
    else {
// left border case
      x = x + n1;
      for (int j = 0; j < np; j++)  v[j] = c[j]*a.data[j];
    }

// multiply resulted v[j] by x-dependent factors (x-k), if (k != j)
    for (int k = 0; k < np; k++) {
      for (int j = 0; j < np; j++) if (j != k) v[j] *= x; 
      x -= 1.; }

    s = 0.;
    for (int j = 0; j < np; j++)  s += v[j]; 
        
    data[i] = (DataType_t)s;
  }

  delete [] c;
  delete [] v;
}

/******************************************************************************
 * copies data from data array of the object wavearray a to *this
 * object data array. Procedure starts from the element "a_pos" in
 * the source array which is copied to the element "pos" in the
 * destination array. "length" is the number of data elements to copy.
 * By default "length"=0, which means copy all source data or if destination
 * array is shorter, copy data until the destination is full.
 *****************************************************************************/
template<class DataType_t> void wavearray<DataType_t>::
cpf(const wavearray<DataType_t> &a, int length, int a_pos, int pos)
{ 
//   if (rate() != a.rate()){
//      cout << "wavearray::cpf() warning: sample rate mismatch.\n";
//      cout<<"rate out: "<<rate()<<"  rate in: "<<a.rate()<<endl;
//   }

   if (length == 0 ) 
      length = ((size() - pos) < (a.size() - a_pos))? 
	 (size() - pos) : (a.size() - a_pos);

   if( length > (int)(size() - pos) ) length = size() - pos; 
   if( length > (int)(a.size() - a_pos) ) length = a.size() - a_pos; 

   for (int i = 0; i < length; i++)
     data[i + pos] = a.data[i + a_pos];

//   rate(a.rate());
}

/******************************************************************************
 * Adds data from data array of the object wavearray a to *this
 * object data array. Procedure starts from the element "a_pos" in
 * the source array which is added starting from the element "pos" in the
 * destination array. "length" is the number of data elements to add.
 * By default "length"=0, which means add all source data or if destination
 * array is shorter, add data up to the end of destination.
 *****************************************************************************/
template<class DataType_t> void wavearray<DataType_t>::
add(const wavearray<DataType_t> &a, int length, int a_pos, int pos)
{
//   if (rate() != a.rate())
//      cout << "wavearray::add() warning: sample rate mismatch.\n";

   if (length == 0 )
      length = ((size() - pos) < (a.size() - a_pos))? 
	 (size() - pos) : (a.size() - a_pos);

   if( length > (int)( size()- pos) ) length = size() - pos; 
   if( length > (int)(a.size() - a_pos) ) length = a.size() - a_pos; 
   
   for (int i = 0; i < length; i++)
      data[i + pos] += a.data[i + a_pos];
}


/******************************************************************************
 * Subtracts data array of the object wavearray a from *this
 * object data array. Procedure starts from the element "a_pos" in
 * the source array which is subtracted starting from the element "pos" in
 * the destination array. "length" is number of data elements to subtract.
 * By default "length"=0, which means subtract all source data or if the 
 * destination array is shorter, subtract data up to the end of destination.
 ******************************************************************************/
template<class DataType_t> void wavearray<DataType_t>::
sub(const wavearray<DataType_t> &a, int length, int a_pos, int pos)
{
//   if (rate() != a.rate())
//      cout << "wavearray::sub() warning: sample rate mismatch.\n";

   if ( length == 0 )
      length = ((size() - pos) < (a.size() - a_pos))? 
	 (size() - pos) : (a.size() - a_pos);

   if( length > (int)(size() - pos) ) length = size() - pos; 
   if( length > (int)(a.size() - a_pos) ) length = a.size() - a_pos; 

   for (int i = 0; i < length; i++)
      data[i + pos] -= a.data[i + a_pos];
}

/*****************************************************************
 * append two wavearrays with the same rate ignoring start time 
 * of input array.
 *****************************************************************/
template<class DataType_t> size_t wavearray<DataType_t>::
append(const wavearray<DataType_t> &a)
{
   size_t n = this->size();
   size_t m = a.size();
   
   if (this->rate() != a.rate())
      cout << "wavearray::append() warning: sample rate mismatch.\n";

   if(m == 0 ) return this->size();
   this->resize(n+m);
   this->cpf(a,m,0,n);
   this->stop(start()+(n+m/rate()));

   return n+m;
}

/*****************************************************************
 * append a data point 
 *****************************************************************/
template<class DataType_t> size_t wavearray<DataType_t>::
append(DataType_t a)
{
   size_t n = this->size();
   this->resize(n+1);
   this->data[n] = a;
   this->Stop += 1./rate();
   return n+1;
}


/*****************************************************************
 * Calculates Fourier Transform for real signal using
 * Fast Fourier Transform (FFT) algorithm. Packs resulting 
 * complex data into original array of real numbers.
 * Calls wavefft() function which is capable to deal with any
 * number of data points. FFT(1) means forward transformation,
 * which is default, FFT(-1) means inverse transformation.
 *****************************************************************/
template<class DataType_t> 
void wavearray<DataType_t>::FFT(int direction) 
{ 
  double *a, *b;
  int N = size();
  int n2 = N/2;

  a=new double[N];
  b=new double[N];

  switch (direction)
  { case 1:

// direct transform
    
    for (int i=0; i<N; i++) { a[i] = data[i]; b[i]=0.;}

    wavefft(a, b, N, N, N, -1);

// pack complex numbers to real array
    for (int i=0; i<n2; i++)
    { data[2*i]   = (DataType_t)a[i]/N;
      data[2*i+1] = (DataType_t)b[i]/N; 
    }

// data[1] is not occupied because imag(F[0]) is always zero and we
// store in data[1] the value of F[N/2] which is pure real for even "N"
      data[1] = (DataType_t)a[n2]/N;

// in the case of odd number of data points we store imag(F[N/2])
// in the last element of array data[N]
      if ((N&1) == 1) data[N-1] = (DataType_t)b[n2]/N;

      break;

    case -1:
// inverse transform

// unpack complex numbers from real array
      for (int i=1;i<n2;i++)
      { a[i]=data[2*i];
        b[i]=data[2*i+1];
        a[N-i]=data[2*i];
        b[N-i]=-data[2*i+1];
       }

      a[0]=data[0];
      b[0]=0.;

      if ((N&1) == 1)
        { a[n2]=data[1]; b[n2]=data[N-1]; }  // for odd n
      else
        { a[n2]=data[1]; b[n2]=0.; }

      wavefft(a, b, N, N, N, 1);             // call FFT for inverse tranform

      for (int i=0; i<N; i++)  
	 data[i] = (DataType_t)a[i];         // copy the result from array "a"
  }

  delete [] b;
  delete [] a;
}

template<class DataType_t> 
void wavearray<DataType_t>::FFTW(int direction) 
{ 
  double *a, *b;
  int N = size();
  int n2 = N/2;

  a=new double[N];
  b=new double[N];

  int sign=0,kind=0;  // fftw dummy parameters 

  switch (direction)
  { case 1:

      if(fftw==NULL) {
        fftw = new TFFTRealComplex(1,&N,false);
        fftw->Init("LS",sign,&kind);          // EX - optimized, ES - least optimized
      }

// direct transform
    
      for (int i=0; i<N; i++) { a[i] = data[i]; b[i]=0.;}

      fftw->SetPoints(a);
      fftw->Transform();
      fftw->GetPointsComplex(a, b);

// pack complex numbers to real array
      for (int i=0; i<n2; i++)
      { data[2*i]   = (DataType_t)a[i]/N;
        data[2*i+1] = (DataType_t)b[i]/N; 
      }

// data[1] is not occupied because imag(F[0]) is always zero and we
// store in data[1] the value of F[N/2] which is pure real for even "N"
      data[1] = (DataType_t)a[n2]/N;

// in the case of odd number of data points we store imag(F[N/2])
// in the last element of array data[N]
      if ((N&1) == 1) data[N-1] = (DataType_t)b[n2]/N;

      break;

    case -1:
// inverse transform

      if(ifftw==NULL) {
        ifftw = new TFFTComplexReal(1,&N,false);
        ifftw->Init("LS",sign,&kind);          // EX - optimized, ES - least optimized
      }

// unpack complex numbers from real array
      for (int i=1;i<n2;i++)
      { a[i]=data[2*i];
        b[i]=data[2*i+1];
        a[N-i]=data[2*i];
        b[N-i]=-data[2*i+1];
      }

      a[0]=data[0];
      b[0]=0.;

      if ((N&1) == 1)
        { a[n2]=data[1]; b[n2]=data[N-1]; }  // for odd n
      else
        { a[n2]=data[1]; b[n2]=0.; }

      ifftw->SetPointsComplex(a,b);
      ifftw->Transform();
      ifftw->GetPoints(a);

      for (int i=0; i<N; i++)  
	 data[i] = (DataType_t)a[i];         // copy the result from array "a"
  }

  delete [] b;
  delete [] a;
}

// Release FFTW memory
template<class DataType_t>
void wavearray<DataType_t>::resetFFTW()
{ 
   if(fftw)  {delete fftw;fftw=NULL;} 
   if(ifftw) {delete ifftw;ifftw=NULL;}
}

/*************************************************************************
 * Stack generates wavearray *this by stacking data from wavearray td.
 * Input data are devided on subsets with with samples "length"
 * then they are added together. Average over the all subsets is saved in
 * *this. 
 *************************************************************************/
template<class DataType_t> 
double wavearray<DataType_t>::
Stack(const wavearray<DataType_t> &td, int length)
{
  double ss, s0, s2;
  rate(td.rate());
  int k = td.size()/length;
  int n = k*length;

  if (k == 0) {
    cout <<" Stack() error: data length too short to contain \n"
         << length << " samples\n";
    return 0.;
  }

  if (size() != (unsigned int)length) resize(length);

// sum (stack) all k periods of frequency f to produce 1 cycle
  s0 = 0.;
  for (int i = 0; i < length; i++) {
    ss = 0.;
    for (int j = i; j < n; j += length) ss += td.data[j];
    data[i] = (DataType_t)ss/k;
    s0 += ss;
  }
  s0 /= (k*length);

// remove constant displacement (zero frequency component) 
  s2 = 0.;
  for (int i = 0; i < length; i++) {
    data[i] -= (DataType_t)s0;
    s2 += data[i]*data[i];
  }
   s2 /= length;

   return s2;        // return stacked signal power (energy/N)
}

/*************************************************************************
 * Another version of Stack:
 * Input data (starting from sample "start") are devided on "k" subsets 
 * with sections "length"
 * Average over the all sections is saved in *this. 
 *************************************************************************/
template<class DataType_t> 
double wavearray<DataType_t>::
Stack(const wavearray<DataType_t> &td, int length, int start)
{
  double avr, rms;
  rate(td.rate());

  if(start+length > (int)td.size()) length = td.size()-start;

  int k = (size()<1) ? 0 : length/size();
  if (k == 0) {
    cout <<" Stack() error: data length too short to contain \n"
         << length << " samples\n";
    return 0.;
  }

  *this = 0;
  for (int i = 0; i<k; i++) add(td, size(), start+i*size());
  *this *= DataType_t(1./k);
  getStatistics(avr,rms);                                
  *this -= DataType_t(avr);

  return rms*rms;        // return stacked signal power
}

// Stack generates wavearray *this by stacking data from wavearray td.
// Input data are devided on subsets with with samples "length"
// then they are added together. Average over the all subsets is saved in
// *this.
template<class DataType_t> 
double wavearray<DataType_t>::
Stack(const wavearray<DataType_t> &td, double window)
{ 
   return this->Stack(td, int(td.rate() * window)); 
}

// wavearray mean
template<class DataType_t>
double wavearray<DataType_t>::mean() const 
{
   size_t i;
   double x = 0;

   if(!size()) return 0.;

   for(i=0; i<size(); i++) x += data[i];
   return x/size();
}

template<class DataType_t>
double wavearray<DataType_t>::mean(double f)
{
// return f percentile mean for data 
// param - f>0 positive side pecentile, f<0 - double-side percentile

   if(!size()) return 0;
   double ff = fabs(f);
   size_t nn = size_t(this->Edge*rate());
   if(ff > 1) ff = 1.;
   if(!nn || 2*nn>=size()-2) {return mean();}
   if(ff==0.) return median(nn,size()-nn-1);

   size_t i;
   size_t N  = size()-2*nn;
   size_t mL = f<0. ? (N-int(N*ff))/2 : 0;               // left boundary
   size_t mR = f<0. ? (N+int(N*ff))/2-1 : int(N*ff)-1;   // right boundary

   double x = 0.;

   wavearray<DataType_t> wtmp = *this;
   if(f>0) wtmp *= wtmp;
   DataType_t *p = wtmp.data;
   DataType_t **pp = (DataType_t **)malloc(N*sizeof(DataType_t*));
   for(i=nn; i<N+nn; i++) pp[i-nn] = p + i;

   if(mL>0)   waveSplit(pp,0,N-1,mL);              // split on left side
   if(mR<N-2) waveSplit(pp,0,N-1,mR);              // split on right side
   for(i=mL; i<=mR; i++) x += this->data[pp[i]-p];

   free(pp);
   return x/(mR-mL+1.);
}

template<class DataType_t>
double wavearray<DataType_t>::mean(const std::slice &s)
{
   size_t i;
   double x = 0.;
   DataType_t *p = data + s.start();
   size_t N = s.size();
   size_t m = (s.stride()<=0) ? 1 : s.stride();
   if(size()<limit(s)) N = (limit(s) - s.start() - 1)/m;
   for(i=0; i<N; i++) { x += *p; p += m; }
   return (N==0) ? 0. : x/N;
}

// running mean
template<class DataType_t>
void wavearray<DataType_t>::mean(double t, 
				 wavearray<DataType_t> *pm,
                                 bool clean, 
				 size_t skip)
{
// calculate running mean of data (x) with window of t seconds
// put result in input array *pm if specified
// subtract median from x if clean=true otherwise replace x with median
// move running window by skip samples
   
   DataType_t* p=NULL;
   DataType_t* q=NULL;
   DataType_t* xx;
   double sum = 0.;
   
   size_t i,last;
   size_t step = Slice.stride();
   size_t N = Slice.size();            // number of samples in wavearray
   size_t n = size_t(t*rate()/step);   // # of samples in the window
   
   if(n<4) {
      cout<<"wavearray<DataType_t>::mean() short time window"<<endl;
      return;
   }   

   if(n&1) n--;                        // # of samples in the window - 1

   size_t nM = n/2;                    // index of median sample
   size_t nL = N-nM-1;

   if(pm){
      pm->resize(N/skip);
      pm->start(start());
      pm->rate(rate());
   }

   xx = (DataType_t  *)malloc((n+1)*sizeof(DataType_t));

   p = data+Slice.start();
   q = data+Slice.start();
   for(i=0; i<=n; i++) { 
      xx[i] = *p; 
      sum += xx[i]; 
      p += step;
   }
   last = 0;

   for(i=0; i<N; i++){

      if(pm) {
	    pm->data[i/skip]  = DataType_t(sum/(n+1.));
	 if(clean) q[i*step] -= DataType_t(sum/(n+1.));
      }
      else {
	 if(clean) q[i*step] -= DataType_t(sum/(n+1.));
	 else      q[i*step]  = DataType_t(sum/(n+1.));
      }

      if(i>=nM && i<nL) {              // copy next sample into last
	 sum -= xx[last]; 
	 sum += *p; 
	 xx[last++] = *p; 
	 p += step;
      }

      if(last>n) last = 0;
      
   } 

   free(xx);
   return;
}


template<class DataType_t>
double wavearray<DataType_t>::rms() 
{
   size_t i;
   double x = 0.;
   double y = 0.;
   size_t N = (size()>>2)<<2;
   DataType_t *p = data + size() - N;

   if(!size()) return 0.;

   for(i=0; i<size()-N; i++) { x += data[i]; y += data[i]*data[i]; }
   for(i=0; i<N; i+=4){ 
      x += p[i] + p[i+1] + p[i+2] + p[i+3];
      y += p[i]*p[i] + p[i+1]*p[i+1] + p[i+2]*p[i+2] + p[i+3]*p[i+3];
   }
   x /= size();
   return sqrt(y/size()-x*x);
}

// running 50% percentile rms
template<class DataType_t>
void wavearray<DataType_t>::rms(double t, 
				wavearray<DataType_t> *pm, 
				bool clean,
				size_t skip)
{
   
   DataType_t*  p=NULL;
   DataType_t*  q=NULL;
   DataType_t** pp;
   DataType_t*  xx;
   DataType_t rm = 1;
   
   size_t i,last;
   size_t step = Slice.stride();
   size_t N = Slice.size();            // number of samples in wavearray
   size_t n = size_t(t*rate()/step);
   
   if(n<4) {
      cout<<"wavearray<DataType_t>::median() short time window"<<endl;
      return;
   }   

   if(n&1) n--;                        // # of samples - 1

   size_t nM = n/2;                    // index of median sample
   size_t nL = N-nM-1;

   if(pm){
      pm->resize(N/skip);
      pm->start(start());
      pm->rate(rate());
   }

   pp = (DataType_t **)malloc((n+1)*sizeof(DataType_t*));
   xx = (DataType_t  *)malloc((n+1)*sizeof(DataType_t));

   p = data+Slice.start();
   q = data+Slice.start();
   for(i=0; i<=n; i++) { 
      xx[i] = *p>0 ? *p : -(*p); 
      pp[i] = xx+i; 
      p += step;
   }
   last = 0;

   for(i=0; i<N; i++){

      if(i==(i/skip)*skip) {
	 waveSplit(pp,0,n,nM);   // median split
	 rm=*pp[nM];
      }

      if(pm) {
	     pm->data[i/skip] = DataType_t(rm/0.6745);
	 if(clean) q[i*step] *= DataType_t(0.6745/rm);
      }
      else {
	 if(clean) q[i*step] *= DataType_t(0.6745/rm);
	 else      q[i*step]  = DataType_t(rm/0.6745);
      }

      if(i>=nM && i<nL) {              // copy next sample into last
	 xx[last++] = *p>0 ? *p : -(*p); 
	 p += step;
      }

      if(last>n) last = 0;
      
   } 

   
   free(pp);
   free(xx);
   return;
}


template<class DataType_t>
double wavearray<DataType_t>::rms(const std::slice &s) 
{
   size_t i;
   double a = 0.;
   double x = 0.;
   double y = 0.;
   DataType_t *p = data + s.start();
   size_t n = s.size();
   size_t m = (s.stride()<=0) ? 1 : s.stride();

   if(size()<limit(s)) n = (limit(s) - s.start() - 1)/m;
   if(!n) return 0.;
   size_t N = (n>>2)<<2; 

   for(i=0; i<n-N; i++) {
      a = *p; x += a; y += a*a; p += m;
   }

   for(i=0; i<N; i+=4) { 
      a = *p; x += a; y += a*a; p += m; 
      a = *p; x += a; y += a*a; p += m; 
      a = *p; x += a; y += a*a; p += m; 
      a = *p; x += a; y += a*a; p += m; 
   }
   x /= N;
   return sqrt(y/N-x*x);
}

template<class DataType_t>
DataType_t wavearray<DataType_t>::max() const
{
   if(!size()) return 0;
   unsigned int i;
   DataType_t x = data[0];
   for(i=1; i<size(); i++) { if(x<data[i]) x=data[i]; }
   return x;
}

template<class DataType_t>
void wavearray<DataType_t>::max(wavearray<DataType_t> &x)
{
   if(!size() ||
      Slice.size()!=x.Slice.size() ||
      Slice.start()!=x.Slice.start() ||
      Slice.stride()!=x.Slice.stride()) {
      cout<<"wavearray::max(): illegal imput array\n";
      return;
   }

   size_t n;  
   size_t K = Slice.stride();
   size_t I = Slice.start();
   size_t N = Slice.size();

   DataType_t* pin = x.data+I;
   DataType_t* pou = this->data+I;
   for(n=0; n<N; n+=K) { 
      if(pou[n]<pin[n]) pou[n]=pin[n]; 
   }
   return;
}

template<class DataType_t>
DataType_t wavearray<DataType_t>::min() const
{
   if(!size()) return 0;
   size_t i;
   DataType_t x = data[0];
   for(i=1; i<size(); i++) { if(x>data[i]) x=data[i]; }
   return x;
}


template<class DataType_t>
int wavearray<DataType_t>::getSampleRank(size_t n, size_t l, size_t r) const
{
   DataType_t v;
   int i = l-1;           // save left boundary 
   int j = r;             // save right boundary 
   
   v = data[n];                    // pivot
   data[n]=data[r]; data[r]=v;     // store pivot
   
   while(i<j)
   {
      while(data[++i]<v && i<j);
      while(data[--j]>v && i<j);
   }
   data[r]=data[n]; data[n]=v;     // put pivot back
   
   return i-int(l)+1;              // rank ranges from 1 to r-l+1
}

template<class DataType_t>
int wavearray<DataType_t>::getSampleRankE(size_t n, size_t l, size_t r) const
{
   DataType_t v,vv;
   int i = l-1;           // save left boundary 
   int j = r;             // save right boundary 
   
   v = data[n];                    // pivot
   data[n]=data[r]; data[r]=v;     // store pivot
   vv = v>0 ? v : -v;              // sort absolute value   

   while(i<j)
   {
      while((data[++i]>0 ? data[i] : -data[i])<vv && i<j);
      while((data[--j]>0 ? data[j] : -data[j])>vv && i<j);
   }
   data[r]=data[n]; data[n]=v;     // put pivot back
   
   return i-int(l)+1;              // rank ranges from 1 to r-l+1
}


template<class DataType_t>
void wavearray<DataType_t>::waveSort(DataType_t** pin, size_t l, size_t r) const
{
// sort data array using quick sorting algorithm between left (l) and right (r) index
// DataType_t** pin is pointer to array of data pointers
// sorted from min to max: *pin[l]/*pin[r] points to min/max 

   size_t k;

   DataType_t v;
   DataType_t* p;
   DataType_t** pp = pin;

   if(pp==NULL || !this->size()) return;  
   if(!r) r = this->size()-1;
   if(l>=r) return;  

   size_t i = (r+l)/2;         // median
   size_t j = r-1;             // pivot storage index

// sort l,i,r
   if(*pp[l] > *pp[i]) {p=pp[l]; pp[l]=pp[i]; pp[i]=p;}
   if(*pp[l] > *pp[r]) {p=pp[l]; pp[l]=pp[r]; pp[r]=p;}
   if(*pp[i] > *pp[r]) {p=pp[i]; pp[i]=pp[r]; pp[r]=p;}
   if(r-l < 3) return;                  // all sorted
   
   v = *pp[i];                          // pivot
   p=pp[i]; pp[i]=pp[j]; pp[j]=p;       // store pivot
   i = l;
   
   for(;;)
   {
      while(*pp[++i] < v);
      while(*pp[--j] > v);
      if(j<i) break;
      p=pp[i]; pp[i]=pp[j]; pp[j]=p;    // swap i,j
   }
   
   p=pp[i]; pp[i++]=pp[r-1]; pp[r-1]=p; // return pivot back
   
   if(j-l > 2) waveSort(pp,l,j);
   else if(j>l){                        // sort l,k,j
      k = l+1;
      if(*pp[l] > *pp[k]) {p=pp[l]; pp[l]=pp[k]; pp[k]=p;}
      if(*pp[l] > *pp[j]) {p=pp[l]; pp[l]=pp[j]; pp[j]=p;}
      if(*pp[k] > *pp[j]) {p=pp[k]; pp[k]=pp[j]; pp[j]=p;}
   }

   if(r-i > 2) waveSort(pp,i,r);
   else if(r>i){                        // sort i,k,r
      k = i+1;
      if(*pp[i] > *pp[k]) {p=pp[i]; pp[i]=pp[k]; pp[k]=p;} 
      if(*pp[i] > *pp[r]) {p=pp[i]; pp[i]=pp[r]; pp[r]=p;}
      if(*pp[k] > *pp[r]) {p=pp[k]; pp[k]=pp[r]; pp[r]=p;}
   }

   return;
}

template<class DataType_t>
void wavearray<DataType_t>::waveSort(size_t l, size_t r)
{
// sort wavearray using quick sorting algorithm between left (l) and right (r) index
// sorted from min to max: this->data[l]/this->data[r] is min/max 
   size_t N = this->size();
   DataType_t  *pd = (DataType_t  *)malloc(N*sizeof(DataType_t));
   DataType_t **pp = (DataType_t **)malloc(N*sizeof(DataType_t*));
   for(size_t i=0; i<N; i++) {
      pd[i] = this->data[i];
      pp[i] = pd + i;
   }   
   waveSort(pp,l,r);
   for(size_t i=0; i<N; i++) this->data[i] = *pp[i];

   free(pd);
   free(pp);
}

template<class DataType_t>
void wavearray<DataType_t>::waveSplit(DataType_t** pp, size_t l, size_t r, size_t m) const
{
// split input array of pointers pp[i] between l <= i <= r so that:
// *pp[i] < *pp[m] if i < m
// *pp[i] > *pp[m] if i > m  
   DataType_t v;
   DataType_t* p;
   size_t i = (r+l)/2;
   size_t j = r-1;

// sort l,i,r
   if(*pp[l] > *pp[i]) {p=pp[l]; pp[l]=pp[i]; pp[i]=p;}
   if(*pp[l] > *pp[r]) {p=pp[l]; pp[l]=pp[r]; pp[r]=p;}
   if(*pp[i] > *pp[r]) {p=pp[i]; pp[i]=pp[r]; pp[r]=p;}
   if(r-l < 3) return;                // all sorted
   
   v = *pp[i];                        // pivot
   p=pp[i]; pp[i]=pp[j]; pp[j]=p;     // store pivot
   i = l;
   
   for(;;)
   {
      while(*pp[++i] < v);
      while(*pp[--j] > v);
      if(j<i) break;
      p=pp[i]; pp[i]=pp[j]; pp[j]=p;  // swap i,j
   }
   p=pp[i]; pp[i]=pp[r-1]; pp[r-1]=p; // put pivot  
   
        if(i > m) waveSplit(pp,l,i,m);
   else if(i < m) waveSplit(pp,i,r,m);

   return;
}

template<class DataType_t>
DataType_t wavearray<DataType_t>::waveSplit(size_t l, size_t r, size_t m)
{
// split wavearray between l and r & at index m so that:
// *this->data[i] < *this->datap[m] if i < m
// *this->data[i] > *this->datap[m] if i > m
   size_t N = this->size();
   DataType_t  *pd = (DataType_t  *)malloc(N*sizeof(DataType_t));
   DataType_t **pp = (DataType_t **)malloc(N*sizeof(DataType_t*));
   for(size_t i=0; i<N; i++) {
      pd[i] = this->data[i];
      pp[i] = pd + i;
   }   
   waveSplit(pp,l,r,m);
   for(size_t i=0; i<N; i++) this->data[i] = *pp[i];

   free(pd);
   free(pp);
   return this->data[m];
}

template<class DataType_t>
void wavearray<DataType_t>::waveSplit(size_t m)
{
// split wavearray at index m so that:
// *this->data[i] < *this->datap[m] if i < m
// *this->data[i] > *this->datap[m] if i > m
   size_t N = this->size();
   DataType_t  *pd = (DataType_t  *)malloc(N*sizeof(DataType_t));
   DataType_t **pp = (DataType_t **)malloc(N*sizeof(DataType_t*));
   for(size_t i=0; i<N; i++) {
      pd[i] = this->data[i];
      pp[i] = pd + i;
   }   
   waveSplit(pp,0,N-1,m);
   for(size_t i=0; i<N; i++) this->data[i] = *pp[i];

   free(pd);
   free(pp);
}

template<class DataType_t>
double wavearray<DataType_t>::median(size_t l, size_t r) const
{
   if(!r) r = size()-1;
   if(r<=l) return 0.;

   size_t i;
   size_t N = r-l+1;
   size_t m = N/2+(N&1);  // median

   double x = 0.;

   DataType_t **pp = (DataType_t **)malloc(N*sizeof(DataType_t*));
   for(i=l; i<=r; i++) pp[i-l] = data + i;
   
   waveSplit(pp,0,N-1,m);
   x = *pp[m];

   free(pp);
   return x;
}


template<class DataType_t>
void wavearray<DataType_t>::median(double t, 
				   wavearray<DataType_t> *pm, 
				   bool clean,
				   size_t skip)
{
// calculate running mean of data (x) with window of t seconds
// put result in input array pm if specified
// subtract median from x if clean=true otherwise replace x with median
// move running window by skip samples
   
   DataType_t*  p=NULL;
   DataType_t*  q=NULL;
   DataType_t** pp;
   DataType_t*  xx;
   DataType_t   am=0;
   
   size_t i,last;
   size_t step = Slice.stride();
   size_t N = Slice.size();            // number of samples in wavearray
   size_t n = size_t(t*rate()/step);   // # of samples in running window
   
   if(n<4) {
      cout<<"wavearray<DataType_t>::median() short time window"<<endl;
      return;
   }   

   if(n&1) n--;                        // # of samples - 1

   size_t nM = n/2;                    // index of median sample
   size_t nL = N-nM-1;

   if(pm){
      pm->resize(N/skip);
      pm->start(start());
      pm->rate(rate()/skip);
   }

   pp = (DataType_t **)malloc((n+1)*sizeof(DataType_t*));
   xx = (DataType_t  *)malloc((n+1)*sizeof(DataType_t));

   p = data+Slice.start();
   q = data+Slice.start();
   for(i=0; i<=n; i++) { 
      xx[i] = *p; 
      pp[i] = xx+i; 
      p += step;
   }
   last = 0;

   for(i=0; i<N; i++){

      if(i==(i/skip)*skip) {
	 waveSplit(pp,0,n,nM);      // median split
	 am=*pp[nM];
      }   

      if(pm) {
	     pm->data[i/skip] = am;
	 if(clean) q[i*step] -= am;
      }
      else {
	 if(clean) q[i*step] -= am;
	 else      q[i*step]  = am;
      }

      if(i>=nM && i<nL) {              // copy next sample into last
	 xx[last++] = *p; 
	 p += step;
      }

      if(last>n) last = 0;
      
   } 

   free(pp);
   free(xx);
   return;
}


template<class DataType_t>
void wavearray<DataType_t>::exponential(double t) 
{
   
   DataType_t*   p=NULL;
   DataType_t*   q=NULL;
   
   size_t i;
   double r;
   size_t last,next;   
   size_t N = Slice.size();            // number of samples in wavearray
   size_t step = Slice.stride();
   size_t n = size_t(t*rate()/step);
   
   if(n<4) {
      cout<<"wavearray<DataType_t>::median() short time window"<<endl;
      return;
   }   

   if(n&1) n--;                        // # of samples in running window

   size_t nM = n/2;                    // index of median sample
   size_t nL = N-nM-1;

   DataType_t** pp = (DataType_t **)malloc((n+1)*sizeof(DataType_t*));
   wavearray<DataType_t> xx(n+1);

   p = data+Slice.start();
   q = data+Slice.start();
   for(i=0; i<=n; i++) { 
      xx.data[i] = *p; 
      pp[i] = xx.data+i;
      p += step;
   }
   last = 0;
   next = 0;

   for(i=0; i<N; i++){

      r = (xx.getSampleRank(next,0,n)-double(nM)-1.)/(nM+1.);
      q[i*step]  = DataType_t(r>0. ? -log(1.-r) : log(1.+r));

      if(i>=nM && i<nL) {              // copy next sample into last
	 xx.data[last++] = *p; p+=step; 
      }
      
      next++;  
      if(next>n) next = 0;
      if(last>n) last = 0;
    
   } 

   free(pp);
   return;
}


template<class DataType_t>
DataType_t wavearray<DataType_t>::rank(double f) const
{
   int i;
   int n = size();
   DataType_t out = 0;
   DataType_t **pp = NULL;
   if(f<0.) f = 0.;
   if(f>1.) f = 1.;

   if(n)
      pp = (DataType_t **)malloc(n*sizeof(DataType_t*));
   else
      return out;

   for(i=0; i<n; i++) pp[i] = data + i;
   qsort(pp, n, sizeof(DataType_t *), 
         (qsort_func)&wavearray<DataType_t>::compare);

   i =int((1.-f)*n);
   if(i==0) out = *(pp[0]);
   else if(i>=n-1) out = *(pp[n-1]);
   else out = (*(pp[i])+*(pp[i+1]))/2;

   for(i=0; i<n; i++) *(pp[i]) = n-i;

   free(pp);
   return out;
}


// symmetric prediction error signal produced with 
// adaptive split lattice algoritm
template<class DataType_t>
void wavearray<DataType_t>::spesla(double T, double w, double oFFset)
{
   int k,j;
   double p,q,xp,xm;

   int K = int(rate()*T+0.5);              // filter duration
   int M = int(rate()*w+0.5);              // integration window
   if(M&1) M++;                            // make M even
   int m = M/2;

   int offset = int(oFFset*rate());        // data offset
   int N  = size();                        // data size
   int nf = offset+m;
   int nl = N-nf;

   wavearray<DataType_t> x0;
   wavearray<DataType_t> x1;

   x0 = *this;                   // previous X iteration
   x1 = *this; 
   x1.add(x0,x0.size()-1,0,1);   // current  X iteration
   x1 *= DataType_t(0.5);                   

//   cout<<nf<<" "<<nl<<" "<<offset<<" "<<K<<" "<<M<<" "<<x0.rms()<<" "<<x1.rms();

   for(k=1; k<K; k++) {

      p = q = 0.;
      for(j=offset; j<offset+M; j++) {  
	 p += x1.data[j]*x1.data[j];
	 q += x0.data[j]*x0.data[j];
      }

      for(j=1; j<N; j++) {
	 if(j>=nf && j<nl) {      // update p & q
	    xp = x1.data[j+m];
	    xm = x1.data[j-m];
	    p += (xp-xm)*(xp+xm);
	    xp = x0.data[j+m];
	    xm = x0.data[j-m];
	    q += (xp-xm)*(xp+xm);
	 }
	 xp = k==1 ? 2*p/q : p/q;
	 data[j]  = x1.data[j]+x1.data[j-1]-DataType_t(x0.data[j-1]*xp);
      }
      
      x0 = x1;
      x1 = *this;

   }      
   return;
}


// apply filter
template<class DataType_t>
void wavearray<DataType_t>::lprFilter(wavearray<double>& w)
{
   int i,j;
   int N = size();
   int m = w.size();
   wavearray<DataType_t> x;
   x = *this;

   for(i=0; i<N; i++) {
      for(j=1; j<m && (i-j)>=0; j++) {
	 data[i] += DataType_t(w.data[j]*x.data[i-j]);
      }
   }
   return;
}



// calculate and apply lpr filter
template<class DataType_t>
void wavearray<DataType_t>::lprFilter(double T, int mode, double stride,
				      double oFFset, int tail)
{
   int i,j,l,n,nf,nl;
   double duration = stop()-start()-2*oFFset;
   double a=0.;
   double b=1.;

   int N = size();
   int M = int(rate()*T+0.5);              // filter duration
   int k = stride>0 ? int(duration/stride) : 1;   // number of intervals

   if(!k) k++;
   int m = int((N-2.*oFFset*rate())/k);
   if(m&1) m--;                            // make m even

   int offset = int(oFFset*rate()+0.5);    // data offset
   if(offset&1) offset++;

   wavearray<DataType_t> w(m);
   wavearray<DataType_t> x = *this;
   wavearray<double> f;
   w.rate(rate());

   for(l=0; l<k; l++) {

      n = l*m+(N-k*m)/2;
      w.cpf(x,m,n);
      f = w.getLPRFilter(M,0,tail);
      
      nf = l==0   ? 0 : n;
      nl = l==k-1 ? N : n+m;
      n  = offset+M;

      if(mode == 0 || mode == 1) {            // forward LP
	 for(i=nf; i<nl; i++) {            
	    if(i < n) continue; 
	    b = (!mode && i<N-n) ? 0.5 : 1.;  // symmetric LP 
	    for(j=1; j<M; j++) {
	       a = double(f.data[j]*x.data[i-j]);
	       data[i] += DataType_t(a*b);
	    }
	 }
      }

      if(mode == 0 || mode == -1) {           // backward LP
	 for(i=nf; i<nl; i++) {
	    if(i >= N-n) continue; 
	    b = (!mode && i>=n) ? 0.5 : 1.;   // symmetric LP 
	    for(j=1; j<M; j++) {
	       a = double(f.data[j]*x.data[i+j]);
	       data[i] += DataType_t(a*b);  
	    }
	 }
      }
      
   }

   return;
}

//**************************************************************
// calculate autocorrelation function and
// solve symmetric Yule-Walker problem
//**************************************************************
template<class DataType_t>
wavearray<double> wavearray<DataType_t>::getLPRFilter(int M, int offset, int tail)
{
  // tail=-1 exclude left tail
  // tail= 0 exclude left and right tail
  // tail= 1 exclude right tail
  int i,m;
  double f  = tail!=0 ? 0.06 : 0.03;   // exclude tail
  int  N = size()-2*offset;
  int LL = int(f*N+0.5);               // excluded tail
  int nL = tail<1 ? LL : 1;            // left percentile
  int nR = tail>-1 ? N-LL-1 : 1;       // right percentile
  int nn = N/2;                        // median
  int n  = size()-offset;

  if(size()<=offset) {
     cout<<"wavearray<DataType_t>::getLPRFilter() invalid input parameters\n";
     wavearray<double> a(1);
     return a;
  }

  wavearray<double> r(M);
  wavearray<double> a(M);
  wavearray<double> b(M);

  wavearray<DataType_t> x = *this;

  DataType_t ** pp = (DataType_t **)malloc(N*sizeof(DataType_t*));

  for(i=offset; i<n; i++) pp[i-offset] = x.data + i;

  waveSplit(pp,0,N-1,nn);
  waveSplit(pp,0,nn,nL);
  waveSplit(pp,nn,N-1,nR);

  x -= *pp[nn];                      // subtract median 
  for(i=0;  i<nL; i++) *pp[i] = 0;   // exclude tails
  for(i=nR; i<N;  i++) *pp[i] = 0;   // exclude tails

// autocorrelation

  offset += M;
  n = size()-offset;

  for (m=0; m<M; m++) {
    r.data[m] = 0.;
    for (i=offset; i<n; i++) {
      r.data[m] += x.data[i]*(x.data[i-m] + x.data[i+m])/2.;
    }
  }


// Levinson algorithm: P.Delsarte & Y.Genin, IEEE, ASSP-35, #5 May 1987

  double p,q;
  double s = r.data[0];

  a = 0.; a.data[0] = 1.; 

  for (m=1; m<M; m++) {

    q = 0.;
    for (i=0; i<m; i++) q -= a.data[i] * r.data[m-i];

    p  = q/s;            // reflection coefficient
    s -= q*p;

    for (i=1; i<=m; i++) b.data[i] = a.data[i] + p*a.data[m-i];
    for (i=1; i<=m; i++) a.data[i] = b.data[i];

  }

/*
  double tmp, num, den;
  M--;

  a.data[1] = - r.data[1] / r.data[0];

  for (m=1; m<M; m++) {

    num = r.data[m + 1];
    den = r.data[0];

    for (i=1; i<=m; i++) {
      num += a.data[i] * r.data[m+1-i];
      den += a.data[i] * r.data[i];
    }

    a.data[m+1] = - num / den;

    for (i=1; i <= ((m+1)>>1); i++) {
      tmp = a.data[i] + a.data[m+1] * a.data[m+1-i];
      a.data[m+1-i] = a.data[m+1-i] + a.data[m+1] * a.data[i];
      a.data[i] = tmp;
    }

  }
  a.data[0] = 1;
*/

  free(pp);
  return a;
}


//**************************************************************
// normalize data by noise variance
// offset  | stride |      
// |*****|*X********X********X********X********X********X*|*****|
//         ^              ^                        ^
//     measurement        |<-      interval      ->|
//**************************************************************
template<class DataType_t>
wavearray<double> 
wavearray<DataType_t>::white(double t, int mode, double oFFset, double stride) const
{
   int i,j;
   int N = size();

   double segT = size()/rate();            // segment duration
   if(t<=0.) t = segT-2.*oFFset;
   int  offset = int(oFFset*rate()+0.5);   // offset in samples
   if(offset&1) offset--;                  // make offset even

   if(stride > t || stride<=0.) stride = t;                  
   int K = int((segT-2.*oFFset)/stride);   // number of noise measurement minus 1
   if(!K) K++;                             // special case   

   int n =  N-2*offset;                    // total number of samples used for noise estimate
   int k =  n/K;                           // shift in samples
   if(k&1) k--;                            // make k even

   int m  = int(t*rate()+0.5);             // number of samples in one interval
   int mm = m/2;                           // median m
   int mL = int(0.15865*m+0.5);            // -sigma index (CL=0.31732)
   int mR = m-mL-1;                        // +sigma index
   int jL = (N-k*K)/2;                     // array index for first measurement
   int jR = N-offset-m;                    // array index for last measurement
   int jj = jL-mm;                         // start of the first interval

   wavearray<double> meDIan(K+1);
   wavearray<double> norm50(K+1);

   meDIan.rate(rate()/k);
   meDIan.start(start()+jL/rate());
   meDIan.stop(start()+(N-offset)/rate());

   norm50.rate(rate()/k);
   norm50.start(start()+jL/rate()); 
   norm50.stop(start()+(N-offset)/rate());

   mode = abs(mode);

   if(m<3 || mL<2 || mR>m-2) {
     cout<<"wavearray::white(): too short input array."<<endl;
     return mode!=0 ? norm50 : meDIan;
   }

   DataType_t *p = data;
   wavearray<DataType_t> w(m);
   double x;
   double r;

   DataType_t ** pp = (DataType_t **)malloc(m*sizeof(DataType_t*));

   for(j=0; j<=K; j++) {

      if(jj < offset)   p  = data + offset;     // left boundary
      else if(jj >= jR) p  = data + jR;         // right boundary
      else  p  = data + jj;
      jj += k;                                  // update jj

      if(p+m>data+N) {cout<<"wavearray::white(): error1\n"; exit(1);}

      for(i=0; i<m; i++) pp[i] = p + i;
      waveSplit(pp,0,m-1,mm);
      waveSplit(pp,0,mm,mL);
      waveSplit(pp,mm,m-1,mR);
      meDIan[j] = mode ? *pp[mm] : sqrt((*pp[mm])*0.7191);
      norm50[j] = (*pp[mR] - *pp[mL])/2.;

   }

   p = data;

   if(mode) {

     mm = jL;
     for(i=0; i<mm; i++){
       x  = double(*p)-meDIan.data[0];
       r  = norm50.data[0];
       *(p++) = mode==1 ? DataType_t(x/r) : DataType_t(x/r/r);
     }
     
     for(j=0; j<K; j++) {
       for(i=0; i<k; i++) {
	 x  = double(*p)-(meDIan.data[j+1]*i + meDIan.data[j]*(k-i))/k;
	 r  = (norm50.data[j+1]*i + norm50.data[j]*(k-i))/k;
	 *(p++) = mode==1 ? DataType_t(x/r) : DataType_t(x/r/r);
       }
     }
     
     mm = (data+N)-p;
     for(i=0; i<mm; i++){
       x  = double(*p)-meDIan.data[K];
       r  = norm50.data[K];
       *(p++) = mode==1 ? DataType_t(x/r) : DataType_t(x/r/r);
     }
     
   }

   free(pp);
   return mode!=0 ? norm50 : meDIan;
}

//: returns mean and root mean square of the signal.
template<class DataType_t>
double wavearray<DataType_t>::getStatistics(double &m, double &r) const
{
   size_t i;
   double a;
   double b;
   DataType_t *p = const_cast<DataType_t *>(data);
   double y = 0.;
   size_t N = size() - 1 + size_t(size()&1);

   if(!size()) return 0.;

   m = p[0];
   r = p[0]*p[0];
   if(N < size()){
      m += p[N];
      r += p[N]*p[N];
      y += p[N]*p[N-1];
   }

   for(i=1; i<N; i+=2) { 
      a = p[i]; b = p[i+1];
      m += a + b; 
      r += a*a + b*b; 
      y += a*(p[i-1]+b);
   }

   N  = size();
   m = m/double(N);
   r = r/double(N) - m*m;

   y  = 4.*(y/N - m*m + m*(p[0]+p[i]-m)/N);
   y /= 4.*r - 2.*((p[0]-m)*(p[0]-m)+(p[i]-m)*(p[i]-m))/N;
   r = sqrt(r);

   a = (fabs(y) < 1) ? sqrt(0.5*(1.-fabs(y))) : 0.;

   return y>0 ? -a : a;
}

template <class DataType_t> 
void wavearray<DataType_t>::Streamer(TBuffer &R__b)
{
   // Stream an object of class wavearray<DataType_t>.
   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      TNamed::Streamer(R__b);
      R__b >> Size;
      R__b >> Rate;
      R__b >> Start;
      R__b >> Stop;
      if(R__v>1) R__b >> Edge;
      R__b.StreamObject(&(Slice),typeid(slice));
      this->resize(Size);
      R__b.ReadFastArray((char*)data,Size*sizeof(DataType_t));
      R__b.CheckByteCount(R__s, R__c, wavearray<DataType_t>::IsA());
   } else {
      R__c = R__b.WriteVersion(wavearray<DataType_t>::IsA(), kTRUE);
      TNamed::Streamer(R__b);
      R__b << Size;
      R__b << Rate;
      R__b << Start;
      R__b << Stop;
      R__b << Edge;
      R__b.StreamObject(&(Slice),typeid(slice));
      R__b.WriteFastArray((char*)data,Size*sizeof(DataType_t));
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//: save to file operator  >>
template <class DataType_t>
char* wavearray<DataType_t>::operator>>(char* fname)
{
  TString name = fname;
  name.ReplaceAll(" ","");
  TObjArray* token = TString(fname).Tokenize(TString("."));
  TObjString* ext_tok = (TObjString*)token->At(token->GetEntries()-1);
  TString ext = ext_tok->GetString();
  if(ext=="dat") {
    //: Dump data array to a binary file
    DumpBinary(fname,0);
  } else
  if(ext=="txt") {
    //: Dump data array to an ASCII file
    Dump(fname,0);
  } else
  if(ext=="root") {
    //: Dump object into root file
    DumpObject(fname);
  } else {
    cout << "wavearray operator (>>) : file type " << ext.Data() << " not supported" << endl;
  }
  return fname;
}

//: save to file operator  >>
template <class DataType_t>
void wavearray<DataType_t>::print()
{
  cout << endl << endl;
  cout.precision(14);
  cout << "Size\t= "  << this->Size  << " samples" << endl;
  cout << "Rate\t= "  << this->Rate  << " Hz" << endl;
  cout << "Start\t= " << this->Start << " sec" << endl;
  cout << "Stop\t= "  << this->Stop  << " sec" << endl;
  cout << "Length\t= "<< this->Stop-this->Start  << " sec" << endl;
  cout << "Edge\t= "  << this->Edge << " sec" << endl;
  cout << "Mean\t= "  << this->mean() << endl;
  cout << "RMS\t= "   << this->rms() << endl;
  cout << endl;

  return;
}

#ifdef _USE_DMT

template<class DataType_t>
wavearray<DataType_t>& wavearray<DataType_t>::
operator=(const TSeries &ts)
{
   Interval ti = ts.getTStep();
   double Tsample = ti.GetSecs();

   unsigned int n=ts.getNSample();
   if(n != size()) resize(n);

   if ( Tsample > 0. )
      rate(double(int(1./Tsample + 0.5)));
   else {
      cout <<" Invalid sampling interval = 0 sec.\n";
   }

   start(double(ts.getStartTime().totalS()));

   TSeries r(ts.getStartTime(), ti, ts.getNSample());
   r = ts;
   float *vec_ref;
   vec_ref= (float*)(r.refData());

   for ( unsigned int i=0; i<n; i++ ) 
      data[i] = (DataType_t) (vec_ref[i]);
   return *this;
}

#endif

// instantiations

#define CLASS_INSTANTIATION(class_) template class wavearray< class_ >;

CLASS_INSTANTIATION(int)
CLASS_INSTANTIATION(short)
CLASS_INSTANTIATION(long)
CLASS_INSTANTIATION(long long)
CLASS_INSTANTIATION(unsigned int)
CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)
//CLASS_INSTANTIATION(std::complex<float>)
//CLASS_INSTANTIATION(std::complex<double>)

#undef CLASS_INSTANTIATION


#if !defined (__SUNPRO_CC)
template wavearray<double>::
wavearray(const double *, unsigned int, double);

template wavearray<double>::
wavearray(const float *, unsigned int, double );

template wavearray<double>::
wavearray(const short *, unsigned int, double );

template wavearray<float>::
wavearray(const double *, unsigned int, double );

template wavearray<float>::
wavearray(const float *, unsigned int, double );

template wavearray<float>::
wavearray(const short *, unsigned int, double );

#else
// FAKE CALLS FOR SUN CC since above instatinations are 
// not recognized
static void fake_instatination_SUN_CC () 
{
   double x;
   float  y;
   short  s;
   wavearray<double> wvdd (&x, 1, 0);
   wavearray<double> wvdf (&y, 1, 0);
   wavearray<double> wvds (&s, 1, 0);
   wavearray<float>  wvfd (&x, 1, 0);
   wavearray<float>  wvff (&y, 1, 0);
   wavearray<float>  wvfs (&s, 1, 0);
}

#endif









