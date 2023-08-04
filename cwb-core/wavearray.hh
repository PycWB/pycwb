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


/**********************************************************
 * Package: 	Wavelet Analysis Tool
 * File name: 	wavearray.h
 **********************************************************/

#ifndef WAVEARRAY_HH
#define WAVEARRAY_HH

#include <iostream>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <valarray>
#include "Wavelet.hh"

#ifdef _USE_DMT
  #include "Time.hh"
  #include "Interval.hh"
  #include "TSeries.hh"
#endif

#include "TFFTComplexReal.h"
#include "TFFTRealComplex.h"
#include "TNamed.h"

template<class DataType_t> 
class wavearray : public TNamed
{

  public:

  wavearray(int);                                // Constructor

  wavearray();                                   // Default constructor

  wavearray(const wavearray<DataType_t>&);       // copy Constructor

// explicit construction from array
  template <class T>
  wavearray(const T *, unsigned int, double=0.); 

  virtual ~wavearray();                          // Destructor

// operators

  wavearray<DataType_t>& operator= (const wavearray<DataType_t> &);

// operator[](const slice &) sets the Slice object of the wavearray class
// the operators above do not use Slice object.
  virtual wavearray<DataType_t>& operator[](const std::slice &);
  virtual DataType_t & operator[](const unsigned int);

// check if two wavearrays overlap
  inline virtual size_t limit() const; 
  inline virtual size_t limit(const std::slice &) const; 
  inline virtual size_t limit(const wavearray<DataType_t> &) const; 

// the Slice object is used in the operators below and set to 
// slice(0,N,1) when an operator has been executed for both arrays

  virtual wavearray<DataType_t>& operator+=(wavearray<DataType_t> &);
  virtual wavearray<DataType_t>& operator-=(wavearray<DataType_t> &);
  virtual wavearray<DataType_t>& operator*=(wavearray<DataType_t> &);
  virtual wavearray<DataType_t>& operator<<(wavearray<DataType_t> &);

          wavearray<DataType_t>& operator= (const DataType_t);
  virtual wavearray<DataType_t>& operator+=(const DataType_t);
  virtual wavearray<DataType_t>& operator-=(const DataType_t);
  virtual wavearray<DataType_t>& operator*=(const DataType_t);

  virtual                 char*  operator>>(char*);

#ifdef _USE_DMT
  virtual wavearray<DataType_t>& operator= (const TSeries &);
#endif

// member functions

  //: Dump data array to an ASCII file 
  virtual void Dump(const char*, int=0);

  //: Dump data array to a binary file 
  virtual void DumpBinary(const char*, int = 0);

  //: Dump data array from a binary file as 16bit words 
  virtual void DumpShort(const char*, int=0);

  //: Dump object array into file root
  virtual void DumpObject(const char*);

  //: Read data array from a binary file 
  virtual void ReadBinary(const char*, int=0);

  //: Read data array from a binary file 
  virtual void ReadShort(const char*);

  virtual void FFT(int = 1);		// fast Fourier transform
  virtual void FFTW(int = 1);		// fast Fourier transform West
  virtual void resetFFTW();		// release FFTW memory

          void Resample(const wavearray<DataType_t> &, double, int=6);
          void resample(const wavearray<DataType_t> &, double, int=6);
  virtual void resample(double, int=6);
  virtual void Resample(double);	// resample with FFTW  

  //: data time shift based on FFTW 
  //: T = time shift (sec) 
  virtual void delay(double T);		

  virtual void   start(double s) {Start=s; if(Rate>0.) Stop=Start+Size/Rate;};
  virtual double start() const   { return Start; };
  virtual void   stop(double s) {Stop = s; };
  virtual double stop() const   { return Stop; };
  virtual void   rate(double r)  {Rate = fabs(r); Stop = Start+Size/Rate; };
  virtual double rate()  const   { return Rate; };
  virtual void   edge(double s) {Edge = s; };
  virtual double edge() const   { return Edge; };
  virtual size_t size() const   { return Size; };
  virtual void   setSlice(const std::slice &s) { Slice=s; };
  virtual std::slice getSlice() const { return Slice; };

  //: return median
  //: for data between index1 and index2 (including) 
  virtual double median(size_t=0, size_t=0) const;

  //: calculate running median of data (x) with window of t seconds (par1)
  //: put result in input array in if specified
  //: subtract median from x if fl=true otherwise replace x with median
  //: move running window by n samples
  virtual void median(double t, wavearray<DataType_t>* in, 
		      bool fl=false, size_t n=1);

  //: return mean
  virtual double mean() const;

  //: return f percentile mean for data 
  // f - >0 positive side pecentile, <0 - double-side percentile
  virtual double mean(double f);

  //: return mean for wavearray slice
  virtual double mean(const std::slice&);

  //: calculate running mean of data (x) with window of t seconds
  //: put result in input array in if specified
  //: subtract mean from x if fl=true otherwise replace x with mean
  //: move running window by n samples
  virtual void mean(double t, wavearray<DataType_t>* in,
		    bool fl=false, size_t n=1);

  //: return sqrt<x^2>
  virtual double rms();

  //: return sqrt<x^2> for wavearray slice
  virtual double rms(const std::slice&);

  //: calculate running rms of data (x) with window of t seconds
  //: put result in input array in if specified
  //: subtract rms from x if fl=true otherwise replace x with rms
  //: move running window by n samples
  virtual void rms(double t, wavearray<DataType_t>* in, 
		   bool fl=false, size_t n=1);

  //: return maximum for wavearray
  virtual DataType_t max() const;

  //: store max elements max(this->data[i],in.data[i]) in this
  //: param: input wavearray
   virtual void max(wavearray<DataType_t> &);

  //: return minimum for wavearray
  virtual DataType_t min() const;

  //: take square root of wavearray elements
  virtual void SQRT() {
      for(size_t i=0; i<this->size(); i++) 
         if(this->data[i]>0.) this->data[i]=(DataType_t)sqrt(double(this->data[i]));
      return;
   }

  // multiply wavearray by Hann window
  inline void hann(void);

  // sort wavearray using quick sorting algorithm
  // return  DataType_t** pp pointer to array of wavearray pointers
  // sorted from min to max
  virtual void waveSort(DataType_t** pp, size_t l=0, size_t r=0) const;
  // sort wavearray using quick sorting algorithm between left (l) and right (r) index
  // sorted from min to max: this->data[l]/this->data[r] is min/max 
  virtual void waveSort(size_t l=0, size_t r=0);

  // split input array of pointers pp[i] between l <= i <= r so that:
  // *pp[i] < *pp[m] if i < m
  // *pp[i] > *pp[m] if i > m  
  virtual void waveSplit(DataType_t** pp, size_t l, size_t r, size_t m) const;
  // split wavearray between l and r & at index m
  virtual DataType_t waveSplit(size_t l, size_t r, size_t m);
  // split wavearray at index m so that:
  // *this->data[i] < *this->datap[m] if i < m
  // *this->data[i] > *this->datap[m] if i > m
  virtual void waveSplit(size_t m);

  //: return rank of sample n ranked against samples between
  //: left (l) and right (r) boundaries 
  //: rank ranges from 1 to r-l+1
  virtual int getSampleRank(size_t n, size_t l, size_t r) const;

  //: rank sample n against absolute value of samples between 
  //: left (l) and right (r) boundaries 
  //: rank ranges from 1 to r-l+1
  virtual int getSampleRankE(size_t n, size_t l, size_t r) const;

  // calculate rank
  // input  - fraction of laudest samples
  // output - min amplitude of laudest samples 
  DataType_t rank(double=0.5) const;

  // Linear Predictor Filter coefficients
  // calculate autocorrelation function excluding 3% tails and
  // solve symmetric Yule-Walker problem
  // param 1: sifter size
  // param 2: boundary offset
   virtual wavearray<double> getLPRFilter(int, int=0, int=0);

  // generate Symmetric Prediction Error with Split Lattice Algorith  
  // param: filter length in seconds
  // param: window for filter training
  // param: boundary offset to account for wavelet artifacts
  virtual void spesla(double,double,double=0.);

  // apply lpr filter defined in input wavearray<double>
  virtual void lprFilter(wavearray<double>&);

  // calculate and apply lpr filter to this 
  // param: filter length in seconds
  // param: filter mode: -1/0/1 - backward/symmetric/forward
  // param: stride for filter training (0 - train on whole TS)
  // param: boundary offset to account for wavelet artifacts
   virtual void lprFilter(double,int=0,double=0.,double=0.,int=0);

  // normalization by 31% percentile amplitude 
  // param 1 - time window dT. if = 0 - dT=T, where T is wavearray duration
  // param 2 - 0 - no whitening, 1 - single whitening, >1 - double whitening
  // param 3 - boundary offset 
  // param 4 - noise sampling interval (window stride)  
  //           the number of measurements is k=int((T-2*offset)/stride)
  //           if stride=0, then stride is set to dT
  // return: noise array if param2>0, median if param2=0
  virtual wavearray<double> white(double, int=0,double=0.,double=0.) const;

  // turn wavearray distribution into exponential using rank statistics 
  // input - time window around a sample to rank the sample
  virtual void exponential(double);

  // get sample with index i from wavearray
  inline DataType_t get(size_t i) { return data[i]; } 
  // get rms sample value  averaged over the window t-dt : t+dt
  inline DataType_t get(double t, double dt=0.);

  inline long uniform(){ return random(); }
  inline long rand48(long k=1024){ return long(k*drand48()); }

  double getStatistics(double &mean, double &rms) const;

  virtual void   resize(unsigned int);

  //: copy, add and subtruct wavearray array from *this 
  void cpf(const wavearray<DataType_t> &, int=0, int=0, int=0); 
  void add(const wavearray<DataType_t> &, int=0, int=0, int=0);
  void sub(const wavearray<DataType_t> &, int=0, int=0, int=0);

  //: append to this *this
  size_t append(const wavearray<DataType_t> &);
  //: append to this *this
  size_t append(DataType_t);

  // count number of pixels above x
   size_t wavecount(double x, int n=0){
     size_t N=0;
     for(int i=n;i<size()-n;i++) if(data[i]>x) N++;
     return N;
  }

  //: cut data on shorter intervals and average
  double Stack(const wavearray<DataType_t> &, int);
  double Stack(const wavearray<DataType_t> &, int, int);
  double Stack(const wavearray<DataType_t> &, double);

  // print wavearray parameters
  void print();             // *MENU*
  virtual void Browse(TBrowser *b) {print();}

  DataType_t *data;             //! data array

//   private:

  size_t Size;                  // number of elements in the data array
  double Rate;		        // data sampling rate
  double Start;		        // start time
  double Stop;		        // end time
  double Edge;		        // buffer length in seconds in the beginning and the end 
  std::slice Slice;             // the data slice structure

  TFFTRealComplex*  fftw;       //! pointer to direct  fftw object
  TFFTComplexReal* ifftw;       //! pointer to inverse fftw object

  inline static int compare(const void *x, const void *y){
     DataType_t a = *(*(DataType_t**)x) - *(*(DataType_t**)y);
     if(a > 0) return 1;
     if(a < 0) return -1;
     return 0;
  }

  ClassDef(wavearray,2)
};

template<class DataType_t>
size_t wavearray<DataType_t>::limit() const 
{ return Slice.start() + (Slice.size()-1)*Slice.stride() + 1; }

template<class DataType_t>
size_t wavearray<DataType_t>::limit(const std::slice &s) const 
{ return s.start() + (s.size()-1)*s.stride() + 1; }

template<class DataType_t>
size_t wavearray<DataType_t>::limit(const wavearray<DataType_t> &a) const 
{
   size_t N = a.Slice.size(); 
   if(N>Slice.size()) N=Slice.size();
   return Slice.start() + (N-1)*Slice.stride() + 1;
}

template<class Tout, class Tin>
inline void waveAssign(wavearray<Tout> &aout, wavearray<Tin> &ain) 
{
   size_t N = ain.size();
   aout.rate(ain.rate()); 
   aout.start(ain.start()); 
   aout.stop(ain.stop()); 
   aout.Slice = std::slice(0,N,1);
   if (N != aout.size()) aout.resize(N);
   for (unsigned int i=0; i < N; i++) aout.data[i] = (Tout)ain.data[i];
}

template<class DataType_t>
inline void wavearray<DataType_t>::hann() 
{
  double phi = 2*3.141592653589793/size();
  double www = sqrt(2./3.);
  int nn = size();
  for(int i=0; i<nn; i++) data[i] *= DataType_t(www*(1.-cos(i*phi)));
}

template<class DataType_t>
inline DataType_t wavearray<DataType_t>::get(double t, double dT) 
{
// dt>0 - average value^2 over the window 2dt around the time t 
  t = t-this->Start;
  double dt = fabs(dT)+0.51/Rate;
  int n = int((t-dt)*Rate);
  if(n<0) n=0; if(n>=Size) n=Size-1;
  int m = int((t+dt)*Rate);
  if(m<0) m=0; if(m>=Size) m=Size-1;
  if(int(t*Rate)<0 || int(t*Rate)>=Size) {
     printf("wavearray<DataType_t>::get(double t): time out of range\n");
     exit(1);
  }
  if(n>=m) return data[size_t(t*Rate)];
  double sum = 0.;
  for(int i=n; i<=m; i++) sum += data[i]*data[i];
  return sqrt(sum/(m-n+1));
}

#endif // WAVEARRAY_HH












