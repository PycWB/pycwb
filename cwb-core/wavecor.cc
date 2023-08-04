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


// ++++++++++++++++++++++++++++++++++++++++++++++
// S. Klimenko, University of Florida
// WAT cross-correlation class
// ++++++++++++++++++++++++++++++++++++++++++++++

#define WAVECOR_CC
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "wavecor.hh"

ClassImp(wavecor)	 // used by THtml doc

using namespace std;

// constructors

wavecor::wavecor()
{
  shift = 0.;
  ifo   = 0;
  run   = 0;
  window= 0.;
  lagint= 0.;
}

wavecor::wavecor(const wavecor& value)
{
   cList.clear();
   cList = value.cList;
   xcor  = value.xcor;
   xlag  = value.xlag;
   shift = value.shift;
   ifo   = value.ifo;
   run   = value.run;
   window= value.window;
   lagint= value.lagint;
}

// destructor

wavecor::~wavecor(){}

//: operator =

wavecor& wavecor::operator=(const wavecor& value)
{
   cList = value.cList;
   xcor  = value.xcor;
   xlag  = value.xlag;
   shift = value.shift;
   ifo   = value.ifo;
   run   = value.run;
   window= value.window;
   lagint= value.lagint;
   return *this;
}



//**************************************************************************
// initialize wavecor from two time series with Kendall x-correlation
//**************************************************************************
void wavecor::kendall(wavearray<double>& a, wavearray<double>& b, 
		      double w, double t, size_t skip)
{
   cList.clear();
   ifo=0; shift=0.;
   window = w;
   lagint = t;

   if(a.start()!=b.start() || a.size()!=b.size() || a.rate()!=b.rate()) {
      cout<<"wavecor::init() invalid arguments"<<endl;
      return;
   }

   int l;
   short sign;
   size_t i,j,k;
   size_t last=0;
   size_t N = a.size();

   double x,y;
   double sum = 0.;
   double* pa=NULL;
   double* pb=NULL;

   size_t n = size_t(fabs(t)*a.rate());      // # of time lags
   n = n/2;                                  // # of time lags on one side
  
   size_t m = size_t(fabs(w)*a.rate());      // # of samples in running window
   if(!(m&1)) m++;                           // # of samples in running window

   short** P = (short **)malloc(m*sizeof(short*));
   for(i=0; i<m; i++) P[i] = (short*)malloc(m*sizeof(short));

   xcor.start(a.start());                 // set start time
   xlag.start(a.start());                 // set start time
   xcor.rate(a.rate()/skip);
   xlag.rate(a.rate()/skip);
   if(xcor.size()!=N/skip) xcor.resize(N/skip);
   if(xlag.size()!=N/skip) xlag.resize(N/skip);

   xcor = 0.;

   for(l=-int(n); l<=int(n); l++){

      pa = l<0 ? a.data-l : a.data;    // a shifted right
      pb = l>0 ? b.data+l : b.data;    // b shifted right

      sum = 0.;
      for(i=0; i<m; i++){
	 x = pa[i];
	 y = pb[i];
	 for(j=i; j<m; j++){
	    sign  = x>pa[j] ? 1 : -1;
	    sign *= y>pb[j] ? 1 : -1;
	    P[i][j] = sign;
	    P[j][i] = sign;
	    sum += 2.*sign;
	 }
	 sum -= 2.;
      }
      
//      cout<<sum<<"  "<<m<<"  "<<n<<endl;
      
      last = 0;
      for(i=0; i<N; i++){

	 k = i/skip;
	 if(i==k*skip) {
	    if(fabs(xcor.data[k]) < fabs(sum)) { 
	       xcor.data[k] = sum;
	       xlag.data[k] = float(l);
	    }
	 }
	 
	 if(i+m+abs(l)>=N) continue;
 
	 x = pa[i+m];
	 y = pb[i+m];

	 for(j=0; j<m; j++) {
	    sum -= P[last][j]; 
	    sign  = x>pa[i+j+1] ? 1 : -1;
	    sign *= y>pb[i+j+1] ? 1 : -1;
	    P[last][j] = sign;
	    sum += double(sign); 
	 }

	 last++;
	 if(last>=m) last=0;
      }
      
   }
   xcor *= 1./double(m*(m-1));
   for(i=0; i<m; i++) delete P[i];
   delete P;
   return;
}


//**************************************************************************
// initialize wavecor from two time series
//**************************************************************************
void wavecor::init(wavearray<double>& a, wavearray<double>& b, 
		   double w, double t, size_t skip)
{
   cList.clear();
   ifo=0; shift=0.;
   window = w;
   lagint = t;

   if(a.start()!=b.start() || a.size()!=b.size() || a.rate()!=b.rate()) {
      cout<<"wavecor::init() invalid arguments"<<endl;
      return;
   }

   size_t i,j;
   size_t N = a.size();
   size_t m = size_t(fabs(w)*a.rate());   // samples in integration window
   size_t n = size_t(fabs(t)*a.rate()/2); // one side time lag interval
   size_t M = N/skip;                     // take every skip sample from x
   float r  = a.rate();
   float var;

   if(!(m&1)) m++;                        // # of samples in integration window

   xcor.start(a.start());                 // set start time
   xlag.start(a.start());                 // set start time
   xcor.rate(a.rate()/skip);
   xlag.rate(a.rate()/skip);
   if(xcor.size()!=M) xcor.resize(M);
   if(xlag.size()!=M) xlag.resize(M);


   wavearray<double> x;
   x.rate(a.rate());

   x  = a;
   x *= b;

   if(w<0.) x.median(fabs(w),NULL,false,skip);
   else     x.mean(fabs(w),NULL);

   for(i=0; i<M; i++) {
      xcor.data[i] = float(x.data[i*skip]);
      xlag.data[i] = 0.;
   }

   for(i=0; i<n; i++){
      x.cpf(b,N-n+i,n-i);       // channel1 start = start 
                                // channel2 start = start+lag (positive)
      for(j=N-n+i; j<N; j++) x.data[j]=0.;

      x *= a; 

      if(w<0.) x.median(fabs(w),NULL,false,skip);
      else     x.mean(fabs(w),NULL);

      for(j=0; j<M; j++) {
	 var = float(x.data[j*skip]);
	 if(fabs(var)>fabs(xcor.data[j])) {
	    xcor.data[j] = var;
	    xlag.data[j] = float(n-i)/r;  // positive lag
	 }
      }

      x.cpf(a,N-n+i,n-i);       // channel2 start = start 
                                // channel1 start = start-lag (negative) 
      for(j=N-n+i; j<N; j++) x.data[j]=0.;

      x *= b; 
      if(w<0.) x.median(fabs(w),NULL,false,skip);
      else     x.mean(fabs(w),NULL);

      for(j=0; j<M; j++) {
	 var = float(x.data[j*skip]);
	 if(fabs(var)>fabs(xcor.data[j])) {
	    xcor.data[j] = var;
	    xlag.data[j] = -float(n-i)/r;  // negative lag
	 }
      }
   }
   xcor *= sqrt(double(m))/2.;
}




//**************************************************************************
// select x-correlation samples above threshold T
//**************************************************************************
double wavecor::select(double T)
{
   if(T <= 0.) return 1.;

   size_t i;
   size_t nonZero=0;
   size_t N = xcor.size();

   for(i=0; i<N; i++) {
      if(fabs(xcor.data[i]) < T) xcor.data[i]=0.;
      else nonZero++;
   }
   return double(nonZero)/xcor.size();
}


//**************************************************************************
// select x-correlation samples
//**************************************************************************
double wavecor::coincidence(double w, wavecor* pw)
{
   if(w<=0. || pw==NULL) return 0.;

   size_t i,last;
   size_t nonZero=0;
   size_t N = xcor.size();
   size_t n = size_t(fabs(w)*xcor.rate());
   size_t count=0;
   size_t nM = n/2;                    // index of median sample
   size_t nL = N-int(nM+1);

   if(n&1) n--;

   wavearray<double> x(n+1);
   float* px = pw->xcor.data;
   x.rate(xcor.rate());

   for(i=0; i<=n; i++) { 
      x.data[i] = fabs(*(px++)); 
      if(x.data[i] > 0.) count++;       // number of non-zero samples in x
   }

   last = 0;
   nonZero = 0;

   for(i=0; i<N; i++){

      if(!count) xcor.data[i]=0.;

      if(i>=nM && i<nL) {              // copy next sample into last
	 if(x.data[last] > 0.) count--; 
       	 x.data[last] = fabs(*(px++)); 
	 if(x.data[last++] > 0.) count++; 
      }

      if(last>n) last = 0;      
      if(xcor.data[i]!=0.) nonZero++;
   } 
   return double(nonZero)/xcor.size();
}












