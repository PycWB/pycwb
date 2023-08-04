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


// Wavelet Analysis Tool
//--------------------------------------------------------------------
// V. Necula & S. Klimenko, University of Florida
// Implementation of Fast Wilson-Daubechies-Meyer transform
// Reference: http://iopscience.iop.org/1742-6596/363/1/012032
//--------------------------------------------------------------------

#include <strstream>
#include <stdexcept>
#include <xmmintrin.h>
#include "WDM.hh"
#include "sseries.hh"
#include "wavefft.hh"
#include <iostream>
#include <stdio.h>
#include "TFFTComplexReal.h"
#include "TFFTRealComplex.h"
#include "TMath.h"
#include <complex>

using namespace std;


extern "C" void sse_dp4();
extern "C" void sse_dp5();
extern "C" void sse_dp6();
extern "C" void sse_dp7();
extern "C" void sse_dp8();
extern "C" void sse_dp9();
extern "C" void sse_dp10();
extern "C" void sse_dp11();

extern float* watasm_data; 
extern float* watasm_filter;
extern float watasm_xmm0[4];

ClassImp(WDM<double>)

static const double Pi = 3.14159265358979312;

//template<class DataType_t> double* WDM<DataType_t>::Cos[MAXBETA];
//template<class DataType_t> double* WDM<DataType_t>::Cos2[MAXBETA];
//template<class DataType_t> double* WDM<DataType_t>::SinCos[MAXBETA];
//template<class DataType_t> double  WDM<DataType_t>::CosSize[MAXBETA];
//template<class DataType_t> double  WDM<DataType_t>::Cos2Size[MAXBETA];
//template<class DataType_t> double  WDM<DataType_t>::SinCosSize[MAXBETA];
//template<class DataType_t>    int  WDM<DataType_t>::objCounter = 0;
      
// exp^{-...} i.e. negative sign
void FFT(double* a, double* b, int n)
{ 
   wavefft(a, b, n, n, n, -1);
}

void InvFFT(double* a, double* b, int n)
{  
  wavefft(a,b, n,n,n, 1);
}

// precalculated transform scaling functions
template<class DataType_t>
void WDM<DataType_t>::initFourier() 
{ 
  #include "FourierCoefficients.icc"
}

template<class DataType_t>
WDM<DataType_t>::WDM() :
WaveDWT<DataType_t>(1, 1, 0, B_CYCLE)
{	
// default constructor

  if(++objCounter==1) initFourier(); 
  this->m_WaveType = WDMT;
  this->m_Heterodine = 1;
  this->BetaOrder = 4;
  this->precision = 10;
  this->m_Level = 0;
  this->m_Layer = 0;
  this->KWDM = 0; 
  this->LWDM = 0; 
  wdmFilter.resize(0);
  this->m_L = 0;
  this->m_H = 0;
  TFMap00 = TFMap90 = 0;
  td_buffer=0;
}

template<class DataType_t>
WDM<DataType_t>::WDM(int M, int K, int iNu, int Precision) :
WaveDWT<DataType_t>(1, 1, 0, B_CYCLE)
{	
// constructor
// M + 1: number of bands
// K    : K=n*M, where n is integer. K defines the width of the 'edge' 
//        of the basis function in Fourier domain (see the paper)
//        larger n, longer the WDM filter, lower the spectral leakage between the bands.
// iNu  : defines the sharpness of the 'edge' of the basis function in Fourier domain (see paper)
// Precison : defines filter length by truncation error quantified by 
// P = -log10(1 - norm_of_filter) (see the paper)
// after forward transformation the structure of the WDM sliced array is the following:
//  f    phase 0       phase 90  
//  M  *  * ...  *   *   * ...  *
//    ... ... ... ... ...  ... ...
//  1  *  * .... *   *   * ...  *
//  0  *  * ...  *   *   * ...  *
//  t  1  2  ... n  n+1 n+2... 2n
// where t/f is the time/frequency index
// the global TF index in the linear array is i = t*(M+1)+f 

  if(++objCounter==1) initFourier(); 

  this->m_WaveType = WDMT;
  this->m_Heterodine = 1;
  this->BetaOrder = iNu;
  this->precision = Precision;
  this->m_Level = 0;
  this->m_Layer = M;
  this->KWDM = K; 
  this->LWDM = 0; 
  
  if(iNu<2){
    printf("iNu too small, reset to 2\n");
    iNu = 2;
  }
  if(iNu>7){
    printf("iNu too large, reset to 7\n");
    iNu = 7;
  }
  
  int nMax = 3e5;
  int M2 = M*2; 
  int N = 1;
  
  wavearray<double> filter = this->getFilter(nMax);
  
  double* tmp = filter.data;
  double residual = 1 - tmp[0]*tmp[0];
  double prec = pow(10., -double(Precision));
  
  do {
    residual -= 2*tmp[N]*tmp[N];
    N++;
    //printf("%d %e\n", N, residual);
  } while(residual>prec || (N-1)%M2 || N/M2<3);

//  printf("Filter length = %d,  norm = %.16f\n", N, 1.-residual);
  
  wdmFilter.resize(N);
  wdmFilter.cpf(filter,N);
  this->m_L = 0;
  this->m_H = N;
  TFMap00 = TFMap90 = 0;
  td_buffer=0;
}

template<class DataType_t>
WDM<DataType_t>::WDM(int m) :
WaveDWT<DataType_t>(1, 1, 0, B_CYCLE)
{	
// constructor
// M + 1: number of bands
// K    : K=n*M, where n is integer. K defines the width of the 'edge' 
//        of the basis function in Fourier domain (see the paper)
//        larger n, longer the WDM filter, lower the spectral leakage between the bands.
// after forward transformation the structure of the WDM sliced array is the following:
//  f    phase 0       phase 90  
//  M  *  * ...  *   *   * ...  *
//    ... ... ... ... ...  ... ...
//  1  *  * .... *   *   * ...  *
//  0  *  * ...  *   *   * ...  *
//  t  1  2  ... n  n+1 n+2... 2n
// where t/f is the time/frequency index
// the global TF index in the linear array is i = t*(M+1)+f 

  if(++objCounter==1) initFourier(); 
  int M = abs(m);

  this->m_WaveType = WDMT;
  this->m_Heterodine = 1;
  this->BetaOrder = 2;
  this->m_Level = 0;
  this->m_Layer = M;
  this->KWDM = M; 
  this->LWDM = 0; 
  
  int nMax = 3e5;
  int M2 = M*6;
  int K  = m<0 ? M : 2*M-int(92*M/256.);          // trancate basis function 
  
  wavearray<double> filter = this->getFilter(M2+6);
  
  double* tmp = filter.data;
  double residual = 1 - tmp[0]*tmp[0];
  //cout<<filter.size()<<" "<<M2<<" "<<K<<endl;
  
  for(int i=1; i<=M2; i++) {
     if(i<K) {
	residual -= 2*tmp[i]*tmp[i];
	//printf("%d %e %e\n", i, residual, tmp[i]);
     } else {
	tmp[i] = 0.;
     }
  }
  
  //printf("Filter length = %d,  norm = %.16f\n", M2+1, 1.-residual);
  
  wdmFilter.resize(M2+1);
  filter *= 1./sqrt(1.-residual);  
  this->precision = -log10(residual);
  wdmFilter.cpf(filter,M2+1);
  this->m_L = 0;
  this->m_H = M2+1;
  TFMap00 = TFMap90 = 0;
  td_buffer=0;
}

template<class DataType_t> WDM<DataType_t>::
WDM(const WDM<DataType_t> &w)  
{
// copy constructor
// w : other WDM object

  ++objCounter;
  this->m_Heterodine = 1;
  this->m_WaveType = WDMT;
  this->BetaOrder = w.BetaOrder;
  this->precision = w.precision;
  this->m_Level = w.m_Level;
  this->m_Layer = w.m_Layer;
  this->m_L = w.m_L;
  this->m_H = w.m_H;
  this->KWDM = w.KWDM; 
  this->LWDM = w.LWDM; 
  this->nSTS = w.nSTS;                // WaveDWT parent class
  this->wdmFilter = w.wdmFilter;
  this->T0 = w.T0;
  this->Tx = w.Tx;
  TFMap00 = TFMap90 = 0;
  sinTD = w.sinTD;
  cosTD = w.cosTD;
  sinTDx = w.sinTDx;
  td_buffer = 0;
  if(T0.Last()) { 
     initSSEPointers();
     for(int i=0; i<6; ++i) {
        td_halo[i].Resize(T0[0].Last());
        td_halo[i].ZeroExtraElements();
     }
  }

}

template<class DataType_t>
WDM<DataType_t>::~WDM()
{	 
// Destructor

   if(--objCounter==0) {
      for(int i=2; i<MAXBETA; ++i){
         if(Cos[i]) delete [] Cos[i]; 
         if(Cos2[i]) delete [] Cos2[i]; 
         if(SinCos[i]) delete [] SinCos[i];
      }
   }
   
   if(TFMap00) delete [] TFMap00; 
   if(TFMap90) delete [] TFMap90;
   if(td_buffer)free(td_buffer);
}

//clone
template<class DataType_t>
WDM<DataType_t>* WDM<DataType_t>::Clone() const
{	
// Clone the object
   
   return new WDM<DataType_t>(*this);
}


template<class DataType_t>
WDM<DataType_t>* WDM<DataType_t>::Init() const
{	
// light-weight clone without TD filters

   WDM<DataType_t>* pwdm = new WDM<DataType_t>();
   pwdm->m_Heterodine = this->m_Heterodine;        
   pwdm->m_WaveType = this->m_WaveType;       
   pwdm->BetaOrder = this->BetaOrder;
   pwdm->precision = this->precision;
   pwdm->m_Level = this->m_Level;    
   pwdm->m_Layer = this->m_Layer;    
   pwdm->m_L = this->m_L;            
   pwdm->m_H = this->m_H;            
   pwdm->KWDM = this->KWDM;          
   pwdm->LWDM = this->LWDM;          
   pwdm->nSTS = this->nSTS;          
   pwdm->wdmFilter = this->wdmFilter;
   pwdm->TFMap00 = pwdm->TFMap90 = 0;  
   return pwdm;
}

template<class DataType_t>
int WDM<DataType_t>::getBaseWave(int m, int n, SymmArray<double>& w)
{  
//  computes basis function (sampled)
//  m : frequency index
//  n : time index
//  w : where to store the values
//  return value: indicates the time translation from origin needed by w (n=0 corresponds to t=0)

   int N = wdmFilter.size() ;
   int M = this->m_Layer;
   w.Resize(N-1);
   if(m==0){
      if(n%2)for(int i=-N+1; i<N; ++i)w[i] = 0;
      else{
         for(int i=1; i<N; ++i)w[-i] = w[i] = wdmFilter[i];
         w[0] = wdmFilter[0];
      }
   }
   else if(m==M){
      if( (n-M)%2) for(int i=-N+1; i<N; ++i)w[i] = 0;
      else {
         int s = (M%2) ? -1 : 1 ; //sign
         w[0] = s*wdmFilter[0];
         for(int i=1; i<N; ++i){
            s = -s;
            w[-i] = w[i] = s*wdmFilter[i];
         }
      }
   }
   else {
      double ratio = m*Pi/M;
      double sign = sqrt(2);
      //if((m*n)%2)sign = -sqrt(2);      
      
      if((m+n)%2){
         w[0] = 0;
         for(int i=1; i<N; ++i){
            w[i] = - sign*sin(i*ratio)*wdmFilter[i];
            w[-i] = - w[i];
         }
      }
      else{
         w[0] = sign*wdmFilter[0];
         for(int i=1; i<N; ++i)
            w[i] = w[-i] = sign*cos(i*ratio)*wdmFilter[i];
     }
   }
   return n*M;
}

template<class DataType_t>
int WDM<DataType_t>::getBaseWaveQ(int m, int n, SymmArray<double>& w)
{  
//  computes Quadrature basis function (sampled)
//  m : frequency index
//  n : time index
//  w : where to store the values
//  return value: indicates the time translation from origin needed by w (n=0 corresponds to t=0)

   int N = wdmFilter.size() ;
   int M = this->m_Layer;
   w.Resize(N-1);
   if(m==0){
      if(n%2){
         for(int i=1; i<N; ++i)w[-i] = w[i] = wdmFilter[i];
         w[0] = wdmFilter[0];
      }
      else for(int i=-N+1; i<N; ++i)w[i] = 0;  
   }
   else if(m==M){
      if( (n-M)%2){
         int s = 1;
         w[0] = wdmFilter[0];
         for(int i=1; i<N; ++i){
            s = -s;
            w[-i] = w[i] = s*wdmFilter[i];
         }
      }
      else for(int i=-N+1; i<N; ++i)w[i] = 0;
   }
   else {
      double ratio = m*Pi/M;
      double sign = sqrt(2);
      //if((m*n)%2)sign = -sqrt(2);
      
      if((m+n)%2){
         w[0] = sign*wdmFilter[0];
         for(int i=1; i<N; ++i)
            w[i] = w[-i] = sign*cos(i*ratio)*wdmFilter[i];
      }
      else{
         w[0] = 0;
         for(int i=1; i<N; ++i){
            w[i] = sign*sin(i*ratio)*wdmFilter[i];
            w[-i] = - w[i];
         }
      }
   }
   return n*M;
}


template<class DataType_t>
int WDM<DataType_t>::getBaseWave(int j, wavearray<double>& w, bool Quad)
{  
//  computes basis function (sampled)
//  j : TF index
//  w : where to store the values
//  Quad: true to return Quadrature basis function
//  return value: indicates the time translation from origin needed by w (j in [0,M] correspond to t=0)

   int M1 = this->m_Layer+1;
   int m = j%M1;
   int n = j/M1;
   SymmArray<double> w2;
   int shift = Quad? getBaseWaveQ(m, n, w2) : getBaseWave(m, n, w2);
   int nn = w2.Last();
   w.resize(2*nn+1);
   for(int i=-nn; i<=nn; ++i) w[nn+i] = w2[i];
   return shift-nn; 
}


template<class DataType_t>
void  WDM<DataType_t>::SetTFMap()
{  
// not used

   int M1 = this->m_Layer+1;
   int N = this->nWWS/2/M1;
   TFMap00 = new DataType_t*[N];
   TFMap90 = new DataType_t*[N];
   for(int i=0; i<N; ++i){
      TFMap00[i] = this->pWWS + M1*i;
      TFMap90[i] = TFMap00[i] + M1*N;
   }
}


template<class DataType_t>
std::slice WDM<DataType_t>::getSlice(const double index)
{
// access function for a frequency layer described by index
// return slice corresponding to the selected frequency band

  int M = this->m_Level;
  int N = this->nWWS;
  int layer = int(fabs(index)+0.001);

  if(layer>M){
    printf("WDM::getSlice(): illegal argument %d. Should be no more than %d\n",layer,M);
    exit(0);
  }
  
  if(!this->allocate()){
    std::invalid_argument("WDM::getSlice(): data is not allocated");
    return std::slice(0,1,1);
  }
  
  size_t n = N/(M+1);                         // number of samples
  size_t k = N/this->nSTS>1 ? 1 : 0;          // power flag when = 0
  size_t s = M+1;                             // slice step
  size_t i = index<0 ? layer+k*N/2 : layer;   // first sample
  
  n /= k+1;                                   // adjust size
  
  if(i+(n-1)*s+1 > this->nWWS){
    std::invalid_argument("WaveDWT::getSlice(): invalide arguments");
    return std::slice(0,1,1);
  }
  
  return std::slice(i,n,s);
}

template<class DataType_t>
wavearray<double> WDM<DataType_t>::getFilter(int n)
{  
// internal WDM function
// computes the WDM transformation filter
// n - number of the filter coefficients defined by WDM
//     parameters and precision.

  wavearray<double> tmp(n);
  double* Fourier = Cos[BetaOrder];
  int nFourier = CosSize[BetaOrder];
  int M = this->m_Layer;
  int K = this->KWDM;
  double B = Pi/K;
  double A = (K-M)*Pi/2./K/M;
  double K2 = K;
  K2 *= K2; 
  double* filter = tmp.data;
  double gNorm = sqrt(2*M)/Pi;
  double fNorm = 0.;
  
  double* fourier = new double[nFourier]; 
  for(int i=0; i<nFourier; ++i) fourier[i] = Fourier[i];

  filter[0] = (A + fourier[0]*B)*gNorm;  // do n = 0 first
  fNorm = filter[0]*filter[0];
  fourier[0] /= sqrt(2);                 // this line requires to copy Fourier to fourier  		      

  for(int i=1; i<n; ++i){ 		 // other coefficients
    double di = i;
    double i2 = di*di;
    double sumEven = 0, sumOdd = 0;
    for(int j=0; j<nFourier; ++j){       // a_j in the Fourier expansion
      if(i%K);
      else if(j==i/K)continue;
      if(j&1)sumOdd += di/(i2 - j*j*K2)*fourier[j];
      else sumEven += di/(i2 - j*j*K2)*fourier[j];
    }

    double intAB = 0;                    //integral on [A, A+B]
    if(i%K == 0)
      if(i/K < nFourier) intAB = fourier[i/K]*B/2*cos(i*A); 

    intAB += 2*(sumEven*sin(i*B/2)*cos(i*Pi/(2*M)) - sumOdd*sin(i*Pi/(2*M))*cos(i*B/2));
    filter[i] = gNorm* ( sin(i*A)/i + sqrt(2)*intAB );  //sqrt2 b/c of the Four. expansion
    fNorm += 2*filter[i]*filter[i];
  }
  
  delete [] fourier;
  return tmp;
  
}

template<class DataType_t> 
wavearray<double> WDM<DataType_t>::getTDFilter1(int n, int L)  // ex: L=8 -> tau/8 step
{  
// computes 1st integral needed for TD filters (see the reference)
// n : defines the number of TD filter coefficients
// L : upsample factor the same as for setTDFilter

  double* Fourier = Cos2[BetaOrder];
  int nFourier = Cos2Size[BetaOrder];
  int M = this->m_Layer;
  int K = this->KWDM;
  double B = Pi/K;
  double A = (K-M)*Pi/2./K/M;
  double K2 = K*K; 
  double gNorm = 2*M/Pi;
  wavearray<double> tmp(n*L);
  double* filter = tmp.data;
  
  double* fourier = new double[nFourier]; 
  for(int i=0; i<nFourier; ++i) fourier[i] = Fourier[i];

  filter[0] = (A + fourier[0]*B)*gNorm;  // do n = 0 first
  fourier[0] /= sqrt(2);                 // this line requires to copy Fourier to fourier  		      

  for(int i=1; i<n*L; ++i){ 		 // other coefficients
    double di = i*(1./L);
    double i2 = di*di;
    double sumEven = 0, sumOdd = 0;
    for(int j=0; j<nFourier; ++j){       // a_j in the Fourier expansion
      if(j*K*L==i)continue;
      if(j&1)sumOdd += di/(i2 - j*j*K2)*fourier[j];
      else sumEven += di/(i2 - j*j*K2)*fourier[j];
    }

    double intAB = 0;                    //integral on [A, A+B]
    if(i%(K*L) == 0)if(i/(K*L) < nFourier) 
      intAB = fourier[i/(K*L)]*B/2*cos(di*A); 
    intAB += 2*(sumEven*sin(di*B/2)*cos(di*Pi/(2*M)) - sumOdd*sin(di*Pi/(2*M))*cos(di*B/2));
    filter[i] = gNorm* ( sin(di*A)/di + sqrt(2)*intAB );  //sqrt2 b/c of the Four. expansion
  }

  delete [] fourier;
  return tmp;
}


template<class DataType_t>
wavearray<double> WDM<DataType_t>::getTDFilter2(int n, int L) // [-B/2, B/2]
{  
// computes 2nd integral needed for TD filters (see paper)
// n : defines the number of TD filter coefficients
// L : upsample factor the same as for setTDFilter

  wavearray<double> tmp(n*L);
  double* res = tmp.data;
  int M = this->m_Layer;
  int K = this->KWDM;
  double B = Pi/K;
  double K2 = K*K;
  
  double* Fourier = SinCos[BetaOrder];
  int nFourier = SinCosSize[BetaOrder];
  double gNorm = M/Pi; 	
  
  double aux = Fourier[0];
  res[0] = aux*B*gNorm;  
  Fourier[0] /= sqrt(2);   		
  for(int i=1; i<n*L; ++i){ 			// psi_i (l in the writeup)
     double di = i*(1./L);
     double i2 = di*di;
     double sum = 0;
     for(int j=0; j<=nFourier; j+=2){	   //j indexing the Fourier coeff
        if(j*K*L==i)continue;
        sum += di/(i2 - j*j*K2)*Fourier[j];
     }
     sum *= 2*sin(di*B/2);
     if(i%(2*K*L) == 0)
        if(i/(K*L) <= nFourier) { // j*K*L = i case (even j)
           if( (i/(2*K*L)) & 1 ) sum -= Fourier[i/K/L]*B/2; 
           else sum += Fourier[i/K/L]*B/2;
        }
     res[i] = gNorm*sqrt(2)*sum;   		//sqrt2 b/c of the Four. expansion
  }
  Fourier[0] = aux; 
  return tmp;
}


template<class DataType_t>
void WDM<DataType_t>::setTDFilter(int nCoeffs, int L) 
{
// initialization of the time delay filters
// nCoeffs : define the number of the filter coefficients 
// L : upsample factor, defines the fundamental time delay step  
//     dt = tau/L , where tau is the sampling interval of the original
//     time series

   int M = this->m_Layer;
   this->LWDM = L;
   T0.Resize(M*L);
   Tx.Resize(M*L);
   for(int i=0; i<6; ++i) {
      td_halo[i].Resize(nCoeffs);
      td_halo[i].ZeroExtraElements();
   }
  
   wavearray<double> filt1 = getTDFilter1(M*(nCoeffs+1)+1, L);
  
   double* tmp = filt1.data ; //cos2
   for(int i=M*L; i>=-M*L; --i){
      T0[i].Resize(nCoeffs);
      T0[i][0] = tmp[abs(i)];
      for(int j=1; j<=nCoeffs; ++j){
         T0[i][j] = tmp[i+ M*L*j];
         T0[i][-j] = tmp[M*L*j - i];
      }
      T0[i].ZeroExtraElements();
   }
  
   wavearray<double> filt2 = getTDFilter2(M*(nCoeffs+1)+1 , L); //sincos
   tmp = filt2.data;
   for(int i=M*L; i>=-M*L; --i){
      Tx[i].Resize(nCoeffs);
      Tx[i][0] = tmp[abs(i)]/2.; 	// divide by 2 b/c the filter is actually for sin(2x) = 2*sin*cos
      for(int j=1; j<=nCoeffs; ++j){
         Tx[i][j] = tmp[i+ M*L*j]/2.;
         Tx[i][-j] = tmp[M*L*j - i]/2.;
      }
    
      // change some signs to bring i^l factor to C_l form!!
      for(int j=1; j<=nCoeffs; ++j)switch(j%4){
         case 1:  Tx[i][-j] *= -1; break;
         case 2:  Tx[i][j] *= -1; Tx[i][-j] *= -1; break;
         case 3:  Tx[i][j] *= -1; 
      }
      Tx[i].ZeroExtraElements();
   }
   
   initSSEPointers();
      
   sinTD.resize(2*L*M);
   cosTD.resize(2*L*M); 
   for(int i=0; i<2*L*M; ++i){
      sinTD[i] = sin(i*Pi/(L*M));
      cosTD[i] = cos(i*Pi/(L*M));
   }
   sinTDx.resize(4*L*M);
   for(int i=0; i<4*L*M; ++i)sinTDx[i] = sin(i*Pi/(2*L*M));
}

template<class DataType_t>
void WDM<DataType_t>::initSSEPointers()
{  
// required before using SSE instructions to speed up time delay filter calculations

   if(td_buffer)free(td_buffer);
   switch(T0[0].SSESize()){   
      case 64: SSE_TDF = sse_dp4; posix_memalign((void**)&td_buffer, 16, 64+16); break;
      case 80: SSE_TDF = sse_dp5; posix_memalign((void**)&td_buffer, 16, 80+16);  break;
      case 96: SSE_TDF = sse_dp6; posix_memalign((void**)&td_buffer, 16, 96+16); break;
      case 112: SSE_TDF = sse_dp7; posix_memalign((void**)&td_buffer, 16, 112+16);  break;
      case 128: SSE_TDF = sse_dp8; posix_memalign((void**)&td_buffer, 16, 128+16);  break;
      case 144: SSE_TDF = sse_dp9; posix_memalign((void**)&td_buffer, 16, 144+16);  break;
      case 160: SSE_TDF = sse_dp10; posix_memalign((void**)&td_buffer, 16, 160+16);  break;
      case 176: SSE_TDF = sse_dp11; posix_memalign((void**)&td_buffer, 16, 176+16);  break;
      default: printf("initSSEPointer error, size not found. Contact V.Necula\n"); SSE_TDF=0;
   }
   td_data = td_buffer + T0[0].SSESize()/8 + 4; 
}

/*
template<class DataType_t>
double WDM<DataType_t>::PrintTDFilters(int m, int dt, int nC)
{	SymmArray<double>& sa0 = T0[dt];
SymmArray<double>& sax = Tx[dt];
int M = this->m_Layer;
double sin0 = sin((m*Pi*dt)/M);
double cos0 = cos((m*Pi*dt)/M);
double sinP = sin((2*m+1)*dt*Pi/(2.*M));
double sinM = sin((2*m-1)*dt*Pi/(2.*M));
if(nC<=0 || nC>sa0.Last())nC = sa0.Last();

double res = 0, t0, tp, tm;
for(int i=-nC; i<=nC; ++i){
if(i&1)t0 = sa0[i]*sin0;
else t0 = sa0[i]*cos0;
tp = sax[i]*sinP;
tm = sax[i]*sinM;
//printf("i=%3d  %+le  %+le  %+le\n", i, t0, tp, tm);
res += t0*t0 + tm*tm + tp*tp;
}
return res;
}
*/

// Quadrature: equivalent to multiplying C_l by -i (plus shifting the 0,M layers)

template<class DataType_t>
double WDM<DataType_t>::getPixelAmplitude(int m, int n, int dT, bool Quad)
{
//  computes one time dalay for one pixel without SSE instructions
//  m : frequency index (0...M)
//  n : time index (in pixels)
//  dT : time delay (in units of the sampling period or fractions of it, 
//       depending on the TD Filters setting
//  Quad: true to compute the delayed amplitude for the quadrature [m,n] pixel


  int M = this->m_Layer;
  int M1 = M+1;
  int odd = n&1;
  bool cancelEven = odd ^ Quad; 
  double dt = dT*(1./this->LWDM);
  if(m==0 || m==M)if(cancelEven) return 0; // DOESN'T WORK FOR ODD M!!
  DataType_t* TFMap = this->pWWS;
  if(Quad) TFMap += this->nWWS/2;
  
  //double* tfc = TFMap[m]+n;
  DataType_t* tfc = TFMap + n*M1 + m;
  double res, res1;
 
  //SymmArray<double>& sa = T0[dT];
  SymmArraySSE<float>& sa = T0[dT];
  double sumOdd = 0;
  double sumEven = tfc[0]*sa[0];
  for(int i=2; i<=sa.Last(); i+=2) sumEven += (tfc[i*M1]*sa[i] +tfc[-i*M1]*sa[-i]);
  for(int i=1; i<=sa.Last(); i+=2) sumOdd += (tfc[i*M1]*sa[i] +tfc[-i*M1]*sa[-i]);
  if(m==0 || m==M) {
    if(cancelEven)sumEven = 0;
    else sumOdd = 0;
  }
  
  //res = 3.2*sumEven + sumOdd;
  if(odd) res = sumEven*cos((m*Pi*dt)/M) - sumOdd*sin((m*Pi*dt)/M);
  else res = sumEven*cos((m*Pi*dt)/M) + sumOdd*sin((m*Pi*dt)/M);
  
  //SymmArray<double>& sax = Tx[dT]; 
  SymmArraySSE<float>& sax = Tx[dT];
  if(m>0){ // m-1 case
    // tfc = TFMap[m-1]+n;
    tfc--;
    sumOdd = 0;
    sumEven = tfc[0]*sax[0];
    for(int i=2; i<=sax.Last(); i+=2) sumEven += (tfc[i*M1]*sax[i] +tfc[-i*M1]*sax[-i]);
    for(int i=1; i<=sax.Last(); i+=2) sumOdd += (tfc[i*M1]*sax[i] + tfc[-i*M1]*sax[-i]);
    
    if(m==1) {
      if(cancelEven)sumEven = 0;
      else sumOdd = 0;
    }
    //res1 = 2.1*sumEven + sumOdd;
    if(odd) res1 = (sumEven - sumOdd)*sin((2*m-1)*dt*Pi/(2.*M));
    else res1 = (sumEven + sumOdd)*sin((2*m-1)*dt*Pi/(2.*M));      
    if(m==1 || m == M) res1 *= sqrt(2);
    if((m+n)&1) res -= res1;
    else res += res1;
  }
   
  if(m<M){ // m+1 case
    //tfc = TFMap[m+1]+n;
    tfc = TFMap + n*M1 + m + 1;
    sumOdd = 0;
    sumEven = tfc[0]*sax[0];
    for(int i=2; i<=sax.Last(); i+=2) sumEven += (tfc[i*M1]*sax[i] + tfc[-i*M1]*sax[-i]);
    for(int i=1; i<=sax.Last(); i+=2) sumOdd  += (tfc[i*M1]*sax[i] + tfc[-i*M1]*sax[-i]);

    if(m==M-1) {
      if(cancelEven)sumEven = 0;
      else sumOdd = 0;
    }
    //res1 = 2.3*sumEven + sumOdd;
    if(odd) res1 = (sumEven + sumOdd)*sin((2*m+1)*dt*Pi/(2.*M));
    else res1 = (sumEven - sumOdd)*sin((2*m+1)*dt*Pi/(2.*M));  //two (-1)^l cancel
    if(m==0 || m==M-1) res1 *= sqrt(2);
    if((m+n)&1) res -= res1;
    else res += res1;
  }
  return res;
}


// Quadrature: equivalent to multiplying C_l by -i (plus shifting the 0,M layers)
template<class DataType_t>
double WDM<DataType_t>::getPixelAmplitudeSSEOld(int m, int n, int dT, bool Quad)
{ 
//  computes one time dalay for one pixel using SSE instructions
//  m : frequency index (0...M)
//  n : time index (in pixels)
//  dT : time delay (in units of the sampling period or fractions of it, 
//       depending on the TD Filters setting
//  Quad: true to compute the delayed amplitude for the quadrature [m,n] pixel
   

   static const double sqrt2 = sqrt(2); 
   int M = this->m_Layer;
  int M1 = M+1;
  int L = this->LWDM;
  int odd = n&1;
  bool cancelEven = odd ^ Quad; 
  
  if(m==0 || m==M)if(cancelEven) return 0; // DOESN'T WORK FOR ODD M!!
  DataType_t* TFMap = this->pWWS;
  if(Quad) TFMap += this->nWWS/2;
  DataType_t* tfc = TFMap + n*M1 + m;
   
  double res, res1;
  
  SymmArraySSE<float>& sa = T0[dT];
  watasm_data = td_buffer + 4;
  watasm_filter = sa.SSEPointer();
  
  const int last = sa.Last();
  for(int i=-last, j = -last*M1; i<=last; ++i){td_data[i] = tfc[j] ; j+=M1;}
  SSE_TDF();
  
  float sumEven = watasm_xmm0[0] + watasm_xmm0[2];
  float sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
    
  if(m==0 || m==M) {
    if(cancelEven)sumEven = 0;
    else sumOdd = 0;
  }
 
  //res = sumEven + sumOdd;
  int index = (m*dT)%(2*M*L);
  if(index<0) index +=2*M*L;
  
  if(odd) res = sumEven*cosTD[index] - sumOdd*sinTD[index];
  else res = sumEven*cosTD[index] + sumOdd*sinTD[index];
  
  SymmArraySSE<float>& sax = Tx[dT];
  watasm_filter = sax.SSEPointer();
  
  if(m>0){ // m-1 case
    tfc--;  
    for(int i=-last, j = -last*M1; i<=last; ++i){td_data[i] = tfc[j] ; j+=M1;}
    SSE_TDF();
    sumEven = watasm_xmm0[0] + watasm_xmm0[2];
    sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
     
    if(m==1) {
      if(cancelEven)sumEven = 0;
      else sumOdd = 0;
    }
    
    index = (2*m-1)*dT%(4*M*L);
    if(index<0)index += 4*M*L;
    if(odd) res1 = (sumEven - sumOdd)*sinTDx[index];
    else res1 = (sumEven + sumOdd)*sinTDx[index];  
        
    if(m==1 || m == M) res1 *= sqrt2;
    if((m+n)&1) res -= res1;
    else res += res1;
  }
  
  if(m<M){ // m+1 case
    tfc = TFMap + n*M1 + m + 1;
    for(int i=-last, j = -last*M1; i<=last; ++i){td_data[i] = tfc[j] ; j+=M1;}
    SSE_TDF();
    
    sumEven = watasm_xmm0[0] + watasm_xmm0[2];
    sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
        
    if(m==M-1) {
      if(cancelEven)sumEven = 0;
      else sumOdd = 0;
    }
    index = (2*m+1)*dT%(4*M*L);
    if(index<0)index += 4*M*L;
    if(odd) res1 = (sumEven + sumOdd)*sinTDx[index];
    else res1 = (sumEven - sumOdd)*sinTDx[index];  //two (-1)^l cancel
    if(m==0 || m==M-1) res1 *= sqrt2;
    if((m+n)&1) res -= res1;
    else res += res1;
  }
  return res;
}

template<class DataType_t>
float WDM<DataType_t>::getPixelAmplitudeSSE(int m, int n, int dT,  bool Quad)
{  
//  computes one time dalay for one pixel using SSE instructions; requires td_halo be initialized
//  m : frequency index (0...M)
//  n : time index (in pixels)
//  dT : time delay (in units of the sampling period or fractions of it, 
//       depending on the TD Filters setting
//  Quad: true to compute the delayed amplitude for the quadrature [m,n] pixel

   static const double sqrt2 = sqrt(2);
   int M = this->m_Layer;
   int L = this->LWDM;
   int odd = n&1;
   bool cancelEven = odd ^ Quad; 
   if(m==0 || m==M)if(cancelEven)return 0;  // DOESN'T WORK FOR ODD M!!
         
   int QuadShift = Quad? 3:0;
   double res, res1;
   
   SymmArraySSE<float>& sa = T0[dT];
   watasm_data = td_halo[QuadShift + 1].SSEPointer();
   watasm_filter = sa.SSEPointer();
   SSE_TDF();
   float sumEven = watasm_xmm0[0] + watasm_xmm0[2];
   float sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
  
   if(m==0 || m==M) {
      if(cancelEven)sumEven = 0;
      else sumOdd = 0;
   }
 
   int index = (m*dT)%(2*M*L);
   if(index<0) index +=2*M*L;
   if(odd) res = sumEven*cosTD[index] - sumOdd*sinTD[index];
   else res = sumEven*cosTD[index] + sumOdd*sinTD[index];
  
   SymmArraySSE<float>& sax = Tx[dT];
   watasm_filter = sax.SSEPointer();
  
   if(m>0){ // m-1 case
      watasm_data = td_halo[QuadShift + 0].SSEPointer();  
      SSE_TDF();
      sumEven = watasm_xmm0[0] + watasm_xmm0[2];
      sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
         
      if(m==1) {
         if(cancelEven)sumEven = 0;
         else sumOdd = 0;
      }
    
      index = (2*m-1)*dT%(4*M*L);
      if(index<0)index += 4*M*L;
      if(odd) res1 = (sumEven - sumOdd)*sinTDx[index];
      else res1 = (sumEven + sumOdd)*sinTDx[index];  
        
      if(m==1 || m == M) res1 *= sqrt2;
      if((m+n)&1) res -= res1;
      else res += res1;
   }
   
   if(m<M){ // m+1 case
      watasm_data = td_halo[QuadShift + 2].SSEPointer();
      SSE_TDF();
      sumEven = watasm_xmm0[0] + watasm_xmm0[2];
      sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
      
      if(m==M-1) {
         if(cancelEven)sumEven = 0;
         else sumOdd = 0;
      }
      index = (2*m+1)*dT%(4*M*L);
      if(index<0)index += 4*M*L;
         
      if(odd) res1 = (sumEven + sumOdd)*sinTDx[index];
      else res1 = (sumEven - sumOdd)*sinTDx[index];  //two (-1)^l cancel
      if(m==0 || m==M-1) res1 *= sqrt2;
      if((m+n)&1) res -= res1;
      else res += res1;
   }
   return res;
}

template<class DataType_t>
void WDM<DataType_t>::getPixelAmplitudeSSE(int m, int n, int t1, int t2, float* r, bool Quad)
{  
//  computes time dalays for one pixel
//  m : frequency index (0...M)
//  n : time index (in pixels)
//  t1, t2 : time delays range [t1, t1+1, .. t2]
//  r : where to store the time delayed amplitude for pixel [m,n]
//  Quad : true to request time delays for the quadrature amplitude

   static const double sqrt2 = sqrt(2);
   int M = this->m_Layer;
   int L = this->LWDM;
   int odd = n&1;
   bool cancelEven = odd ^ Quad; 
   if(m==0 || m==M)if(cancelEven){ // DOESN'T WORK FOR ODD M!!
      for(int dT = t1 ; dT<=t2; ++dT)*r++ = 0;
      return;
   } 
         
   double res, res1;
   int QuadShift = Quad? 3:0;
   int _2ml = 2*M*L;
   int _4ml = 2*_2ml;
   
   for(int dT = t1 ; dT<=t2; ++dT){
      
      watasm_data = td_halo[QuadShift + 1].SSEPointer();
      watasm_filter = T0[dT].SSEPointer();
      SSE_TDF();
      float sumEven = watasm_xmm0[0] + watasm_xmm0[2];
      float sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
  
      if(m==0 || m==M) {
         if(cancelEven)sumEven = 0;
         else sumOdd = 0;
      }
 
      int index = (m*dT)%_2ml;
      if(index<0) index +=_2ml;
      if(odd) res = sumEven*cosTD[index] - sumOdd*sinTD[index];
      else res = sumEven*cosTD[index] + sumOdd*sinTD[index];
  
      SymmArraySSE<float>& sax = Tx[dT];
      watasm_filter = sax.SSEPointer();
  
      if(m>0){ // m-1 case
         watasm_data = td_halo[QuadShift + 0].SSEPointer();  
         SSE_TDF();
         sumEven = watasm_xmm0[0] + watasm_xmm0[2];
         sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
         
         if(m==1) {
            if(cancelEven)sumEven = 0;
            else sumOdd = 0;
         }
    
         index = (2*m-1)*dT%_4ml;
         if(index<0)index += _4ml;
         if(odd) res1 = (sumEven - sumOdd)*sinTDx[index];
         else res1 = (sumEven + sumOdd)*sinTDx[index];  
        
         if(m==1 || m == M) res1 *= sqrt2;
         if((m+n)&1) res -= res1;
         else res += res1;
      }
   
      if(m<M){ // m+1 case
         watasm_data = td_halo[QuadShift + 2].SSEPointer();
         SSE_TDF();
         sumEven = watasm_xmm0[0] + watasm_xmm0[2];
         sumOdd = watasm_xmm0[1] + watasm_xmm0[3];
      
         if(m==M-1) {
            if(cancelEven)sumEven = 0;
            else sumOdd = 0;
         }
         
         index = (2*m+1)*dT%_4ml;
         if(index<0)index += _4ml;
         if(odd) res1 = (sumEven + sumOdd)*sinTDx[index];
         else res1 = (sumEven - sumOdd)*sinTDx[index];  //two (-1)^l cancel
         if(m==0 || m==M-1) res1 *= sqrt2;
         if((m+n)&1) res -= res1;
         else res += res1;
      }
      *r++ = res;
   }
}


template<class DataType_t>
float WDM<DataType_t>::getTDamp(int j, int k, char c)
{
// override getTDamp from WaveDWT
// return time-delayed amplitude for delay index k:
// j - global index in the TF map
// k - delay index assuming time delay step 1/(LWDM*rate)  
// c - mode: 'a'/'A' - returns 00/90 amplitude, 'p'or'P' - returns power

   int N = (int)this->nWWS/2;
   if(this->nWWS/this->nSTS!=2) { 
      printf("WDM:getTDamp() - time delays can not be produced with this TF data.");
      exit(0);
   }
   if(!this->Last()) { 
      printf("WDM:getTDamp() - time delay filter is not set");
      exit(0);
   }
   if(j>=N) j -= N;

   int M = this->m_Layer;
   int J = M*LWDM;                     // number of delays in one pixel
   int n = j/(M+1);                    // time index                      
   int m = j%(M+1);                    // frequency index
   int wdmShift = k/J;
   k %= J;
   
   if(n<0 || n>N/(M+1)) {
      cout<<"WDM::getTDamp(): index outside TF map"<<endl; exit(1);
   }


   if(c=='a'){
      if(wdmShift%2){      // not working well for odd M when m=0/M !!
         if((n+m)%2) return -getPixelAmplitude(m, n-wdmShift, k, true);
         else return getPixelAmplitude(m, n-wdmShift, k, true);
      }
      return getPixelAmplitude(m, n-wdmShift, k, false);
   }
   
   if(c=='A'){
      if(wdmShift%2){      // not working well for odd M when m=0/M !!
         if((n+m)%2) return getPixelAmplitude(m, n-wdmShift, k, false);
         else return -getPixelAmplitude(m, n-wdmShift, k, false);
      }
      return getPixelAmplitude(m, n-wdmShift, k, true);
   }
   
   double a00 = getPixelAmplitude(m, n-wdmShift, k, false);
   double a90 = getPixelAmplitude(m, n-wdmShift, k, true);
   return float(a00*a00+a90*a90)/2;
}



// set array of time-delay amplitudes/energy
template<class DataType_t>
wavearray<float> WDM<DataType_t>::getTDvec(int j, int K, char c)
{
// override getTDAmp from WaveDWT
// return array of time-delayed amplitudes in the format:
// [-K*dt , 0,  K*dt,  -K*dt,  0,  K*dt] - total 2(2K+1) amplitudes  
// or array of time-delayed power in the format:
// [-K*dt , 0,  K*dt] - total (2K+1) amplitudes  
// j - index in TF map
// K - range of unit delays dt
// c - mode: 'a','A' - amplitude, 'p','P' - power

   if(this->nWWS/this->nSTS!=2) { 
      printf("WDM:getTDAmp() - time delays can not be produced with this TF data.");
      exit(0);
   }
   if(!this->Last()) { 
      printf("WDM:getTDAmp() - time delay filter is not set");
      exit(0);
   }
   if(j>=(int)this->nWWS/2) j -= this->nWWS/2;

   int mode = (c=='a' || c=='A') ? 2 : 1;
   wavearray<float> amp(mode*(2*K+1));

   float* p00 = amp.data;
   float* p90 = amp.data + (mode-1)*(2*K+1);
   int i = 0;

   for(int k=-K; k<=K; k++) {
      if(mode==2) { 
         p00[i] = float(getTDamp(j, k, 'a')); 
         p90[i] = float(getTDamp(j, k, 'A')); 
      }
      else  p00[i] = getTDamp(j, k, 'p');
      i++;
   }
   return amp;
}
   

// set array of time-delay amplitudes/energy
template<class DataType_t>
wavearray<float> WDM<DataType_t>::getTDvecSSE(int j, int K, char c, SSeries<double>* pSS)
{
// return array of time-delayed amplitudes in the format:
// [-K*dt , 0,  K*dt,  -K*dt,  0,  K*dt] - total 2(2K+1) amplitudes  
// or array of time-delayed power in the format:
// [-K*dt , 0,  K*dt] - total (2K+1) amplitudes  
// j - index in TF map
// K - range of unit delays dt
// c - mode: 'a'/'A' - amplitude, 'p','P' - power

   if(!this->Last()) { 
      printf("WDM:getTDAmp() - time delay filter is not set");
      exit(0);
   }
   
   int M = this->m_Layer;
   int n = j/(M+1);                    // time index                      
   int m = j%(M+1);                    // frequency index
   
   int J = M*LWDM;                     
   int max_wdmShift = ((K + J)/(2*J))*2;
   int max_k = (K + J)%(2*J) - J; 
   int min_wdmShift = (-K + J)/(2*J);
   int min_k = (-K + J)%(2*J);
   if(min_k<0){
      min_k += 2*J;
      --min_wdmShift;
   }
   min_wdmShift *=2; 
   min_k -= J;
   
   //printf("minShift = %d   maxShift = %d   mink = %d   maxk = %d\n", 
   //   min_wdmShift, max_wdmShift, min_k, max_k);
   
   int mode = (c=='a' || c=='A') ? 2 : 1;
   wavearray<float> amp(mode*(2*K+1));
   wavearray<float> aux;
   if(mode==1) aux.resize(2*K+1);
   
   float* p00 = amp.data;
   float* p90 = aux.data;
   if(mode==2) p90 = amp.data + (mode-1)*(2*K+1);
   
   if(min_wdmShift==max_wdmShift){
      pSS->GetSTFdata(j, td_halo);
      getPixelAmplitudeSSE(m, n, min_k, max_k, p00, false);
      getPixelAmplitudeSSE(m, n, min_k, max_k, p90, true);
   }
   else{
      pSS->GetSTFdata(j - min_wdmShift*(M+1), td_halo);
      getPixelAmplitudeSSE(m, n-min_wdmShift, min_k, J-1, p00, false);
      getPixelAmplitudeSSE(m, n-min_wdmShift, min_k, J-1, p90, true);
      p00 += J - min_k;
      p90 += J - min_k;
   
      for(int i=min_wdmShift+2; i<max_wdmShift; i+=2){
         pSS->GetSTFdata(j - i*(M+1), td_halo);
         getPixelAmplitudeSSE(m, n-i, -J, J-1, p00, false);
         getPixelAmplitudeSSE(m, n-i, -J, J-1, p90, true);
         p00 += 2*J;
         p90 += 2*J;
      }
      pSS->GetSTFdata(j - max_wdmShift*(M+1), td_halo);
      getPixelAmplitudeSSE(m, n-max_wdmShift, -J, max_k, p00, false);
      getPixelAmplitudeSSE(m, n-max_wdmShift, -J, max_k, p90, true);
   }
   
   if(mode==1){
      p00 = amp.data;
      p90 = aux.data;
      for(int i=0; i<=2*K+1; ++i)p00[i] = (p00[i]*p00[i] + p90[i]*p90[i])/2;
   }
   return amp; 
}

template<class DataType_t>
void WDM<DataType_t>::getTFvec(int j, wavearray<float>& r)
{
// fill r with amplitudes needed to compute time-delays (for both quadratures)
// j - global index in the TF map

   if(this->nWWS/this->nSTS!=2) { 
      printf("WDM:getTDAmp() - time delays can not be produced with this TF data.");
      exit(0);
   }
   if(!this->Last()) { 
      printf("WDM:getTDAmp() - time delay filter is not set");
      exit(0);
   }
   DataType_t* TFMap = this->pWWS;
   if(j>=(int)this->nWWS/2) j -= this->nWWS/2;
   
   int M = this->m_Layer;
   int M1 = M+1;
   int L = this->LWDM;
   int BlockSize = T0[0].SSESize()/4; 
   r.resize(6*BlockSize);
   float* data = r.data;
   data += BlockSize/2;
   for(int i=-L; i<=L; ++i)data[i] = TFMap[j + i*M1];
   data += BlockSize;
   --j;
   for(int i=-L; i<=L; ++i)data[i] = TFMap[j + i*M1];
   data += BlockSize;
   j+=2;
   for(int i=-L; i<=L; ++i)data[i] = TFMap[j + i*M1];
   
   j+= this->nWWS/2 - 1;      // 90 degree phase
   data += BlockSize;
   for(int i=-L; i<=L; ++i)data[i] = TFMap[j + i*M1];
   data += BlockSize;
   --j;
   for(int i=-L; i<=L; ++i)data[i] = TFMap[j + i*M1];
   data += BlockSize;
   j+=2;
   for(int i=-L; i<=L; ++i)data[i] = TFMap[j + i*M1];
}
    
   
template<class DataType_t>
double WDM<DataType_t>::TimeShiftTest(int dt)
{  
// a rough CPU time test for time delays
   double res = 0;
   int M = this->m_Layer;
   int N = this->nWWS/2/(M+1);
   int nCoeff = T0[0].Last();
   for(int k=0; k<100; ++k)
   for(int i=nCoeff; i<N-nCoeff; ++i)
      for(int j=0; j<=M; ++j) res += getPixelAmplitude(j, i, dt);
   return res; 
}


template<class DataType_t>
double WDM<DataType_t>::TimeShiftTestSSE(int dt)
{  
// a rough CPU time test for time delays using SSE instructions

   double res; 
   int M = this->m_Layer;
   int N = this->nWWS/2/(M+1);
   int nCoeff = T0[0].Last();
   for(int k=0; k<100; ++k)
   for(int i=nCoeff; i<N-nCoeff; ++i)
      for(int j=0; j<=M; ++j) res += getPixelAmplitudeSSEOld(j, i, dt);
   return res; 
}


template<class DataType_t>
int WDM<DataType_t>::getMaxLevel()
{	
// legacy function 

  return this->m_Level;
}

template<class DataType_t>
int WDM<DataType_t>::getOffset(int level, int layer)
{
// legacy function
	
  return layer;
}

template<class DataType_t>
void WDM<DataType_t>::forward(int level, int layer)
{	
// legacy function, should not be called

  printf("ERROR, WDM::forward called\n");
}

template<class DataType_t>
void WDM<DataType_t>::inverse(int level,int layer)
{	
// legacy function, should not be called

  printf("ERROR, WDM::inverse called\n");
}


template<class DataType_t>
void WDM<DataType_t>::t2w(int MM)
{  
// direct transform
// MM = 0 requests power map of combined quadratures (not amplitudes for both)
   
   //pWWS contains the TD data, nWWS the size;
   static const double sqrt2 = sqrt(2);
  
   int n, m, j, J;
   int M  = this->m_Layer;    // max layer
   int M1 = M+1;
   int M2 = M*2;
   int nWDM = this->m_H;
   int nTS = this->nWWS;
   int KK = MM;
   
   if(MM<=0) MM = M;
   
   // adjust nWWS
   this->nWWS += this->nWWS%MM ? MM-this->nWWS%MM : 0;      
   
   // initialize time series with boundary conditions (mirror)
   m = this->nWWS+2*nWDM;
   double* ts = (double*) _mm_malloc(m*sizeof(double), 16);  // align to 16-byte for SSE
   for(n=0; n<=nWDM; n++) ts[nWDM-n] = (double)(this->pWWS[n]);
   for(n=0; n < nTS; n++) ts[nWDM+n] = (double)(this->pWWS[n]);
   for(n=0; n<int(m - nWDM-nTS); n++) ts[n+nWDM+nTS] = (double)(this->pWWS[nTS-n-1]);

   double* pTS = ts + nWDM;
   
   
   // create aligned symmetric arrays
   double* wdm = (double*) _mm_malloc((nWDM)*sizeof(double),16);  // align to 16-byte for SSE
   double* INV = (double*) _mm_malloc((nWDM)*sizeof(double),16);  // align to 16-byte for SSE
   for(n=0; n<nWDM; n++) {
      wdm[n] = wdmFilter.data[n];
      INV[n] = wdmFilter.data[nWDM-n-1];
   }
   double* WDM = INV+nWDM-1;
   
   // reallocate TF array
   int N = int(this->nWWS/MM);
   int L = KK<0 ? 2*N*M1 : N*M1;
   this->m_L = KK<0 ? this->m_H : 0;
   DataType_t* pWDM = (DataType_t *)realloc(this->pWWS,L*sizeof(DataType_t));
   this->release();
   this->allocate(size_t(L), pWDM);
   
   double* re = new double[M2];
   double* im = new double[M2];

   double* reX = (double*) _mm_malloc(M2 * sizeof(double), 16);  // align to 16-byte for SSE
   double* imX = (double*) _mm_malloc(M2 * sizeof(double), 16);  // align to 16-byte for SSE
 
   DataType_t* map00 = pWDM;
   DataType_t* map90 = pWDM + N*M1;
    
   int odd = 0;
   int sign,kind;  sign=0; // dummy parameters
   TFFTRealComplex fft(1,&M2,false);
   fft.Init("ES",sign, &kind);          // EX - optimized, ES - least optimized
      
   for(n=0; n<N; n++){
      /* 
      for(m=0; m<M2; m++) {re[m] = 0; im[m] = 0.;}    
      for(j=0; j<nWDM-1; j+=M2) {
         J = M2 + j; 
         for(m=0; m<M2; m++) 
            re[m]    += *(pTS+j+m)*wdm[j+m] + *(pTS-J+m)*wdm[J-m];
      }
      re[0] += wdm[J]*pTS[J];           
      printf("%d %.16f  %.16f  /",n,re[0],re[1]);
      */

      for(m=0; m<M2; m++) {reX[m] = 0; imX[m] = 0.;}    
      
      for(j=0; j<nWDM-1; j+=M2) {
         J = M2 + j;         
         
         __m128d* PR = (__m128d*) reX;
         for(m=0; m<M2; m+=2) {
            __m128d ptj = _mm_loadu_pd(pTS+j+m);             // non-aligned pTS array
            __m128d pTJ = _mm_loadu_pd(pTS-J+m);
            __m128d pwj = _mm_load_pd(wdm+j+m);              // aligned wdm and WDM arrays
            __m128d pWJ = _mm_load_pd(WDM-J+m);
            __m128d pjj = _mm_mul_pd(ptj,pwj);
            __m128d pJJ = _mm_mul_pd(pTJ,pWJ);
            *PR = _mm_add_pd(*PR, _mm_add_pd(pjj,pJJ));
            PR++;
         }     
                  
      }
      
      reX[0] += wdm[J]*pTS[J];           
//      printf("%.16f  %.16f  \n",reX[0],reX[1]);

      fft.SetPoints(reX);
      fft.Transform();
      fft.GetPointsComplex(reX, imX);
/*
      wavefft(re,im, M2,M2,M2, 1); // InvFFT(reX, imX, M2);

      printf("%d %.16f  %.16f  /",n,re[0],im[0]);
      printf("%.16f  %.16f  \n",reX[0],imX[0]);
      printf("%d %.16f  %.16f  /",n,re[1],im[1]);
      printf("%.16f  %.16f  \n",reX[1],imX[1]);
      printf("%d %.16f  %.16f  /",n,re[M],im[M]);
      printf("%.16f  %.16f  \n",reX[M],imX[M]);
*/    
      reX[0] = imX[0] = reX[0]/sqrt2; 
      reX[M] = imX[M] = reX[M]/sqrt2; 
    
// with fftw replace im with -im

      if(KK<0) {
         for(int m=0; m<=M; ++m) {
            if((m+odd) & 1){
               map00[m] =  sqrt2*imX[m];
               map90[m] =  sqrt2*reX[m];
            }
            else{
               map00[m] =  sqrt2*reX[m];
               map90[m] = -sqrt2*imX[m];
            }
         }
      }

      else                    // power map
         for(int m=0; m<=M; m++) 
            map00[m] = reX[m]*reX[m]+imX[m]*imX[m];
      

      odd = 1 - odd; 
      pTS += MM;
      map00 += M1;
      map90 += M1;
   }
   
  this->m_Level = this->m_Layer;
      
  _mm_free(reX); 
  _mm_free(imX); 
  _mm_free(wdm); 
  _mm_free(INV); 
  _mm_free(ts); 
  
  delete [] re; 
  delete [] im;
  
}



template<class DataType_t>
void WDM<DataType_t>::w2t(int flag)
{  
// inverse transform
// flag = -2 indicates that the Quadrature coefficients will be used

   if(flag==-2){
      w2tQ(0);
      return;
   }
  int n, j;
  int M = this->m_Layer;
  int M2 = M*2;
  int N = this->nWWS/(M2+2);
  int nWDM = this->m_H; 
  int nTS = M*N + 2*nWDM;

  double  sqrt2 = sqrt(2.); 
  double* reX = new double[M2];
  double* imX = new double[M2];
  double* wdm = wdmFilter.data;
  DataType_t* TFmap = this->pWWS;	
	
  if(M*N/this->nSTS!=1) {
    printf("WDM<DataType_t>::w2t: Inverse is not defined for the up-sampled map\n");
    exit(0);
  }

  wavearray<DataType_t> ts(nTS);
  DataType_t* tmp = ts.data + nWDM;

  int odd = 0;
  int sign,kind;  sign=0; // dummy parameters
  TFFTComplexReal ifft(1,&M2,false);
  ifft.Init("ES",sign, &kind);          // EX - optimized, ES - least optimized

  // Warning : if wavefft is used than reX/imX must be multiplied by sqrt(2)
  //           if fftw    is used than reX/imX must be divided    by sqrt(2)  
  for(n=0; n<N; ++n){
    for(j = 0; j<M2; ++j) reX[j] = imX[j] = 0;
    for(j=1; j<M; ++j)
      if( (n+j) & 1 ) imX[j] = TFmap[j]/sqrt2;
      else  if(j&1) reX[j] = - TFmap[j]/sqrt2;
      else reX[j] = TFmap[j]/sqrt2;
    
    /* works only for EVEN M now
    if(n & 1);  
    else{
      reX[0] = TFmap[0];
      reX[M] = TFmap[M];
    }
    */
    
    if( (n & 1) == 0 )reX[0] = TFmap[0];
    if( ((n+M) & 1) == 0 )
      if(M&1) reX[M] = -TFmap[M];
      else  reX[M] = TFmap[M];

    ifft.SetPointsComplex(reX,imX);
    ifft.Transform();
    ifft.GetPoints(reX);

//    InvFFT(reX, imX, M2);
    DataType_t* x = tmp + n*M;
    int m = M*(n&1);
    int mm = m;
    
    x[0] += wdm[0]*reX[m];	
    for(j = 1; j<nWDM; ++j){
      ++m;
      if(m==2*M)m=0;
      x[j] += wdm[j]*reX[m]; 
      if(mm==0)mm = 2*M - 1;
      else --mm;
      x[-j] += wdm[j]*reX[mm];
    }
    TFmap += M+1;
  }
  
  DataType_t* pTS = (DataType_t *)realloc(this->pWWS,this->nSTS*sizeof(DataType_t));
  this->release();
  this->allocate(size_t(this->nSTS), pTS);
  for(n=0; n<int(this->nSTS); n++) pTS[n] = tmp[n];
  this->m_Level = 0;    // used in getSlice() function

  delete [] reX;
  delete [] imX;
}


// extra -i in C_k...

template<class DataType_t>
void WDM<DataType_t>::w2tQ(int)
{  
// inverse transform using the quadrature coefficients

  int n, j;
  int M = this->m_Layer;
  int M1 = M+1;
  int M2 = M*2;
  int N = this->nWWS/(M2+2);
  int nWDM = this->m_H; 
  int nTS = M*N + 2*nWDM;

  double  sqrt2 = sqrt(2.); 
  double* reX = new double[M2];
  double* imX = new double[M2];
  double* wdm = wdmFilter.data;
  //Data ype_t* TFmap = this->pWWS;	
	
  DataType_t* map90 = this->pWWS + N*M1;
  
  if(M*N/this->nSTS!=1) {
    printf("WDM<DataType_t>::w2tQ: Inverse is not defined for the up-sampled map\n");
    exit(0);
  }

  wavearray<DataType_t> ts(nTS);
  DataType_t* tmp = ts.data + nWDM;

  int odd = 0;
  int sign,kind;  sign=0; // dummy parameters
  TFFTComplexReal ifft(1,&M2,false);
  ifft.Init("ES",sign, &kind);          // EX - optimized, ES - least optimized

  // Warning : if wavefft is used than reX/imX must be multiplied by sqrt(2)
  //           if fftw    is used than reX/imX must be divided    by sqrt(2)  
  for(n=0; n<N; ++n){
    for(j = 0; j<M2; ++j) reX[j] = imX[j] = 0;
    for(j=1; j<M; ++j)
      if( (n+j) & 1 ) reX[j] = map90[j]/sqrt2;
      else  if(j&1) imX[j] = map90[j]/sqrt2;
      else imX[j] = -map90[j]/sqrt2;
    
    /* works only for EVEN M now
    if(n & 1){
      reX[0] = map90[0];
      reX[M] = map90[M];
    }
    */
    
    if((n+M) & 1)reX[M] = map90[M];
    if(n & 1)reX[0] = map90[0];

    ifft.SetPointsComplex(reX,imX);
    ifft.Transform();
    ifft.GetPoints(reX);
    
//    InvFFT(reX, imX, M2);
    DataType_t* x = tmp + n*M;
    int m = M*(n&1);
    int mm = m;
    
     x[0] += wdm[0]*reX[m];	
    for(j = 1; j<nWDM; ++j){
      ++m;
      if(m==2*M)m=0;
      x[j] += wdm[j]*reX[m]; 
      if(mm==0)mm = 2*M - 1;
      else --mm;
      x[-j] += wdm[j]*reX[mm];
    }
    
    map90 += M+1;
  }
  
  DataType_t* pTS = (DataType_t *)realloc(this->pWWS,this->nSTS*sizeof(DataType_t));
  this->release();
  this->allocate(size_t(this->nSTS), pTS);
  for(n=0; n<int(this->nSTS); n++) pTS[n] = tmp[n];
  this->m_Level = 0;    // used in getSlice() function

  delete [] reX;
  delete [] imX;
}



#define CLASS_INSTANTIATION(class_) template class WDM< class_ >;

CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)

#undef CLASS_INSTANTIATION
