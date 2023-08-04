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


// $Id: wseries.cc,v 1.5 2005/07/01 02:25:58 klimenko Exp $

#define WSeries_CC
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "wseries.hh"

#include "Haar.hh"
#include "Biorthogonal.hh"
#include "Daubechies.hh"
#include "Symlet.hh"
#include "Meyer.hh"
#include "WDM.hh"

ClassImp(WSeries<double>)

using namespace std;

// constructors

template<class DataType_t>
WSeries<DataType_t>::WSeries() : wavearray<DataType_t>()
{
   this->pWavelet = new WaveDWT<DataType_t>();
   this->pWavelet->allocate(this->size(),this->data);
   this->bpp = 1.;
   this->wRate = 0.;
   this->f_low = 0.;
   this->f_high = 0.;
   this->w_mode = 0;
}

template<class DataType_t>
WSeries<DataType_t>::WSeries(const Wavelet &w) : 
wavearray<DataType_t>()
{ 
   this->pWavelet = NULL;
   this->setWavelet(w);
   this->bpp = 1.;
   this->wRate = 0.;
   this->f_low = 0.;
   this->f_high = 0.;
   this->w_mode = 0;
}  

template<class DataType_t>
WSeries<DataType_t>::WSeries(const wavearray<DataType_t>& value, const Wavelet &w) : 
wavearray<DataType_t>(value)
{   
   this->pWavelet = NULL;
   this->setWavelet(w);
   this->bpp = 1.;
   this->wRate = value.rate();
   this->f_low = 0.;
   this->f_high = value.rate()/2.;
   this->w_mode = 0;
}

template<class DataType_t>
WSeries<DataType_t>::WSeries(const WSeries<DataType_t>& value) : 
wavearray<DataType_t>(value)
{
   this->pWavelet = NULL;
   this->setWavelet(*(value.pWavelet));
   this->bpp = value.getbpp();
   this->wRate = value.wRate;
   this->f_low = value.getlow();
   this->f_high = value.gethigh();
   this->w_mode = value.w_mode;
}

// destructor

template<class DataType_t>
WSeries<DataType_t>::~WSeries()
{
   if(this->pWavelet) this->pWavelet->release();
   if(this->pWavelet) delete this->pWavelet;
}

// metadata methods

template<class DataType_t>
int WSeries<DataType_t>::getMaxLevel()
{ 
  int maxlevel = 0;

  if(pWavelet->allocate())
    maxlevel = pWavelet->getMaxLevel();
  
  return maxlevel;
}

// Accessors

//: Get central frequency of a layer
template<class DataType_t> 
double WSeries<DataType_t>::frequency(int i){
// Get central frequency of a layer
// l - layer number. 
//     TF map       binary   dyadic    WDM
// zero layer Fc     dF/2      Fo       0
// non-zero   Fc    +n*dF      Fn     +n*dF
   int I = maxLayer()+1;
   double df = this->rate()/I/4.;
   if(I==1) return this->rate()/2.;
   if(pWavelet->m_WaveType==WDMT) return i*this->rate()/(I-1)/2.;
   if(pWavelet->BinaryTree()) return 2*df*(i+0.5);
   df = this->rate()/(1<<I);
   double f = df;
   while(i--){ f += 2*df; df*=2; }
   return f;
}

template<class DataType_t> 
void WSeries<DataType_t>::mul(WSeries<DataType_t>& w){
// multiply this and w layaer by layer 
// requires the same number of layer samples in this and w
   int I = this->maxLayer()+1;         // layers in this 
   int J = w.maxLayer();               // layers-1 in w
   int i = 0;
   int j = 0;
   double df = (w.frequency(1)-w.frequency(0))/2.;
   double f,F;
   wavearray<DataType_t> x,y;
   bool error = false;
   w.getLayer(x,j);                        // get x array
   while(i<I) {
      f = w.frequency(j)+df;
      F = this->frequency(i);
      if(f<F){
	 if(j==J) {error = true; break;}
	 w.getLayer(x,++j);               // update x array
      }
      this->getLayer(y,i);                // extract layer from this
      if(x.size()!=y.size()) {
	 cout<<"wseries::mul(): "<<x.size()<<" "<<y.size()<<endl;
	 error=true; break;               // error!
      }
      y *= x; 
      this->putLayer(y,i++);              // replace layer in this     
   }
   if(error) {cout<<"wseries::mul() error!\n"; exit(1);}
   return;
}

template<class DataType_t> 
int WSeries<DataType_t>::layer(double f){
//: Get layer index for input frequency
// f - frequency Hz 
//     TF map       binary   dyadic    WDM
// zero layer Fc     dF/2      Fo       0
// non-zero   Fc    +n*dF      Fn     +n*dF

   int I = int(maxLayer()+1);
   if(I<2 || f>=this->rate()/2.) return I;
   double df = this->rate()/(I-1)/2.;
   if(pWavelet->m_WaveType==WDMT) return int((f+df/2.)/df);
   df = this->rate()/I/2.;
   if(pWavelet->BinaryTree()) return int(f/df);
   df = this->rate()/(1<<I);
   double ff = df;
   int i = 0;
   while(ff<f){ ff += 2*df; df*=2; i++; }
   return i;

}

// access to data in wavelet domain
// get wavelet coefficients from a layer with specified frequency index
// if index>0 - get coefficients from the layer = |index| and 0 phase
// if index<0 - get coefficients from the layer = |index| and 90 phase
template<class DataType_t> 
int WSeries<DataType_t>::getLayer(wavearray<DataType_t> &value, double index)
{
  int n = int(fabs(index)+0.001); 
  if(n > maxLayer()) index = maxLayer();   
  slice s = pWavelet->getSlice(index);
  
  if(this->limit(s) <= this->size()){
    value.resize(s.size());
    value.rate(this->wrate());
    value.start(this->start());
    value.stop(this->start()+s.size()/value.rate());
    value.edge(this->edge());
    value.Slice = slice(0,s.size(),1);
    value << (*this)[s];               // get slice of wavelet valarray
    return n;
  }
  else{
    cout<<"WSeries::getLayer(): data length mismatch: "<<this->limit(s)<<" "<<this->size()<<"\n";
    return -1;
  }
}

// access to data in wavelet domain
// put wavelet coefficients into layer with specified frequency index
// if index<0 - put coefficients into layer = |index|
template<class DataType_t> 
void WSeries<DataType_t>::putLayer(wavearray<DataType_t> &value, double index)
{
  slice s = this->pWavelet->getSlice(index);
  
  if( (s.size() < value.size()) || (this->limit(s) > this->size()) ){
    cout<<"WSeries::putLayer(): invalid array size.\n";
  }      
  else{
    (*this)[s] << value;    // put slice into wavelet valarray
  }
}

// mutators

template<class DataType_t>
void WSeries<DataType_t>::setWavelet(const Wavelet &w)
{ 
  if(pWavelet){              // delete old wavelet object
    pWavelet->release();
    delete pWavelet; 
  } 
  
  pWavelet = (WaveDWT<DataType_t> *)w.Clone();
  pWavelet->allocate(this->size(), this->data);
}

template<class DataType_t>
void WSeries<DataType_t>::Forward(int k)
{
   if(pWavelet->allocate()){
      pWavelet->nSTS = pWavelet->nWWS;
      pWavelet->t2w(k);
      if(pWavelet->pWWS != this->data || pWavelet->nWWS != this->Size){
         this->data = pWavelet->pWWS;
         this->Size = pWavelet->nWWS;
         this->Slice = std::slice(0,pWavelet->nWWS,1);
      }
      std::slice s = this->getSlice(0);
      this->wrate(s.size()/(this->stop()-this->start()));
   }
   else{
      throw std::invalid_argument
         ("WSeries::Forward(): data is not allocated");
   }
}

template<class DataType_t> 
void WSeries<DataType_t>::Forward(wavearray<DataType_t> &x, int k)
{
   wavearray<DataType_t>* p = this;
   if(pWavelet->allocate()) pWavelet->release();
   *p = x;
   this->wrate(x.rate());
   f_high = x.rate()/2.;
   pWavelet->allocate(this->size(), this->data);
   pWavelet->reset();
   Forward(k);
}

template<class DataType_t> 
void WSeries<DataType_t>::Forward(wavearray<DataType_t> &x, Wavelet &w, int k)
{
   wavearray<DataType_t>* p = this;
   if(pWavelet->allocate()) pWavelet->release();
   *p = x;
   this->wrate(x.rate());
   f_high = x.rate()/2.;
   setWavelet(w);
   Forward(k);
}

template<class DataType_t> 
void WSeries<DataType_t>::Inverse(int k)
{ 
   if(pWavelet->allocate()){
      pWavelet->w2t(k);
      if(pWavelet->pWWS != this->data || pWavelet->nWWS != this->Size) { 
	this->data = pWavelet->pWWS;
	this->Size = pWavelet->nWWS;
        this->Slice = std::slice(0,pWavelet->nWWS,1);
        this->wrate(this->rate());
      } 
      else {
         std::slice s = this->getSlice(0);
         this->wrate(s.size()/(this->stop()-this->start()));
      }
   }
   else{
      throw std::invalid_argument
      ("WSeries::Inverse(): data is not allocated");
   }
}

template<class DataType_t> 
void WSeries<DataType_t>::bandpass(wavearray<DataType_t> &ts, double flow, double fhigh, int n)
{
// bandpass data and store in TF domain, do not use for WDM
// ts - input time series
// flow - low frequence boundary
// fhigh - high frequency boundary
// n - decomposition level
   if(!this->pWavelet) {
      cout<<"WSeries::bandpass ERROR: no transformation is specified"<<endl;
      exit(1);
   }

   this->Forward(ts,n);
   
   double freq;
   wavearray<DataType_t> wa;
   size_t I = this->maxLayer()+1;
   for(size_t i=0; i<I; i++) {
      freq = this->frequency(i);
      if(freq<flow || freq>fhigh) {
         this->getLayer(wa,i); wa = 0;
         this->putLayer(wa,i);
     }
   }
}

template<class DataType_t> 
void WSeries<DataType_t>::bandpass(double f1, double f2, double a)
{
// set wseries coefficients to a in layers between
// f1 - low frequence boundary
// f2 - high frequency boundary
// f1>0, f2>0 -  zzzzzz..........zzzzzz 	band pass
// f1<0, f2<0 -  ......zzzzzzzzzz...... 	band cut 
// f1<0, f2>0 -  ......zzzzzzzzzzzzzzzz 	low  pass
// f1>0, f2<0 -  zzzzzzzzzzzzzzzz...... 	high pass
// a - value
   int i;
   double dF = this->frequency(1)-this->frequency(0);              // frequency resolution
   double fl = fabs(f1)>0. ? fabs(f1) : this->getlow();
   double fh = fabs(f2)>0. ? fabs(f2) : this->gethigh();
   size_t n  = this->pWavelet->m_WaveType==WDMT ? size_t((fl+dF/2.)/dF+0.1) : size_t(fl/dF+0.1);
   size_t m  = this->pWavelet->m_WaveType==WDMT ? size_t((fh+dF/2.)/dF+0.1)-1 : size_t(fh/dF+0.1)-1;
   size_t M  = this->maxLayer()+1;
   wavearray<DataType_t> w;

   if(n>m) return;

   for(i=0; i<int(M); i++) {                          	// ......f1......f2......

     if((f1>=0 && i>n) && (f2>=0 && i<=m))  continue; 	// zzzzzz..........zzzzzz 	band pass
     if((f1<0 && i<n) || (f2<0 && i>m))     continue; 	// ......zzzzzzzzzz...... 	band cut 
     if((f1<0 && f2>=0 && i<n))             continue; 	// ......zzzzzzzzzzzzzzzz 	low  pass
     if((f1>=0 && f2<0 && i>=m))            continue; 	// zzzzzzzzzzzzzzzz...... 	high pass

     this->getLayer(w,i+0.01); w=a; this->putLayer(w,i+0.01);
     this->getLayer(w,-i-0.01); w=a; this->putLayer(w,-i-0.01);
   }
   return;
}


template<class DataType_t>
double WSeries<DataType_t>::wdmPacket(int patt, char c, TH1F* hist)    
{
// wdmPacket: converts this to WDM packet series described by pattern
// c = 'e' / 'E' - returns pattern / packet energy
// c = 'l' / 'L' - returns pattern / packet likelihood
// c = 'a' / 'A' - returns packet amplitudes
// patt = pattern
// patterns: "/" - chirp, "\" - ringdown, "|" - delta, "*" - pixel 
// param: pattern =  0 - "*" single pixel standard search
// param: pattern =  1 - "3|"  packet
// param: pattern =  2 - "3-"  packet
// param: pattern =  3 - "3/"  packet - chirp
// param: pattern =  4 - "3\"  packet - ringdown
// param: pattern =  5 - "5/"  packet - chirp
// param: pattern =  6 - "5\"  packet - ringdown
// param: pattern =  7 - "3+"  packet
// param: pattern =  8 - "3x"  cross packet
// param: pattern =  9 - "9p"  9-pixel square packet
//        pattern = else - "*" single pixel standard search
// mean of the packet noise distribution is mu=2*K+1, where K is 
// the effective number of pixels in the packet (K may not be integer)
//
   int n,m;
   double aa,ee,EE,uu,UU,dd,DD,em,ss,cc,nn;
   double shape,mean,alp;
   int pattern = abs(patt);
   int J = this->size()/2;          // energy map size
   if(pattern==0) return 1.;
   if(!this->isWDM() || (2*J)/this->xsize()==1) exit(0);

   WSeries<DataType_t> in = *this;
   int M  = in.maxLayer()+1;                        // number of layers
   int jb = int(in.Edge*in.wrate()/4.)*M;
   if(jb<4*M) jb = 4*M;
   int je = J-jb;
   int p[] = {0,0,0,0,0,0,0,0,0};
   double df = in.resolution(0);
   int mL = int(in.getlow()/df+0.1);
   int mH = int(in.gethigh()/df+0.1);

   if(c!='a'||c!='A') this->resize(J);                  // convert to energy array             

   if(pattern==1) {                                     // "3|" vertical line 
      p[1]=1; p[2]=-1;                                  // 'p': a=2.8,b=0.90
      shape = mean = 3.;                              	// 'e': a=3.0,b=1.00
      mL+=1; mH-=1;
   } 
   else if(pattern==2) {                                // "3-" horizontal line 
      p[1]=M; p[2]=-M;                                  // 'p': a=2.0,b=1.41
      shape = mean = 3.;                              	// 'e': a=2.2,b=1.38
   } 
   else if(pattern==3) {                                // "3/" chirp packet 
      p[1]=M+1; p[2]=-M-1;                              // 'p': a=2.7,b=0.91
      shape = mean = 3.;                              	// 'e': a=3.0,b=1.00
      mL+=1; mH-=1;
   } 
   else if(pattern==4) {                                // "3\" chirp packet 
      p[1]=-M+1; p[2]=M-1;                              // 'p': a=2.7,b=0.91
      shape = mean = 3.;                              	// 'e': a=3.0,b=1.00
      mL+=1; mH-=1;
   } 
   else if(pattern==5) {                                // "5/"-pattern=5 
      p[1]=M+1; p[2]=-M-1; p[3]=2*M+2; p[4]=-2*M-2;     // 'e': a=5.0,b=1.00            
      shape = mean = 5.; //shape=4.8; mean=4.8*0.95;      // 'p': a=4.8,b=0.95
      mL+=2; mH-=2;
   } 
   else if(pattern==6) {                                // "5\"-pattern=-5 
      p[1]=-M+1; p[2]=M-1; p[3]=-2*M+2; p[4]=2*M-2;     // 'e': a=5.0,b=1.00            
      shape = mean = 5.;                              	// 'p': a=4.8,b=0.95
      mL+=2; mH-=2;
   } 
   else if(pattern==7) {                                // "3+"  packet
      p[1]=1; p[2]=-1; p[3]=M; p[4]=-M;                 // 'e': a=3.9,b=1.28
      shape = mean = 5.;                                // 'p': a=3.6,b=1.28
      mL+=1; mH-=1;
   } 
   else if(pattern==8) {                                // "3x" cross packet 
      p[1]=M+1; p[2]=-M+1; p[3]= M-1; p[4]=-M-1;        // 'p': a=2.8,b=0.90         
      shape = mean = 5.;                                // 'e': a=3.0,b=1.00
      mL+=1; mH-=1;
   } 
   else if(pattern==9) {                                // "9*" 9-pixel square 
      p[1]=1; p[2]=-1; p[3]=M; p[4]=-M;                 // 'p': a=5.6,b=1.57 
      p[5]=M+1; p[6]=M-1; p[7]=-M+1; p[8]=-M-1;         // 'e': a=5.8,b=1.55        
      shape = mean = 9.;                              
      mL+=1; mH-=1;
   } else { shape = mean = 1.; }

   DataType_t *q;                         // pointers to 00 phase 
   DataType_t *Q;                         // pointers to 90 phase 

   for(int j=jb; j<je; j++) {
      m = j%M;    
      if(m<mL || m>mH) {this->data[j]=0.; continue;}
      q = in.data+j; Q = q+J; 
      ss = q[p[1]]*Q[p[1]]+q[p[2]]*Q[p[2]]+q[p[3]]*Q[p[3]]+q[p[4]]*Q[p[4]]
         + q[p[5]]*Q[p[5]]+q[p[6]]*Q[p[6]]+q[p[7]]*Q[p[7]]+q[p[8]]*Q[p[8]];
      ee = q[p[1]]*q[p[1]]+q[p[2]]*q[p[2]]+q[p[3]]*q[p[3]]+q[p[4]]*q[p[4]]
         + q[p[5]]*q[p[5]]+q[p[6]]*q[p[6]]+q[p[7]]*q[p[7]]+q[p[8]]*q[p[8]];
      EE = Q[p[1]]*Q[p[1]]+Q[p[2]]*Q[p[2]]+Q[p[3]]*Q[p[3]]+Q[p[4]]*Q[p[4]]
         + Q[p[5]]*Q[p[5]]+Q[p[6]]*Q[p[6]]+Q[p[7]]*Q[p[7]]+Q[p[8]]*Q[p[8]];
      ss+= q[p[0]]*Q[p[0]]*(mean-8);
      ee+= q[p[0]]*q[p[0]]*(mean-8);
      EE+= Q[p[0]]*Q[p[0]]*(mean-8);

      cc = ee-EE; ss*=2;
      nn = sqrt(cc*cc+ss*ss);
      if(ee+EE<nn) nn=ee+EE;
      aa = sqrt((ee+EE+nn)/2) + sqrt((ee+EE-nn)/2);
      em = (c=='e'||c=='l'||mean==1.) ? (ee+EE)/2. : aa*aa/4;
      alp = shape-log(shape)/3; 
      if(c=='l'||c=='L') {
      	 em*=shape/mean; 
	 if(em<alp) {em=0; continue;}
      	 em-= alp*(1+log(em/alp));
      }
      if(c=='a'||c=='A') {
	 cc/=nn; ss/=nn;
	 nn = sqrt((1+cc)*(1+cc)+ss*ss)/2; 
	 this->data[j]=aa*cc/nn; this->data[j+J]=aa*ss/nn/2;
      } else {this->data[j] = em;} 
      if(hist && em>0.01) hist->Fill(sqrt(em));
   }
   return shape;
}


template<class DataType_t> 
double WSeries<DataType_t>::maxEnergy(wavearray<DataType_t> &ts, Wavelet &w, 
				      double dT, int N, int pattern, TH1F* hist)
{
// maxEnergy: put maximum energy of delayed samples in this 
// param: wavearray - input time series
// param: wavelet   - wavelet used for the transformation
// param: double    - range of time delays
// param: int       - downsample factor to obtain coarse TD steps
// param: int       - clustering pattern
   wavearray<DataType_t> xx; xx=ts;
   WSeries<DataType_t> tmp;
   int K = int(ts.rate()*fabs(dT));   // half number of time delays
   double shape = 1.;

   if(w.m_WaveType != WDMT) {
      cout<<"wseries::maxEnergy(): illegal wavelet\n";
      exit(0);
   }

   this->Forward(ts,w,0);

   if(abs(pattern)) {
      *this = 0.;
      tmp.Forward(ts,w);
      tmp.setlow(this->getlow());
      tmp.sethigh(this->gethigh());
      shape=tmp.wdmPacket(pattern,'E');   
      this->max(tmp);      
      for(int k=N; k<=K; k+=N) {
	 xx.cpf(ts,ts.size()-k,k);
	 tmp.Forward(xx,w);
	 tmp.setlow(this->getlow());
	 tmp.sethigh(this->gethigh());
	 tmp.wdmPacket(pattern,'E');   
	 this->max(tmp);      
	 xx.cpf(ts,ts.size()-k,0,k);
	 tmp.Forward(xx,w);
	 tmp.setlow(this->getlow());
	 tmp.sethigh(this->gethigh());
	 tmp.wdmPacket(pattern,'E');      
	 this->max(tmp);      
      } 
   } else {                        // extract single pixels
      for(int k=N; k<=K; k+=N) {
	 xx.cpf(ts,ts.size()-k,k);
	 tmp.Forward(xx,w,0);
	 this->max(tmp);   
	 xx.cpf(ts,ts.size()-k,0,k);
	 tmp.Forward(xx,w,0);
	 this->max(tmp);      
      }
   }

   int M  = tmp.maxLayer()+1;
   this->getLayer(xx,0.1); xx=0;
   this->putLayer(xx,0.1);
   this->getLayer(xx,M-1); xx=0;
   this->putLayer(xx,M-1);

   int m = abs(pattern);
   if(m==5 || m==6 || m==9) {
      this->getLayer(xx,1); xx=0;
      this->putLayer(xx,1);
      this->getLayer(xx,M-2); xx=0;
      this->putLayer(xx,M-2);
   }

   if(!pattern) return 1.;
   return Gamma2Gauss(hist);
}

template<class DataType_t> 
double WSeries<DataType_t>::Gamma2Gauss(TH1F* hist)
{
// finds parameters shape and scale for noise Gamma statistic
// convert from Gamma to Gaussian statistic 

   WSeries<DataType_t> tmp = *(this);
   int M  = tmp.maxLayer()+1;
   int nL = size_t(tmp.Edge*tmp.wrate()*M);
   int nn = tmp.size();
   int nR = nn-nL-1;                                       // right boundary
   double fff = (nR-nL)*tmp.wavecount(0.001)/double(nn);   // zero fraction
   double med = tmp.waveSplit(nL,nR,nR-int(0.5*fff));      // distribution median
   double amp, aaa, bbb, rms, alp;
   
   aaa = bbb = 0.; nn = 0;
   for(int i=nL; i<nR; i++) {                              // get Gamma shape
      amp = (double)this->data[i];
      if(amp>0.01 && amp<20*med) {
	 aaa += amp; bbb += log(amp); nn++;
      }
   }
   alp = log(aaa/nn)-bbb/nn;
   alp = (3-alp+sqrt((alp-3)*(alp-3)+24*alp))/12./alp;
   double avr = med*(3*alp+0.2)/(3*alp-0.8);               // get Gamma mean
   //cout<<"debug0: "<<fff<<" "<<avr<<" "<<med<<" "<<alp<<endl;

   double ALP = med*alp/avr;
   for(int i=0; i<this->size(); i++) {
      amp = (double)this->data[i]*alp/avr;
      if(amp<ALP) {this->data[i]=0.; continue;}
      this->data[i] = amp-ALP*(1+log(amp/ALP));
      //if(hist && i>nL && i<nR) hist->Fill(this->data[i]);
   }
   tmp = *(this);                                         
   fff = tmp.wavecount(1.e-5,nL);                          // number of events excluding 0
   rms = 1./tmp.waveSplit(nL,nR,nR-int(0.3173*fff));       // 1 over distribution rms
   //cout<<"debug1: "<<fff<<"  "<<avr<<" "<<rms<<" "<<aaa<<endl;
   for(int i=0; i<this->size(); i++) {
      this->data[i] *= rms;
      if(hist && i>nL && i<nR) hist->Fill(sqrt(this->data[i]));
   }
   return ALP;
}


template<class DataType_t> 
void WSeries<DataType_t>::wavescan(WSeries<DataType_t>** pws, int N, TH1F* hist)
{
// create a wavescan object
// produce multi-resolution TF series of input time series x
// pws   - array of pointers to input  time-frequency series
// N     - number of resolutions
// hist  - diagnostic histogram

   std::vector<int> vM;                             // vector to store number of layers
   std::vector<int> vJ;                             // vector to store TF map size
   std::vector<double> vR;                          // vector to store number of layers
   std::vector<double> vF;                          // vector to store resolutions
   int mBand, mRate, level;
   int nn,j,J;
   double time, freq, a,A,ee,EE,ss,cc,gg,b,B;
   double mean  = 2.*N-1;                          // Gamma distribution mean
   double shape = N-log(N)/N;                      // Gamma distribution shape parameter

   mBand = 0; mRate = 0;
   for(int n=0; n<N; n++) {                        // get all TF data
      vF.push_back(pws[n]->resolution());         // save frequency resolution
      vR.push_back(pws[n]->wrate());               // save frequency resolution
      vM.push_back(pws[n]->maxLayer()+1);
      vJ.push_back(pws[n]->size()/2);
      if(vM[n]>mBand) {mBand = vM[n]; nn=n;}
      if(pws[n]->wrate()>mRate) mRate = pws[n]->wrate();
   }

// set super TF map

   time = pws[nn]->size()/pws[nn]->wrate()/2;        // number of ticks
   J = mRate*int(time+0.1);
   cout<<"debug1: "<<mBand<<" "<<mRate<<" "<<J<<" "<<pws[nn]->size()<<endl;
   level = 0;
   *this = *pws[nn];
   this->resize(J);
   this->setLevel(mBand-1);
   this->wrate(mRate);
   //this->rate(mRate*(mBand-1));
   this->pWavelet->nWWS = J;
   this->pWavelet->nSTS = (J/mBand)*(mBand-1);

   int nL = int(this->Edge*mRate*mBand);              // left boundary
   int nR = this->size()-nL-1;                        // right boundary

   for(int i=0; i<this->size(); i++) {                // loop over super-TF-map
      time = double(i/mBand)/mRate;                   // discrete time in seconds
      freq = (i%mBand)*vF[nn];                        // discrete frequency
      ss=ee=EE=0.;
      for(int n=1; n<N; n++) {                        // loop over TF-maps
	 j = int(time*vR[n-1]+0.1)*vM[n-1];           // pixel tick index in TF map
	 j+= int(freq/vF[n-1]+0.1);                   // add pixel frequency index
	 a = pws[n]->data[j];                         // 00 phase amplitude 
	 A = pws[n]->data[j+vJ[n-1]];                 // 90 phase amplitude 
	 j = int(time*vR[n]+0.1)*vM[n];               // pixel tick index in TF map
	 j+= int(freq/vF[n]+0.1);                     // add pixel frequency index
	 b = pws[n]->data[j];                         // 00 phase amplitude 
	 B = pws[n]->data[j+vJ[n]];                   // 90 phase amplitude 
	 ss = 2*(a*A+b*B);
	 ee = 2*(a*a+b*b); 
	 EE = 2*(a*a+b*b); 
EE+=A*A;
      cc = ee-EE;
      gg = sqrt(cc*cc+4*ss*ss);
      a = sqrt((ee+EE+gg)/2);
      A = sqrt((ee+EE-gg)/2);
      }


      this->data[i] = (a+A)*(a+A)/mean/2.;
      if(hist && i>nL && i<nR) hist->Fill(this->data[i]);
   }
   //this->Gamma2Gauss(shape,hist);
   return;
}


//: operators =

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator=(const wavearray<DataType_t>& a)
{
   wavearray<DataType_t>* p = this;
   if( pWavelet->allocate() ) pWavelet->release();
   if(p->size() != a.size()) pWavelet->reset();
   *p = a;
   this->rate(a.rate());
   this->wrate(0.);
   f_high = a.rate()/2.;
   pWavelet->allocate(this->size(), this->data);
   return *this;
}

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator=(const WSeries<DataType_t>& a)
{
   const wavearray<DataType_t>* p = &a;
   wavearray<DataType_t>* q = this;
   *q = *p;
   setWavelet(*(a.pWavelet));
   bpp = a.getbpp();
   wRate = a.wrate();
   f_low = a.getlow();
   f_high = a.gethigh();
   w_mode = a.w_mode;
   return *this;
}

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator[](const std::slice& s)
{
   this->Slice = s;
   if(this->limit() > this->size()){
      cout << "WSeries::operator[]: Illegal argument: "<<this->limit()<<" "<<this->size()<<"\n";
      this->Slice = std::slice(0,this->size(),1);
   }
   return *this;
}

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator=(const DataType_t a)
{ this->wavearray<DataType_t>::operator=(a); return *this; }

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator*=(const DataType_t a)
{ this->wavearray<DataType_t>::operator*=(a); return *this; }

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator-=(const DataType_t a)
{ this->wavearray<DataType_t>::operator-=(a); return *this; }

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator+=(const DataType_t a)
{ this->wavearray<DataType_t>::operator+=(a); return *this; }

//template<class DataType_t>
//WSeries<DataType_t>& WSeries<DataType_t>::
//operator*=(wavearray<DataType_t> &a)
//{ this->wavearray<DataType_t>::operator*=(a); return *this; }

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator-=(wavearray<DataType_t> &a)
{ this->wavearray<DataType_t>::operator-=(a); return *this; }

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator+=(wavearray<DataType_t> &a)
{ this->wavearray<DataType_t>::operator+=(a); return *this; }

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator*=(WSeries<DataType_t>& a)
{
   size_t i;
   wavearray<DataType_t> x;
   wavearray<DataType_t>* p  = (wavearray<DataType_t>*)this;
   wavearray<DataType_t>* pa = (wavearray<DataType_t>*)&a;
   size_t max_layer = (maxLayer() > a.maxLayer()) ? a.maxLayer() : maxLayer();

   if(pWavelet->m_TreeType != a.pWavelet->m_TreeType){
     cout<<"WSeries::operator* : wavelet tree type mismatch."<<endl;
     return *this;
   }

   if(this->size()==a.size()) { 
     this->wavearray<DataType_t>::operator*=(*pa); 
     return *this; 
   }

   for(i=0; i<= max_layer; i++)
     (*p)[pWavelet->getSlice(i)] *= (*pa)[a.pWavelet->getSlice(i)];

   return *this;
}

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator+=(WSeries<DataType_t>& a)
{
   size_t i;
   wavearray<DataType_t>* p  = (wavearray<DataType_t>*)this;
   wavearray<DataType_t>* pa = (wavearray<DataType_t>*)&a;
   size_t max_layer = (maxLayer() > a.maxLayer()) ? a.maxLayer() : maxLayer();

   if(pWavelet->m_TreeType != a.pWavelet->m_TreeType){
     cout<<"WSeries::operator+ : wavelet tree type mismatch."<<endl;
     return *this;
   }

   if(this->size()==a.size()) { 
     this->wavearray<DataType_t>::operator+=(*pa); 
     return *this; 
   }

   for(i=0; i<= max_layer; i++)
       (*p)[pWavelet->getSlice(i)] += (*pa)[a.pWavelet->getSlice(i)];

   return *this;
}

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator-=(WSeries<DataType_t>& a)
{
   size_t i;
   wavearray<DataType_t>* p  = (wavearray<DataType_t>*)this;
   wavearray<DataType_t>* pa = (wavearray<DataType_t>*)&a;
   size_t max_layer = (maxLayer() > a.maxLayer()) ? a.maxLayer() : maxLayer();

   if(pWavelet->m_TreeType != a.pWavelet->m_TreeType){
     cout<<"WSeries::operator- : wavelet tree type mismatch."<<endl;
     return *this;
   }

   if(this->size()==a.size()) { 
     this->wavearray<DataType_t>::operator-=(*pa); 
     return *this; 
   }

   for(i=0; i<= max_layer; i++)
       (*p)[pWavelet->getSlice(i)] -= (*pa)[a.pWavelet->getSlice(i)];

   return *this;
}

template<class DataType_t>
WSeries<DataType_t>& WSeries<DataType_t>::operator*=(wavearray<DataType_t>& a)
{
   size_t i;
   wavearray<DataType_t>* p  = (wavearray<DataType_t>*)this;
   size_t max_layer = maxLayer()+1;
   
   if(max_layer == a.size()) {
     for(i=0; i< max_layer; i++) {
       (*p)[pWavelet->getSlice(i)] *= a.data[i];
     }     
   }
   else if(this->size()==a.size()) { 
     this->wavearray<DataType_t>::operator*=(a); 
     return *this; 
   }
   else cout<<"WSeries::operator* - no operation is performed"<<endl;

   return *this;
}


//: Dumps data array to file *fname in ASCII format.
template<class DataType_t>
void WSeries<DataType_t>::Dump(const char *fname, int app)
{
  wavearray<DataType_t> a;
  int i,j;
  int n = this->size();
  int m = this->maxLayer()+1;
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
    fprintf( fp,"# number of layers: -n %d \n", m );
  }

  for (i = 0; i < m; i++) {
    this->getLayer(a,i);
    n = (int)a.size();
    for(j = 0; j < n; j++) fprintf( fp,"%e ", (float)a.data[j]);
    fprintf( fp,"\n");
  }
  fclose(fp); 
}


template<class DataType_t>
void WSeries<DataType_t>::resize(unsigned int n)
{
   wavearray<DataType_t>* p = this;
   if( pWavelet->allocate() ) pWavelet->release();
   p->wavearray<DataType_t>::resize(n);
   pWavelet->allocate(this->size(), this->data);
   pWavelet->reset();
   bpp = 1.;
   f_low = 0.;
   wRate = this->rate();
   f_high = this->rate()/2.;
}

template<class DataType_t>
void WSeries<DataType_t>::resample(double f, int nF)
{
   wavearray<DataType_t>* p = this;
   if( pWavelet->allocate() ) pWavelet->release();
   p->wavearray<DataType_t>::resample(f,nF);
   pWavelet->allocate(this->size(), this->data);
   pWavelet->reset();
   bpp = 1.;
   f_low = 0.;
   wRate = p->wavearray<DataType_t>::rate();
   f_high = p->wavearray<DataType_t>::rate()/2.;
}

template<class DataType_t>
double WSeries<DataType_t>::coincidence(WSeries<DataType_t>& a, int t, int f, double threshold)
{
#if !defined (__SUNPRO_CC)
   int i;
   int j;
   int u;
   int v;
   float* q = NULL;
   float* p = NULL; 

   int is, ie, js, je;
   
   wavearray<DataType_t> x;
   wavearray<DataType_t> y;

   float energy;

   if(!pWavelet->BinaryTree()) return 1.;

   int ni = maxLayer()+1;
   int nj = this->size()/ni;
   int n = ni-1;
   int m = nj-1;

   bool CROSS = t<0 || f<0;

   t = abs(t);
   f = abs(f);

   float A[ni][nj];
   float B[ni][nj];

   for(i=0; i<=n; i++){
      p  = A[i]; 
      q  = B[i];
      a.getLayer(x,i);	
        getLayer(y,i);	

      for(j=0; j<=m; j++){
	 p[j] = (float)x.data[j];
	 q[j] = (float)y.data[j];
      }
   }

   for(i=0; i<=n; i++){
      p  = A[i]; 
      q  = B[i];
      a.getLayer(x,i);
        getLayer(y,i);
      
      for(j=0; j<=m; j++){

	 if(p[j]==0. && q[j]==0.) continue;

	 is = i-f<0 ? 0 : i-f;
	 js = j-t<0 ? 0 : j-t;
	 ie = i+f>n ? n : i+f;
	 je = j+t>m ? m : j+t;

	 energy = 0.;
	 if(x.data[j]!=0.) {
	    for(u=is; u<=ie; u++)
	       for(v=js; v<=je; v++){
		  if(CROSS && !(i==u || j==v)) continue;
		  if(B[u][v]!=0.) energy += log(fabs(B[u][v])); 
	       }
	    if(energy < threshold) x.data[j]=0;
	 }

	 energy = 0.;
	 if(y.data[j]!=0.) {
	    for(u=is; u<=ie; u++)
	       for(v=js; v<=je; v++){
		  if(CROSS && !(i==u || j==v)) continue;
		  if(A[u][v]!=0.) energy += log(fabs(A[u][v])); 
	       }
	    if(energy < threshold) y.data[j]=0;
	 }
	 
	 if(y.data[j]==0. && x.data[j]!=0.) y.data[j]=DataType_t(a.size());

      }

      putLayer(y,i);

   }
#endif
   return 0.;
}

template<class DataType_t>
double WSeries<DataType_t>::Coincidence(WSeries<DataType_t>& a, double w, double So)
{
   size_t i,k,m;
   size_t event = 0;
   size_t step;
   size_t N = a.size();               
   int n;
   
   double E,S;
   bool pass;

   slice x,y;
   DataType_t* p = NULL;
   DataType_t* q = NULL;
   DataType_t* P = NULL;
   DataType_t* Q = NULL;

   if(pWavelet->m_TreeType != a.pWavelet->m_TreeType){
     cout<<"WSeries::operator- : wavelet tree type mismatch."<<endl;
     return 0.;
   }

   size_t max_layer = (maxLayer() > a.maxLayer()) ? a.maxLayer() : maxLayer();

   for(k=0; k<= max_layer; k++){

      x =   getSlice(k);
      y = a.getSlice(k);

      if(x.size()   != y.size())   continue;
      if(x.stride() != y.stride()) continue;
      if(x.start()  != y.start())  continue;

      step = x.stride();
      n = int(w*a.rate()/2./step);     // 2*(n+1) - coincidence window in pixels 
      if(n < 0) n = 0;                 // not negative n
      if(!n && w>=0.) n++;             // min 0 vs min 3 pixels
      S = log(float(n));               // threshold correction
      S = So + 2.*S/3.;                // threshold
//      S = So + 0.5*log(2*n+1.);      // threshold
//      S = So + S*S/(S+1);            // threshold
      P = a.data+x.start();
      Q = a.data+x.start()+step*(x.size()-1);
      n *= step;

      for(i=x.start(); i<N; i+=step)
      {
	 if(this->data[i] == 0.) continue;

	 p = a.data+i-n;               // start pointer
	 if(p<P) p = P;
	 q = a.data+i+n;               // stop pointer
	 if(q>Q) q = Q;

// calculate total likelihood and number of black pixels
// and set threshold on significance

	 pass = false;
	 E = 0.; m = 0;

	 while(p <= q) {
//	    if(*p>S) { pass = true; break; }
	    if(*p>0) { E += *p; m++; }
	    p += step;
	 }

	 if(m>0 && !pass) {
	    if(gammaCL(E,m) > S-log(double(m))) pass = true;
	 }

	 if(pass) event++;
	 else     this->data[i] = 0;
      }
   }

// handle wavelets with different number of layers

   if(size_t(maxLayer())>max_layer){
      wavearray<DataType_t>* pw  = (wavearray<DataType_t>*)this;
      for(k=max_layer+1; k<= size_t(maxLayer()); k++) {
	(*pw)[getSlice(k)] = 0;
      }
   }

   return double(event)/this->size();
}


template<class DataType_t>
void WSeries<DataType_t>::median(double t, bool r)
{
   int i;
   int M = maxLayer()+1;

   for(i=0; i<M; i++){
      this->setSlice(getSlice(i)); 
      wavearray<DataType_t>::median(t,NULL,r);
   }

   std::slice S(0,this->size(),1);
   this->setSlice(S);
   return;
}


template<class DataType_t>
void WSeries<DataType_t>::lprFilter(double T, int mode, double stride, double offset)
{
   if(offset<T) offset = T;
   size_t i;
   size_t M = maxLayer()+1;

   wavearray<DataType_t> a;

   for(i=0; i<M; i++){
      getLayer(a,i);
      if(mode<2) a.lprFilter(T,mode,stride,offset);
      else       a.spesla(T,stride,offset);
      putLayer(a,i);
   } 
   return;
}



template<class DataType_t>
WSeries<double> WSeries<DataType_t>::white(double t, int mode, double offset, double stride)
{
//: tracking of noise non-stationarity and whitening. 
// param 1 - time window dT. if = 0 - dT=T, where T is wavearray duration - 2*offset
// param 2 - mode: 0 - no whitening, 1 - single whitening, >1 - double whitening
//           mode < 1 - whitening of guadrature (WDM wavelet only) 
// param 3 - boundary offset 
// param 4 - noise sampling interval (window stride)  
//           the number of measurements is k=int((T-2*offset)/stride)
//           if stride=0, then stride is set to dT
// return: noise array if param2>0, median if param2=0
// what it does: each wavelet layer is devided into k intervals.
// The data for each interval is sorted and the following parameters 
// are calculated: median and the amplitude 
// corresponding to 31% percentile (wp). Wavelet amplitudes (w) are 
// normalized as  w' = (w-median(t))/wp(t), where median(t) and wp(t)
// is a linear interpolation between (median,wp) measurements for
// each interval. 
   int i;
   double segT = this->stop()-this->start();
   if(t <= 0.) t = segT-2.*offset;
   int M = maxLayer()+1;
   int m = mode<0 ? -1 : 1;

   this->w_mode = abs(mode);                  
   if(stride > t || stride<=0.) stride = t;                  
   size_t K = size_t((segT-2.*offset)/stride);   // number of noise measurement minus 1
   if(!K) K++;                                   // special case   

   Wavelet* pw = pWavelet->Clone();
   wavearray<DataType_t> a;
   wavearray<DataType_t> aa;
   wavearray<double>     b(M*(K+1));
   WSeries<double>       ws(b,*pw);

   for(i=0; i<M; i++){
      getLayer(a,(i+0.01)*m);
      if(!w_mode) {                               // use square coefficients
         getLayer(aa,-(i+0.01)); aa*=aa; a*=a; a+=aa; 
      }
      b = a.white(t,w_mode,offset,stride);
      if(b.size() != K+1) cout<<"wseries::white(): array size mismatch\n";
      ws.putLayer(b,i);
      if(w_mode) putLayer(a,i*m);
   } 

   ws.start(b.start());
   ws.stop(b.stop());
   ws.wrate(b.rate());
   ws.rate(this->rate());
   ws.setlow(0.);
   ws.sethigh(this->rate()/2.);

   delete pw;
   return ws;
}

template<class DataType_t>
bool WSeries<DataType_t>::white(WSeries<double> nRMS, int mode)
{
// whiten wavelet series by using TF noise rms array
// if mode <0 - access qudrature WDM data
// if abs(mode) <2 - single, otherwise double whitening

   size_t i,j,J,K;

  if(!nRMS.size()) return false;

  slice S,N;

  int k;
  int m = mode<0 ? -1 : 1;
  size_t In = nRMS.maxLayer()+1;
  size_t I  = this->maxLayer()+1;
  double To = nRMS.start()-this->start();
  double dT = 1./nRMS.wrate();               // rate in a single noise layer!
  double R  = this->wrate();                 // rate in a single data layer 
  double r;
  wavearray<DataType_t> xx;                  // data layer
  wavearray<double> na;                      // nRMS layer

  this->w_mode = abs(mode);                  

  if(I!=In) {
    cout<<"wseries::white error: mismatch between WSeries and nRMS\n";
    exit(0);
  }
   
  for(i=(1-m)/2; i<I; i++){                    // loop over layers
     this->getLayer(xx,(i+0.01)*m);            // layer of WSeries 
     nRMS.getLayer(na,int(i));                 // layer of noise array
     J  = xx.size();                           // number of data samples
     K  = na.size()-1;                         // number of noise samples - 1
     k  = 0.; 

     for(j=0; j<J; j++) {
        double t = j/R;                        // data sample time
        double T = To+k*dT;
        
        if(t >= To+K*dT) r = na.data[K];       // after last noise point
        else if(t <= To) r = na.data[0];       // before first noise point
        else {
           if(t > T) {k++; T+=dT;}
           r  = (na.data[k-1]*(dT-T+t) + na.data[k]*(T-t))/dT;
        }
        xx.data[j] *= abs(mode)<=1 ? DataType_t(1./r) : DataType_t(1./r/r);        
     }
     this->putLayer(xx,(i+0.01)*m);            // update layer in WSeries 
     
  }
  return true;
}

template<class DataType_t>
wavearray<double> WSeries<DataType_t>::filter(size_t n)
{
   size_t i;
   size_t M = maxLayer()+1;
   size_t k = 1<<n;
   double x;

   wavearray<DataType_t> a;
   wavearray<double>     b;
   wavearray<double>     v(M);      // effective variance

   if(!pWavelet->BinaryTree()) { v = 1.; return v; }
   else                        { v = 0.;           }

   Forward(n);                      // n decomposition steps 

   for(i=0; i<M; i++){
      getLayer(a,i); 
      b = a.white(1);
      x = b.data[0];
      v.data[i/k] += x>0. ? 1./x/x : 0.;
      putLayer(a,i);
   } 

   Inverse(n);                      // n reconstruction steps 

   for(i=0; i<v.size(); i++) 
     v.data[i] = sqrt(k/v.data[i]);

   v.start(this->start());

   return v;
}


template<class DataType_t>
WSeries<float> WSeries<DataType_t>::variability(double t, double fl, double fh)
{
   if(fl<0.) fl = this->getlow();
   if(fh<0.) fh = this->gethigh();
   double tsRate = this->rate();

   size_t i,j;
   size_t  M = maxLayer()+1;        // number of layers
   size_t  N = this->size()/M;      // zero layer length
   size_t ml = size_t(2.*M*fl/tsRate+0.5);    // low layer 
   size_t mh = size_t(2.*M*fh/tsRate+0.5);    // high layer

   if(mh>M) mh = M;

   size_t nL = ml+int((mh-int(ml))/4.+0.5); // left sample (50%)
   size_t nR = mh-int((mh-int(ml))/4.+0.5); // right sample (50%)
   size_t n  = size_t(fabs(t)*tsRate/M);

   DataType_t* p;
   DataType_t* pp[M];               // sorting
   size_t inDex[M];                 // index(layer)
   size_t laYer[M];                 // layer(index)
   WSeries<float> v;                // effective variance
   WSeries<float> V;                // variance correction
   slice S;
   v.resize(N);

   if(!pWavelet->BinaryTree() || mh<ml+8 || nL<1) 
        { v = 1.; return v; }
   else { v = 0.;           }

// calculate frequency order of layers

   for(j=0; j<M; j++){ 
      S = getSlice(j);
      inDex[j] = S.start();   // index in the array (layer)
      laYer[inDex[j]] = j;    // layer (index in the array)
   }

// calculate variability for each wavelet collumn

   for(i=0; i<N; i++){
     p = this->data + i*M;
     for(j=0; j<M; j++){ pp[j] = &(p[inDex[j]]); }
     this->waveSplit(pp,ml,mh-1,nL-1);          // left split
     this->waveSplit(pp,nL,mh-1,nR);            // right split
     v.data[i]  = float(*pp[nR] - *pp[nL-1])/2./0.6745;
   } 

   v.start(this->start());
   v.rate(this->wrate());
   v.setlow(fl);
   v.sethigh(fl);

   if(n<2) return v;

// calculate variability correction

   V=v; V-=1.;
   V.lprFilter(fabs(t),0,0.,fabs(t)+1.);
   v -= V;

// correct variability
   
   p = this->data;

   for(i=0; i<N; i++){             
     for(j=0; j<M; j++){
	if(laYer[j]>=ml && laYer[j]<mh) *p /= (DataType_t)v.data[i];
	p++; 
     }
   }

   return v;
}


template<class DataType_t>
double WSeries<DataType_t>::fraction(double t, double f, int mode)
{
   slice S;
   DataType_t*  p=NULL;
   DataType_t*  P=NULL;
   DataType_t** pp;
   DataType_t A, aL, aR;
   size_t i,j,k;
   size_t nS, kS, nL, nR, lS;
   size_t nZero = 0;
   size_t n0 = 1;
   size_t nsub = t>0. ? size_t(this->size()/this->wrate()/t+0.1) : 1;
   long r;

   if(!nsub) nsub++;

   f = fabs(f);
   if((f>1. || bpp!=1.) && mode) { 
      cout<<"WSeries fraction(): invalid bpp: "<<bpp<<" fraction="<<f<<endl;
      return bpp;
   }
   if(f>0.) bpp = f;

   size_t M = maxLayer()+1;

   n0 = 1;
   pp = (DataType_t **)malloc(sizeof(DataType_t*));
   wavearray<DataType_t> a(n0);

// percentile fraction

   if(mode && f>0.){  
      for(i=0; i<M; i++){

	  S = getSlice(i);	      
	 nS = S.size()/nsub;                           // # of samles in sub-interval
	 kS = S.stride();                              // stride for this layer
	 lS = nS*nsub<S.size() ? S.size()-nS*nsub : 0; // leftover

// loop over subintervals

	 for(k=0; k<nsub; k++) {

	    p = this->data + nS*k*kS + S.start();  // beginning of subinterval

	    if(k+1 == nsub) nS += lS;     // add leftover to last interval

	    nL = nS&1 ? nS/2 : nS/2-1; 
	    nL = size_t(f*nL);            // set left boundary
	    nR = nS - nL - 1;             // set right boundary
	    if(nL<1 || nR>nS-1) { 
	       cout<<"WSeries::fraction() error: too short wavelet layer"<<endl; 
	       return 0.;
	    }

	    if(nS!=n0) {                      // adjust array length
	       pp = (DataType_t **)realloc(pp,nS*sizeof(DataType_t*));
	       a.resize(nS);       
	       n0 = nS;
	    }

	    for(j=0; j<nS; j++) pp[j] = p + j*kS;

	    this->waveSplit(pp,0,nS-1,nL);           // left split
	    this->waveSplit(pp,nL,nS-1,nR);          // right split
	    aL = *pp[nL]; aR = *pp[nR];

	    for(j=0; j<nS; j++){
	       P =  pp[j]; A = *P;

	            if(j<nL) *P = (DataType_t)fabs(A - aL);
	       else if(j>nR) *P = (DataType_t)fabs(A - aR);
	            else   { *P = 0; nZero++; }
	    
	       if(mode > 1) {                   // do initialization for scrambling
		  a.data[j] = *P;               // save data
		  *P = 0;                       // zero sub-interval
	       }
	    }

	    if(mode == 1) continue;             // all done for mode=1

// scramble	 

	    for(j=0; j<nS; j++){
	       if(a.data[j] == 0.) continue;
	       do{ r = int(nS*drand48()-0.1);}
	       while(p[r*kS] != 0);
	       p[r*kS] = a.data[j]; 
	    }
	 }	
      }
   }
   
   else if(f>0.){                            // random fraction
      M = this->size();
      for(i=0; i<M; i++)
	 if(drand48() > f) { this->data[i] = 0; nZero++; }
   }

   else{                                     // calculate zero coefficients
      M = this->size();
      for(i=0; i<M; i++) {
	 if(this->data[i]==0.) nZero++;
      }      
   }

   free(pp);
   return double(this->size()-nZero)/double(this->size());
}


template<class DataType_t>
double WSeries<DataType_t>::significance(double T, double f)
{
   slice S;
   DataType_t*  p=NULL;
   DataType_t** pp;
   double tsRate = this->rate();
   size_t i,j,k,l,m;
   size_t nS,nP,nB;
   size_t M = maxLayer()+1;
   size_t il = size_t(2.*M*getlow()/tsRate);
   size_t ih = size_t(2.*M*gethigh()/tsRate+0.5);
   int nZero = 0;
   double ratio = double(this->size());

   if(ih>M) ih = M;
   if(il>=ih) { 
      cout<<"WSeries::significance(): invalid low and high:  ";
      cout<<"low = "<<il<<"  high = "<<ih<<endl;
      il = 0;
      ih = M;
   }

// zero unused layers   

   for(i=0; i<M; i++){

      if(i>=il && i<=ih) continue;

      S = getSlice(i);
      k = S.size();
      m = S.stride();
      p = this->data+S.start();
      ratio -= double(k);

      for(j=0; j<k; j++) p[j*m] = 0; 
   }
   ratio /= this->size();            // fraction of pixels between high and low

// calculate number of sub-intervals

   S = getSlice(0);                   // layer 0

   size_t n = size_t(fabs(T)*tsRate/S.stride()/ratio+0.1); // number of towers
   if(n<1) n = S.size();

   k = S.size()/n;                     // number of sub-intervals
   m = this->size()/S.size();                // number of samples in each tower

   f = fabs(f);
   if(f>1.) f = 1.;
   if(f>0. && f<bpp) bpp = f; 
   nS = n*m;                           // # of samples in one sub-interval
   nB = size_t(bpp*nS*ratio);          // expected number of black pixels

   if(!nS || !nB || tsRate<=0. || m*S.size()!=this->size()) {
      cout<<"WSeries::significance() error: invalid parameters"<<endl; 
      return 0.;
   } 
   
   l = (S.size()-k*n)*m;               // leftover
   if(l) k++;                          // add one more subinterval

//   cout<<"k="<<k<<" m="<<m<<" bpp="<<bpp<<" nS="<<nS<<endl;

   pp = (DataType_t **)malloc(nS*sizeof(DataType_t*));

   p = this->data;

   for(i=0; i<k; i++){

// fill pp and a

      nP = 0;

      for(j=0; j<nS; j++) {
	 if(*p == 0.) {p++; continue;}
	 *p = (DataType_t)fabs(double(*p));
	 pp[nP++] = p++;
	 nZero++;                                // count non-Zero pixels
      }

      if(nP>2) this->waveSort(pp,0,nP-1);        // sort black pixels

      for(j=0; j<nP; j++) {
	 if(!i && l && pp[j]>=this->data+l) continue;  // handle leftover
	 *pp[j] = nP<nB ? (DataType_t)log(double(nP)/(nP-j)) :  
	                  (DataType_t)log(double(nB)/(nP-j));
	 if(*pp[j] < 0) { 
	    *pp[j] = 0;
	    nZero--;
	 } 
      }

      p = this->data+i*nS+l;
      if(!l) p += nS;
   }
   
   free(pp);
   return double(nZero)/ratio/this->size();
}


template<class DataType_t>
double WSeries<DataType_t>::rsignificance(size_t n, double f)
{
   DataType_t*   p=NULL;
   DataType_t*  px=NULL;
   DataType_t** pp;
   DataType_t** qq;
   DataType_t*  xx;
   DataType_t*  yy;

   double aL, aR;

   size_t i,j,m;
   size_t last, next;
   size_t nS,nB,nL,nR;
   size_t nBlack = 0;
   size_t index;
 
   slice S=getSlice(0);                // layer 0
   size_t N = S.size();                // number of towers in WSeries

   m = this->size()/S.size();                // number of samples in each tower

   f = fabs(f);
   if(f>1.) f = 1.;
   if(f>0. && f<bpp) bpp = f; 
   nS = (2*n+1)*m;                     // # of samples in one sub-interval
   nB = size_t(bpp*nS);                // expected number of black pixels
   if(nB&1) nB++;
   nL = nB/2;                          // left bp boundary
   nR = nS - nL;                       // right bp boundary

   if(!nS || !nB || this->rate()<=0. || m*S.size()!=this->size()) {
      cout<<"WSeries::significance() error: invalid WSeries"<<endl; 
      return 0.;
   } 
   
   pp = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   xx = (DataType_t  *)malloc(nS*sizeof(DataType_t));
   qq = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   yy = (DataType_t  *)malloc(nS*sizeof(DataType_t));

   p = this->data;
   for(j=0; j<nS; j++){ 
      xx[j] = *p; 
      pp[j] = xx+j;
      qq[j] = yy+j;      
      *p++ = 0;
   }
   last = 0;
   next = 0;

   for(i=0; i<N; i++){

      this->waveSplit(pp,0,nS-1,nL-1);       // left split
      this->waveSplit(pp,nL,nS-1,nR);        // right split
      aL = *pp[nL]; aR = *pp[nR];

      for(j=0;  j<nL; j++) yy[j]       = (DataType_t)fabs(*pp[j] - aL);
      for(j=nR; j<nS; j++) yy[j+nL-nR] = (DataType_t)fabs(*pp[j] - aR);

      this->waveSort(qq,0,nB-1);       // sort black pixels

      for(j=0; j<nB; j++) {
	 index = qq[j]-yy;             // index in yy
	 if(index>nL) index+=nR-nL;    // index in pp  
	 index = pp[index]-xx;         // index in xx
	 if(next != index/m) continue; // compare with current tower index in xx
	 this->data[index-next*m+i*m] = (DataType_t)log(double(nB)/(nB-j));
	 nBlack++;
      }

      if(i>=n && i<N-n) {              // copy next tower into last
	 px = xx+last*m;
	 for(j=0; j<m; j++) { *(px++) = *p; *p++ = 0;}
	 last++;                       // update last tower index in array xx
      }

      next++;                          // update current tower index in array xx
      if(next>2*n) next = 0;
      if(last>2*n) last = 0;
      
   } 

   free(pp);
   free(qq);
   free(xx);
   free(yy);

   return double(nBlack)/double(this->size());
}


template<class DataType_t>
double WSeries<DataType_t>::rSignificance(double T, double f, double t)
{
   wavearray<DataType_t> wa;

   DataType_t*   p=NULL;
   DataType_t*  xx;    // buffer for data
   DataType_t*  yy;    // buffer for black pixels
   DataType_t** px;    // pointers to xx
   DataType_t** py;    // pointers to yy
   DataType_t** pp;    // pointers to this->data, so *pp[j] == xx[j]

   double aL, aR;
   double tsRate = this->rate();

   size_t i,j,m,l,J;
   size_t last, next;
   size_t nS,nB,nL,nR;
   size_t nBlack = 0;
   size_t index;

   size_t M = maxLayer()+1;            // number of samples in a tower  
   size_t il = size_t(2.*M*getlow()/tsRate);
   size_t ih = size_t(2.*M*gethigh()/tsRate+0.5);

   if(ih>M) ih = M;
   if(il>=ih) { 
      cout<<"WSeries::significance(): invalid low and high:  ";
      cout<<"low = "<<il<<"  high = "<<ih<<endl;
      il = 0;
      ih = M;
   }

   m = ih-il;                          // # of analysis samples in a tower
 
   for(j=0; j<il; j++){ this->getLayer(wa,j); wa=1234567891.; this->putLayer(wa,j); }
   for(j=ih; j<M; j++){ this->getLayer(wa,j); wa=1234567891.; this->putLayer(wa,j); }

   t *= double(M)/m; 
   T *= double(M)/m; 

   slice S=getSlice(0);                // layer 0
   size_t N = S.size();                // number of towers in WSeries
   size_t k = size_t(t*this->wrate());      // sliding step in towers
   size_t n = size_t(T*this->wrate()/2.);   // 1/2 sliding window in towers

   if(t<=0. || k<1) k = 1;
   if(T<=0. || n<1) n = 1;

   size_t Nnk = N-n-k;
   size_t nW  = (2*n+k)*M;             // total # of samples in the window

   f = fabs(f);
   if(f>1.) f = 1.;
   if(f>0. && f<bpp) bpp = f; 
   nS = (2*n+k)*m;                     // # of analysis samples in the window
   nB = size_t(bpp*nS);                // expected number of black pixels
   if(nB&1) nB++;
   nL = nB/2;                          // left bp boundary
   nR = nS - nL;                       // right bp boundary

   if(!nS || !nB || this->rate()<=0. || M*S.size()!=this->size()) {
      cout<<"WSeries::significance() error: invalid WSeries"<<endl; 
      return 0.;
   } 
   
   pp = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   px = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   py = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   xx = (DataType_t  *)malloc(nS*sizeof(DataType_t));
   yy = (DataType_t  *)malloc(nS*sizeof(DataType_t));

   p = this->data;
   J = 0;
   for(i=0; i<nW; i++){ 
      if(*p != 1234567891.){
	 xx[J] = *p; 
	 pp[J] =  p; 
	 px[J] = xx+J;
	 py[J] = yy+J;
	 J++;
      }      
      *p++ = 0;
   }
   last = 0;
   next = 0;

   if(J != nS) {
     cout<<"wseries::rSignificance() error 1 - illegal sample count"<<endl;
     exit(0);
   }

   for(i=0; i<N; i+=k){

      this->waveSplit(px,0,nS-1,nL-1);       // left split
      this->waveSplit(px,nL,nS-1,nR);        // right split
      aL = *px[nL]; aR = *px[nR];

      for(j=0;  j<nL; j++) yy[j]       = (DataType_t)fabs(*px[j] - aL);
      for(j=nR; j<nS; j++) yy[j+nL-nR] = (DataType_t)fabs(*px[j] - aR);

      if(nB != nS-nR+nL) {
	cout<<"wseries::rSignificance:  nB="<<nB<<",  N="<<nS-nR+nL<<endl;
	nB = nS-nR+nL;
      }

      this->waveSort(py,0,nB-1);       // sort black pixels

      for(j=0; j<nB; j++) {            // save rank in *this
	 index = py[j]-yy;             // index in yy
	 if(index>nL) index+=nR-nL;    // index in xx  
	 index = px[index]-xx;         // index in pp
         index = pp[index]-this->data; // index in WS data array

         if(index>=i*M && index<(i+k)*M) {  // update pixels in window t
           if(this->data[index]!=0) {
             cout<<"WSeries::rSignificance error: "<<this->data[index]<<endl;
           }
           this->data[index] = DataType_t(log(double(nB)/(nB-j))); // save rank
           nBlack++;
         }
      }

      for(l=i; l<i+k; l++){             // copy towers
	 if(l>=n && l<Nnk) {            // copy next tower into last
	    J = last*m;                 // pointer to last tower storage at xx
	    for(j=0; j<M; j++) { 
	       if(*p != 1234567891.) { xx[J] = *p; pp[J++]=p; }
	       *p++ = 0;
	    }
	    last++;                    // update last tower index in array xx
	    if(J != last*m) { 
	       cout<<"wseries::rSignificance() error 2 - illegal sample count"<<endl;
	       exit(0);
	    }
	 }

	 next++;                       // update current tower index in array xx
	 if(next==2*n+k) next = 0;
	 if(last==2*n+k) last = 0;
      }
   } 

   free(pp);
   free(px);
   free(py);
   free(xx);
   free(yy);

   return double(nBlack)/double(this->size());
}



template<class DataType_t>
double WSeries<DataType_t>::gSignificance(double T, double f, double t)
{
   wavearray<DataType_t> wa;

   DataType_t*   p=NULL;
   DataType_t*  xx;    // buffer for data
   DataType_t*  yy;    // buffer for black pixels
   DataType_t** px;    // pointers to xx
   DataType_t** py;    // pointers to yy
   DataType_t** pp;    // pointers to this->data, so *pp[j] == xx[j]

   double aL, aR;

   size_t i,j,m,l,J;
   size_t last, next;
   size_t nS,nB,nL,nR;
   size_t nBlack = 0;
   size_t index;

   size_t M = maxLayer()+1;            // number of samples in a tower  
   size_t il = size_t(2.*getlow()/this->wrate());
   size_t ih = size_t(2.*gethigh()/this->wrate()+0.5);

   if(ih>M) ih = M;
   if(il>=ih) { 
      cout<<"WSeries::significance(): invalid low and high:  ";
      cout<<"low = "<<il<<"  high = "<<ih<<endl;
      il = 0;
      ih = M;
   }

   m = ih-il;                          // # of analysis samples in a tower
 
   for(j=0; j<il; j++){ this->getLayer(wa,j); wa=1234567891.; this->putLayer(wa,j); }
   for(j=ih; j<M; j++){ this->getLayer(wa,j); wa=1234567891.; this->putLayer(wa,j); }

   t *= double(M)/m; 
   T *= double(M)/m; 

   slice S=getSlice(0);                // layer 0
   size_t N = S.size();                // number of towers in WSeries
   size_t k = size_t(t*this->wrate());      // sliding step in towers
   size_t n = size_t(T*this->wrate()/2.);   // 1/2 sliding window in towers

   if(t<=0. || k<1) k = 1;
   if(T<=0. || n<1) n = 1;

   size_t Nnk = N-n-k;
   size_t nW  = (2*n+k)*M;             // total # of samples in the window

   f = fabs(f);
   if(f>1.) f = 1.;
   if(f>0. && f<bpp) bpp = f; 
   nS = (2*n+k)*m;                     // # of analysis samples in the window
   nB = size_t(bpp*nS);                // expected number of black pixels
   if(nB&1) nB++;
   nL = nB/2;                          // left bp boundary
   nR = nS - nL;                       // right bp boundary

   if(!nS || !nB || this->rate()<=0. || M*S.size()!=this->size()) {
      cout<<"WSeries::gSignificance() error: invalid WSeries"<<endl; 
      return 0.;
   } 
   
   pp = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   px = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   py = (DataType_t **)malloc(nS*sizeof(DataType_t*));
   xx = (DataType_t  *)malloc(nS*sizeof(DataType_t));
   yy = (DataType_t  *)malloc(nS*sizeof(DataType_t));

   p = this->data;
   J = 0;
   for(i=0; i<nW; i++){ 
      if(*p != 1234567891.){
	 xx[J] = *p; 
	 pp[J] =  p; 
	 px[J] = xx+J;
	 py[J] = yy+J;
	 J++;
      }      
      *p++ = 0;
   }
   last = 0;
   next = 0;

   if(J != nS) {
     cout<<"wseries::gSignificance() error 1 - illegal sample count"<<endl;
     exit(0);
   }

   for(i=0; i<N; i+=k){

      this->waveSplit(px,0,nS-1,nL-1);       // left split
      this->waveSplit(px,nL,nS-1,nR);        // right split
      aL = *px[nL]; aR = *px[nR];

      for(j=0;  j<nL; j++) yy[j]       = (DataType_t)fabs(*px[j]);
      for(j=nR; j<nS; j++) yy[j+nL-nR] = (DataType_t)fabs(*px[j]);

      if(nB != nS-nR+nL) {
	cout<<"wseries::gSignificance:  nB="<<nB<<",  N="<<nS-nR+nL<<endl;
	nB = nS-nR+nL;
      }

      this->waveSort(py,0,nB-1);       // sort black pixels

      for(j=0; j<nB; j++) {            // save significance in *this
	 index = py[j]-yy;             // index in yy
	 if(index>nL) index+=nR-nL;    // index in pp  
	 index = px[index]-xx;         // index in xx
	 *(pp[index]) = pow(*py[j]+1.11/2,2)/2./1.07 + log(bpp); // save significance
      }
      nBlack += nB;

      for(l=i; l<i+k; l++){             // copy towers
	 if(l>=n && l<Nnk) {            // copy next tower into last
	    J = last*m;                 // pointer to last tower storage at xx
	    for(j=0; j<M; j++) { 
	       if(*p != 1234567891.) { xx[J] = *p; pp[J++]=p; }
	       *p++ = 0;
	    }
	    last++;                    // update last tower index in array xx
	    if(J != last*m) { 
	       cout<<"wseries::gSignificance() error 2 - illegal sample count"<<endl;
	       exit(0);
	    }
	 }

	 next++;                       // update current tower index in array xx
	 if(next==2*n+k) next = 0;
	 if(last==2*n+k) last = 0;
      }
   } 

   free(pp);
   free(px);
   free(py);
   free(xx);
   free(yy);

   return double(nBlack)/double(this->size());
}



template<class DataType_t>
double WSeries<DataType_t>::pixclean(double S)
{
   size_t k;
   size_t event = 0;
   int i, j, n;
   int nm, np, mp, mm;
   
   bool one;

   wavearray<DataType_t>  a;
   wavearray<DataType_t>  am;
   wavearray<DataType_t>  ac;
   wavearray<DataType_t>  ap;
   wavearray<DataType_t>* p;
   wavearray<DataType_t>* pm;
   wavearray<DataType_t>* pc;
   wavearray<DataType_t>* pp;

   size_t max_layer = maxLayer()+1;

   pc = &ac; pp = &ap; pm = p = NULL;
   mp = mm = 1;
   getLayer(a,0);
   ac = a;

   for(k=1; k<=max_layer; k++){

      if(k<max_layer) getLayer(*pp,k);  // next layer
      else pp = NULL;
      
      if(pp!=NULL) mp = pp->size()/pc->size();  // scale for upper layer
      if(pm!=NULL) mm = pc->size()/pm->size();  // scale for lower layer

      n  = pc->size()-1; 

      for(i=0; i<=n; i++) {
	 one = true;

	 if(pc->data[i] == 0.)        continue;
	 if(pc->data[i] > 9.7) cout<<"pixclean: "<<pc->data[i]<<endl;
	 event++;
	 if(i>0 && pc->data[i-1]!=0.) continue;
	 if(i<n && pc->data[i+1]!=0.) continue;

// work on upper (+) layer

	 if(pp!=NULL) {
	    nm = i*mp-1;              // left index for + layer
	    np = i*mp+2;              // right index for + layer
	    if(nm < 0) nm = 0; 
	    if(np > n) np = n;
	    
	    for(j=nm; j<np; j++) {
	       if(pp->data[j] != 0) {
		  one = false;
		  break;
	       }
	    } 
	 }
	 if(!one) continue;

// work on lower (-) layer

	 if(pm!=NULL) {
	    nm = i/mm-1;                // left index for + layer
	    np = i/mm+2;              // right index for + layer
	    if(nm < 0) nm = 0; 
	    if(np > n) np = n;
	    
	    for(j=nm; j<np; j++) {
	       if(pm->data[j] != 0) {
		  one = false;
		  break;
	       }
	    } 
	 }
	 if(!one) continue;

	 if(pc->data[i]<S) {a.data[i]=0; event--;}
      }

      putLayer(a,k-1);

// shaffle layers

      if(pp==NULL) break; 

       a = *pp;
       p = pm==NULL ? &am : pm;
      pm = pc;
      pc = pp;
      pp = p;
   }

   return double(event)/this->size();
}


template<class DataType_t>
double WSeries<DataType_t>::percentile(double f, int mode, WSeries<DataType_t>* pin)
{
   slice S;
   DataType_t*  p=NULL;
   DataType_t*  P=NULL;
   DataType_t** pp;
   double A, aL, aR;
   double x;
   size_t i,j;
   size_t nS, kS, mS, nL, nR;
   size_t nZero = 0;
   long r;

   f = fabs(f);
   if((f>=1. || bpp!=1.) && mode) { 
      cout<<"WSeries percentile(): invalid bpp: "<<bpp<<" fraction="<<f<<endl;
      return bpp;
   }
   bpp = f;

   if(pin) *this = *pin;                    // copy input wavelet if specified

   size_t M = maxLayer()+1;
   WaveDWT<DataType_t>* pw = pWavelet;

   S=pw->getSlice(0);	      
   size_t n0 = S.size();
   if(n0) pp = (DataType_t **)malloc(n0*sizeof(DataType_t*));
   else return 0.;

   wavearray<DataType_t> a(n0);
   wavearray<DataType_t> b;

   if(mode && f>0.){                                 // percentile fraction
      for(i=0; i<M; i++){

	  S = pw->getSlice(i);	      
	 nS = S.size();
	 kS = S.stride();
	 mS = S.start();
	  p = this->data+S.start();
	 nL = size_t(f*nS/2.+0.5);
	 nR = nS - nL;

	 if(nL<2 || nR>nS-2) { 
	   cout<<"WSeries::percentile() error: too short wavelet layer"<<endl; 
	   return 0.;
	 }
	 
	 if(nS!=n0) {
	    pp = (DataType_t **)realloc(pp,nS*sizeof(DataType_t*));
	    a.resize(nS);       
	 }

	 for(j=0; j<nS; j++) pp[j] = p + j*kS;

	 this->waveSplit(pp,0,nS-1,nL-1);            // left split
	 this->waveSplit(pp,nL,nS-1,nR);             // right split
	 aL = double(*pp[nL-1]); aR = double(*pp[nR]);

	 for(j=0; j<nS; j++){
	    P =  pp[j]; A = double(*P);

//   	         if(j<nL) *P = -sqrt(A*A - aL*aL);
//	    else if(j>nR) *P =  sqrt(A*A - aR*aR);

   	         if(j<nL) *P = DataType_t(fabs(A - aL));
	    else if(j>nR) *P = DataType_t(fabs(A - aR));
		 else   { *P = 0; nZero++; }
	    
	    if(mode == -1) continue;            // all done for mode = -1
	    if(pin) pin->data[mS+P-p] = *P;     // update pin slice
	    if(j>nL && j<nR) continue;          // skip zero amplitudes
	    a.data[(P-p)/kS] = *P;              // save data
	    if(j<nL) *P *= -1;                  // absolute value
	    if(j>=nR) pp[nL+j-nR] = P;          // copy right sample
	 }

	 if(mode == -1) continue;               // keep wavelet amplitudes

	 nL *= 2;
	 this->waveSort(pp,0,nL-1);             // sort black pixels

	 if(abs(mode)!=1) b = a;

	 for(j=0; j<nL; j++){
	    r = (pp[j]-p)/kS;
//	    x = a.data[r]<0. ? -double(nL)/(nL-j) : double(nL)/(nL-j);
	    x = log(double(nL)/(nL-j));
	    *pp[j] = mode==1 ? DataType_t(x) : 0;
	    if(mode>1) a.data[r]=DataType_t(x);  // save data for random sample
	 }

	 if(abs(mode)==1) continue;

// scramble	 

	 for(j=0; j<nL; j++){
	    P = pp[j];
	    do{ r = int(nS*drand48()-0.1);}
	    while(p[r*kS] != 0);
	    p[r*kS] = a.data[(P-p)/kS]; 
	    if(pin) pin->data[mS+r*kS] = b.data[(P-p)/kS]; 
	 }
      }	
   }
   
   else if(f>0.){                            // random fraction
      M = this->size();
      for(i=0; i<M; i++)
	 if(drand48() > f) { this->data[i] = 0; nZero++; }
   }

   else{                                     // calculate zero coefficients
      M = this->size();
      for(i=0; i<M; i++)
	 if(this->data[i]==0) nZero++;
   }

   free(pp);
   return double(this->size()-nZero)/double(this->size());
}


template<class DataType_t>
WSeries<double> WSeries<DataType_t>::calibrate(size_t n, double df,
					       d_complex* R, d_complex* C,
					       wavearray<double> &a,
					       wavearray<double> &g,
					       size_t channel)
{
   size_t i,k,m,N;
   size_t count, step;
   size_t M = maxLayer()+1;

   double tsRate = this->rate();
   double left, righ;
   double tleft, trigh;
   double reH, imH;
   double c, t, dt;
   double cstep = 1./a.rate();
   double sTart = this->start();
   double sTop  = this->start()+this->size()/tsRate;
   DataType_t* p;
   slice S;

   wavecomplex* pR=R;
   wavecomplex* pC=C;

   Wavelet* pw = pWavelet->Clone();
   wavearray<double> alp;
   wavearray<double> gam;
   wavearray<double> reR(M);
   wavearray<double> reC(M);
   wavearray<double> imR(M);
   wavearray<double> imC(M);

   alp = a; alp.start(0.);
   gam = a; gam.start(0.);

// select alpha to be in WB segment
   count=0;
   for(i=0; i<a.size(); i++) {
      if(a.start()+i/a.rate() < sTart) continue;  // skip samples in front
      if(a.start()+i/a.rate() > sTop)  break;     // skip samples in the back
      if(alp.start()==0.) alp.start(a.start()+i/a.rate());
      alp.data[count++] = a.data[i];
   }
   alp.resize(count);

// select gamma to be in WB segment
   count=0;
   for(i=0; i<g.size(); i++) {
      if(g.start()+i/g.rate() < sTart) continue;  // skip samples in front
      if(g.start()+i/g.rate() > sTop)  break;     // skip samples in the back
      if(gam.start()==0.) gam.start(g.start()+i/g.rate());
      gam.data[count++] = g.data[i];
   }
   gam.resize(count);

   if(gam.size()>alp.size()) gam.resize(alp.size());
   if(gam.size()<alp.size()) alp.resize(gam.size());

   wavearray<double>     x(M*alp.size());
   WSeries<double>       cal(x,*pw);

   if(!alp.size() || a.rate()!=g.rate()) { 
      cout<<"WSeries<DataType_t>::calibrate() no calibration data\n";
      return cal;
   }

   cal = 0.;
   left = righ = 0.;

   reR = 0.; reC = 0.;
   imR = 0.; imC = 0.;
   for(k=0; k<M; k++){

      S = getSlice(k);
      left  = righ;                       // left border
      righ += tsRate/2./S.stride();       // right border
      if(righ > n*df) break;

// average R and C

      count = 0;
      while(left+count*df < righ){
	 reR.data[k] += pR->real();
	 imR.data[k] += pR->imag();
	 reC.data[k] += pC->real();
	 imC.data[k] += pC->imag();
	 count++; pR++; pC++;
      }
      reR.data[k] /= count; reC.data[k] /= count;
      imR.data[k] /= count; imC.data[k] /= count;

// calculate calibration constants

      cal.getLayer(x,k);
      for(i=0; i<alp.size(); i++){

	 if(alp.data[i]<=0. || gam.data[i]<=0.) {
	    cout<<"WSeries<DataType_t>::calibrate() zero alpha error\n";
	    alp.data[i] = 1.;
	    gam.data[i] = 1.;
	 }

	 reH = reR.data[k]*reC.data[k]-imR.data[k]*imC.data[k];
	 imH = reR.data[k]*imC.data[k]+imR.data[k]*reC.data[k];
	 reH = 1.+(reH-1.)*gam.data[i];
	 imH*= gam.data[i];
	 x.data[i]  = sqrt(reH*reH+imH*imH);
	 x.data[i] /= sqrt(reC.data[k]*reC.data[k]+imC.data[k]*imC.data[k]);
	 x.data[i] /= channel==0 ? alp.data[i] : gam.data[i];
      }
      cal.putLayer(x,k);

// apply energy calibration

      S    = getSlice(k);
      step = S.stride();
      N    = S.size();
      p    = this->data + S.start();
      dt   = step/tsRate;        // sampling time interval
      t    = this->start();      // time stamp
      sTart = alp.start();       // first calibration point
      sTop  = alp.start()+(alp.size()-1)*cstep;         // last calibration sample

      tleft = sTart;             // left and right borders defining beginning and end
      trigh = sTart+cstep;       // of the current calibration cycle.
      m = 0;

      for(i=0; i<N; i++){
	 t += dt;
	 if(t <= sTart) *p *= (DataType_t)x.data[0];
	 else if(t >= sTop) *p *= (DataType_t)x.data[alp.size()-1];
	 else {
	    if(t>trigh) { tleft=trigh; trigh+=cstep; m++; }
	    c = (t-tleft)/cstep;
	    *p *= DataType_t(x.data[m]*(1-c) + x.data[m+1]*c);
	 }	    
	 p += step;
      }

   }

   return cal;
}

template<class DataType_t>
void WSeries<DataType_t>::print()
{
   pWavelet->print();
   wavearray<DataType_t>::print();
}

//______________________________________________________________________________
template <class DataType_t> 
void WSeries<DataType_t>::Streamer(TBuffer &R__b)
{
   // Stream an object of class WSeries<DataType_t>.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      wavearray<DataType_t>::Streamer(R__b);
      R__b >> w_mode;
      R__b >> bpp;
      R__b >> wRate;
      R__b >> f_low;
      R__b >> f_high;

      bool bWWS;
      int m_WaveType;                                               
      R__b >> m_WaveType;                                           
      R__b >> bWWS;
      if(!bWWS) m_WaveType=-1;
      WaveDWT<DataType_t> *pW;
      switch(m_WaveType) {                                          
      case HAAR :                                                   
        pW = (WaveDWT<DataType_t>*)(new Haar<DataType_t>);    
        break;                                                      
      case BIORTHOGONAL :                                           
        pW = (WaveDWT<DataType_t>*)(new Biorthogonal<DataType_t>);
        break;                                                          
      case DAUBECHIES :                                                 
        pW = (WaveDWT<DataType_t>*)(new Daubechies<DataType_t>);  
        break;                                                          
      case SYMLET :                                                     
        pW = (WaveDWT<DataType_t>*)(new Symlet<DataType_t>);      
        break;                                                          
      case MEYER :                                                      
        pW = (WaveDWT<DataType_t>*)(new Meyer<DataType_t>);       
        break;                                                          
      case WDMT :                                                       
        pW = (WaveDWT<DataType_t>*)(new WDM<DataType_t>);         
        break;                                                          
      default :                                                         
        pW = new WaveDWT<DataType_t>;                             
      }                                                                 
      pW->Streamer(R__b);
      this->setWavelet(*pW);
      delete pW; 

      R__b.CheckByteCount(R__s, R__c, WSeries<DataType_t>::IsA());
   } else {
      R__c = R__b.WriteVersion(WSeries<DataType_t>::IsA(), kTRUE);
      wavearray<DataType_t>::Streamer(R__b);
      R__b << w_mode;
      R__b << bpp;
      R__b << wRate;
      R__b << f_low;
      R__b << f_high;

      R__b << pWavelet->m_WaveType;
      bool bWWS = ((pWavelet->pWWS==NULL)&&(pWavelet->m_WaveType==HAAR)) ? false : true;
      R__b << bWWS;	
      pWavelet->Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

// instantiations

#define CLASS_INSTANTIATION(class_) template class WSeries< class_ >;

CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)

#undef CLASS_INSTANTIATION

