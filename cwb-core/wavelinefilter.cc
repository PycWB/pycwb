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


/************************************************************************ 
 * Sergey Klimenko, University of Florida                  
 *
 * v-1.63: 08/07/01 
 *    - addded flag reFine (force to do fScan if !reFine)
 *      set true by default, set false if SNR<0
 *    - corrected situation when getOmega fails to find frequency
 *      getOmega always uses FilterID=1
 *    - correct frequency band fBand if start not from the first harmonic
 * v-2.0: 11/08/01 
 *    - svitched to new class of wavelet transforms
 *      replace wavearray<double> class with wavearray class
 ************************************************************************/

#include "wavelinefilter.hh"
//#include "DVector.hh"
#include "Biorthogonal.hh"
#include <iostream>

using namespace std;

linefilter::linefilter() : 
  FilterID(1), Frequency(60.), Window(0.0), Stride(1.), nFirst(1), nLast(0),
  nStep(1), nScan(20), nBand(5), nSubs(1), fBand(0.45), nLPF(-1), nWave(16), 
  clean(false), badData(false), noScan(false), nRIF(6), SNR(2.), reFine(true), 
  dumpStart(0),  FilterState(0), SeedFrequency(60.), CurrentTime(0.), 
  StartTime(0.), Sample(0.0) 
{ 
   reset();
}

/* linefilter constructor
 *  f - approximate line frequency
 *  w - time interval to remove lines.               
 *fid - select filter ID (-1/0/1), -1 by default
 * nT - number of subdivisions of time interval w
 */
linefilter::linefilter(double f, double w, int fid, int nT) : 
  FilterID(1), Frequency(60.), Window(0.0), Stride(1.), nFirst(1), nLast(0),
  nStep(1), nScan(20), nBand(5), nSubs(1), fBand(0.45), nLPF(-1), nWave(16), 
  clean(false), badData(false), noScan(false), nRIF(6), SNR(2.), reFine(true), 
  dumpStart(0), FilterState(0), SeedFrequency(60.), CurrentTime(0.), 
  StartTime(0.), Sample(0.0) 
{ 
   reset(); 
   Frequency = fabs(f); 
   SeedFrequency = Frequency;
   FilterID  = fid;
   Window = w; 
   if(f < 0.) clean = true;
   nSubs = (nT > 0) ? nT : 1;
}

linefilter::~linefilter() { reset(); }

linefilter::linefilter(const linefilter& x) 
  : FilterID(x.FilterID), Frequency(x.Frequency), Window(x.Window), 
    Stride(x.Stride), nFirst(x.nFirst), nLast(x.nLast), nStep(x.nStep), 
    nScan(x.nScan), nBand(x.nBand), nSubs(x.nSubs), fBand(x.fBand), 
    nLPF(x.nLPF), nWave(x.nWave), clean(x.clean), badData(false), 
    noScan(x.noScan), nRIF(x.nRIF), SNR(x.SNR), reFine(x.reFine), 
    dumpStart(0), FilterState(0), SeedFrequency(60.), CurrentTime(0.), 
    StartTime(0.), Sample(0.)
{}

linefilter*
linefilter::clone(void) const {
    return new linefilter(*this);
}

unsigned int linefilter::maxLine(int L) 
{
   unsigned int imax = (nLPF>0) ? int(L/2)+1 : int(L/4)+1;
   if(nFirst > imax) std::cout<<"linefilter: Invalid harmonic number.\n";
   if(imax>nLast && nLast>0) imax = nLast+1;
   if(imax <= nFirst) imax = nFirst+1;
   if(imax > (unsigned)L/2) imax = L/2;
   return imax;
}


/***setFilter*********************************************************
 * nF - first harmonic
 * nL - last harmonic
 * nS - skip harmonics
 * nD - decimation depth (resampling)
 * nB - number of frequency bins to estimate noise spectral density
 * nR - order of interpolating filter for resample
 * nW - lifting wavelet order
 *********************************************************************/
  void linefilter::setFilter(int nF,
			     int nL,
			     int nS,
			     int nD,
			     int nB,
			     int nR,
			     int nW){

    reset();
    if(nS == 0) nS++;
    nStep  = nS;
    nFirst = nF;
    nLast  = nL;

    if(nS<0){
       nFirst *= -nStep;
       nLast  *= -nStep;
       Frequency /= -nStep;
       SeedFrequency = Frequency;
    }

    nBand     = (nB>=2) ? nB : 2;
    nWave     = nW;
    nLPF      = nD;
    nRIF      = nR;
    return;
}

void linefilter::reset() {
    FilterState = 0;
    CurrentTime = 0.;
    StartTime   = 0.;
    Sample      = 0.0;
    badData     = false;
    lineList.clear();
    dumpStart = 0;
    Frequency = SeedFrequency;
}

void linefilter::resize(size_t dS) {

    if(dS == 0){           // clear the list
      lineList.clear();
      dumpStart = 0;
    }

    else{                 // pop entries from the beginnig of the list

      if(lineList.size() <= dS)
	dumpStart = lineList.size();

      else
	do { lineList.pop_front(); }
	while(lineList.size() > dumpStart);

    }

    return;
}


void linefilter::setFScan(double f, 
			  double sN, 
			  double fB, 
			  int nS)
{
   noScan = true; 
   fBand = fabs(fB);
   nScan = nS;
   SNR = fabs(sN);
   reFine = (sN>0) ? true : false;
   if(f != 0.) { 
      Frequency = nStep>0 ? fabs(f) : fabs(f/nStep);
      noScan = (f < 0.) ? true : false; 
   }
   return;
}

/****************************************************************************
 * Function estimates Power Spectral Density for input TD time series.
 * Return WaveData object with average Spectral Density near harmonics 
 * (f,2*f,3*f,..imax*f).
 * Spectral Density is estimated using nb Fourier bins around harmonics. 
 ****************************************************************************/

WaveData linefilter::getPSD(const WaveData &TD, int nb)
{
   int i,j,k;
   double a, b, c;
   int L = int(TD.rate()/Frequency + 0.5);  // # of samples in one line cycle
   int m = int(TD.size()/nSubs);            // length of 1/nSubs TD
   int n = int(m/(nb*L))*L;                 // length of 1/nb/nSubs TD

   WaveData td2(2*L);   // S.K. wavefft 9/10/20002 
   WaveData tds(L);
   WaveData tdn(L);
   WaveData tdx(n);
   WaveData psd(L/2);
   
   if(nb < 1) nb = 1;

   psd = 0.;
   if (n/L == 0) {
      cout <<" linefilter::getPSD error: time series is too short to contain\n"
	   <<" one cycle of fundamental harmonic " << Frequency << "\n";
      return psd;
   }
  
   c = (nb>1) ? 1./(nb-1) : 1.;             
   c *= Window/nSubs/nb/nSubs;   // before 03/24/01 was c *= double(m)/nSubs/TD.rate()/nb;
   psd.rate(TD.rate()); 

//   cout<<"c="<<c<<"  n="<<n<<"  m="<<m<<"  L="<<L<<"\n";

   for(k = 0; k < nSubs; k++){
      psd.data[0] += tdn.Stack(TD, m, m*k); 

      for(j = 0; j < nb; j++){
	 if(nb > 1) {
	    psd.data[0] -= tds.Stack(TD, n, n*j+m*k);
	    tds -= tdn;
	 }
	 else
	    tds = tdn;

       	 tds.hann();
//     	 tds.FFTW(1);                        // calculate FFT for stacked signal
	 	 
// SK wavefft fix 9/10/2002
	 td2.rate(tds.rate());              
	 td2.cpf(tds); td2.cpf(tds,L,0,L);  
	 td2.FFTW(1);                     
	 tds[slice(0,L/2,2)] << td2[slice(0,L/2,4)];
	 tds[slice(1,L/2,2)] << td2[slice(1,L/2,4)];
// SK wavefft fix 9/10/2002: end
	  
	 for (i = 2; i < L-1; i+=2){        // always neglect last harmonic
	    a = tds.data[i]; b = tds.data[i+1];
	    psd.data[i/2] += c*(a*a+b*b);   // save spectral density 
	    //	    cout<<"nb="<<nb<<"  i="<<i<<"   PSD="<<psd.data[i/2]<<endl;
	 }
      }
   }

   return psd;
}

/******************************************************************
 * makeFilter calculates comb filter. 
 * Returns the line intensity at the filter output
 * nFirst    - "number" of the first harmonic
 * nLast     - "number" of the last harmonic
 * nStep     - skip harmonics
 * FilterID  - select the filter #, 1 by default
 * nBand     - number of frequency bins used to estimate the noise SD 
 *****************************************************************/  
double linefilter::makeFilter(const WaveData &TD, int FID)
{
   if (badData) {
      cout <<" linefilter::MakeFilter() error: badData flag is on\n";
    return 0.;
  }

   double S, N;
   unsigned int i;
   int L = int(TD.rate()/Frequency+0.5);   // number of samples per one cycle
   int k = int(TD.size()/L);

   if (k == 0) {
    cout <<" linefilter::MakeFilter() error: data length too short to contain\n"
         <<" at least one cycle of target frequency = " << Frequency << " Hz\n";
    badData = true;
    return 0.;
  }

   unsigned int imax = maxLine(L);       // max number of harmonics 

   if((int)Filter.size() < L/2) Filter.resize(L/2);
   Filter = 0.;
   for (i = nFirst; i < imax; i += abs(nStep)) Filter.data[i] = 1.;

   LineSD = getPSD(TD);                  // Line energy + NSD

   if (FID == 1) {
      NoiseSD = getPSD(TD,nBand);        // Line energy + NSD*nBand
      for (i = nFirst; i < imax; i += abs(nStep)) {
	 S = LineSD.data[i];
	 N = NoiseSD.data[i];
	 Filter.data[i] = (S>N && S>0) ? (1-N/S) : 0.;
	 //	 cout<<"S="<<S<<"  N="<<N<<"  F="<<Filter.data[i]<<endl;
      }
   }

   S = 0.;
   N = 0.;
   for (i = nFirst; i < imax; i += abs(nStep)) {
      S += LineSD.data[i]*Filter.data[i]*Filter.data[i];
      N += (FID == 1) ? NoiseSD.data[i]*Filter.data[i] : 0.;
   }

   if(S < SNR*N || S <= 0.) badData = true;
   //   if(badData) cout << S << "  FID=" << FID << endl;
   return S;
}

/*******************************************************************
 * getLine() replaces the original data TD with calculated 
 * interference signal. Filter F is applied. Returns linedata object
 * with the line summary information.  Works with resampled data
 *******************************************************************/

linedata linefilter::getLine(WaveData &TD)
{
   double a,b;
   int iF;
   double F;
   linedata v;

   v.frequency = 0.;
   v.intensity = 0.;
   v.T_current = CurrentTime;

   if (Frequency <= 0) {
      cout << " getLine() error: invalid interference frequency"
	   << " :  " << Frequency << " Hz\n";
      return v;
   }

   int L  = int(TD.rate()/Frequency + 0.5);   // number of samples per cycle
   int n  = int(TD.size()/nSubs);             // data section length

   unsigned int imax = maxLine(L);            // max number of harmonics 

   if (n/L == 0 || L < 4) {
      cout << " getLine() error: input data length too short to contain\n"
	   << " one cycle of target frequency = " << Frequency << " Hz\n";
      return v;
   }

   WaveData td2(2*L);                   // SK wavefft fix 9/10/2002
   WaveData tds(L);
   WaveData amp(L);
   amp *= 0.;

   double p = axb(Frequency, n/TD.rate());
   double phase = 0.;

   v.frequency = Frequency;
   v.intensity = 0.;

   for (int k = 0; k < nSubs; k++) {
      tds.Stack(TD, n, n*k);           // stack signal

      if(!clean) tds.hann();

// SK wavefft fix 9/10/2002
	 td2.rate(tds.rate());              
	 td2.cpf(tds); td2.cpf(tds,L,0,L);  
	 td2.FFTW(1);                     
	 tds[slice(0,L/2,2)]<<td2[slice(0,L/2,4)];
	 tds[slice(1,L/2,2)]<<td2[slice(1,L/2,4)];
// SK wavefft fix 9/10/2002: end

//*************************************************************
// reconstruction of filtered interference signal
//*************************************************************
      //      cout<<"tds.size="<<tds.size()<<"  L="<<L-1<<"  imax="<<imax
      //  <<"  nFirst="<<nFirst<<"   nLast="<<nLast<<"  Filter.size="<<Filter.size()<<endl;

// apply the filter
      for (unsigned int i = 0; i < (unsigned)L-1; i+=2) {
	 F = Filter.data[i/2];
	 tds.data[i]   *= F;
	 tds.data[i+1] *= F;
	 if(F > 0.){
	    a = tds.data[i]; b = tds.data[i+1];
	    amp.data[i]   += (a*a + b*b)/nSubs;

	    if(k==0){
	       phase = arg(f_complex(a,b));
	       amp.data[i+1] = phase;
	    }
	    else{
	       b  = arg(f_complex(a,b));
	       a  = (b-phase)/2./PI - axb(p,double(k*(i/2)));
	       a -= (a>0.) ? long(a+0.5) : long(a-0.5);
	       amp.data[i+1] += 2*PI*a/nSubs;
	       phase = b;
	    }
	 }    
      }
      if((L&1) == 1) tds.data[L-1] = 0.; // take care of odd L

//      for(int jj=0; jj<tds.size(); jj++) cout<<"i="<<jj<<"  tds="<<tds.data[jj]<<endl;

// SK wavefft fix 9/10/2002
      td2=0.;
      td2[slice(0,L/2,4)]<<tds[slice(0,L/2,2)];
      td2[slice(1,L/2,4)]<<tds[slice(1,L/2,2)];
      td2.FFTW(-1);                     
      tds.cpf(td2,L);
// SK wavefft fix 9/10/2002: end

      tds.getStatistics(a,b);
      v.intensity += b*b;

      iF = (k == nSubs-1) ? TD.size() : n*k+n;
      if(clean)                  // reconstruct interference to clean data  
	 for (int i = 0; i < L; i++)
	    for(int j = n*k+i; j < iF; j += L)
	       TD.data[j] = tds.data[i];
   }

// add amplitudes and noise spectral density into the linedata
   iF = imax-nFirst;
   v.amplitude.resize(iF);
   v.line.resize(iF);
   v.noise.resize(iF);
   v.filter.resize(iF);

   for (unsigned int i = nFirst; i < imax; i += abs(nStep)) {
      iF = i-nFirst;
      v.line[iF]  = LineSD.data[i];
      v.noise[iF]  = (FilterID == 0) ? 0. : NoiseSD.data[i];
      v.filter[iF]  = Filter.data[i];
      a = amp.data[2*i];
      b = amp.data[2*i+1];
      v.amplitude[iF] = float(2.*float(sqrt(a)))*exp(f_complex(0.,b));
      if(!clean) v.amplitude[iF] *= sqrt(1.5);  // correct for Hann (9/3/02)
   }

// calculate inverse FFT to reproduce one cycle of filtered signal
   v.intensity /= nSubs;
   if(!clean) v.intensity *= 1.5;  // correct for Hann (9/3/02)
   v.first = nFirst;

   return v;
}

/*******************************************************************
 * heterodyne() estimates single line amplitude and phase by using
 * straight heterodyning. It replaces the original data TD with calculated 
 * interference signal. Returns linedata object
 * with the line summary information.  Works with original data
 *******************************************************************/

linedata linefilter::getHeteroLine(WaveData &TD)
{
   long i,k,m;
   double a,b,u,v;
   double C,S, c,s, ch,sh, x;
   double omega;
   long N = TD.size();
   double* p = TD.data; 
   size_t mode;
   linedata Q;

   Q.frequency = 0.;
   Q.intensity = 0.;
   Q.T_current = CurrentTime;

   if (Frequency <= 0) {
      cout << " getLine() error: invalid interference frequency"
	   << " :  " << Frequency << " Hz\n";
      return Q;
   }

   int L  = int(TD.rate()/Frequency);         // number of samples per cycle
   int n  = int(TD.size()/nSubs);             // data sub-section length
   size_t imax = L;                           // max number of harmonics 
   size_t M = imax-nFirst;                    // number of modes

   omega = 2*PI*nFirst*Frequency/TD.rate();

   // tabulate sin and cos
 
   if(!lineList.size()) {  
     ct.resize(n);
     st.resize(n);
     wt.resize(n);
     for(i=0; i<n; i++){
       ct.data[i] = cos(i*omega);
       st.data[i] = sin(i*omega);
       wt.data[i] = 1.-cos(i*2.*PI/double(n));
     }
   }

   WaveData amp(nSubs*M);
   WaveData phi(nSubs*M);

   amp = 0.;
   phi = 0.;

   Q.frequency = Frequency;
   Q.intensity = 0.;

   for (mode = nFirst; mode < imax; mode += abs(nStep)) {
     m = nSubs*(mode-nFirst)/abs(nStep);
     omega = 2*PI*mode*Frequency/TD.rate();
     c = cos(omega);
     s = sin(omega);
     p = TD.data;

     ch = cos(2*PI/double(n));
     sh = sin(2*PI/double(n));

     for (k = 0; k < nSubs; k++) {
       a = b = 0.;
       x = omega*k*n;
       C = cos(x);
       S = sin(x);

       if(FilterID==-1){
	 for (i = 0; i < n; i++) {
	   a += C * *p;
	   b += S * *(p++);
	   x = c*C-s*S; 
	   S = c*S+s*C;
	   C = x;
	 }
       }

       else if(FilterID==-2 || mode>nFirst){
	 u = 1.;
	 v = 0.;
	 for (i = 0; i < n; i++) {
	   x = (1-u) * *(p++);
	   a += C*x;
	   b += S*x;
	   x = c*C-s*S; 
	   S = c*S+s*C;
	   C = x;
	   x = ch*u-sh*v; 
	   v = ch*v+sh*u;
	   u = x;
	 }
       }

       else if(FilterID==-3){
	 for (i = 0; i < n; i++) {
	   x = wt.data[i] * *(p++);
	   a += ct.data[i]*x;
	   b += st.data[i]*x;
	 }
       }
 
       amp.data[k+m] = 2.*sqrt(a*a + b*b)/double(n);
       phi.data[k+m] = atan2(b,a);
       //      cout<<m<<" a="<<a<<"  b="<<b<<endl;
       //      cout<<m<<" "<<amp.data[k+m]<<"  "<<phi.data[k+m]<<endl;
     }   
   }
  
//*************************************************************
// reconstruction of filtered interference signal
//*************************************************************
   Q.amplitude.resize(M);
   Q.line.resize(M);
   Q.noise.resize(M);
   Q.filter.resize(M);

   if(clean) TD = 0.;

   for (mode = nFirst; mode < imax; mode += abs(nStep)) {
     m = nSubs*(mode-nFirst)/abs(nStep);
     omega = 2*PI*mode*Frequency/TD.rate();
     c = cos(omega);
     s = sin(omega);

     a = b = 0.;
     for (k = 0; k < nSubs; k++) {
       a += amp.data[k+m]*amp.data[k+m];
       b += phi.data[k+m];
     }

     b /=nSubs;
     Q.line[mode-nFirst]  = 10.;
     Q.noise[mode-nFirst]  = 1.;
     Q.filter[mode-nFirst]  = 1;
     Q.amplitude[mode-nFirst] = float(sqrt(float(a)/nSubs))*exp(f_complex(0.,b));
     Q.intensity += a/nSubs/2.;

     if(clean){
       p = TD.data;
	 
       a = amp.data[m];
       x = phi.data[m];
       C = cos(x);
       S = sin(x);
       for (i = 0; i < n/2; i++){
	   *(p++) -= a*C; 
	   x = c*C-s*S; 
	   S = c*S+s*C;
	   C = x;
       }

       for (k = 0; k < nSubs-1; k++) {
	 //	 cout<<"k="<<k<<" m="<<m<<"  "<<amp.data[k+m]<<endl;
	 x = phi.data[k+m]  +omega*(k*n+n/2);
	 C = cos(x);
	 S = sin(x);
	 x = phi.data[k+m+1]+omega*(k*n+n/2);
	 u = cos(x);
	 v = sin(x);
	 for (i = 0; i < n; i++) {
           a = amp.data[k+m]  * C;
	   b = amp.data[k+1+m]* u;
	   x = c*C-s*S; 
	   S = c*S+s*C;
	   C = x;
	   x = c*u-s*v; 
	   v = c*v+s*u;
	   u = x;
	   *(p++) -= (a*(n-i) + b*i)/n;
	 }
       }

       a = amp.data[nSubs-1+m];
       x = phi.data[nSubs-1+m]+omega*(TD.size()-n/2);
       C = cos(x);
       S = sin(x);
       for (i = TD.size()-n/2; i<N; i++){
	 *(p++) -= a*C; 
	 x = c*C-s*S; 
	 S = c*S+s*C;
	 C = x;
       }

     }
   }

   if(clean){
     b = TD.rms();
     Q.intensity = b*b;
   }
   Q.first = nFirst;

   //   cout<<"hetero: freq="<<Frequency<<"  intensity="<<Q.intensity<<endl;

   return Q;
}


/*******************************************************************
 * getOmega(WaveData &TD) refines the fundamental frequency using
 * phase difference (p2+w*s/2) - (p1+w*s/2), where pi+w*s/2 is a 
 * phase in the middle of time interval s.
 * It works with resampled data
 *******************************************************************/

double linefilter::getOmega(const WaveData &TD, int nsub)
{
   double a,b;
   double F;

   if(noScan)  return Frequency;
   if(!reFine) return -Frequency;

   if(nsub<2) nsub = 2;

   if (Frequency <= 0) {
      cout << " getOmega() error: invalid interference frequency"
	   << " :  " << Frequency << " Hz\n";
      return 0.;
   }

// use Wiener filter if FID = -1 or 1;
//   int FID = (FilterID == 0) ? 0 : 1;
// starting 08/07/01 getOmega always works with FilterID=1
   int FID = 1;     

   WaveData TDR(1);
   TDR.resample(TD, newRate(TD.rate()), 6);
   makeFilter(TDR, FID);
   if(badData) return -Frequency;

   int L  = int(TDR.rate()/Frequency + 0.5);   // number of samples per cycle
   int n  = int(TDR.size()/nsub);                // data section length

   unsigned int imax = maxLine(L);          // max number of harmonics 

   if (n/L == 0 || L < 4) {
      cout << " getOmega() error: input data length too short to contain\n"
	   << " one cycle of target frequency = " << Frequency << " Hz\n";
      return 0.;
   }

   WaveData td2(2*L);                   // SK wavefft fix 9/10/2002
   WaveData tds(L);
   WaveData amp(L);
   WaveData phi(L);
   amp *= 0.;
   phi *= 0.;

   double step = n/TDR.rate();
   double wt = Frequency * step; 
   double phase = 0.;
   double FSNR = SNR/(1.+SNR);          // Filter threshold 

   for (int k = 0; k < nsub; k++) {
      tds.Stack(TDR, n, n*k);           // stack signal

      tds.hann();
// SK wavefft fix 9/10/2002
      td2.rate(tds.rate());              
      td2.cpf(tds); td2.cpf(tds,L,0,L);  
      td2.FFTW(1);                     
      tds[slice(0,L/2,2)]<<td2[slice(0,L/2,4)];
      tds[slice(1,L/2,2)]<<td2[slice(1,L/2,4)];
// SK wavefft fix 9/10/2002: end

//*************************************************************
// estimate frequency from the phase difference
// nsub always > 1 - calculate frequency from the data TD 
//*************************************************************

      for (unsigned int i = 2; i < (unsigned)L-1; i+=2) {
	 F = Filter.data[i/2];
	 a = tds.data[i]*F;
	 b = tds.data[i+1]*F;

	 if(F > FSNR){
	    amp.data[i] += a*a+b*b;
	    phase  = arg(f_complex(a,b))/2./PI;
	    phase += axb(wt/2.,double(i/2));
	    phase -= intw(phase);                    // phase(t)

	    if(k>0){
	       F = phase - phi.data[i+1];
	       F -= intw(F);                         // phase(t2) - phase(t1)
	       phi.data[i] += (long(wt*(i/2)+0.5) + F)/step/(i/2);
	    }
	    else
	       phi.data[i] = 0.; 

 	    phi.data[i+1] = phase;
	 }
      }
   }

// average frequency

   double omega = 0.;
   double weight = 0.;

   for (unsigned int i = nFirst; i < imax; i += abs(nStep)) {
      F = Filter.data[i];
      if(F>FSNR){
	 a = 1 - F;
	 if(a < 0.0001) a = 0.0001;
	 a = 1./a;
 	  omega += a*phi.data[2*i];
	 weight += a;
      }
   }

// was before 08/08/01
// 	  omega += amp.data[2*i]*phi.data[2*i];
//	 weight += amp.data[2*i];

   omega = (weight>1.) ? omega/weight/(nsub-1) : -Frequency;
   return omega;
}


/***********************************************************************
 * fScan(WaveData &) finds the line fundamental frequency 
 * (saves in Frequency) and the interference amplitude.
 * Uses:
 * Frequency - seed value for line frequency
 * fBand     - defines frequency band (f-df,f+df), where to scan for peak
 ***********************************************************************/
double linefilter::fScan(const WaveData &TD)
{
#define np 6    // number of points in interpolation scheme in resample()

   badData = false;

   if(noScan) return Frequency;

   WaveData td2(1);

   int    n = TD.size();
//  int    FID = (FilterID<0) ? abs(FilterID) : 0;  // set filter ID for makeFilter()
   int    FID = 0;  // starting on 03/24/01 for fScan always use FID=0

   double d, s, sp, fc;
   double fwhm = 1.;
   double delta = 0.;
   double am = 1., ac = 0.;
   double dfft = nSubs*TD.rate() / n;
   double e1 = 1.e-3;

// double fw = fBand*dfft;            before 08/03/01
   double fw = fBand*dfft/nFirst;

   double ff = Frequency;      // seed frequency before tunning 
   double fp = Frequency;      // central frequency.          

// ff - seed interference frequency;
// fc - central frequency
// fp - peak's frequency estimated via scan
// fnew - new sampling rate which is multiple of target frequency f
// ac - calculated frequency correction (relative to fc) in uints of fw
// am - maximum allowed deviation of frequebcy in units of fw
// aw - am/FWHM of the peak
// ic = +1, -1 stores the sign of ac at previous step
// dfft - Fourier frequency resolution for given sample
// fw - frequency band width for 3 sample points to build parabola
// e1  - limit iteration accuracy of the frequency deviation
// e2  - limit iteration accuracy for frequency window width

   if (TD.rate() <= 0.) {
      cout << " fScan() error: invalid sampling rate = "
	   << TD.rate() << " Aborting calculation.\n";
      badData = true;
      return ff;
   }

   if ( ff <= 0.) {
      cout << " fScan() error: invalid interference frequency = "
	   << ff << " Aborting calculation.\n";
      badData = true;
      return ff;
   }


//*******************************************************
// detailed scan of the region (ff-fband, ff+fband)
//*******************************************************

  int mScan = -nScan;
  if ( mScan > 0 ) {
     WaveData sw(mScan);
     sp = 0.;
    
    cout <<" Scanning frequency from "<< ff-mScan*fw/2. << " Hz to "
         << ff+mScan*fw/2. <<" Hz\n";

    int ip = 0;
    for (int i=0; i < mScan && !badData; i++) {
       Frequency = ff + (i - mScan/2)*fw;

       td2.resample(TD, newRate(TD.rate()), np);
       s = makeFilter(td2,FID); 

       sw.data[i] = s;
       if(s > sp) {sp = s; fp = Frequency; ip = i;}    
       printf(" Frequency = %f Hz, sqrt(<E>) = %f \n", Frequency, s);
    }

//  improve peak approximation by taking 3 points close to peak
    if (ip > 0 && ip < (mScan-1) && !badData) {
       d = 2.*sw.data[ip] - sw.data[ip + 1] - sw.data[ip - 1];
       fp += (d > 0.) ? 0.5*fw*(sw.data[ip + 1] - sw.data[ip - 1])/d : 0.;
    }
  }

//*******************************************************
// start tuning frequency
//*******************************************************
  int k = 3;
  int mode;
  double ss[3];
  int ks[3] = {1,1,1};

  fc = fp;
  while (!badData) {

// calculate energies for 3 values of frequency : fp-fw, fp, fp+fw
    for (int m = 0; m < 3; m++) {
       if(ks[m]){
	  Frequency = fc + fw*(m - 1);
	  td2.resample(TD, newRate(TD.rate()), np);
	  ss[m] = makeFilter(td2,FID);
	  ks[m] = 0;
       }
       if(badData) break;
    }

    if(k++ > nScan){ 
       badData = true;   // limit number of iterations
//       cout << "fScan hits a limit on number of iterations\n";
    }
    if(badData) break;

// find minimum of parabola using three values ss[i]
// if d > 0 - parabola has maximum, if d < 0 - minimum

    d=2.*ss[1]-(ss[2]+ss[0]);

    if (d > 0.) {
       ac = 0.5 * (ss[2] - ss[0]);         // d * (f-fc)/fw
       fwhm = sqrt(ac*ac+2.*ss[1]*d)/d;    // FWHM/fw 
       fwhm *= fw/dfft;                    // FWHM/dfft 
       ac /= d;                            // (f-fc)/fw
    }
    else {
       ac = (ss[2] > ss[0])? am : -am;
       fwhm = 1.;
    }

//    cout<<"        fp="<<fp<<"  fc="<<fc<<"  fw="<<fw<<"  ac="<<ac<< "\n";

    mode = 0;                                     // fw  shift
    if(fabs(ac) < am) mode = 1;                   // fw/2  shift
    if(fabs(ac)<am/4. && fw/dfft>0.1) mode = 2;   // no shift

    delta = 1.;
    if(mode){
       delta = (fc-fp)/fw + ac;           // deviation from the peak frequency
       fp += delta*fw;                    // new peak frequency
    }

    delta *= fw/dfft;                     // deviation in units of dfft

//    cout<<k-1<<"  "<<fp<<"  "<<delta<<"  "<<fwhm<<"  "<<ac*fw/dfft<< "\n";

    if(fabs(delta) < e1) break;                       // limit in units of dfft
    if(fabs(delta*fwhm) < e1 && fw/dfft<0.1) break;   // limits ac in units of FWHM

    switch (mode) {
       case 0:                            // one step shift right or left
	  if(ac > 0.){                    // move one fw step right
	     ss[0] = ss[1]; ss[1] = ss[2]; ks[2] = 1;
	  }
	  else{                           // move one fw step left
	     ss[2] = ss[1]; ss[1] = ss[0]; ks[0] = 1;
	  }
	  fc += (ac>0.) ? fw : -fw;       // new central frequency
	  fp = fc;
	  break;

       case 1:                            // half step shift right or left
	  if(ac > 0.){                 
	     ss[0] = ss[1]; ks[1] = 1;
	  }
	  else{                          
	     ss[2] = ss[1]; ks[1] = 1;
	  }
	  fw *= 0.5;                      // reduce fw by 2 if d>0  
	  fc += (ac>0.) ? fw : -fw;       // new central frequency
	  break;

       case 2:                            // no shift 
	  ks[0] = 1; ks[2] = 1;
	  fw = 4*fw*fabs(ac);                   // reduce fw  
	  if(fw/dfft<0.01) fw = 0.01*dfft;      // reduce fw  
	  k++;
	  break;

    }
	  
  }

  if(badData)
     fp = ff;
//  else if(reFine){
//     Frequency = fp;                    // use value of frequency found by fScan
//     fp = fabs(getOmega(TD, nSubs));    // return refined frequency
//  }
  Frequency = ff;
  return (badData) ? Frequency : fp;    // return refined frequency

}


/***********************************************************************
 * Function returns the interference signal for harmonics from nFirst
 * to nLast with fundamental frequency F  
 * Update lineList trend data
 * Filter = 0 - corresponds to pure "comb" filter
 * Filter = 1 - corresponds to optimal Wiener filter with 1-st method of
 *              noise estimation (default)
 ***********************************************************************/
double linefilter::Interference(WaveData &TD, double omega)
{
  WaveData  td2(1);
  double s = 0.;
  linedata v;
  double seedF = Frequency;

// use Wiener filter if FID = -1 or 1;
  int FID = (FilterID == 0) ? 0 : 1;

  if ( TD.rate() <= 0. || omega <= 0. ) {
    cout << " Interference() error: invalid interference frequency  = "
         << omega << "\n Aborting calculation.\n";

  }

  v.T_current = CurrentTime;
  v.intensity = 0.;
  v.frequency = Frequency;
  v.first = nFirst;

  if (badData) {                            // skip bad data
//     lineList.insert(lineList.end(),v);     // update lineList trend data
//     cout << "Interference(): skip bad data\n";
     return 0.;
  }

  if(FilterID < 0){
    v = getHeteroLine(TD);
    s = v.intensity;
  }
  else {
  
// resample data at new sample rate "fnew" which is multiple of base
// frequency of interference fLine and approximately is two times as high
// as original frequency "Rate"

    Frequency = omega;
    td2.resample(TD, newRate(TD.rate()), nRIF);
    s = makeFilter(td2, FID);
    v = getLine(td2);                     // get interference and linedata

    if(clean){                              // return interference
      if(badData){ 
	TD *= 0.;
      }
      else
	TD.resample(td2, TD.rate(), nRIF);  // resample at original sampling rate
    }
    
  }
  
  if(badData){ 
     v.frequency *= -1.;                  
     Frequency = seedF;                   // do not update frequency;
  }
  if(v.intensity > 0.)
    lineList.push_back(v);                // update lineList trend data

  //   cout<<"intensity="<<sqrt(v.intensity)<<"  s="<<s
  //       <<"  amplitude="<<abs(v.amplitude[0])<<endl;

  return s;
}



/************************************************************************
 * apply(ts) estimate interference and save trend data 
 * if clean=true returns time series with the interference signal removed
 ***********************************************************************/
void linefilter::apply(WaveData& ts) {
//----------------------------------  Check the input data.
    if (!ts.size()) return;
    if (!ts.rate()) return;

    StartTime = ts.start();
    CurrentTime = StartTime;

    Stride = ts.size()/ts.rate();
    double s = Window > 0. ? Window : Stride;
    double delta = s;

    double rate = ts.rate();
    int n = ts.size();

    int LPF = (nLPF>0) ? nLPF : 0;                 // depth of the wavelet LP filter 

    Biorthogonal<wavereal> w(nWave);               // needed if LPF is applied
    WSeries<wavereal> *pw = 0;
   
    WaveData tw;                                   // needed if LPF is applied
    double omega = Frequency;

    int NN = 0;
    int nTS = ts.size();
    
    if(LPF){                        // apply wavelet low path filter
       pw = new WSeries<wavereal>(ts, w);
       NN = (nTS >> LPF) << LPF;    // new number of samples
       if(NN != nTS){               // adjust the number of samples
	  NN += 1<<LPF;
	  pw->resample(NN*rate/nTS,nRIF);   
	  rate = pw->rate();
       }

       pw->Forward(LPF);
       pw->getLayer(tw,0);
       rate /= (1<<LPF);             // rate of decimated data
       tw.rate(rate);
       n = tw.size();
    }

    int nn = Window > 0. ? int(Window*rate) : n;   // if window = 0, take whole ts

//    cout<<"Window="<<Window<<"  rate="<<rate<<"  n="<<n<<"  nn="<<nn<<endl;

    if (nn < int(rate/Frequency)) {
      cout <<" linefilter::apply() error: invalid time window "<<Window<<" sec.\n";
      return;
    }

    WaveData *tx = new WaveData(nn);

//    printf(" Time interval (sec) | Base frequency (Hz) | Sqrt(E_int)\n");

    int i = 0;
    while(i <= n-nn && nn > 0) {

      if((n-i-nn) < nn) {            // the last data interval is too short 
	 delta *= double(n - i)/nn;  // add leftover to the last interval    
	 nn = n - i;                     
      }

      tx->rate(rate);
      if((int)tx->size() != nn) tx->resize(nn);

      if(LPF) tx->cpf(tw,nn,i);
      else    tx->cpf(ts,nn,i);

      if(FilterID>=0){
	if (!reFine || badData || (lineList.size()<3))
	  omega = fScan(*tx);
	else{
	  omega = getOmega(*tx, nSubs);
	  if(omega < 0.) omega = fScan(*tx);
	}
      }

      s = Interference(*tx, omega);
      CurrentTime += delta;

      if(clean && !badData){
	 if(LPF) tw.sub(*tx,nn,0,i);
	 else	 ts.cpf(*tx,nn,0,i);
      }

      //      if(!badData)
//            printf(" %8.3f - %8.3f %12.6f %20.6f\n" ,
//      	     double(i)/rate, double(i + nn)/rate, Frequency*nFirst, sqrt(s));

      i += nn;
    }

    if(clean && LPF){
       pw->putLayer(tw,0);
       pw->Inverse();
       if(NN != nTS)
	  ts.resample((WaveData &) *pw,ts.rate(),nRIF);
       else
	  ts = *pw;

       if(nTS != (int)ts.size())   // check if data has the same length as input data
	  cout << "linefilter::apply(): is "<<ts.size()<<",  should be: "<<nTS<<"\n";
    }

    delete tx;
    if(pw) delete pw;
    return;
}

/*********************************************************************
 * get Trend Data from lineList object
 *
 *********************************************************************/
wavearray<float> linefilter::getTrend(int m, char c)
{
  int l_size = lineList.size();
  int v_size;
  int n = 0;
  wavearray<float> out(l_size);
  wavearray<float> www(l_size);
  list<linedata>::iterator it;
  linedata* v;
  double a=0., F=0.;
  double step = (Window>0.) ? Window : Stride;
  double time = 0.;
  double phase = 0.;
  double averF = 0.;

  out = 0;
  www = 0;

  out.rate((step>0.) ? 1./step : 1.); 

  if(l_size < 2) { return out; }

  if(m < 0) m=0;
  it = lineList.begin();   // set the list at the beginning

  v = &(*(it++));
  v_size = v->amplitude.size();

  if(m > 0 && m <= v_size)
     phase = arg(v->amplitude[m-1]); 
  
  out.start(v->T_current);

  if(c=='p')               // average frequency 
    for(int i=0; i<l_size; i++) {
      v = &(*(it++));
      F = double(v->frequency);
      if(F<=0.) continue;
      F *= (m == 0) ? 1. : (v->first+m-1);
      averF += F; n++;
    }

  if(n) averF /= double(n);

  it = lineList.begin();   // set the list at the beginning

  for(int i=0; i<l_size; i++) {
     v = &(*(it++));
     v_size = v->amplitude.size();   
     switch (c) {

	case 't':          // get start time
	   out.data[i] = v->T_current - out.start(); 
	   break;

	case 'a':          // get harmonic's amplitude
	   if(m == 0 || m > v_size) 
	      out.data[i] = (m==0) ? sqrt(v->intensity) : 0.;
	   else 
	      out.data[i] = abs(v->amplitude[m-1]);	      
	   break;
	   
	case 'p':          // get harmonic's phase
	case 'f':          // get harmonic frequency 
	   if(m == 0 || m > v_size){
	      if(c == 'p') 
		 out.data[i] = v_size;
	      else
		 out.data[i] = v->frequency;
	   }
	   else {
	      F  = fabs(v->frequency * (v->first+m-1));
	      a  = (arg(v->amplitude[m-1])-phase)/2./PI;
	      a += axb(F, Stride/2.);
	      a -= axb(averF, v->T_current-out.start()+Stride/2.);
	      a -= (a>0.) ? long(a+0.5) : long(a-0.5);
	      out.data[i] = 2.*PI*a;
	      www.data[i] = 2.*PI*a;
	      step = v->T_current - time;
	      time = v->T_current;
	      if(c == 'f') out.data[i] = F;
	   }
	   if(c == 'f' && step<1.1*Stride){   // calculate frequency
	      a = (i>0) ? (www.data[i]-www.data[i-1])/2./PI : F*step;
	      a -= (a>0.) ? long(a+0.5) : long(a-0.5);
	      out.data[i] = (long(F*step+0.5) + a)/step;
	   }
	   break;

	case 'F':          // get harmonic frequency 
	   a = double(v->frequency);
	   out.data[i] = (m == 0) ? a : a*(v->first+m-1);
	   break;

	case 'P':          // get harmonic's phase
	   if(m == 0 || m > v_size) 
	      out.data[i] = v_size;
	   else 
	      out.data[i] = arg(v->amplitude[m-1]);	      
	   break;

	case 's':          // get signal power
	   out.data[i] = 0.;
	   if(m > v_size || v_size < 1) break; 
	   if(m == 0) 
	      for(int j=0; j<v_size; j++){
		 F = v->filter[j];
		 out.data[i] += v->line[j] * F*F; 
	      }
	   else{ 
	      F = v->filter[m-1];
	      out.data[i] = v->line[m-1] * F*F;
	   }
	   out.data[i] *= out.rate();
	   break;
	   
	case 'S':          // get signal spectral density
	   out.data[i] = 0.;
	   if(m > v_size || v_size < 1) break; 
	   if(m == 0) 
	      for(int j=0; j<v_size; j++)
		 out.data[i] += v->line[j]; 
	   else 
	      out.data[i] = v->line[m-1];	      
	   break;
	   
	case 'n':          // get noise power
	   out.data[i] = 0.;
	   if(m > v_size || v_size < 1) break; 
	   if(m == 0) 
	      for(int j=0; j<v_size; j++)
		 out.data[i] += v->noise[j] * v->filter[j]; 
	   else 
	      out.data[i] = v->noise[m-1] * v->filter[m-1];	      
	   out.data[i] *= out.rate();
	   break;
	   
	case 'N':          // get noise spectral density
	   out.data[i] = 0.;
	   if(m > v_size || v_size < 1) break; 
	   if(m == 0) 
	      for(int j=0; j<v_size; j++)
		 out.data[i] += v->noise[j]; 
	   else 
	      out.data[i] = v->noise[m-1];	      
	   break;
	   
	case 'K':          // get SNR 
	   a = 0.;
	   out.data[i] = 0.;
	   if(m > v_size || v_size < 1) break; 
	   if(m == 0) 
	      for(int j=0; j<v_size; j++){
		 F = v->filter[j];
		 a += v->noise[j]*F; 
		 out.data[i] += v->line[j] * F*F; 
	      }
	   else{ 
	      F = v->filter[m-1];
	      a = v->noise[m-1]*F; 
	      out.data[i] = v->line[m-1] * F*F;	      
	   }
	   out.data[i] /= (a>0.) ? a : 1.;
	   break;
	   
	case 'W':          //get filter (m>0)
	   out.data[i] = 0.;
	   if(m > v_size || v_size < 1) break; 
	   if(m > 0) 
	      out.data[i] = v->filter[m-1];
	   break;
	   
	default:            // get first harmonic
	   out.data[i] = v->first; 
	   break;
     }	   
  }
  return out;
}  


/*********************************************************************
 * Dumps lineList data into file *fname in binary format and type 
 * "double".
 *********************************************************************/
bool linefilter::DumpTrend(const char *fname, int app)
{
  size_t l_size;
  size_t v_size;
  linedata* v;

  list<linedata>::iterator it = lineList.begin();

  if(dumpStart >= lineList.size()) return false;
  for(size_t i=0; i<dumpStart; i++) it++;
  l_size = lineList.size() - dumpStart;

// calculate the data length

  size_t max_size = 0; 
  for(size_t i=0; i<l_size; i++) {
     v = &(*(it++));
     v_size = v->amplitude.size();
     if(v_size > max_size) max_size = v_size;
  }

  int m = (5*max_size+4);
  size_t n = m*(l_size+1);
  if(n < 4) return false; 

// pack lineList into WaveData out

  wavearray<float>* out = new wavearray<float>(int(n));
  out->data[0] = max_size;       // length of the linedata amplitude vector
  out->data[1] = l_size;         // length of the lineList
  out->data[2] = m;              // total length of the linedata
  out->data[3] = n;              // total lenght
  
  it = lineList.begin();
  for(size_t i=0; i<dumpStart; i++) it++;

//  cout<<"dumpStart = "<<dumpStart;
//  cout<<",    listSize  = "<<lineList.size();
//  cout<<",    Stride = "<<Stride<<endl;

  double gpsTime = it->T_current;

  int gps   = int(gpsTime)/1000;
  out->data[4] = float(gps);                     // int(gps time/1000)
  out->data[5] = gpsTime - 1000.*double(gps);    // rest of gps time
  out->data[6] = (Window>0.) ? Window : Stride;  // filter stride

  for(size_t i=1; i<=l_size; i++) {
     v = &(*(it++));
     v_size = v->amplitude.size();

     out->data[i*m+0]= v->T_current - gpsTime; 
     out->data[i*m+1]= v->frequency; 
     out->data[i*m+2]= v->intensity; 
     out->data[i*m+3]= v->first; 

     for(unsigned int j=0; j<max_size; j++){
	if(j < v_size) {
	   out->data[i*m+4+j*5] = abs(v->amplitude[j]);
	   out->data[i*m+5+j*5] = arg(v->amplitude[j]);
	   out->data[i*m+6+j*5] = v->line[j];
	   out->data[i*m+7+j*5] = v->noise[j];
	   out->data[i*m+8+j*5] = v->filter[j];
	}
	else {
	   out->data[i*m+4+j*5] = 0.;
	   out->data[i*m+5+j*5] = 0.;
	   out->data[i*m+6+j*5] = 0.;
	   out->data[i*m+7+j*5] = 0.;
	   out->data[i*m+8+j*5] = 0.;
	}
     }
  }

  out->DumpBinary(fname, app);

  delete out;
  return true;
}


/*********************************************************************
 * Read trend data from file into the lineList data structure.
 *********************************************************************/
bool linefilter::LoadTrend(const char *fname)
{
  linedata v;
  wavearray<float> in;
  float *p = 0;
  double last = 0.;
  double step = 0.;
  double time = 0.;
  double gpsTime = 0.;
  unsigned int count = 0;

  in.ReadBinary(fname);              // read file
  if(in.size() < 6) return false;

  while(count < in.size()){
     p = &(in.data[count]);
     int m = int(*(p+2) + 0.5);      // total length of the linedata
     if(m <= 1) return false; 

     int l_size = int(*(p+1) + 0.5); // length of the lineList
     if(l_size < 1) return false; 

     int v_size = int(*p + 0.5);     // length of the linedata amplitude vector
     count += int(*(p+3)+0.5);       // add total length of the block

     gpsTime = double(int(*(p+4)+0.5))*1000. + *(p+5);  // block start time
     StartTime = 0.;                             
     CurrentTime = *(p+6);                      
     Stride = *(p+6);
     Window = *(p+6);

     v.amplitude.resize(v_size);
     v.line.resize(v_size);
     v.noise.resize(v_size);
     v.filter.resize(v_size);

     time = *(p+m);
     last = time;
     step = 0.;

// pack lineList from WaveData inv->T_current; 

     for(int i=1; i<=l_size; i++) {
	p += m;
	if(*p != 0.) step = *p - last;
	time += step;
	last = *p;
	v.T_current = time+gpsTime; 
	v.frequency = *(p+1); 
	v.intensity = *(p+2); 
	v.first     = int(*(p+3)+0.5); 

	for(int j=0; j<v_size; j++){
	   v.amplitude[j] = *(p+4+j*5);
	   v.amplitude[j]*= exp(f_complex(0,*(p+5+j*5)));
	   v.line[j]      = *(p+6+j*5);
	   v.noise[j]     = *(p+7+j*5);
	   v.filter[j]    = *(p+8+j*5);
	}

	lineList.insert(lineList.end(),v);     
     }
  }
  return true;
}




