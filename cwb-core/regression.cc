/*
# Copyright (C) 2019 Sergey Klimenko, Vaibhav Tiwari
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


//---------------------------------------------------
// WAT class for regression analysis
// S. Klimenko, University of Florida
//---------------------------------------------------

#include <time.h>
#include <iostream>
#include <stdexcept>
#include "regression.hh"
#include "Biorthogonal.hh"
#include "SymmArray.hh"
#include "WDM.hh"

ClassImp(regression)  // used by THtml doc

regression::regression() : Edge(0.), pOUT(false) 
{
//
// Default constructor
//

  this->clear(); 
}

regression::regression(WSeries<double> &in, char* ch, double fL, double fH) 
{ 
//
// Constructor from WSeries, add target channel
//
// in: TF transform of target
// ch: channel name
// fL, fH:  low and high frequency
//
// It calls the add function
//

  this->add(in,ch,fL,fH); 
}

regression::regression(const regression& value)  :
 Edge(0.), pOUT(false) 
{
//
// Copy constructor
//

  *this = value; 
}

regression& regression::operator=(const regression& value)
{ 
//
// Copy constructor using operator
//

   std::cout<<"I am =\n";
   this->chList = value.chList;   // TF data: 0 - target, >0 - withess   
   this->FILTER = value.FILTER;   // total Wiener filter                 
   this->chName = value.chName;   // channel names                       
   this->chMask = value.chMask;   // vectors of selected/rejected (1/0) layers 
   this->matrix = value.matrix;   // symmetric matrix                    
   this->vCROSS = value.vCROSS;   // cross-correlation vector                    
   this->vEIGEN = value.vEIGEN;   // eigenvaue vector                    
   this->target = value.target;   // target time series                  
   this->rnoise = value.rnoise;   // regressed out noise    
   this->WNoise = value.WNoise;   // Wavelet series for regressed out noise             
   this->Edge   = value.Edge;     // boundary buffer
   this->pOUT   = value.pOUT;     // printout flag

   return *this; 
}

size_t regression::add(WSeries<double> &in, char* ch, double fL, double fH) {
//
// clear and add target channel to the regression list  
//
// in: target channel, time-frequency series
// ch: name - channel name
// fL, fH: -  fL <-- frequency band --> fH. If fL>=fH - set boundaries from input in
//
// return value: number of channels in the regression list 
//

   this->clear();
   size_t I = in.maxLayer();
   wavearray<int> mask(I+1); mask=0;
   WSeries<double> ww;
   ww = in;
   ww.setlow(fL>=fH ? in.getlow() : fL);
   ww.sethigh(fL>=fH ? in.gethigh() : fH);

   for(size_t i=1; i<I; i++) {
      if(ww.frequency(i)<=ww.gethigh() && ww.frequency(i)>=ww.getlow()) 
         mask.data[i] = 1;
   }

   this->chList.push_back(ww);
   this->chName.push_back(ch);
   this->chMask.push_back(mask);
   ww.Inverse();
   this->target = ww;
   return chList.size();
}

size_t regression::add(wavearray<double> &in, char* ch, double fL, double fH) {
//
// add witness channel to the regression list  
//
// in: witness time series
// ch: channel name, if ch name is the same as witness, build LPE filter
// fL, fH:  fL <-- frequency band --> fH 
//
// return value: number of channels in the regression list    
//

   if(!chList.size()) {
      std::cout<<"regression::add() ERROR: add target first\n";
      exit(1);
   }
   if(chList.size()>1 && strcmp(this->chName[0],this->chName[1])==0) {
      std::cout<<"regression::add() ERROR: second witness channel can not be added to LPEF\n";
      exit(1);
   }
   in -= in.mean();
   if(in.rms()==0.) {
      std::cout<<"regression::add() warning: empty witness channel.\n";
      return 0;
   }
   double rate = chList[0].rate();
   size_t size = chList[0].size();
   int n = in.rate()>rate ? int(in.rate())%int(rate) : int(rate)%int(in.rate());
   int m = in.rate()>rate ? int(in.rate()/rate) : -int(rate/in.rate());
   int l = int(log(fabs(m))/log(2.)+0.1);

   if(n || abs(m)!=int(pow(2.,l)+0.1)) {
      std::cout<<"regression::add() ERROR: incompatible target and witness channels\n";
      exit(1);
   }

   Biorthogonal<double> Bio(512);
   WSeries<double> ws(Bio);
   wavearray<double> ts;
   WDM<double>* wdm = (WDM<double>*)this->chList[0].pWavelet->Clone();

   if(m==1) {              // copy witness
      ts = in;
   }
   else if(m>0) {          // downsample witness
      ws.Forward(in,l);
      ws.getLayer(ts,0);      
   }
   else {                 // upsample witness
      ws.resize(abs(m)*in.size());
      ws.rate(abs(m)*in.rate());
      ws.Forward(l); ws=0.;
      ws.putLayer(in,0); 
      ws.Inverse(); ts = ws;
   }
   ws.Forward(ts,*wdm);

   if(ws.rate()!=rate || ws.size() != size) {
      std::cout<<"regression::add() ERROR: incompatible target and witness channels\n";
      exit(1);
   }
  
   ws.setlow(fL>=fH ? 0. : fL);
   ws.sethigh(fL>=fH ? ws.rate()/2. : fH);

   size_t I = ws.maxLayer()+1;
   wavearray<int> mask(I); mask=0;

   for(size_t i=0; i<I; i++) {
      if(ws.frequency(i)<=ws.gethigh() && ws.frequency(i)>=ws.getlow()) 
         mask.data[i] = 1;
      //ws.getLayer(ts,i);
      //if (ws.frequency(i)>500 && ws.frequency(i)<540) std::cout << ws.frequency(i) << " " << ts.mean() << " " << ts.rms() << endl;
      //if (ts.rms()==0) mask.data[i]=0;      
   }

   this->chList.push_back(ws);
   this->chName.push_back(ch);
   this->chMask.push_back(mask);
   
   delete wdm;

   return chList.size();
}

size_t regression::add(int n, int m, char* ch) {
//
// construct witness channel from 2 channels and add
//
// n, m channel indeces in chList
// ch: channel name
//         
// return value: number of channels in the regression list
//


   if(chList.size()>1 && strcmp(this->chName[0],this->chName[1])==0) {
      std::cout<<"regression::add() ERROR: second witness channel can not be added to LPEF\n";
      exit(1);
   }
   if(n>=(int)chList.size() || m>=(int)chList.size()) {
      std::cout<<"regression::add() WARNING: can't construct witness channel.\n";
      return chList.size();
   }

   WSeries<double> wsn = this->chList[n];
   WSeries<double> wsm = this->chList[m];
   wavearray<int>* pMn = &(this->chMask[n]);
   wavearray<int>* pMm = &(this->chMask[m]);
   wavearray<double> ts;
   WDM<double>* wdm = (WDM<double>*)this->chList[0].pWavelet->Clone();

   size_t I = wsn.maxLayer()+1;
   wavearray<double> wa;
   wavearray<int> mask(I); mask=0;

   for(size_t i=0; i<I; i++) {
      if(!(pMn->data[i])) {
         wsn.getLayer(wa,i); wa = 0;
         wsn.putLayer(wa,i);
      }
      if(!(pMm->data[i])) {
         wsm.getLayer(wa,i); wa = 0;
         wsm.putLayer(wa,i);
      }
      if(!(pMn->data[i])) continue;
      for(size_t j=0; j<I; j++) {
         if(!(pMm->data[j])) continue;
         if(i+j < I-1) mask.data[i+j] = 1;
         if(i>j) mask.data[i-j] = 1;
         if(i<j) mask.data[j-i] = 1;
      }
   }

   wsn.setlow(chList[0].getlow());
   wsn.sethigh(chList[0].gethigh());
   for(size_t i=0; i<I; i++) {
      if(wsn.frequency(i)>wsn.gethigh() || wsn.frequency(i)<wsn.getlow()) 
         mask.data[i] = 0;
   }

   wsn.Inverse();
   wsm.Inverse();
   wsm *= wsn; ts = wsm; ts -= ts.mean();
   wsn.Forward(ts, *wdm);
   this->chList.push_back(wsn);
   this->chName.push_back(ch);
   this->chMask.push_back(mask);

   delete wdm;

   return chList.size();
}   

size_t regression::setFilter(size_t K) {
//
// set Wiener filter structure
//
// K: half-size length of a unit filter: each witness layer has filter 2(2*k+1) long
//
// return value: total number of witness layers
//
// FILTER.filter[0] is not used
//
  
   size_t i,j;
   size_t I = this->chList[0].maxLayer();
   size_t count = 0;
   wavearray<int>* pT = &(this->chMask[0]);
   wavearray<int>* pW;
   vectorD f; f.resize(2*K+1);

   this->FILTER.clear(); std::vector<Wiener>().swap(FILTER);
   this->matrix.clear(); std::vector<TMatrixDSym>().swap(matrix);
   this->vCROSS.clear(); std::vector< wavearray<double> >().swap(vCROSS);
   this->vEIGEN.clear(); std::vector< wavearray<double> >().swap(vEIGEN);

   this->kSIZE = K;
   for(i=1; i<I; i++) {
      if(!pT->data[i]) continue;          // check target mask

      Wiener WF;
   
      for(j=0; j<chList.size(); j++) {
         pW = &(this->chMask[j]);
         if(!pW->data[i]) continue;       // check witness mask
         WF.channel.push_back(j); 
         WF.layer.push_back(i); 
         WF.norm.push_back(1.); 
         WF.filter00.push_back(f); 
         WF.filter90.push_back(f); 
         if(j) count++;
      }
      if(WF.channel.size()>1) this->FILTER.push_back(WF);
      else pT->data[i] = 0;               // mask target layer
   }
   return count;
}

void regression::mask(int n, double flow, double fhigh) {
//
// mask (set to 0) Mask vector
//
// n: channel index
// flow, high: low and high frequency bounds
//

   int* pM = this->chMask[n].data;
   WSeries<double>* pW = &(this->chList[n]);
   size_t I = pW->maxLayer()+1;
   for(size_t i=0; i<I; i++) {
      if((pW->frequency(i)<fhigh && pW->frequency(i)>flow) || fhigh<=flow) 
         pM[i] = 0;
   }

}

void regression::unmask(int n, double flow, double fhigh) {
//
// unmask (set to 1) Mask vector
//
// n: channel index
// flow, high: low and high frequency bounds
//

   int* pM = this->chMask[n].data;
   WSeries<double>* pW = &(this->chList[n]);
   size_t I = pW->maxLayer()+1;
   for(size_t i=0; i<I; i++) {
      if((pW->frequency(i)<fhigh && pW->frequency(i)>flow) || fhigh<=flow) 
         pM[i] = 1;
   }

}

wavearray<double> regression::getVEIGEN(int n) { 
//
// extract eigenvalues 
//
// n: channel index. If -1 return all eigenvalues
//
// return wavearray: eigen-values
//

   wavearray<double> E;
   size_t I = this->vEIGEN.size();
   if(n<(int)I && n>=0) return vEIGEN[n];
   for(size_t i=0; i<I; i++) E.append(this->vEIGEN[i]);
   return E;
}
   
wavearray<double> regression::getFILTER(char c, int nT, int nW) { 
//
// extract filter for target index nT and witness nW
//
//  c: 'f' or 'F' - extract 0-phase or 90-phase filter
//     'n' or 'N' - extract 0-phase or 90-phase  norm
//     'a' or 'A' - extract 0-phase - 90-phase  asymmetry 
//  nT: target index. If <0, return full vector
//  nW:  witness index
//
//  return wavearray: filter values
//

   wavearray<double> OUT;
   int N = int(this->FILTER.size());
   int i,I,j,J;

   if(nT>=0 && nT<N) {i=nT; I=nT+1;}
   else {i=0; I=N;}

   for(int n=i ;n<I;n++) {
      int M = int(this->FILTER[n].filter00.size());

      if(nW>0 && nW<M) {j=nW; J=nW+1;}
      else {j=1; J=M;}

      for(int m=j; m<J; m++) {
         int K = int(this->FILTER[n].filter00[m].size());
         wavearray<double> e(K);
         wavearray<double> E(K);

         for(int k=0; k<K; k++) {
            e.data[k] = this->FILTER[n].filter00[m][k];
            E.data[k] = this->FILTER[n].filter90[m][k];
         }

         if(c=='a') {
            double re = e.rms();
            double RE = E.rms();
            OUT.append((re*re-RE*RE)/(re*re+RE*RE));
         }
         else if(c=='n') OUT.append(e.rms());
         else if(c=='N') OUT.append(E.rms());
         else {         
            if(c!='F') OUT.append(e);
            if(c!='f') OUT.append(E);
         }
      }
   }
   return OUT; 
}
   
void regression::setMatrix(double edge, double f) {
//
// set system of linear equations: M * F = V 
// M = matrix array, V =  vector of free coefficients, F = filters
//
// edge: boundary interval in seconds excluded from training
// 1-f: tail fraction to be removed during training 
//

   int i,j,n,m,nn,mm,N;
   int I = this->FILTER.size();                 // number of target layers
   int K = (int)this->kSIZE;                    // unit filter half-size
   int K2 = 2*K;
   int K4 = 2*(2*K+1);
   int k,ii,jj,kk;
   int sIZe,sTEp,k00n,k90n,k00m,k90m;

   double FLTR = !strcmp(this->chName[0],this->chName[1]) ? 0. : 1.;  // LPEF : Wiener
   double fm = fabs(f);

   double fraction, rms;
   WSeries<double>* pTFn;
   WSeries<double>* pTFm;
   wavearray<double> ww;
   wavearray<double> WW;
   wavearray<double> qq;
   std::slice s00n,s90n;
   std::slice s00m,s90m;
   Wiener* pW;
   SymmArray<double> acf(K2);
   SymmArray<double> ccf(K2);

   this->Edge = edge;
   this->target.edge(edge); 
   for(i=0; i<(int)this->chList.size(); i++) {   // loop on channels
      this->chList[i].edge(edge);                // set edge
   }
   if(!I) {std::cout<<"regression::nothing to clean.\n"; return;}

   this->matrix.clear(); std::vector<TMatrixDSym>().swap(matrix);
   this->vCROSS.clear(); std::vector< wavearray<double> >().swap(vCROSS);
   this->vEIGEN.clear(); std::vector< wavearray<double> >().swap(vEIGEN);

   for(i=0; i<I; i++) {                      // loop on target layers
      pW = &FILTER[i];
      N  = pW->layer.size();
      TMatrixDSym MS((N-1)*K4);
      wavearray<double> VC((N-1)*K4);

      pTFm = &(this->chList[0]);             // target
      s00m = pTFm->pWavelet->getSlice(pW->layer[0]);
      s90m = pTFm->pWavelet->getSlice(-pW->layer[0]);
      k00m = s00m.start(); 
      k90m = s90m.start(); 
      sIZe = s00m.size();                    // # of samples in a layer
      sTEp = s00m.stride();                  // layer stide

      for(n=0; n<N; n++) {                   // normalization & cross-vector
         pTFn = &(this->chList[pW->channel[n]]);             

         // normalize target & witness channels

         pTFn->getLayer(qq,pW->layer[n]);
         pTFn->getLayer(ww,pW->layer[n]);
         pTFn->getLayer(WW,-pW->layer[n]);

         for(j=0; j<(int)ww.size(); j++) {
            qq.data[j] = ww.data[j]*ww.data[j]+WW.data[j]*WW.data[j];
         }
         rms = sqrt(qq.mean(fm));
         pW->norm[n] = rms; 
//         std::cout<<i<<" "<<rms<<std::endl;
         if(!n) continue;                    // skip target

         // calculate target-witness cross-vector

         s00n = pTFn->pWavelet->getSlice(pW->layer[n]);
         s90n = pTFn->pWavelet->getSlice(-pW->layer[n]);
         k00n = s00n.start(); 
         k90n = s90n.start(); 

         for(k=-K; k<=K; k++) {

            if(k<0) {                        // n-channel is shifted
               k00n = s00n.start(); 
               k90n = s90n.start(); 
               k00m = s00m.start()-k*sTEp; 
               k90m = s90m.start()-k*sTEp; 
            }
            else {                           // m-channel is shifted
               k00n = s00n.start()+k*sTEp; 
               k90n = s90n.start()+k*sTEp; 
               k00m = s00m.start(); 
               k90m = s90m.start(); 
            }
            for(j=K; j<sIZe-K; j++) {
               jj = j*sTEp;
               ww.data[j] = pTFn->data[k00n+jj]*pTFm->data[k00m+jj];
               ww.data[j]+= pTFn->data[k90n+jj]*pTFm->data[k90m+jj];
               WW.data[j] = pTFm->data[k90m+jj]*pTFn->data[k00n+jj];
               WW.data[j]-= pTFm->data[k00m+jj]*pTFn->data[k90n+jj];
            }            
            
            kk = (n-1)*K4+K+k;
            VC.data[kk]      = ww.mean(fm)/pW->norm[0]/pW->norm[n];          // Chx+Ch'x'
            VC.data[kk+K4/2] = WW.mean(fm)/pW->norm[0]/pW->norm[n];          // Ch'x-Chx'
            if(k==0) {VC.data[kk] *= FLTR; VC.data[kk+K4/2] *= FLTR;}        // handle LPE filter
//            printf("%f ",VC.data[kk+0*K4/2]);
         }
//         std::cout<<std::endl;
      }      
      ww=0.; WW=0.;

      for(n=1; n<N; n++) {                   // loop on witness channels
         nn = (n-1)*K4+K;                    // index of ZERO element of ii block 
         pTFn = &(this->chList[pW->channel[n]]);          // witness channel i
         s00n = pTFn->pWavelet->getSlice(pW->layer[n]);
         s90n = pTFn->pWavelet->getSlice(-pW->layer[n]);
         k00n = s00n.start(); 
         k90n = s90n.start(); 
         sIZe = s00n.size();
         sTEp = s00n.stride();

         for(m=n; m<N; m++) {
            mm = (m-1)*K4+K;                 // index of ZERO element of jj block 
            pTFm = &(this->chList[pW->channel[m]]);       // witness channel j
            s00m = pTFm->pWavelet->getSlice(pW->layer[m]);
            s90m = pTFm->pWavelet->getSlice(-pW->layer[m]);
            k00m = s00m.start(); 
            k90m = s90m.start(); 

//            printf("n/m=%2d/%2d %d %d %d %d\n",n,m,sIZe,sTEp,(int)s00n.start(),(int)s90n.start());

            for(k=-K2; k<=K2; k++) {

               if(k<0) {
                  k00n = s00n.start(); 
                  k90n = s90n.start(); 
                  k00m = s00m.start()-k*sTEp; 
                  k90m = s90m.start()-k*sTEp; 
               }
               else {
                  k00n = s00n.start()+k*sTEp; 
                  k90n = s90n.start()+k*sTEp; 
                  k00m = s00m.start(); 
                  k90m = s90m.start(); 
               }

               for(j=K2; j<sIZe-K2; j++) {
                  jj = j*sTEp;
                  ww.data[j] = pTFn->data[k00n+jj]*pTFm->data[k00m+jj];
                  ww.data[j]+= pTFn->data[k90n+jj]*pTFm->data[k90m+jj];
                  WW.data[j] = pTFm->data[k00m+jj]*pTFn->data[k90n+jj];
                  WW.data[j]-= pTFm->data[k90m+jj]*pTFn->data[k00n+jj];
               }

               acf[k] = ww.mean(fm)/pW->norm[n]/pW->norm[m];      // "auto" function
               ccf[k] = WW.mean(fm)/pW->norm[n]/pW->norm[m];      // "cros" function           

//               printf("%f ",acf[k]);
            }
//            std::cout<<"\n";

            for(ii=-K; ii<=K; ii++) {                        // fill in matrix
               for(jj=-K; jj<=K; jj++) {
                  MS[nn+ii][mm+jj]           = (ii==0 || jj==0) ?  acf[ii-jj]*FLTR : acf[ii-jj];
                  MS[mm+jj][nn+ii]           = (ii==0 || jj==0) ?  acf[ii-jj]*FLTR : acf[ii-jj];
                  MS[nn+ii][mm+jj+K4/2]      = (ii==0 || jj==0) ?  ccf[ii-jj]*FLTR : ccf[ii-jj];
                  MS[mm+jj+K4/2][nn+ii]      = (ii==0 || jj==0) ?  ccf[ii-jj]*FLTR : ccf[ii-jj];
                  MS[nn+ii+K4/2][mm+jj]      = (ii==0 || jj==0) ? -ccf[ii-jj]*FLTR :-ccf[ii-jj];
                  MS[mm+jj][nn+ii+K4/2]      = (ii==0 || jj==0) ? -ccf[ii-jj]*FLTR :-ccf[ii-jj];
                  MS[nn+ii+K4/2][mm+jj+K4/2] = (ii==0 || jj==0) ?  acf[ii-jj]*FLTR : acf[ii-jj];
                  MS[mm+jj+K4/2][nn+ii+K4/2] = (ii==0 || jj==0) ?  acf[ii-jj]*FLTR : acf[ii-jj];
               }
            }
         }
      }
    //  for(int i=0;i<VC.size();i++)
    //    std::cout<<VC[i]<<std::endl;
    //  MS.Print();  
      this->matrix.push_back(MS);
      this->vCROSS.push_back(VC);
   }
}

void regression::solve(double th, int nE, char c) {
//
// solve for eigenvalues and calculate Wiener filters
//
// th: eigenvalue threshold. If <0, than in units of max eigenvalue
// nE: number of selected eigenvalues (minimum is 4)
// c:  regulator h=mild, s=soft, h=hard
//

   int i, j, k;
   int nA = this->FILTER.size();
   int nR = this->matrix.size();
   int nC = this->vCROSS.size();
   int ne;

   if(!nA) {std::cout<<"regression::nothing to clean.\n"; return;}
   if (nA!=nR || nA!=nC || nA==0) {
      std::cout << "Error, filter not initialized\n"; exit(1);
   }

   int K = (int) this->kSIZE;
   int K4=2*(2*K+1);
   double temp;

   this->vEIGEN.clear(); std::vector< wavearray<double> >().swap(vEIGEN);
   
   for (i=0; i<nA; i++) {                        //loop on wavelet layers
       int J = FILTER[i].layer.size()-1;         //number of witness channels
       ne = nE<=0 ? K4*J : nE-1;
       if(ne>=K4*J) ne = K4*J-1;
       if(ne<1) ne = 1;

       wavearray<double> lambda(K4*J);           //inverse of regulators
       TMatrixDSym R(this->matrix[i]);
       wavearray<double>* pC = &(this->vCROSS[i]);
       TMatrixDSymEigen QP(R);
       TVectorD eigenval(QP.GetEigenValues());
       TMatrixD eigenvec(QP.GetEigenVectors());
 
       //select threshold
       double last = 0.;
       double TH = th<0. ? -th*eigenval[0] : th+1.e-12;
       int nlast = -1;
       for(j=0; j<K4*J; j++) {
          lambda.data[j] = eigenval[j];
          if(eigenval[j]>=TH) nlast++;
	  if(j<K4*J-1)
	     if(eigenval[j+1]>eigenval[j]) 
		std::cout<<eigenval[j]<<" "<<eigenval[j+1]<<endl;
       }
       if(nlast<1) nlast = 1;

       this->vEIGEN.push_back(lambda);
       if(nlast>ne) nlast = ne;
       lambda = 0.;

       //std::cout<<i<<" "<<J<<" "<<TH<<" "<<nlast<<" "<<K<<" "<<eigenval[nlast]<<std::endl;

       switch(c) {   //regulators
          case 'h':
            last = 0.; 
            break;
          case 's':
            last = eigenval[nlast]>0. ? 1./eigenval[nlast] : 0.; 
            break;
          case 'm':
            last = 1./eigenval[0]; 
            break;
        }
       for(j=0;j<=nlast;j++) lambda[j] = eigenval[j]>0. ? 1./eigenval[j] : 0.;
       for(j=nlast+1;j<K4*J;j++) lambda[j] = last;

       //calculte filters
       wavearray<double> vv(K4*J);           // lambda * eigenvector * cross vector
       wavearray<double> aa(K4*J);           //  eigenvector^T * lambda * eigenvector * cross vector
       
       for(j=0;j<K4*J;j++) {
           temp=0.0;
           for(k=0;k<K4*J;k++) temp += eigenvec[k][j]*pC->data[k];
           vv.data[j]=temp*lambda.data[j];   
        }

       for(j=0;j<K4*J;j++) {
           temp=0.0;
           for(k=0;k<K4*J;k++) temp += eigenvec[j][k]*vv.data[k];
           aa.data[j]=temp;   
        }
      
       //save filters coefficients 
       for (j=0; j<J; j++) {
          for(k=0;k<=2*K;k++) {
             FILTER[i].filter00[j+1][k] = aa.data[j*K4+k];
             FILTER[i].filter90[j+1][k] = aa[j*K4+k+K4/2];
          }
       }
   }    
}


void regression::apply(double threshold, char c) { 
//
// apply filter to target channel and produce noise TS
//
// threshold: filter is not applied if regressed noise rms < threshold
//            mask channels with rms<threshold, if c='m' or c='M'.
//            mask changes configuration of witness channels 
//  c: 'n' or 'N' - either 0-phase or 90-phase noise
//     'a' - (0-phase + 90-phase)/2 noise, standard RMS threshold  
//     'A' - (0-phase + 90-phase)/2 noise, differential RMS threshold  
//     'm' - (0-phase + 90-phase)/2 noise, mask, standard RMS threshold  
//     'M' - (0-phase + 90-phase)/2 noise, mask, differential RMS threshold  
//

   int n,m;
   int nA = this->FILTER.size();
   int nR = this->matrix.size();
   int nC = this->vCROSS.size();

   if(!nA) {std::cout<<"regression::nothing to clean.\n"; return;}
   if (nA!=nR || nA!=nC || nA==0) {
      std::cout << "Error, filter is not initialized\n"; exit(1);
   }

   int K = (int) this->kSIZE;
   int N = this->FILTER.size();               // number of target layers

   wavearray<double> tmp(N); tmp=0.;                   //RANK
   int L = this->chList.size();                        //RANK
   for(n=0; n<L; n++) this->vrank.push_back(tmp);      //RANK

   wavearray<double> ww;
   wavearray<double> WW;
   wavearray<double> qq;
   wavearray<double> QQ;
   WSeries<double>* pT;
   Wiener* pF;

   double sum, SUM;
   WSeries<double> wnoise(this->chList[0]);
   WSeries<double> WNOISE(this->chList[0]);
   wnoise=0; WNOISE=0.;

   for(n=0; n<N; n++) {                        // loop on target layers
      std::vector< wavearray<double> > wno;
      std::vector< wavearray<double> > WNO;
      _apply_(n,wno,WNO);

      pF = &FILTER[n];                         // pointer to Weiner structure                    
      pT = &(this->chList[0]);                 // pointer to target WS
      pT->getLayer(ww,pF->layer[0]);           // get target 00 data
      pT->getLayer(WW,-pF->layer[0]);          // get target 90 data
      ww = 0.; WW = 0.;                        // zero total noise arrays 

      double freq = pT->frequency(pF->layer[0]);   //RANK
      vfreq.append(freq);                          //RANK

      int KK = int(ww.rate()*this->Edge);      // calculate # of edge samples
      if(KK<K) KK = K; KK++;
      std::slice S(KK,ww.size()-2*KK,1);       // slice for RMS calculation
      
      double trms = wno[0].rms(S);
      double TRMS = WNO[0].rms(S);
      int M = pF->layer.size();                // number of witness channels + 1 

      for(m=1; m<M; m++) {                     // loop over witnwss channels
         int j = (int)pF->channel[m];          // witness channel index
         int* pM = this->chMask[j].data;       // poiner to j's mask

         if(c=='A' || c=='M') {
            qq=wno[0]; qq-=wno[m];             // channel j subtracted
            QQ=WNO[0]; QQ-=WNO[m];             // channel j subtracted                
            sum = qq.rms(S);
            SUM = QQ.rms(S);
            sum = trms*trms-sum*sum;
            SUM = TRMS*TRMS-SUM*SUM;
         }
         else {
            sum = pow(wno[m].rms(S),2);        // rms 00
            SUM = pow(WNO[m].rms(S),2);        // rms 90
         }
         if(sum<0.) sum = 0.;
         if(SUM<0.) SUM = 0.;

         this->vrank[j].data[n]  = sum+SUM;       //RANK
         this->vrank[0].data[n] += sum+SUM;       //RANK

         if(sum+SUM < threshold*threshold) { 
            if(c=='m' || c=='M') pM[pF->layer[m]] = 0; 
            continue; 
         }
         ww += wno[m]; WW += WNO[m];
         //std::cout << "RMS: M: " << m << " " << wno[m].rms(S) << " " << WNO[m].rms(S) << std::endl;
      }
      if(c=='A' || c=='M') this->vrank[0].data[n] = trms*trms+TRMS*TRMS;   //RANK
      //std::cout << "RMS: N: " << n << " " << ww.rms() << " " << WW.rms() << std::endl;
      ww *= pF->norm[0];
      WW *= pF->norm[0];
      //std::cout << "RMS: N: " << n << " " << ww.rms() << " " << WW.rms() << " " << pF->norm[0] << std::endl;
      wnoise.putLayer(ww,pF->layer[0]);
      wnoise.putLayer(WW,-pF->layer[0]);
   }
   WNOISE = wnoise;
   this->WNoise = wnoise;
   wnoise.Inverse();
   WNOISE.Inverse(-2);

        if(c=='n') this->rnoise = wnoise;
   else if(c=='N') this->rnoise = WNOISE;
   else{
      this->rnoise  = wnoise; 
      this->rnoise += WNOISE; 
      this->rnoise *= 0.5;
   }
}

wavearray<double> regression::rank(int nbins, double fL, double fH) {
//
// get RMS for predicted noise (per layer/bin) for all witness channels
// (including masked)
//
// nbins: number of loudest frequency layers to calculate rms/layer. 
//        nbins=0 - take all layers
//        if positive, return standard RMS
//        if negative, return differential RMS
// fL, fH: frequency range: low, high
//
// return wavearray: rank values
//

   int n,m;
   int nA = this->FILTER.size();
   int nR = this->matrix.size();
   int nC = this->vCROSS.size();

   if(!nA) {
      std::cout<<"regression::nothing to clean.\n"; 
      return wavearray<double>();
   }
   if (nA!=nR || nA!=nC) {
      std::cout << "Error, filter not initialized\n"; exit(1);
   }

   int N = this->FILTER.size();               // number of target layers
   int L = this->chList.size();

   wavearray<double> tmp(N); tmp=0.;
   wavearray<double> rms(L); rms=0.;
   std::vector< wavearray<double> > vrms;
   for(n=0; n<L; n++) {                        // loop on channels
      vrms.push_back(tmp);
      int tsize=this->vfreq.size();
      for (int i=0; i<tsize; i++) {
         double freq = vfreq.data[i];
         if(fL<fH && (freq>fH || freq<fL)) continue;
         vrms[n].data[i]=vrank[n].data[i];
      }
   }

   for(n=0; n<L; n++) {                        // loop on channels
      vrms[n].waveSort();
      nA = 0;
      for(m=N-1; m>=N-abs(nbins); m--) {            // loop over loud layers
         rms.data[n] += vrms[n].data[m];
         if(vrms[n].data[m]>0.) nA++;
      }
      if(nA) rms.data[n] /= nA;
      rms.data[n] = sqrt(rms.data[n]);      
   }
   return rms;
}

void regression::_apply_(int n, 
                           std::vector< wavearray<double> > &wno,
                           std::vector< wavearray<double> > &WNO)
{
//
// internal regression apply function 
//

   int m,k,i;
   wavearray<double> qq;
   wavearray<double> QQ;
   wavearray<double> ww;
   wavearray<double> WW;
   WSeries<double>* pT;
   WSeries<double>* pW;
   Wiener* pF;
   std::vector<double> *ff, *FF;
   double val, VAL;

   pF = &FILTER[n];                         // pointer to Weiner structure                    
   pT = &(this->chList[0]);                 // pointer to target WS
   pT->getLayer(ww,pF->layer[0]);           // get target 00 data
   pT->getLayer(WW,-pF->layer[0]);          // get target 90 data

   ww = 0.; WW = 0.;                        // zero total noise arrays 
   wno.clear();
   WNO.clear();
   wno.push_back(ww);
   WNO.push_back(WW);

   int K = (int) this->kSIZE;
   int M = pF->layer.size();                // number of witness channels + 1 
   for(m=1; m<M; m++) {                     // loop over witnwss channels
      int j = (int)pF->channel[m];          // witness channel index
      pW = &(this->chList[j]);              // pointer to witness WS            
      pW->getLayer(qq,pF->layer[m]);        // get witness 00 data
      pW->getLayer(QQ,-pF->layer[m]);       // get witness 90 data
      ff = &(pF->filter00[m]);              // get 00 filter pointer 
      FF = &(pF->filter90[m]);              // get 90 filter pointer 
      wavearray<double> nn(qq.size());      // array for witness prediction data 
      wavearray<double> NN(qq.size());      // array for witness prediction data 
      nn.rate(qq.rate());
      NN.rate(QQ.rate());
      qq *= 1./pF->norm[m];
      QQ *= 1./pF->norm[m];
      
      for(i=0; i<(int)qq.size(); i++) {
         val = VAL = 0.;
         if(i>=K && i<int(qq.size())-K) { 
            for(k=-K; k<=K; k++) {
               val += (*ff)[k+K]*qq.data[i+k] - (*FF)[k+K]*QQ.data[i+k];
               VAL += (*FF)[k+K]*qq.data[i+k] + (*ff)[k+K]*QQ.data[i+k];
            }
         }
         nn.data[i]  = val;
         NN.data[i]  = VAL;
         wno[0].data[i] += val;
         WNO[0].data[i] += VAL;
      }
      wno.push_back(nn);
      WNO.push_back(NN);
   }
}





