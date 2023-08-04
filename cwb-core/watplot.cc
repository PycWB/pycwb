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


#define WATPLOT_CC
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "watplot.hh"
#include "WDM.hh"
#include "TSystem.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TPaletteAxis.h"

ClassImp(watplot)       // used by THtml doc


//______________________________________________________________________________
/* Begin_Html
<center><h2>watlpot class</h2></center>
The class watlpot is used to plot different objects types (wavearray, WSeries, skymap, netcluster, ...)

<p>
<h3><a name="example">Example</a></h3>
<p>
The macro <a href="./tutorials/wat/testWDM_3.C.html">testWDM_3.C</a> is an example which shown how to plot wavearray object.<br>
<p>

End_Html

Begin_Macro
testWDM_3.C
End_Macro 

Begin_Html
The macro <a href="./tutorials/wat/testWDM_1.C.html">testWDM_1.C</a> is an example which shown how to plot WSeries object.<br>
<p>

End_Html

Begin_Macro
testWDM_1.C
End_Macro 

Begin_Html
The macro <a href="./tutorials/wat/DrawSkymapWithSkyplot.C.html">DrawSkymapWithSkyplot.C</a> is an example which shown how to plot skymap object.<br>
<p>

End_Html

Begin_Macro
DrawSkymapWithSkyplot.C
End_Macro 

Begin_Html
The macro <a href="./tutorials/wat/DrawClusterWithSkyplot.C.html">DrawClusterWithSkyplot.C</a> is an example which shown how to plot netcluster object.<br>
<p>

End_Html

Begin_Macro
DrawClusterWithSkyplot.C
End_Macro */


// ++++++++++++++++++++++++++++++++++++++++++++++
// S. Klimenko, University of Florida
// WAT plot class
// ++++++++++++++++++++++++++++++++++++++++++++++

watplot::watplot(char* name, int i1, int i2, int i3, int i4) { 
// 
//  Default constructor
//
//  Input parameters are used to create a new canvas.
//
//  i1,i2 are the pixel coordinates of the top left corner of
//  the canvas (if i1 < 0) the menubar is not shown)
//  i3 is the canvas size in pixels along X
//  i4 is the canvas size in pixels along Y
// 
//  default values are : name=NULL, i1=200, i2=20, i3=800, i4=600
//

  this->title  = "";
  this->xtitle = "";
  this->ytitle = "";
  this->ncol   = 0;
  this->opt    = "";
  this->col    = 1;
  this->t1     = 0.;
  this->t2     = 0.;
  this->fft    = false;
  this->f1     = 0.;
  this->f2     = 0.;
  this->hist2D = NULL;
  
  char defn[16] = "watplot";
  if (name && gROOT->FindObject(name)!=NULL) {  
    printf("watplot : Error Canvas Name %s already exist",name);
    exit(1);
  }

  null();
  if(!name) name = defn;
  canvas= new TCanvas(name, name, i1, i2, i3, i4);
  canvas->Clear();
  canvas->ToggleEventStatus();
  canvas->SetGridx();
  canvas->SetGridy();
  canvas->SetFillColor(kWhite);
  canvas->SetRightMargin(0.10);
  canvas->SetLeftMargin(0.10);
  canvas->SetBottomMargin(0.13);
  canvas->SetBorderMode(0);

  // remove the red box around canvas
  gStyle->SetFrameBorderMode(0);
  gROOT->ForceStyle();

  gStyle->SetTitleH(0.050);
  gStyle->SetTitleW(0.95);
  gStyle->SetTitleY(0.98);
  gStyle->SetTitleFont(12,"D");
  gStyle->SetTitleColor(kBlue,"D");
  gStyle->SetTextFont(12);
  gStyle->SetTitleFillColor(kWhite);
  gStyle->SetLineColor(kWhite);
  gStyle->SetNumberContours(256);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetStatBorderSize(1);

}

void watplot::plot(wavearray<double> &td, char* opt, int col, 
                   double t1, double t2, bool fft, float f1, float f2, bool psd, float t3, bool oneside) {
//
// Plot wavearray<double> time series 
//
// td    : wavearray time series
// opt   : TGraph::Draw options
// col   : set TGraph line color       
// t1    : start of time interval in seconds
// t2    : end of time interval in seconds
// fft   : true -> plot fft
// f1    : set begin frequency (Hz)
// f2    : set end frequency (Hz)
// psd   : true -> plot psd using blackmanharris window
// t3    : is the chunk length (sec) used to produce the psd 
// oneside : true/false -> oneside/doubleside
//

  if ( t2 < t1) {
    printf(" Plot error: t2 must be greater then t1.\n");
    return;
  }
  if ( f2 < f1) {
    printf(" Plot error: f2 must be greater then f1.\n");
    return;
  }

  if(t2==0.) { t1 = td.start(); t2 = t1+td.size()/td.rate(); }

  double Ts = (td.rate() == 0)? 1.: 1./td.rate();
  int    i1 = int((t1-td.start())/Ts); 
  int    i2 = (t2 == 0.)? td.size() : int((t2-td.start())/Ts);
  //int  nmax = i2-i1;
  int  nmax = (i2-i1)-(i2-i1)%2;  // nmax even 
  wavearray<double> _x(nmax);
  wavearray<double> _y(nmax);
  double xmin, xmax;
  double ymin, ymax;
  xmin=0.;
  xmax=(nmax-1)*Ts;
  ymin=1.e40;  
  ymax=0.;
  double x0 = i1*Ts+td.start(); 
  TGraph* G;

  for (int i=0; i < nmax; i++) {
    _x.data[i] = x0 + Ts*i - 0.;
    _y.data[i] = td.data[i+i1];
    if (ymin > _y.data[i]) ymin=_y.data[i];
    if (ymax < _y.data[i]) ymax=_y.data[i];
  }

  if(fft) {  
    if(f2==0.) { f1 = 0.; f2 = td.rate()/2.; }
    if(psd) {  // power spectrum density
      t3 = t3<0. ? 0. : t3;
      t3 = t3<(_y.size()/td.rate()) ? t3 : (_y.size()/td.rate());
      int bsize = t3*td.rate();
      bsize-=bsize%2;
      int loops = int(_y.size()/bsize)==0 ? 1 : int(_y.size()/bsize);
      double Fs=(double)td.rate()/(double)bsize;
      wavearray<double> w(bsize);
      wavearray<double> u(bsize);
      wavearray<double> psd(bsize); psd=0;
      blackmanharris(w);
      for (int n=0;n<loops;n++) {
        int shift=n*bsize;
        //cout << "shift: " << shift << endl;
        for (int i=0;i<bsize;i++) u[i]=_y[i+shift];
        for (int i=0;i<bsize;i++) u[i]*=w[i];
        u.FFTW(1);
        for (int i=0;i<bsize;i+=2) psd[i/2]+=pow(u[i],2)+pow(u[i+1],2);
      }
      for (int i=0;i<bsize/2; i++) psd[i]=sqrt(psd[i]/(double)loops)*sqrt(1./Fs);  // double side spectra 1/sqrt(Hz);
      if(oneside) psd*=sqrt(2.);  						   // one side spectra *sqrt(2);
      _y=psd;nmax=bsize;
      _x.resize(nmax);
      for (int i=0;i<nmax/2;i++) _x.data[i]=i*Fs;
    } else {
      wavearray<double> _z(nmax);
      double Fs=(double)td.rate()/(double)_y.size();
      _y.FFTW(1);
      for (int i=0;i<nmax;i+=2)  _z.data[i/2]=sqrt(pow(_y.data[i],2)+pow(_y.data[i+1],2));
      for (int i=0;i<nmax/2;i++) _y.data[i]=_z.data[i]*(1./Fs);     // double side spectra 1/Hz
      for (int i=0;i<nmax/2;i++) _x.data[i]=i*Fs;
      if(oneside) for(int i=0;i<nmax/2;i++) _y.data[i]*=sqrt(2.);  // one side spectra 1/Hz
    }
    nmax/=2;
    ymin=1.e40;
    ymax=0.;
    for (int i=0;i<nmax;i++) {
      if ((_x.data[i]>f1)&&(_x.data[i]<f2)) {
        if (ymin > _y.data[i]) ymin=_y.data[i];
        if (ymax < _y.data[i]) ymax=_y.data[i];
      }
    }
  }

  G = new TGraph(nmax,_x.data,_y.data);
  G->SetLineColor(col);
  G->SetMarkerColor(col);  
//  G->Draw(opt);  
//  canvas->Update();  
  G->GetHistogram()->SetTitle("");  
  G->SetFillColor(kWhite);

  //G->GetXaxis()->SetNdivisions(70318);
  G->GetXaxis()->SetTitleFont(42);
  G->GetXaxis()->SetLabelFont(42);
  G->GetXaxis()->SetLabelOffset(0.012);
  G->GetXaxis()->SetTitleOffset(1.5);

  //G->GetYaxis()->SetNdivisions(508);
  G->GetYaxis()->SetTitleFont(42);
  G->GetYaxis()->SetLabelFont(42);
  G->GetYaxis()->SetLabelOffset(0.01);
  G->GetYaxis()->SetTitleOffset(1.4);
//
  if(fft) {  
    G->GetHistogram()->GetXaxis()->SetRangeUser(f1,f2);
    //G->GetHistogram()->GetYaxis()->SetRangeUser(ymin/2.,ymax*10.);  
    G->GetHistogram()->GetYaxis()->SetRangeUser(ymin/2.,ymax*1.1);  
    G->GetHistogram()->SetXTitle("freq, hz");
  } else {
    G->GetHistogram()->SetXTitle("time, s");
  }
  G->GetHistogram()->SetYTitle("magnitude");
  G->Draw(opt);  
  graph.push_back(G);
//  graph->Fit("fit","VMR");

  return;
}

void watplot::plsmooth(WSeries<double> &w, int sfact, double t1, double t2, char* opt, int pal, int palId)  
{

  t1 = t1==0. ? w.start() : t1;
  t2 = t2==0. ? w.start()+w.size()/w.rate() : t2;

  int ni = 1<<w.pWavelet->m_Level;
  int nb = int((t1-w.start())*w.rate()/ni);
  int nj = int((t2-t1)*w.rate())/ni;
  int ne = nb+nj;
  double rate=w.rate();
  double freq=rate/2/ni;
  rate = rate/ni;

  double** iw = new double*[nj]; 
  for(int j=0;j<nj;j++) iw[j] = new double[ni]; 

//  double wmax=0.;
  wavearray<double> wl;
  for(int i=0;i<ni;i++) { 
    w.getLayer(wl,i);	
//    for(int j=nb;j<ne;j++) iw[j-nb][i] = fabs(wl.data[j]);

    for(int j=nb;j<ne;j++) {
//printf("DEB14 %d %d,%d,%d %d\n",j,j-nb,i,nj,wl.size());
      iw[j-nb][i] = wl.data[j]*wl.data[j];
//      if(wmax<iw[j-nb][i]) wmax=iw[j-nb][i];
    }

  }

  int smfact=1; 
  for(int n=0;n<sfact;n++) smfact*=3; 
  double** sw = NULL;
  int sni=ni;
  int snj=nj;

  if(sfact==0) {
    sw = new double*[snj]; 
    for(int j=0;j<snj;j++) sw[j] = new double[sni]; 
    for(int i=0;i<sni;i++) { 
      for(int j=0;j<snj;j++) {
        sw[j][i] = iw[j][i];
      }
    }
  }

  for(int n=1;n<=sfact;n++) { 
    if(sw!=NULL) {
      for(int j=0;j<snj;j++) delete [] sw[j]; 
      delete [] sw;
    }
    sni*=3;
    snj*=3;
    sw = new double*[snj]; 
    for(int j=0;j<snj;j++) sw[j] = new double[sni]; 
    for(int i=0;i<sni;i++) for(int j=0;j<snj;j++) sw[j][i] = 0.;
    for(int i=1;i<sni-1;i++) { 
      for(int j=1;j<snj-1;j++) {
        for(int j3=j-1;j3<=j+1;j3++) {
          for(int i3=i-1;i3<=i+1;i3++) {
//printf("%d %d %d %d\n",i,j,i3/3,j3/3);
            sw[j][i] += iw[(j3-j3%3)/3][(i3-i3%3)/3];
          }
        }
        sw[j][i]/=9.;
      }
    }
    for(int j=0;j<nj;j++) delete [] iw[j]; 
    delete [] iw;
    iw = new double*[snj]; 
    for(int j=0;j<snj;j++) iw[j] = new double[sni]; 
    for(int i=0;i<sni;i++) { 
      for(int j=0;j<snj;j++) {
        iw[j][i] = sw[j][i];
      }
    }
  }

  for(int j=0;j<snj;j++) delete [] iw[j]; 
  delete [] iw;

  if(hist2D) { delete hist2D; hist2D=NULL; }
  //hist2D=new TH2F("WTF","", snj, t1-w.start(), t2-w.start(), sni, 0., freq*ni);
  hist2D=new TH2F("WTF","", snj, 0., t2-t1, sni, 0., freq*ni);
  hist2D->SetXTitle("time, sec");
  hist2D->SetYTitle("frequency, Hz");

  Int_t colors[30]={101,12,114,13,115,14,117,15,16,17,166,18,19,
                   167,0,0,167,19,18,166,17,16,15,117,14,115,13,114,12,101};
  if(pal==0) gStyle->SetPalette(30,colors);
  else if(pal<0) {SetPlotStyle(palId,pal);}  
  else {gStyle->SetPalette(1,0);gStyle->SetNumberContours(pal);}
/*
  double swmax=0.;
  for(int i=0;i<sni;i++) { 
    for(int j=0;j<snj;j++) {
      if(swmax<sw[j][i]) swmax=sw[j][i];
    }
  }
*/

  int snb=smfact*nb;
  int sne=smfact*ne;
  double srate=smfact*rate;
  double sfreq=freq/smfact;
  for(int i=0;i<sni;i++) { 
    for(int j=snb;j<sne;j++) {
      double x = sw[j-snb][i];
      //double x = sw[j-snb][i]*wmax/swmax;
      //double x = sqrt(sw[j-snb][i]);
      //double x = sqrt(sw[j-snb][i]*wmax/swmax);
      //hist2D->Fill(double(j)/srate,(i+0.5)*sfreq,x);
      hist2D->Fill(double(j-snb)/srate,(i+0.5)*sfreq,x);
    }
  }

  for(int j=0;j<snj;j++) delete [] sw[j]; 
  delete [] sw;

  hist2D->SetStats(kFALSE);
  hist2D->SetFillColor(kWhite);
  hist2D->GetXaxis()->SetTitleFont(42);
  hist2D->GetXaxis()->SetLabelFont(42);
  hist2D->GetXaxis()->SetLabelOffset(0.012);
  hist2D->GetXaxis()->SetTitleOffset(1.1);
  hist2D->GetYaxis()->SetTitleFont(42);
  hist2D->GetYaxis()->SetLabelFont(42);
  hist2D->GetYaxis()->SetLabelOffset(0.01);
  hist2D->GetZaxis()->SetLabelFont(42);
  hist2D->SetTitleOffset(1.3,"Y");
  if(opt) hist2D->Draw(opt);
  else    hist2D->Draw("COLZ");

  // change palette's width
  canvas->Update();
  TPaletteAxis *palette = (TPaletteAxis*)hist2D->GetListOfFunctions()->FindObject("palette");
  palette->SetX1NDC(0.91);
  palette->SetX2NDC(0.933);
  palette->SetTitleOffset(0.92);
  palette->GetAxis()->SetTickSize(0.01);
  canvas->Modified();

  return;
}


double watplot::getmax(WSeries<double> &w, double t1, double t2)  
{
//
// return max value of WSeries w object
//
// t1     : start of time interval in seconds
// t2     : end of time interval in seconds
//

  float x;

  t1 = t1==0. ? w.start() : t1;
  t2 = t2==0. ? w.start()+w.size()/w.rate() : t2;

  int ni = 1<<w.pWavelet->m_Level;
  int nb = int((t1-w.start())*w.rate()/ni);
  int nj = int((t2-t1)*w.rate())/ni;
  int ne = nb+nj;
  double rate=w.rate();
  rate = rate/ni;

  wavearray<double> wl;
  double xmax;

  for(int i=0;i<ni;i++) { 
    w.getLayer(wl,i);	
    for(int j=nb;j<ne;j++) {
      x=fabs(wl.data[j]);
      if(x>xmax) xmax=x;
    }
  }
  return xmax;
}

void watplot::plot(WSeries<double> &w, int mode, double t1, double t2, char* opt, int pal, int palId)  
{
//
// Plot TF series
//
// w      : WSeries<double> object
// mode   : plot type 
//          if WSeries is WDM
//            0 : sqrt((E00+E90)/2)
//            1 : (E00+E90)/2
//            2 : sqrt((E00+E90)/2)
//            3 : amplitude:00
//            4 : energy:00
//            5 : |amplitude:00|
//            6 : amplitude:90
//            7 : energy:90
//            8 : |amplitude:90|
//          if WSeries is wavelet
//            0 : amplitude
//            1 : energy
//            2 : |amplitude|
//
// t1     : start of time interval in seconds
// t2     : end of time interval in seconds
// opt    : root draw options 
// pal    : palette 
//

  t1 = t1==0. ? w.start() : t1;
  t2 = t2==0. ? w.start()+w.size()/w.rate() : t2;

  if(t2 <= t1) {
    printf("watplot::plot error: t2 must be greater then t1.\n");
    exit(1);
  }
  if((t1 < w.start()) || (t1 > w.start()+w.size()/w.rate())) {
    printf("watplot::plot error: t1 must be in this range [%0.12g,%0.12g]\n",
           w.start(),w.start()+w.size()/w.rate());
    exit(1);
  }
  if((t2 < w.start()) || (t2 > w.start()+w.size()/w.rate())) {
    printf("watplot::plot error: t2 must be in this range [%0.12g,%0.12g]\n",
           w.start(),w.start()+w.size()/w.rate());
    exit(1);
  }

  Int_t colors[30]={101,12,114,13,115,14,117,15,16,17,166,18,19,
                   167,0,0,167,19,18,166,17,16,15,117,14,115,13,114,12,101};
  if(pal==0) gStyle->SetPalette(30,colors);
  else if(pal<0) {SetPlotStyle(palId,pal);}  
  else {gStyle->SetPalette(1,0);gStyle->SetNumberContours(pal);}

  float x;
  TString ztitle("");

  if(w.isWDM()) {

    double A00=1; 
    double A90=1; 

    if(mode==0) mode=2;
    if(mode==1) {ztitle="(E00+E90)/2";}
    if(mode==2) {ztitle="sqrt((E00+E90)/2)";}

    if(mode==3) {ztitle="amplitude:00";A90=0;}
    if(mode==4) {ztitle="energy:00";A00=sqrt(2.);A90=0;}
    if(mode==5) {ztitle="|amplitude:00|";A90=0;}

    if(mode==6) {ztitle="amplitude:90";A00=0;}
    if(mode==7) {ztitle="energy:90";A00=0;A90=sqrt(2.);}
    if(mode==8) {ztitle="|amplitude:90|";A00=0;}

    mode*=-1;

    WDM<double>* wdm = (WDM<double>*) w.pWavelet;
    int M = w.getLevel();
    double* map00 = wdm->pWWS;
    double tsRate = w.rate();
    int mF = int(w.size()/wdm->nSTS);
    int nTC = w.size()/(M+1)/mF;                   // # of Time Coefficients
    double* map90 = map00 + (mF-1)*(M+1)*nTC;

    //printf("nTC = %d, rate = %d  M = %d\n", nTC, (int)w.rate(), M);

    // make Y bins:
    double* yBins = new double[M+2];
    double dF = tsRate/M/2.;
    yBins[0] = 0;
    yBins[1] = dF/2;
    for(int i=2; i<=M; ++i) yBins[i] = yBins[1] + (i-1)*dF;
    yBins[M+1] = tsRate/2.;

    const double scale = 1./w.wrate();
    if(hist2D) { delete hist2D; hist2D=NULL; }
    hist2D=new TH2F("WTF", "", 2*nTC, 0., nTC*scale, M+1, yBins);
    hist2D->SetXTitle("time, sec");
    hist2D->SetYTitle("frequency, Hz");
    hist2D->GetXaxis()->SetRangeUser(t1-w.start(),t2-w.start());

    double v;
    int it1 = (t1-w.start())/scale + 1;
    int it2 = (t2-w.start())/scale + 1;
    if(it2<=it1 || it2>nTC)it2 = nTC;

    map00+=it1*(M+1); map90+=it1*(M+1);
    for(int i=it1; i<it2; ++i){
      if(i){
         v = procOpt(mode%3, A00*map00[0], A90*map90[0]); //first half-band
         hist2D->SetBinContent( 2*i , 1, v);
         hist2D->SetBinContent( 2*i+1 , 1, v);

         v = procOpt(mode%3, A00*map00[M], A90*map90[M]); //last half-band
         hist2D->SetBinContent( 2*i , M+1, v);
         hist2D->SetBinContent( 2*i+1 , M+1, v);
      }
      for(int j=1; j<M; ++j){
         v = procOpt(mode%3, A00*map00[j], A90*map90[j]); 
         hist2D->SetBinContent( 2*i , j+1, v);
         hist2D->SetBinContent( 2*i+1 , j+1, v);
      }
      map00+=M+1; map90+=M+1;
    }

    delete [] yBins;

  } else {

    if(mode==0) ztitle="amplitude";
    if(mode==1) ztitle="energy";
    if(mode==2) ztitle="|amplitude|";

    int ni = 1<<w.pWavelet->m_Level;
    int nb = int((t1-w.start())*w.rate()/ni);
    int nj = int((t2-t1)*w.rate())/ni;
    int ne = nb+nj;
    double rate=w.rate();
    double freq=rate/2/ni;
    rate = rate/ni;

//  cout<<rate<<endl;

    if(hist2D) { delete hist2D; hist2D=NULL; }
    hist2D=new TH2F("WTF","", nj, t1-w.start(), t2-w.start(), ni, 0., freq*ni);
    hist2D->SetXTitle("time, sec");
    hist2D->SetYTitle("frequency, Hz");

    wavearray<double> wl;
    double avr,rms;

    if(mode==0) 
    for(int i=0;i<ni;i++) { 
      w.getLayer(wl,i);	
      for(int j=nb;j<=ne;j++) {  
        x=wl.data[j];
        hist2D->Fill((j+0.5)/rate,(i+0.5)*freq,x);  
      }
    }
  
    if(mode==1)
    for(int i=0;i<ni;i++) { 
      w.getLayer(wl,i);	
      for(int j=nb;j<=ne;j++) {  
        x=wl.data[j]*wl.data[j];
        hist2D->Fill((j+0.5)/rate,(i+0.5)*freq,x);  
      }
    }
  
    if(mode==2)
    for(int i=0;i<ni;i++) {   
      w.getLayer(wl,i);	
      for(int j=nb;j<=ne;j++) {
        x = fabs(wl.data[j]);
        hist2D->Fill((j+0.5)/rate,(i+0.5)*freq,x);  
      }
    }
  
    if(mode==4)
    for(int i=0;i<ni;i++) {   
      w.getLayer(wl,i);	
      for(int j=nb;j<=ne;j++) {  
        x=(wl.data[j] > 0.) ? 1. : -1.;
        hist2D->Fill((j+0.5)/rate,(i+0.5)*freq,x);  
      }
    }
  
    if(mode==5)
    for(int i=0;i<ni;i++) {   
      w.getLayer(wl,i);	
      for(int j=nb;j<=ne;j++) {
        x=(wl.data[j] > 0.) ? 1. : -1.;
        if(wl.data[j] == 0.) x=0.;
        hist2D->Fill((j+0.5)/rate,(i+0.5)*freq,x);  
      }
    }

    double sum = 0.;
    int nsum = 0;

    if(mode==3){
      for(int i=0;i<ni;i++) { 
        w.getLayer(wl,i);
        wl.getStatistics(avr,rms);
        for(int j=nb;j<ne-0;j++) {
  	x=(wl.data[j]-avr)/rms;
  	x*=x;
  	if(x>1){ sum += x; nsum++; }
  	hist2D->Fill(j/rate,(i+0.5)*freq,x);
        }
      }
    }
  } 

  hist2D->SetStats(kFALSE);
  hist2D->SetTitleFont(12);
  hist2D->SetFillColor(kWhite);

  char title[256];
  sprintf(title,"Scalogram (%s)",ztitle.Data());
  hist2D->SetTitle(title);

  hist2D->GetXaxis()->SetNdivisions(506);
  hist2D->GetXaxis()->SetLabelFont(42);
  hist2D->GetXaxis()->SetLabelOffset(0.014);
  hist2D->GetXaxis()->SetTitleOffset(1.4);
  hist2D->GetYaxis()->SetTitleOffset(1.2);
  hist2D->GetYaxis()->SetNdivisions(506);
  hist2D->GetYaxis()->SetLabelFont(42);
  hist2D->GetYaxis()->SetLabelOffset(0.01);
  hist2D->GetZaxis()->SetLabelFont(42);
  hist2D->GetZaxis()->SetNoExponent(false);
  hist2D->GetZaxis()->SetNdivisions(506);

  hist2D->GetXaxis()->SetTitleFont(42);
  hist2D->GetXaxis()->SetTitle("Time (sec)");
  hist2D->GetXaxis()->CenterTitle(true);
  hist2D->GetYaxis()->SetTitleFont(42);
  hist2D->GetYaxis()->SetTitle("Frequency (Hz)");
  hist2D->GetYaxis()->CenterTitle(true);

  hist2D->GetZaxis()->SetTitleOffset(0.6);
  hist2D->GetZaxis()->SetTitleFont(42);
  //hist2D->GetZaxis()->SetTitle(ztitle);
  hist2D->GetZaxis()->CenterTitle(true);

  hist2D->GetXaxis()->SetLabelSize(0.03);
  hist2D->GetYaxis()->SetLabelSize(0.03);
  hist2D->GetZaxis()->SetLabelSize(0.03);


  if(opt) hist2D->Draw(opt);
  else    hist2D->Draw("COLZ");

  // change palette's width
  canvas->Update();
  TPaletteAxis *palette = (TPaletteAxis*)hist2D->GetListOfFunctions()->FindObject("palette");
  palette->SetX1NDC(0.91);
  palette->SetX2NDC(0.933);
  palette->SetTitleOffset(0.92);
  palette->GetAxis()->SetTickSize(0.01);
  canvas->Modified();

  return;
}

void watplot::plot(skymap &sm, char* opt, int pal) {
//
// Draw skymap 
//
// sm     : skymap object
// opt    : root draw options 
// pal    : palette 
//

  int ni = sm.size(0);         // number of theta layers 
  int nj = 0;                  // number of phi collumns

  for(int i=1; i<=ni; i++) if(nj<int(sm.size(i))) nj = sm.size(i);

  double t1 = sm.theta_1;
  double t2 = sm.theta_2;
  double p1 = sm.phi_1;
  double p2 = sm.phi_2;
  double dt = ni>1 ? (t2-t1)/(ni-1) : 0.; 
  double dp = nj>0 ? (p2-p1)/nj : 0.; 

  if(hist2D) { delete hist2D; hist2D=NULL; }
  hist2D=new TH2F("WTS","", nj,p1,p2, ni,90.-t2,90.-t1);
  hist2D->SetXTitle("phi, deg.");
  hist2D->SetYTitle("theta, deg.");

  Int_t colors[30]={101,12,114,13,115,14,117,15,16,17,166,18,19,
                   167,0,0,167,19,18,166,17,16,15,117,14,115,13,114,12,101};
  if(pal==0) gStyle->SetPalette(30,colors);
  else {gStyle->SetPalette(1,0);gStyle->SetNumberContours(pal);}

  wavearray<double> wl;
  double theta, phi;

  for(int i=0; i<ni; i++) { 
    theta = (t2+t1)/2.+(i-ni/2)*dt;
    for(int j=0; j<nj; j++) {
      phi = (p2+p1)/2.+(j-nj/2)*dp;
      hist2D->Fill(phi,90.-theta,sm.get(theta,phi));
    }
  }

  hist2D->SetStats(kFALSE);
  hist2D->SetFillColor(kWhite);

  //hist2D->GetXaxis()->SetNdivisions(70318);
  hist2D->GetXaxis()->SetTitleFont(42);
  hist2D->GetXaxis()->SetLabelFont(42);
  hist2D->GetXaxis()->SetLabelOffset(0.012);
  hist2D->GetXaxis()->SetTitleOffset(1.1);

  //hist2D->GetYaxis()->SetNdivisions(508);
  hist2D->GetYaxis()->SetTitleFont(42);
  hist2D->GetYaxis()->SetLabelFont(42);
  hist2D->GetYaxis()->SetLabelOffset(0.01);

  hist2D->GetZaxis()->SetLabelFont(42);
//
  if(opt) hist2D->Draw(opt);
  else    hist2D->Draw("COLZ");

  // change palette's width
  canvas->Update();
  TPaletteAxis *palette = (TPaletteAxis*)hist2D->GetListOfFunctions()->FindObject("palette");
  palette->SetX1NDC(0.91);
  palette->SetX2NDC(0.933);
  palette->SetTitleOffset(0.92);
  palette->GetAxis()->SetTickSize(0.01);
  canvas->Modified();

  return;
}

void watplot::plot(netcluster* pwc, int cid, int nifo, char type, int irate, char* opt, int pal, bool wp) {
//
// monster event display (only for 2G analysis) 
// display pixels of all resolution levels
//
// pwc    : pointer to netcluster object
// cid    : cluster id
// nifo   : number of detectors
// type   : 'L' -> plot event likelihood pixels, 'N' -> plot event null pixels
// irate  : select pixel rate
//          0 : all rates
//          1 : optimal rate
//          x : rate x
// opt    : root draw options 
// pal    : palette 
// wp     : true -> wavelet packet
//

  bool isPCs = std::isupper(type);       		// are Principal Components ?
  type = std::toupper(type);

  if(type != 'L' && type != 'N') return;

  double RATE = pwc->rate;                      	// original rate

  std::vector<int>* vint = &(pwc->cList[cid-1]); 	// pixel list

  int V = vint->size();                         	// cluster size
  if(!V) return;                                               

  // check if WDM (2G), else exit
  netpixel* pix = pwc->getPixel(cid,0);
  int mp= size_t(pwc->rate/pix->rate+0.1);
  int mm= pix->layers;                   		// number of wavelet layers
  if(mm==mp) {       					// wavelet 
    cout << "watplot::plot - Error : monster event display is enabled only for WDM (2G)" << endl;
    exit(1);
  } 

  bool optrate=false;
  if(irate==1) {					// extract optimal rate
    optrate=true;
    vector_int* pv = pwc->cRate.size() ? &(pwc->cRate[cid-1]) : NULL;
    irate = pv!=NULL ? (*pv)[0] : 0;                
  }

  int minLayers=1000;
  int maxLayers=0;   
  double minTime=1e20;
  double maxTime=0.;  
  double minFreq=1e20;
  double maxFreq=0.;  
  for(int j=0; j<V; j++) {                      // loop over the pixels
    netpixel* pix = pwc->getPixel(cid,j);                               
    if(!pix->core) continue;                                           

    if((irate)&&(irate != int(pix->rate+0.5))) continue;  // if irate!=0 skip rate!=irate

    if(pix->layers<minLayers) minLayers=pix->layers;
    if(pix->layers>maxLayers) maxLayers=pix->layers;

    double dt = 1./pix->rate;
    double time = int(pix->time/pix->layers)/double(pix->rate); 	// central bin time
    time -= dt/2.; 							// begin bin time
    if(time<minTime) minTime=time;                   
    if(time+dt>maxTime) maxTime=time+dt;                   

    double freq = pix->frequency*pix->rate/2.; 
    if(freq<minFreq) minFreq=freq;     
    if(freq>maxFreq) maxFreq=freq;     
  }                                    

  int minRate=RATE/(maxLayers-1);
  int maxRate=RATE/(minLayers-1);

  double mindt = 1./maxRate;
  double maxdt = 1./minRate;
  double mindf = minRate/2.;
  double maxdf = maxRate/2.;

  //cout << "minRate : " << minRate << "\t\t\t maxRate : " << maxRate << endl;
  //cout << "minTime : " << minTime << "\t\t\t maxTime : " << maxTime << endl;
  //cout << "minFreq : " << minFreq << "\t\t\t maxFreq : " << maxFreq << endl;
  //cout << "mindt   : " << mindt   << "\t\t\t maxdt   : " << maxdt << endl;
  //cout << "mindf   : " << mindf   << "\t\t\t maxdf   : " << maxdf << endl;

  double iminTime = minTime-maxdt;
  double imaxTime = maxTime+maxdt;
  int nTime = (imaxTime-iminTime)*maxRate;

  if(hist2D) { delete hist2D; hist2D=NULL; }
  hist2D=new TH2F("WTF", "WTF", nTime, iminTime, imaxTime, 2*(maxLayers-1), 0, RATE/2);
  hist2D->SetXTitle("time, sec");                                                  
  hist2D->SetYTitle("frequency, Hz");                                              

  Int_t colors[30]={101,12,114,13,115,14,117,15,16,17,166,18,19,
                   167,0,0,167,19,18,166,17,16,15,117,14,115,13,114,12,101};
  if(pal==0) gStyle->SetPalette(30,colors);
  else {gStyle->SetPalette(1,0);gStyle->SetNumberContours(pal);}

  hist2D->SetStats(kFALSE);
  hist2D->SetTitleFont(12);
  hist2D->SetFillColor(kWhite);

  hist2D->GetXaxis()->SetNdivisions(506);
  hist2D->GetXaxis()->SetLabelFont(42);  
  hist2D->GetXaxis()->SetLabelOffset(0.014);
  hist2D->GetXaxis()->SetTitleOffset(1.4);  
  hist2D->GetYaxis()->SetTitleOffset(1.2);  
  hist2D->GetYaxis()->SetNdivisions(506);   
  hist2D->GetYaxis()->SetLabelFont(42);     
  hist2D->GetYaxis()->SetLabelOffset(0.01); 
  hist2D->GetZaxis()->SetLabelFont(42);     
  hist2D->GetZaxis()->SetNoExponent(false); 
  hist2D->GetZaxis()->SetNdivisions(506);   

  hist2D->GetXaxis()->SetTitleFont(42);
  hist2D->GetXaxis()->SetTitle("Time (sec)");
  hist2D->GetXaxis()->CenterTitle(true);     
  hist2D->GetYaxis()->SetTitleFont(42);      
  hist2D->GetYaxis()->SetTitle("Frequency (Hz)");
  hist2D->GetYaxis()->CenterTitle(true);         

  hist2D->GetZaxis()->SetTitleOffset(0.6);
  hist2D->GetZaxis()->SetTitleFont(42);   
  hist2D->GetZaxis()->CenterTitle(true);  

  hist2D->GetXaxis()->SetLabelSize(0.03);
  hist2D->GetYaxis()->SetLabelSize(0.03);
  hist2D->GetZaxis()->SetLabelSize(0.03);

  double dFreq = (maxFreq-minFreq)/10.>2*maxdf ? (maxFreq-minFreq)/10. : 2*maxdf ;
  double mFreq = minFreq-dFreq<0 ? 0 : minFreq-dFreq;
  double MFreq = maxFreq+dFreq>RATE/2 ? RATE/2 : maxFreq+dFreq;
  hist2D->GetYaxis()->SetRangeUser(mFreq, MFreq);              

  double dTime = (maxTime-minTime)/10.>2*maxdt ? (maxTime-minTime)/10. : 2*maxdt ;
  double mTime = minTime-dTime<iminTime ? iminTime : minTime-dTime;
  double MTime = maxTime+dTime>imaxTime ? imaxTime : maxTime+dTime;
  hist2D->GetXaxis()->SetRangeUser(mTime,MTime);

  int npix=0;
  double Null=0;
  double Likelihood=0;
  for(int n=0; n<V; n++) {
    netpixel* pix = pwc->getPixel(cid,n);
    if(!pix->core) continue;            

    if((irate)&&(irate != int(pix->rate+0.5))) continue;  // if irate!=0 skip rate!=irate

    double like=0;
    double null=0;
    if(wp) {  					// likelihoodWP 
      like = pix->likelihood>0. ? pix->likelihood : 0.;
      null = pix->null>0. ? pix->null : 0.;
    } else {					// likelihood2G
      double sSNR=0;
      double wSNR=0;
      for(int m=0; m<nifo; m++) {                 
        sSNR += pow(pix->getdata('S',m),2);     // snr whitened reconstructed signal 00
        sSNR += pow(pix->getdata('P',m),2);     // snr whitened reconstructed signal 90
        wSNR += pow(pix->getdata('W',m),2);     // snr whitened at the detector 00
        wSNR += pow(pix->getdata('U',m),2);     // snr whitened at the detector 90
      }       
      if(!isPCs) {sSNR/=2;wSNR/=2;}		// if not principal components we use (00+90)/2
      like = sSNR;
      null = wSNR-sSNR;                                      
    }                                                               

    int iRATE = int(pix->rate+0.5); 
    int M=maxRate/iRATE;              
    int K=2*(maxLayers-1)/(pix->layers-1);
    double dt = 1./pix->rate;
    double itime = int(pix->time/pix->layers)/double(pix->rate); 	// central bin time
    itime -= dt/2.;							// begin bin time
    int i=(itime-iminTime)*maxRate;                            
    int j=pix->frequency*K;                                    
    if(iRATE!=irate && irate!=0) continue;                     
    Null+=null;                                                
    Likelihood+=like;                               
    int L=0;int R=1;while (R < iRATE) {R*=2;L++;}
    for(int m=0;m<M;m++) {                                     
      for(int k=0;k<K;k++) {
        if(null<0) null=0; 
        double A = hist2D->GetBinContent(i+1+m,j+1+k-K/2);
        if(type=='L') hist2D->SetBinContent(i+1+m,j+1+k-K/2,like+A);
        else          hist2D->SetBinContent(i+1+m,j+1+k-K/2,null+A);           
      }                                                               
    }                                                                 

    if(type=='L' && like>0) npix++;
    if(type=='N' && null>0) npix++;
  }                                                                   

  char gtitle[256];
  if(type=='L') sprintf(gtitle,"Likelihood %3.0f - dt(ms) [%.6g:%.6g] - df(hz) [%.6g:%.6g] - npix %d",
                Likelihood,1000*mindt,1000*maxdt,mindf,maxdf,npix);
  else          sprintf(gtitle,"Null %3.0f - dt(ms) [%.6g:%.6g] - df(hz) [%.6g:%.6g] - npix %d",
                Null,1000*mindt,1000*maxdt,mindf,maxdf,npix);

  hist2D->SetTitle(gtitle);                                                             

  //cout << "Event Likelihood : " << Likelihood << endl;
  //cout << "Event Null       : " << Null << endl;      

  if(opt) hist2D->Draw(opt);
  else    hist2D->Draw("COLZ");

  // change palette's width
  canvas->Update();
  TPaletteAxis *palette = (TPaletteAxis*)hist2D->GetListOfFunctions()->FindObject("palette");
  palette->SetX1NDC(0.91);
  palette->SetX2NDC(0.933);
  palette->SetTitleOffset(0.92);
  palette->GetAxis()->SetTickSize(0.01);
  canvas->Modified();

  return;
}

void watplot::plot(clusterdata* pcd, double inj_mchirp) {
//
// chirp event display (only for 2G analysis) : WARNING : Add check 2G !!!
// display frequency vs time 
//
// pcd        : pointer to clusterdata structure
// inj_mchirp : injected mchirp value
//

  static TGraphErrors egraph;

  double x,y;
  double ex,ey;
  int np = pcd->chirp.GetN();
  egraph.Set(np);
  for(int i=0;i<np;i++) {
    pcd->chirp.GetPoint(i,x,y);
    ex=pcd->chirp.GetErrorX(i);
    ey=pcd->chirp.GetErrorY(i);
    egraph.SetPoint(i,x,y);
    egraph.SetPointError(i,ex,ey);
  }
  TF1* fit = &(pcd->fit);

  char title[256];
  if(inj_mchirp) {
    sprintf(title,"chirp mass : rec = %3.3f [%3.2f] inj = %3.3f , chi2 = %3.2f",
            pcd->mchirp,pcd->mchirperr,inj_mchirp,pcd->chi2chirp);
  } else {
    sprintf(title,"chirp mass : rec = %3.3f [%3.2f] , chi2 = %3.2f",
            pcd->mchirp,pcd->mchirperr,pcd->chi2chirp);
  }
  egraph.SetTitle(title);
  egraph.GetHistogram()->SetStats(kFALSE);
  egraph.GetHistogram()->SetTitleFont(12);
  egraph.SetFillColor(kWhite);
  egraph.SetLineColor(kBlack);
  egraph.GetXaxis()->SetNdivisions(506);
  egraph.GetXaxis()->SetLabelFont(42);
  egraph.GetXaxis()->SetLabelOffset(0.014);
  egraph.GetXaxis()->SetTitleOffset(1.4);
  egraph.GetYaxis()->SetTitleOffset(1.2);
  egraph.GetYaxis()->SetNdivisions(506);
  egraph.GetYaxis()->SetLabelFont(42);
  egraph.GetYaxis()->SetLabelOffset(0.01);
  egraph.GetXaxis()->SetTitleFont(42);
  egraph.GetXaxis()->SetTitle("Time (sec)");
  egraph.GetXaxis()->CenterTitle(true);
  egraph.GetYaxis()->SetTitleFont(42);
  egraph.GetYaxis()->SetTitle("Frequency^{-8/3}");
  egraph.GetYaxis()->CenterTitle(true);
  egraph.GetXaxis()->SetLabelSize(0.03);
  egraph.GetYaxis()->SetLabelSize(0.03);

  egraph.Draw("AP");
  fit->Draw("same");

  return;
}

void watplot::SetPlotStyle(int paletteId, int NCont) {
//
// set palette type
//

  const Int_t NRGBs = 5;
//  const Int_t NCont = 255;
  NCont=abs(NCont);

  if (fabs(paletteId)==1) {
    Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
    Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
    if (paletteId<0) {
      TColor::CreateGradientColorTable(NRGBs, stops, blue, green, red, NCont);
    } else {
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    }
  } else
  if (fabs(paletteId)==2) {
    Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
    //Double_t red[NRGBs]   = { 0.00, 0.00, 0.00, 1.00, 1.00 };
    Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    //Double_t green[NRGBs] = { 0.00, 1.00, 1.00, 1.00, 0.00 };
    Double_t blue[NRGBs]  = { 1.00, 1.00, 0.00, 0.00, 0.00 };
    if (paletteId<0) {
      TColor::CreateGradientColorTable(NRGBs, stops, blue, green, red, NCont);
    } else {
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    }
  } else
  if (fabs(paletteId)==3) {
    Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    Double_t red[NRGBs]   = { 0.00, 0.09, 0.18, 0.09, 0.00 };
    Double_t green[NRGBs] = { 0.01, 0.02, 0.39, 0.68, 0.97 };
    Double_t blue[NRGBs]  = { 0.17, 0.39, 0.62, 0.79, 0.97 };
    if (paletteId<0) {
      TColor::CreateGradientColorTable(NRGBs, stops, blue, green, red, NCont);
    } else {
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    }
  } else
  if (fabs(paletteId)==4) {
    Double_t stops[NRGBs] = { 0.00, 0.50, 0.75, 0.875, 1.00 };
    Double_t red[NRGBs]   = { 1.00, 1.00, 1.00, 1.00, 1.00 };
    Double_t green[NRGBs] = { 1.00, 0.75, 0.50, 0.25, 0.00 };
    Double_t blue[NRGBs]  = { 0.00, 0.00, 0.00, 0.00, 0.00 };
    if (paletteId<0) {
      TColor::CreateGradientColorTable(NRGBs, stops, blue, green, red, NCont);
    } else {
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    }
  } else
  if (fabs(paletteId)==5) {  // Greyscale palette
    Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    Double_t red[NRGBs]   = { 1.00, 0.84, 0.61, 0.34, 0.00 };
    Double_t green[NRGBs] = { 1.00, 0.84, 0.61, 0.34, 0.00 };
    Double_t blue[NRGBs]  = { 1.00, 0.84, 0.61, 0.34, 0.00 };
    if (paletteId<0) {
      TColor::CreateGradientColorTable(NRGBs, stops, blue, green, red, NCont);
    } else {
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    }
  }
  gStyle->SetNumberContours(NCont);

  return;
}

// wavescan (2D histogram)
/*
void watplot::scan(wavearray<double>& x, int M1, int M2, double=t1, double=t2, char* =c, int=nc)
{
  int i,j,k;
  WDM<double>* wdm[10];

  if(M2-M1 > 10) M2 = M1+9;
  for(i=M1; i<=M2; i++) wdm[i-M1] = new WDM<double>(i,i);


  for(i=M1; i<=M2; i++) delete wdm[i-M1];

}
*/


void
watplot::blackmanharris(wavearray<double>& w) {
//
// blackmanharris window
//
// w   : in/out wavearray - must be initialized with w.size() & w.rate()
//

  int size = (int)w.size();

  for (int i = 0; i < size; i++)
  {
    double f = 0.0;
    f = ((double) i) / ((double) (size - 1));
    w[i] = 0.35875 -
      0.48829 * cos(2.0 * TMath::Pi() * f) +
      0.14128 * cos(4.0 * TMath::Pi() * f) -
      0.01168 * cos(6.0 * TMath::Pi() * f);
  }

  double norm = 0;
  for (int i=0;i<size;i++) norm += pow(w[i],2);
  norm /= size;
  for (int i=0;i<size;i++) w[i] /= sqrt(norm);
}

void 
watplot::print(TString fname) {
//
// Save plot with name -> fname
//

  if(canvas!=NULL && fname!="") {
    if(fname.Contains(".png")) {
      TString gname(fname);
      gname.ReplaceAll(".png",".gif");
      canvas->Print(gname);
      char cmd[1024];
      sprintf(cmd,"convert %s %s",gname.Data(),fname.Data());
      //cout << cmd << endl;
      gSystem->Exec(cmd);
      sprintf(cmd,"rm %s",gname.Data());
      //cout << cmd << endl;
      gSystem->Exec(cmd);
    } else {
      canvas->Print(fname);
    }
  }
}

void 
watplot::goptions(char* opt, int col, double t1, double t2, bool fft, float f1, float f2, bool psd, float t3, bool oneside) {
//
// Set graphical options
//
// opt   : TGraph::Draw options
// col   : set TGraph line color       
// t1    : start of time interval in seconds
// t2    : end of time interval in seconds
// fft   : true -> plot fft
// f1    : set begin frequency (Hz)
// f2    : set end frequency (Hz)
// psd   : true -> plot psd using blackmanharris window
// t3    : is the chunk length (sec) used to produce the psd 
// oneside : true/false -> oneside/doubleside
//

  this->ncol= 0;
  this->opt = opt;
  this->col = col;
  this->t1  = t1;
  this->t2  = t2;
  this->fft = fft;
  this->f1  = f1;
  this->f2  = f2;
  this->psd = psd;
  this->t3  = t3;
  this->oneside  = oneside;

  for(size_t i=0; i<graph.size(); i++) {
    if(graph[i]) delete graph[i];
  }
  graph.clear();
}

void 
watplot::gtitle(TString title, TString xtitle, TString ytitle) {
//
// Set TGraph titles 
//
// title  : graph title 
// xtitle : x axis name
// ytitle : y axis name
//

  this->title=title;
  this->xtitle=xtitle;
  this->ytitle=ytitle;
  if(graph.size()>0) { 
    if(title!="")  graph[0]->SetTitle(title);
    if(xtitle!="") graph[0]->GetHistogram()->SetXTitle(xtitle);
    if(ytitle!="") graph[0]->GetHistogram()->SetYTitle(ytitle);
    canvas->Update();
  }
}

wavearray<double>& operator >> (watplot& graph, wavearray<double>& x) {
//
// assignement operator
//
// graph >> x : fill x wavearray with watplot object graph with values
//

  x=graph.data;
  return x;
}

watplot& operator >> (wavearray<double>& x, watplot& graph) {
//
// assignement operator
//
// x >> graph : fill watplot object graph with x wavearray values
//

  if(x.size()==0) {
    cout << "watplot::operator(>>) : input array with size=0" << endl;
    exit(1);
  }

  graph.opt.ToUpper();
  bool logx = false; if(graph.opt.Contains("LOGX")) {logx=true;graph.opt.ReplaceAll("LOGX","");}
  bool logy = false; if(graph.opt.Contains("LOGY")) {logy=true;graph.opt.ReplaceAll("LOGY","");}

  double t1 = graph.t1==0. ? x.start()                   : graph.t1;
  double t2 = graph.t2==0. ? x.start()+x.size()/x.rate() : graph.t2;
  double f1 = graph.f1==0. ? 0.                          : graph.f1;
  double f2 = graph.f2==0. ? x.rate()/2.                 : graph.f2;

  TString opt = graph.ncol ? "SAME" : graph.opt;

  graph.data=x;
  if(logx) graph.canvas->SetLogx();
  if(logy) graph.canvas->SetLogy();
  graph.plot(x, const_cast<char*>(opt.Data()), graph.col+graph.ncol, t1, t2, graph.fft, f1, f2, graph.psd, graph.t3, graph.oneside);
  if(graph.graph.size()>0) { 
    if(graph.title!="")  graph.graph[0]->SetTitle(graph.title);
    if(graph.xtitle!="") graph.graph[0]->GetHistogram()->SetXTitle(graph.xtitle);
    if(graph.ytitle!="") graph.graph[0]->GetHistogram()->SetYTitle(graph.ytitle);
    graph.canvas->Update();
  }
  graph.ncol++;
  return graph;
}

TString& operator >> (watplot& graph, TString& fname) {
//
// print operator 
//
// graph >> fname : save watplot graph to fname file
//

  if(graph.graph.size()==0) {
    cout << "watplot::operator (>>) - Error : No graph in the object" << endl;
    return fname;
  }
  TString name = fname;
  name.ReplaceAll(" ","");

  TObjArray* token = TString(name).Tokenize(TString("."));
  TObjString* ext_tok = (TObjString*)token->At(token->GetEntries()-1);
  TString ext = ext_tok->GetString();
  if(ext=="txt") {

     ofstream out;
     out.open(name.Data(),ios::out);
     if (!out.good()) {cout << "watplot::operator (>>) - Error : Opening File : " << name.Data() << endl;exit(1);}

     double x,y;
     int size=graph.graph[0]->GetN();
     for (int j=0;j<size;j++) {
       graph.graph[0]->GetPoint(j,x,y);
       if(x>graph.f1 && x<graph.f2) out << x << "\t" << y << endl;
     }

     out.close();
  } else {
    graph.print(name);
  }
  return fname;
}

char* operator >> (watplot& graph, char* fname) {
//
// print operator 
//
// graph >> fname : save watplot graph to fname file
//

  TString name = fname;
  graph >> name;
  return fname;
}
