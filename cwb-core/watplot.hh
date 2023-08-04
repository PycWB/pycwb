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


/********************************************************/
/* Wavelet Analysis Tool                                */
/* file watplot.hh                                      */
/* wat plot routines                                    */
/********************************************************/

#ifndef WATPLOT_HH
#define WATPLOT_HH

#include <iostream>
#include <vector>
#include "TCanvas.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TGraph.h"
#include "TRandom.h"
#include "TString.h"
#include "wavearray.hh"
#include "wseries.hh"
#include "skymap.hh"
#include "netcluster.hh"

#include "TColor.h"  

class watplot {
public:
  
  // Default constructor
  watplot(char* name=NULL, int=200, int=20, int=800, int=600); 
  
  // Destructor
  virtual ~watplot() {
    clear();
    if(canvas) delete canvas; 
  }
  
  // Clear graph objects
  inline void clear() { 
    for(size_t i=0; i<hist1D.size(); i++) {
      if(hist1D[i]) delete hist1D[i];
    }
    hist1D.clear();
    for(size_t i=0; i<graph.size(); i++) {
      if(graph[i]) delete graph[i];
    }
    graph.clear();
    if(hist2D) { delete hist2D;  hist2D = NULL; }   // this instruction could produce a crash , why???
  }
  
  // Clear graph objects
  inline void null() { 
    canvas = NULL;
    hist2D = NULL;
    graph.clear();
    hist1D.clear();
  }


  // Plot wavearray<double> time series 
  void plot(wavearray<double>&, char* =NULL, int=1, double=0., double=0., 
            bool=false, float=0., float=0., bool=false, float=0., bool=false);
    
  // Plot wavearray<double>* time series : same parameters as plot(wavearray<double>,...)
  void plot(wavearray<double>* ts, char* o=NULL, int c=1, double t1=0., double t2=0.,
            bool fft=false, float f1=0., float f2=0., bool psd=false, float t3=0., bool oneside=false) {
    plot(*ts, o, c, t1, t2, fft, f1, f2, psd, t3, oneside); 
  }

  // Plot wavearray<float>* time series : same parameters as plot(wavearray<double>,...)
  void plot(wavearray<float>* ts, char* o=NULL, int c=1, double t1=0., double t2=0.,
            bool fft=false, float f1=0., float f2=0., bool psd=false, float t3=0., bool oneside=false) {
    wavearray<double> tf;
    waveAssign(tf,*ts);
    plot(tf, o, c, t1, t2, fft, f1, f2, psd, t3, oneside); 
  }

  // Plot wavearray<float> time series : same parameters as plot(wavearray<double>,...)
  void plot(wavearray<float> &ts, char* o=NULL, int c=1, double t1=0., double t2=0.,
            bool fft=false, float f1=0., float f2=0., bool psd=false, float t3=0., bool oneside=false) {
    wavearray<double> tf;
    waveAssign(tf,ts);
    plot(tf, o, c, t1, t2, fft, f1, f2, psd, t3, oneside); 
  }

  // Plot wavearray<int>* time series : same parameters as plot(wavearray<double>,...)
  void plot(wavearray<int>* ts, char* o=NULL, int c=1, double t1=0., double t2=0.,
            bool fft=false, float f1=0., float f2=0., bool psd=false, float t3=0., bool oneside=false) {
    wavearray<double> tf;
    waveAssign(tf,*ts);
    plot(tf, o, c, t1, t2, fft, f1, f2, psd, t3, oneside); 
  }

  // Plot wavearray<int> time series : same parameters as plot(wavearray<double>,...)
  void plot(wavearray<int> &ts, char* o=NULL, int c=1, double t1=0., double t2=0.,
            bool fft=false, float f1=0., float f2=0., bool psd=false, float t3=0., bool oneside=false) {
    wavearray<double> tf;
    waveAssign(tf,ts);
    plot(tf, o, c, t1, t2, fft, f1, f2, psd, t3, oneside); 
  }

  // wavescan (2D histogram)
  //  void scan(wavearray<double>&, int, int, double=0., double=0., char* =NULL, int=256);

  // time-frequency series (2D histogram)
//  void plot(WSeries<double>&, int=0, double=0., double=0., char* =NULL, int=256);
//  void plot(WSeries<double>* tf, int m=0, double t1=0., double t2=0., char* o=NULL, int p=256) {
//    plot(*tf, m, t1, t2, o, p); 
//  }

  void plot(WSeries<double>&, int=0, double=0., double=0., char* =NULL, int=256, int=0);
  void plot(WSeries<double>* tf, int m=0, double t1=0., double t2=0., char* o=NULL, int p=256, int pid=0) {
    plot(*tf, m, t1, t2, o, p, pid); 
  }

  void plsmooth(WSeries<double>&, int=0, double=0., double=0., char* =NULL, int=256, int=0);
  void plsmooth(WSeries<double>* tf, int sfact=0, double t1=0., double t2=0., char* o=NULL, int p=256, int pid=0) {
    plsmooth(*tf, sfact, t1, t2, o, p, pid); 
  }

  // plot skymaps
  void plot(skymap&, char* = NULL, int=256);

  // monster event display (multi resolution analysis)
  // pwc   : pointer to netcluster 
  // cid   : event cluster id
  // nifo  : number of detectors
  // type  : L/N likelihood/null
  // irate : rate to be plotted - if 0 then select all rates 
  // opt   : draw options
  // pal   : draw palette
  // wp    : wavelet packet
  void plot(netcluster* pwc, int cid, int nifo, char type='L', int irate=0, char* opt=NULL, int pal=256, bool wp=false);

  // chirp event display
  // pwc        : pointer to clusterdata 
  // inj_mchirp : chirp mass of injected signal
  void plot(clusterdata* pcd, double inj_mchirp);

  // set palette 
  void SetPlotStyle(int paletteId, int NCont=255);
  // return max val
  double getmax(WSeries<double> &w, double t1, double t2);  
  // return max value of WSeries* w object
  double getmax(WSeries<double>* tf, double t1=0., double t2=0.) {
    return getmax(*tf, t1, t2); 
  }

  void print(TString fname);
  void blackmanharris(wavearray<double>& w);
  void goptions(char* opt=NULL, int col=1, double t1=0., double t2=0., 
                bool fft=false, float f1=0., float f2=0., bool psd=false, float t3=0., bool oneside=false);
  void gtitle(TString title="", TString xtitle="", TString ytitle="");

  // return value according to opt option
  double procOpt(int opt, double val00, double val90=0.) {       
           switch(opt) {
             case  1:  return val00*val00;
             case  2:  return fabs(val00);
             case -1:  return (val00*val00+val90*val90)/2.;
             case -2:  return sqrt((val00*val00+val90*val90)/2.);
             default:  return val00+val90;
           }
         }

  // return graph pointer
  TGraph* getGraph(int n){
          return n<graph.size() ? graph[n] : graph[0];
        }

  // data members
  
  TCanvas* canvas;	// pointer to TCanvas object
  TH2F*    hist2D;	// pointer to TH2F object
  std::vector<TGraph*>  graph;	// vector of pointers to TGraph objects
  std::vector<TH1F*>    hist1D;	// vector of pointers to TH1F objects

  wavearray<double> data;
  TString title;	// graph title
  TString xtitle; 	// x axis name
  TString ytitle;	// y axis name
  int ncol;		// color index of TGraph plots
  TString opt;		// TGraph::Draw options
  int col;		// TGraph line color
  double t1;		// start of time interval in seconds
  double t2;		// end of time interval in seconds
  bool fft;		// true -> plot fft
  double f1;		// set begin frequency (Hz)
  double f2;		// set end frequency (Hz)
  bool psd;		// true -> plot psd using blackmanharris window
  double t3;		// is the chunk length (sec) used to produce the psd
  bool oneside;         // 

  // used by THtml doc
  ClassDef(watplot,1)       
};

// put operator
wavearray<double>& operator >> (watplot& graph, wavearray<double>& x);
// get operator
watplot& operator >> (wavearray<double>& x, watplot& graph);
// print operator
TString& operator >> (watplot& graph, TString& fname);
char* operator >> (watplot& graph, char* fname);

#endif // WATPLOT_HH
