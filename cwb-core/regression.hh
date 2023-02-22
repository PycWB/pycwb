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


//**************************************************************
// Wavelet Analysis Tool  
// Sergey Klimenko, University of Florida
// class for regression analysis of GW data
//**************************************************************

#ifndef REGRESSION_HH
#define REGRESSION_HH

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "wavearray.hh"
#include "wseries.hh"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TVectorD.h"

typedef TMatrixTSym<double> TMatrixDSym; 
typedef std::vector<double> vectorD; 

struct Wiener {                                // filter structure for a single target layer 
   std::vector<int>          channel;  // data channels: 0 - target, >0 witness 
   std::vector<int>          layer;    // data layers: 0 - target, >0 witness 
   std::vector<double>       norm;     // normalization factor 
   std::vector<vectorD>      filter00; // 0-phase filter: 
   std::vector<vectorD>      filter90; // 90-phase filter: 
};

class regression
{
  public:
      
   /* ************** */
   /* constructors   */
   /* ************** */

   regression();

   regression(WSeries<double> &, char*, double fL=0., double fH=0.);
   
   regression(const regression&);
   
   /* ************** */
   /* destructor     */
   /* ************** */

   virtual ~regression() {this->clear();}
   
   /* ************** */
   /* operators      */
   /* ************** */
 
   regression& operator= (const regression&);
   
   /* ************** */
   /* accessors      */
   /* ************** */

   size_t add(WSeries<double> & target, char* name, double fL=0., double fH=0.);

   size_t add(wavearray<double>& witness, char* name, double fL=0., double fH=0.);

   size_t add(int n, int m, char* name);
   
   /* ************** */
   /* mask frequency */
   /* ************** */

   void mask(int n, double flow=0., double fhigh=0.);
   void unmask(int n, double flow=0., double fhigh=0.);

   /* ************** */
   /* compute filter */
   /* ************** */

   size_t setFilter(size_t); 
   
   void setMatrix(double edge=0., double f=1.); 

   void solve(double th, int nE=0, char c='s');

   void apply(double threshold=0., char c='a');

   /* ************** */
   /* get parameters */
   /* ************** */

   // extract matrix
   //
   // n: channel index
   //
   TMatrixDSym getMatrix(size_t n=0)
   { return n<matrix.size() ? matrix[n] : matrix[0]; }
   
   // extract cross vector
   //
   // n: channel index
   //
   wavearray<double> getVCROSS(size_t n=0)
   { return n<vCROSS.size() ? vCROSS[n] : vCROSS[0]; }

   wavearray<double> getVEIGEN(int n=-1);
   
   wavearray<double> getFILTER(char c='a', int nT=-1, int nW=-1);
   
   // get pointer to TF series
   WSeries<double>* getTFmap(int n=0) {return n<(int)chList.size() ? &chList[n] : NULL;}
   
   wavearray<double> rank(int nbins=0, double fL=0., double fH=0.);
   
   //get target-prediction wseries
   inline WSeries <double> getWNoise() { return WNoise; }
	      
   // get target-prediction time series
   inline wavearray<double> getClean() {
      wavearray<double> x = this->target;
      return x -= this->rnoise;
   }

   // get prediction time series
   inline wavearray<double> getNoise() {return rnoise;}      

   // get time series for channel n
   inline wavearray<double> channel(size_t n) {
      WSeries<double> w = n<chList.size() ? chList[n] : chList[0];
      w.Inverse(); return (wavearray<double>)w;
   }

  // get rank values for all frequecy layers
  //
  // n: channel index
  inline wavearray<double> getRank(int n) {                               //RANK
         int tsize=vrank[n].size();                                       //RANK
         wavearray<double> trank(tsize);                                  //RANK
         for (int i=0; i<tsize; i++) trank.data[i]=sqrt(vrank[n].data[i]); //RANK
         return trank;                                                    //RANK
    }                                                                     //RANK

   // clear channel list
   inline void clear() {
      chList.clear(); std::vector< WSeries<double> >().swap(chList);
      chName.clear(); std::vector< char* >().swap(chName);
      chMask.clear(); std::vector< wavearray<int> >().swap(chMask);
      FILTER.clear(); std::vector<Wiener>().swap(FILTER);
      matrix.clear(); std::vector<TMatrixDSym>().swap(matrix);
      vCROSS.clear(); std::vector< wavearray<double> >().swap(vCROSS);
      vEIGEN.clear(); std::vector< wavearray<double> >().swap(vEIGEN);
      vrank.clear(); std::vector< wavearray<double> >().swap(vrank);      //RANK
      vfreq.resize(0);                                                    //RANK
   }
   
   // data members
   
   size_t kSIZE;             // unit filter half-length
   double Edge;              // time offset at the boundaries
   bool   pOUT;              // true/false printout flag
   
   std::vector< WSeries<double> >    chList;   // TF data: 0 - target, >0 - withess   
   std::vector<char*>                chName;   // channel names: 0 - target, >0-witness
   std::vector< wavearray<int> >     chMask;   // layer mask: 0 - target, >0-witness
   std::vector<Wiener>               FILTER;   // total Wiener filter                 
   std::vector<TMatrixDSym>          matrix;   // symmetric matrix                    
   std::vector< wavearray<double> >  vCROSS;   // cross-correlation vector   
   std::vector< wavearray<double> >  vEIGEN;   // vector of eigenvalues   
   wavearray<double>                 target;   // target time series                  
   wavearray<double>                 rnoise;   // regressed out noise          
   WSeries<double>                   WNoise;   // Wavelet series for regressed out noise       
   std::vector< wavearray<double> >  vrank;                   //RANK
   wavearray<double>                 vfreq;                   //RANK

 
private:
   void _apply_(int n,
                std::vector< wavearray<double> > &w, 
                std::vector< wavearray<double> > &W);

   // used by THtml doc
   ClassDef(regression,1)	
  
}; // class regression

#endif // REGRESSION_HH
