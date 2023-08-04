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


//////////////////////////////////////////////////////////
//   class for WaveBurst network event 
//   used as ROOT macro
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef injection_h
#define injection_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "wseries.hh"
#include "network.hh"
#include "detector.hh"
#include "netcluster.hh"

//class detector;
//class netcluster;
//class network;

/* Structure of WaveBurst network event */

#define INJECTION_INIT                                                                          \
      iFile(NULL),run(0),nevent(0),eventID(0),type(0),name(NULL),log(NULL),gps(0.),strain(0),	\
      psi(NULL),iota(NULL),phi(NULL),theta(NULL),bp(NULL),bx(NULL),time(NULL),			\
      duration(NULL),frequency(NULL),bandwidth(NULL),hrss(NULL),snr(NULL),			\
      Deff(NULL),mass(NULL),spin(NULL),pwf(NULL)             

class injection {

//private :

//  detector*       pD[NIFO_MAX];  //!temporary detectors pointers

public :

  TFile          *iFile;          //!root input file cointainig the mdc TTree

  TTree          *fChain;         //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;       //!current Tree number in a TChain
  Int_t           ndim;           //! number of detectors

//Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index, .....
  Int_t           run;            // run ID                                                       
  Int_t           nevent;         // event count                                                  
  Int_t           eventID;        // event ID                                                     
  Int_t           type;           // injection type                                              
  string*         name;           //! injection name
  string*         log;            //! injection log

  Float_t         factor;         // simulation factor				  
  Float_t         distance;       // distance to source in Mpc
  Float_t         mchirp;         // chirp mass in Mo

  Float_t         rp0;	          // eBBH binary distance
  Float_t         e0;	          // eBBH eccentricity
  Float_t         redshift;       // eBBH redshift

  Double_t        gps;            // start time of data segment                         
  Double_t        strain;         // strain of injected simulated signals                         
  Float_t*        psi;            //! source psi angle
  Float_t*        iota;           //! source iota angle
  Float_t*        phi;            //! source phi angle
  Float_t*        theta;          //! source theta angle
  Float_t*        bp;             //! beam pattern coefficients for hp
  Float_t*        bx;             //! beam pattern coefficients for hx 
				  
  Double_t*       time;           //! injection gps time
  Float_t*        duration;       //! estimated duration

  Float_t*        frequency;      //! average center_of_hrss frequency
  Float_t*        bandwidth;      //! estimated bandwidth

  Double_t*       hrss;           //! injected hrss in the detectors
  Float_t*        snr;            //! injected snr in the detectors
  Float_t*        Deff;           //! detector specific effective distance
  Float_t*        mass;           //! [m1,m2], binary mass parameters
  Float_t*        spin;           //! [x1,y1,z1,x2,y2,z2] components of spin vector 

  wavearray<double>** pwf;        //! pointer to the reconstructed waveform 

//List of branches

   TBranch        *b_ndim;   	//!
   TBranch        *b_run;   	//!
   TBranch        *b_nevent;   	//!
   TBranch        *b_eventID;   //!
   TBranch        *b_type;   	//!
   TBranch        *b_name;      //!
   TBranch        *b_log;       //!

   TBranch        *b_factor;	//!
   TBranch        *b_distance;	//!
   TBranch        *b_mchirp;	//!
   TBranch        *b_rp0;	//!
   TBranch        *b_e0;	//!
   TBranch        *b_redshift;	//!
   TBranch        *b_gps;   	//!
   TBranch        *b_strain;   	//!
   TBranch        *b_psi;   	//!
   TBranch        *b_iota;   	//!
   TBranch        *b_phi;   	//!
   TBranch        *b_theta;   	//!
   TBranch        *b_bp;   	//!
   TBranch        *b_bx;   	//!

   TBranch        *b_time;   	//!
   TBranch        *b_duration;  //!

   TBranch        *b_frequency; //!
   TBranch        *b_bandwidth; //!

   TBranch        *b_hrss;   	//!
   TBranch        *b_snr;   	//!
   TBranch        *b_Deff;	//!
   TBranch        *b_mass;	//!
   TBranch        *b_spin;	//!

   injection() : INJECTION_INIT
   { ndim=1; allocate(); init(); return; }

   injection(int n) : INJECTION_INIT
   { ndim=n; allocate(); init(); return; }

   injection(const injection& a) : INJECTION_INIT
   { ndim=a.ndim; allocate(); init(); *this = a; return; }

   injection(TTree *tree, int n) : INJECTION_INIT
   { ndim=n; allocate(); init(); if(tree) Init(tree); }

   injection(TString fName, int n=0) : INJECTION_INIT
   { TTree* tree=Init(fName,n); allocate(); init(); if(tree) Init(tree); return; }

   virtual ~injection() { 

      if (name)        delete name;       // injection name
      if (log)         delete log;        // injection log

      if (psi)         free(psi);         // polarization angles
      if (iota)        free(iota);        // polarization angles
      if (phi)         free(phi);         // source phi angles
      if (theta)       free(theta);       // source theta angles
      if (bp)          free(bp);          // beam pattern coefficients for hp
      if (bx)          free(bx);          // beam pattern coefficients for hx 
      
      if (time)        free(time);        // average center_of_snr time
      if (duration)    free(duration);    // injection duration
      
      if (frequency)   free(frequency);   // average center_of_hrss frequency
      if (bandwidth)   free(bandwidth);   // injection bandwidth
      if (hrss)        free(hrss);        // injected hrss
      if (snr)         free(snr);         // injected snr
      if (Deff)        free(Deff);        // effective distance
      if (mass)        free(mass);        // source mass vector
      if (spin)        free(spin);        // source spin vector

      if (pwf)         free(pwf);         // pointer to the reconstructed waveform  

      if(iFile) {if(fChain) delete fChain; delete iFile;}
   };

   virtual injection& operator=(const injection &);

   Int_t  GetEntry(Int_t);
   Int_t  GetEntries();
   void   allocate();
   void   init();
   TTree* Init(TString fName, int n);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   Bool_t fill_in(network*,int,bool=true);
   void output(TTree*, network*, double, bool=true);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);
   void   Show(Int_t entry = -1);

   // used by THtml doc
   ClassDef(injection,4)	 
};

#endif

