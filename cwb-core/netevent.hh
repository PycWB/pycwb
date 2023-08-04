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
//   Gabriele Vedovato, INFN,  Sezione  di  Padova, Italy
//////////////////////////////////////////////////////////


#ifndef netevent_h
#define netevent_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "watfun.hh"
#include "watversion.hh"
#include "injection.hh"
#include "detector.hh"
#include "netcluster.hh"
#include "network.hh"

//class detector;
//class netcluster;
//class network;

/* Structure of WaveBurst network event */

#define NETEVENT_INIT                                                                          \
      iFile(NULL),fChain(NULL),run(0),nevent(0),eventID(NULL),type(NULL),name(NULL),log(NULL),rate(NULL),\
      volume(NULL),size(NULL),usize(0),gap(NULL),lag(NULL),slag(NULL),strain(NULL),            \
      phi(NULL),theta(NULL),psi(NULL),iota(NULL),bp(NULL),bx(NULL),time(NULL),gps(NULL),       \
      right(NULL),left(NULL),duration(NULL),start(NULL),stop(NULL),frequency(NULL),            \
      low(NULL),high(NULL),bandwidth(NULL),hrss(NULL),noise(NULL),erA(NULL),Psave(0),          \
      Psm(NULL),null(NULL),nill(NULL),netcc(NULL),neted(NULL),rho(NULL),gnet(0.),anet(0.),     \
      ecor(0.),norm(0.),ECOR(0.),penalty(0.),likelihood(0.),factor(0.),range(NULL),         \
      chirp(NULL),eBBH(NULL),Deff(NULL),mass(NULL),spin(NULL),snr(NULL),xSNR(NULL),sSNR(NULL), \
      iSNR(NULL),oSNR(NULL),ioSNR(NULL),fP(NULL) 

class netevent {

public :

  TFile          *iFile;         //!root input file cointainig the analyzed TTree 

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain
  Int_t           ndim;          //! number of detectors
  Int_t           Psave;         //! max size used by allocate() for the probability maps  

// Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index, .....

  Int_t           run;            //! run ID                                                       
  Int_t           nevent;         //! event count                                                  
  Int_t*          eventID;        //! event ID
  Int_t*          type;           //! event type: [0] - prod, [1]-sim            
  string*         name;           //! event name:  "" - prod, mdc_name - sim
  string*         log;            //! event log:   "" - prod, mdc_log  - sim
  Int_t*          rate;           //! 1/rate - wavelet time resolution
				  
  Int_t*          volume;         //! cluster volume                                               
  Int_t*          size;           //! cluster size (black pixels only)                             
  Int_t           usize;          //! cluster union size                                           
				  
  Float_t*        gap;            //! time between consecutive events                              
  Float_t*        lag;            //! time lag [sec]                                                   
  Float_t*        slag;           //! time slag [sec]
  Double_t*       strain;         //! sqrt(h+*h+ + hx*hx)                         
  Float_t*        phi;            //! [0]-reconstructed, [1]-injected phi angle, [2]-RA
  Float_t*        theta;          //! [0]-reconstructed, [1]-injected theta angle, [2]-DEC
  Float_t*        psi;            //! [0]-reconstructed psi or phase of gc, [1]-injected psi angle
  Float_t*        iota;           //! [0]-reconstructed iota angle, [1]-injected iota angle
  Float_t*        bp;             //! beam pattern coefficients for hp
  Float_t*        bx;             //! beam pattern coefficients for hx 
				  
  Double_t*       time;           //! average center_of_gravity time
  Double_t*       gps;            //! segment start GPS time                           
  Float_t*        right;          //! min cluster time relative to segment start
  Float_t*        left;           //! max cluster time relative to segment start
  Float_t*        duration;       //! cluster duration = stopW-startW
  Double_t*       start;          //! GPS start time of the cluster
  Double_t*       stop;           //! GPS stop time of the cluster 
				  
  Float_t*        frequency;      //! average center_of_snr frequency
  Float_t*        low;            //! min frequency 
  Float_t*        high;           //! max frequency 
  Float_t*        bandwidth;      //! high-low 
  Double_t*       hrss;           //! hrss
  Double_t*       noise;          //! noise rms
  Float_t*        erA;            //! error angle
  skymap*         Psm;            //! probability cc skymap
  Float_t*        null;           //! un-biased null statistics
  Float_t*        nill;           //! biased null statistics
  Float_t*        rho;            //! effective correlated SNR 
  Float_t*        netcc;          //! network correlation coefficients: 0-net,1-pc,2-cc,3-net2 
  Float_t*        neted;          //! network energy disbalance: 0 - total, 1 - 00-phase, 2 - 90-phase
                                  // 3 - L00-L90, 4 - total abs ED    

  Float_t         gnet;           // network sensitivity
  Float_t         anet;           // network alignment factor
  Float_t         inet;           // network index
  Float_t         ecor;           // correlated energy
  Float_t         norm;           // norm Factor or ellipticity
  Float_t         ECOR;           // effective correlated energy
  Float_t         penalty;        // penalty factor
  Float_t         likelihood;     // network likelihood

  Float_t         factor;         // Multiplicative amplitude factor - simulation only
  Float_t*        range;          //! range to source: [0/1]-rec/inj
  Float_t*        chirp;          //! chirp array: 0-injmass,1-recmass,2-merr,3-tmrgr,4-terr,5-chi2
  Float_t*        eBBH;           //! eBBH array
  Float_t*        Deff;           //! effective range for each detector 
  Float_t*        mass;           //! mass[2], binary mass parameters
  Float_t*        spin;           //! spin[6], binary spin parameters

  Float_t*        snr;            //! energy/noise_variance                                   
  Float_t*        xSNR;           //! data-signal correlation Xk*Sk
  Float_t*        sSNR;           //! energy of reconstructed responses Sk*Sk 
  Float_t*        iSNR;           //! injected snr waveform
  Float_t*        oSNR;           //! reconstructed snr waveform
  Float_t*        ioSNR;          //! injected reconstructed xcor waveform				  

  FILE*           fP;             //! dump file

  std::vector<detector*> ifoList; // detectors

//List of branches

   TBranch        *b_ndim;   	//!
   TBranch        *b_run;   	//!
   TBranch        *b_nevent;   	//!
   TBranch        *b_eventID;   //!
   TBranch        *b_type;   	//!
   TBranch        *b_name;   	//!
   TBranch        *b_log;   	//!
   TBranch        *b_rate;   	//!

   TBranch        *b_volume;   	//!
   TBranch        *b_size;   	//!
   TBranch        *b_usize;   	//!

   TBranch        *b_gap;   	//!
   TBranch        *b_lag;   	//!
   TBranch        *b_slag;   	//!
   TBranch        *b_strain;   	//!
   TBranch        *b_phi;   	//!
   TBranch        *b_theta;   	//!
   TBranch        *b_psi;   	//!
   TBranch        *b_iota;   	//!
   TBranch        *b_bp;   	//!
   TBranch        *b_bx;   	//!

   TBranch        *b_time;   	//!
   TBranch        *b_gps;   	//!
   TBranch        *b_right;   	//!
   TBranch        *b_left;   	//!
   TBranch        *b_duration;  //!
   TBranch        *b_start;   	//!
   TBranch        *b_stop;   	//!

   TBranch        *b_frequency; //!
   TBranch        *b_low;   	//!
   TBranch        *b_high;   	//!
   TBranch        *b_bandwidth; //!

   TBranch        *b_hrss;   	//!
   TBranch        *b_noise;    	//!
   TBranch        *b_erA; 	//!
   TBranch        *b_Psm;       //!
   TBranch        *b_null; 	//!
   TBranch        *b_nill; 	//!
   TBranch        *b_netcc; 	//!
   TBranch        *b_neted; 	//!
   TBranch        *b_rho; 	//!

   TBranch        *b_gnet;  	//!
   TBranch        *b_anet; 	//!
   TBranch        *b_inet; 	//!
   TBranch        *b_ecor; 	//!
   TBranch        *b_norm; 	//!
   TBranch        *b_ECOR; 	//!
   TBranch        *b_penalty; 	//!
   TBranch        *b_likelihood;//!

   TBranch        *b_factor;   	//!
   TBranch        *b_range;	//!
   TBranch        *b_chirp;	//!
   TBranch        *b_eBBH;	//!
   TBranch        *b_Deff;	//!
   TBranch        *b_mass;	//!
   TBranch        *b_spin;	//!

   TBranch        *b_snr;   	//!
   TBranch        *b_xSNR; 	//!
   TBranch        *b_sSNR; 	//!
   TBranch        *b_iSNR;	//!
   TBranch        *b_oSNR;	//!
   TBranch        *b_ioSNR;	//!

   netevent() : NETEVENT_INIT
     { ndim=1; Psave=0; allocate(); init(); return; }  

   netevent(int n, int Psave=0) : NETEVENT_INIT
     { ndim=n; this->Psave=Psave; allocate(); init(); return; }  

   netevent(const netevent& a) : NETEVENT_INIT
     { ndim=a.ndim; Psave=a.Psave; allocate(); init(); *this = a; return; }  

   netevent(TTree *tree, int n) : NETEVENT_INIT 
     { ndim=n; allocate(); init(); if(tree) Init(tree); return; }  

   netevent(TString fName, int n=0) : NETEVENT_INIT 
     { TTree* tree=Init(fName,n); allocate(); init(); if(tree) Init(tree); return; }  

   virtual ~netevent() { 
//    if (fChain)      free(fChain->GetCurrentFile());
      if (eventID)     free(eventID);     // event ID: 1/2 - prod/sim                            
      if (type)        free(type);        // event type: 1/2 - prod/sim                            
      if (name)        delete name;       // event name:  "" - prod, mdc_name - sim
      if (log)         delete log;        // event log:   "" - prod, mdc_log  - sim
      if (rate)        free(rate);        // 1/rate - wavelet time resolution
      
      if (volume)      free(volume);      // cluster volume                                               
      if (size)        free(size);        // cluster size (black pixels only)                             
      
      if (gap)         free(gap);         // time between consecutive events                              
      if (lag)         free(lag);         // time lag [sec]      
      if (slag)        free(slag);        // time slag [sec]      
      if (strain)      free(strain);      // GW strain: 1/2 - prod/sim                            
      if (phi)         free(phi);         // phi: 1/2 - prod/sim                            
      if (theta)       free(theta);       // theta: 1/2 - prod/sim                            
      if (psi)         free(psi);         // psi: 1/2 - prod/sim                            
      if (iota)        free(iota);        // iota: 0/1 - prod/sim                            
                                            
      if (bp)          free(bp);          // beam pattern coefficients for hp
      if (bx)          free(bx);          // beam pattern coefficients for hx 
      
      if (time)        free(time);        // average center_of_snr time for prod and sim
      if (right)       free(right);       // min cluster time                                   
      if (left)        free(left);        // max cluster time                                    
      if (duration)    free(duration);    // cluster duration = stopW-startW
      if (start)       free(start);       // actual start GPS time
      if (stop)        free(stop);        // actual stop GPS time 
      
      if (frequency)   free(frequency);   // average center_of_snr frequency
      if (low)         free(low);         // min frequency 
      if (high)        free(high);        // max frequency 
      if (bandwidth)   free(bandwidth);   // high-low 
      if (hrss)        free(hrss);        // log10(calibrated hrss)
      if (noise)       free(noise);       // log10(calibrated noise rms)
      if (erA)         free(erA);         // error angle
      if (null)        free(null);        // un-biased null statistics
      if (nill)        free(nill);        // biased null statistics
      if (rho)         free(rho);         // effective correlated SNR
      if (netcc)       free(netcc);       // correlation coefficients: 0-net,1-pc,2-cc,3-net2
      if (neted)       free(neted);       // network energy disbalabce
      if (range)       free(range);       // range array: 0-reconstructed, 1-injected  
      if (chirp)       free(chirp);       // chirp array: 0-injmass,1-recmass,2-merr,3-tmrgr,4-terr,5-chi2 
      if (eBBH)        free(eBBH);        // eBBH array: 0-rec ecc, 1-inj ecc  
      if (Deff)        free(Deff);        // sim effective distance for each detector 
      if (mass)        free(mass);        // mass[2], binary mass parameters
      if (spin)        free(spin);        // spin[6], binary spin parameters
      if (snr)         free(snr);         // energy/noise_variance                                   
      if (xSNR)        free(xSNR);        // x-s snr of the detector responses
      if (sSNR)        free(sSNR);        // signal snr of the detector responses
      if (iSNR)        free(iSNR);        // injected snr
      if (oSNR)        free(oSNR);        // recontructed snr
      if (ioSNR)       free(ioSNR);       // injected recontructed xcor

      if (Psm)         delete Psm;        // probability skymap

      if(iFile) delete iFile;
   };

   virtual netevent& operator=(const netevent &);

   Int_t  GetEntries();
   Int_t  GetEntry(Int_t);
   void   allocate();
   void   init();
   TTree* Init(TString fName, int n);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   void   setSLags(float* slag);  
   void output(TTree* = NULL, network* = NULL, double = 0., size_t = 0, int = -1);
   void output2G(TTree*, network* , size_t, int, double);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);
   void   Show(Int_t entry = -1);

   inline void dopen(const char *fname, char* mode, bool header=true) {
     if(fP != NULL) fclose(fP);
     if((fP = fopen(fname, mode)) == NULL) {
       cout << "netevent::Dump() error: cannot open file " << fname <<". \n";
       return;
     };
     if(header) fprintf(fP,"# WAT Version : %s - GIT Revision : %s - Tag/Branch : %s",watversion('f'),watversion('r'),watversion('b'));  	
   }

   inline void dclose() {
     if(fP!=NULL) fclose(fP);
     fP = NULL;
     return;
   }
  
   inline void Dump(TString analysis="2G") {
     if(fP==NULL || ndim<1) return;
     size_t i;
     size_t I = ndim;
     
     fprintf(fP,"nevent:     %d\n",nevent);  	 
     fprintf(fP,"ndim:       %d\n",ndim);    	 
     fprintf(fP,"run:        %d\n",run);     	 
     fprintf(fP,"name:       %s\n",name->c_str());
     fprintf(fP,"log:        %s\n",log->c_str());
     fprintf(fP,"rho:        %f\n",rho[analysis=="2G"?0:1]);     
     fprintf(fP,"netCC:      %f\n",netcc[analysis=="2G"?0:0]);     
     fprintf(fP,"netED:      %f\n",neted[0]/ecor);     
     fprintf(fP,"penalty:    %f\n",penalty);    
     fprintf(fP,"gnet:       %f\n",gnet);       
     fprintf(fP,"anet:       %f\n",anet);       
     fprintf(fP,"inet:       %f\n",inet);       
     fprintf(fP,"likelihood: %e\n",likelihood); 
     fprintf(fP,"ecor:       %e\n",ecor);       
     fprintf(fP,"ECOR:       %e\n",ECOR);       
     fprintf(fP,"factor:     %f\n",factor);      
     fprintf(fP,"range:      %f\n",range[0]);   
     fprintf(fP,"mchirp:     %f\n",chirp[0]);      
     fprintf(fP,"norm:       %f\n",norm);       
     fprintf(fP,"usize:      %d\n",usize);

     fprintf(fP,"ifo:        ");  for(i=0; i<I; i++) fprintf(fP,"%s ",ifoList[i]->Name);fprintf(fP,"\n");
     fprintf(fP,"eventID:    ");  for(i=0; i<2; i++) fprintf(fP,"%d ",eventID[i]);   fprintf(fP,"\n");
     fprintf(fP,"rho:        ");  for(i=0; i<2; i++) fprintf(fP,"%f ",rho[i]);       fprintf(fP,"\n");
     fprintf(fP,"type:       ");  for(i=0; i<2; i++) fprintf(fP,"%d ",type[i]);      fprintf(fP,"\n");
     fprintf(fP,"rate:       ");  for(i=0; i<I; i++) fprintf(fP,"%d ",rate[i]);      fprintf(fP,"\n"); 
     fprintf(fP,"volume:     ");  for(i=0; i<I; i++) fprintf(fP,"%d ",volume[i]);    fprintf(fP,"\n");
     fprintf(fP,"size:       ");  for(i=0; i<I; i++) fprintf(fP,"%d ",size[i]);      fprintf(fP,"\n");
     fprintf(fP,"lag:        ");  for(i=0; i<I; i++) fprintf(fP,"%f ",lag[i]);       fprintf(fP,"\n");
     fprintf(fP,"slag:       ");  for(i=0; i<I; i++) fprintf(fP,"%f ",slag[i]);      fprintf(fP,"\n");
     fprintf(fP,"phi:        ");  for(i=0; i<4; i++) fprintf(fP,"%f ",phi[i]);       fprintf(fP,"\n");
     fprintf(fP,"theta:      ");  for(i=0; i<4; i++) fprintf(fP,"%f ",theta[i]);     fprintf(fP,"\n");
     fprintf(fP,"psi:        ");  for(i=0; i<2; i++) fprintf(fP,"%f ",psi[i]);       fprintf(fP,"\n");
     fprintf(fP,"iota:       ");  for(i=0; i<2; i++) fprintf(fP,"%f ",iota[i]);      fprintf(fP,"\n");
     fprintf(fP,"bp:         ");  for(i=0; i<I; i++) fprintf(fP,"%7.4f ",bp[i]);     fprintf(fP,"\n");
     fprintf(fP,"inj_bp:     ");  for(i=I; i<2*I; i++) fprintf(fP,"%7.4f ",bp[i]);   fprintf(fP,"\n");
     fprintf(fP,"bx:         ");  for(i=0; i<I; i++) fprintf(fP,"%7.4f ",bx[i]);     fprintf(fP,"\n");
     fprintf(fP,"inj_bx:     ");  for(i=I; i<2*I; i++) fprintf(fP,"%7.4f ",bx[i]);   fprintf(fP,"\n");
     fprintf(fP,"chirp:      ");  for(i=0; i<6; i++) fprintf(fP,"%f ",chirp[i]);     fprintf(fP,"\n");
     fprintf(fP,"range:      ");  for(i=0; i<2; i++) fprintf(fP,"%f ",range[i]);     fprintf(fP,"\n");  
     fprintf(fP,"Deff:       ");  for(i=0; i<I; i++) fprintf(fP,"%f ",Deff[i]);      fprintf(fP,"\n");  
     fprintf(fP,"mass:       ");  for(i=0; i<2; i++) fprintf(fP,"%f ",mass[i]);      fprintf(fP,"\n");  
     fprintf(fP,"spin:       ");  for(i=0; i<6; i++) fprintf(fP,"%f ",spin[i]);      fprintf(fP,"\n");
     fprintf(fP,"eBBH:       ");  for(i=0; i<4; i++) fprintf(fP,"%f ",eBBH[i]);      fprintf(fP,"\n");  
     fprintf(fP,"null:       ");  for(i=0; i<I; i++) fprintf(fP,"%e ",null[i]);      fprintf(fP,"\n");  
     fprintf(fP,"strain:     ");  for(i=0; i<2; i++) fprintf(fP,"%e ",strain[i]);    fprintf(fP,"\n");
     fprintf(fP,"hrss:       ");  for(i=0; i<I; i++) fprintf(fP,"%e ",hrss[i]);      fprintf(fP,"\n");  
     fprintf(fP,"inj_hrss:   ");  for(i=I; i<2*I; i++) fprintf(fP,"%e ",hrss[i]);    fprintf(fP,"\n");  
     fprintf(fP,"noise:      ");  for(i=0; i<I; i++) fprintf(fP,"%e ",noise[i]);     fprintf(fP,"\n");  

     // use seglen of detector at lag=0 to set segment value of the other detectors
     int mdet=0; for(i=0; i<I; i++) if(lag[i]==0) mdet=i;
     fprintf(fP,"segment:    ");  for(i=0; i<I; i++) {
                                     double seglen = left[mdet]+right[mdet]+duration[1];
				     fprintf(fP,"%12.4f %12.4f ",gps[i],gps[i]+seglen); 
                                  }
                                  fprintf(fP,"\n"); 

     fprintf(fP,"start:      ");  for(i=0; i<I; i++) fprintf(fP,"%12.4f ",start[i]);     fprintf(fP,"\n");  
     fprintf(fP,"time:       ");  for(i=0; i<I; i++) fprintf(fP,"%12.4f ",time[i]);      fprintf(fP,"\n");  
     fprintf(fP,"stop:       ");  for(i=0; i<I; i++) fprintf(fP,"%12.4f ",stop[i]);      fprintf(fP,"\n");
     fprintf(fP,"inj_time:   ");  for(i=I; i<2*I; i++) fprintf(fP,"%12.4f ",time[i]);    fprintf(fP,"\n");  
     fprintf(fP,"left:       ");  for(i=0; i<I; i++) fprintf(fP,"%f ",left[i]);      fprintf(fP,"\n");  
     fprintf(fP,"right:      ");  for(i=0; i<I; i++) fprintf(fP,"%f ",right[i]);     fprintf(fP,"\n");  
     fprintf(fP,"duration:   ");  for(i=0; i<2; i++) fprintf(fP,"%f ",duration[i]);  fprintf(fP,"\n");  
     fprintf(fP,"frequency:  ");  for(i=0; i<2; i++) fprintf(fP,"%f ",frequency[i]); fprintf(fP,"\n");  
     fprintf(fP,"low:        ");  for(i=0; i<1; i++) fprintf(fP,"%f ",low[i]);       fprintf(fP,"\n");  
     fprintf(fP,"high:       ");  for(i=0; i<1; i++) fprintf(fP,"%f ",high[i]);      fprintf(fP,"\n");  
     fprintf(fP,"bandwidth:  ");  for(i=0; i<2; i++) fprintf(fP,"%f ",bandwidth[i]); fprintf(fP,"\n");  

     fprintf(fP,"snr:        ");  for(i=0; i<I; i++) fprintf(fP,"%e ",snr[i]);       fprintf(fP,"\n");      
     fprintf(fP,"xSNR:       ");  for(i=0; i<I; i++) fprintf(fP,"%e ",xSNR[i]);      fprintf(fP,"\n");  
     fprintf(fP,"sSNR:       ");  for(i=0; i<I; i++) fprintf(fP,"%e ",sSNR[i]);      fprintf(fP,"\n");  
     fprintf(fP,"iSNR:       ");  for(i=0; i<I; i++) fprintf(fP,"%f ",iSNR[i]);      fprintf(fP,"\n");
     fprintf(fP,"oSNR:       ");  for(i=0; i<I; i++) fprintf(fP,"%f ",oSNR[i]);      fprintf(fP,"\n");
     fprintf(fP,"ioSNR:      ");  for(i=0; i<I; i++) fprintf(fP,"%f ",ioSNR[i]);     fprintf(fP,"\n");

     fprintf(fP,"netcc:      ");  for(i=0; i<4; i++) fprintf(fP,"%f ",netcc[i]);     fprintf(fP,"\n");
     fprintf(fP,"neted:      ");  for(i=0; i<5; i++) fprintf(fP,"%f ",neted[i]);     fprintf(fP,"\n");
     fprintf(fP,"erA:        ");  for(i=0; i<11; i++) fprintf(fP,"%6.3f ",erA[i]);   fprintf(fP,"\n");  
   };

   // used by THtml doc 
   ClassDef(netevent,2) 	 
};
#endif
