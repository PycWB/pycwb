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


// netevent class to process and store cWB triggers in root file
// S.Klimenko, University of Florida, Gainesville, FL
// G.Vedovato, INFN,  Sezione  di  Padova, Italy  

#include "netevent.hh"
#include "wseries.hh"

#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TMath.h"
#include "TMarker.h"
#include "TVector3.h"
#include "TRotation.h"
#include "TPolyLine.h"
#include "Math/Rotation3D.h"
#include "Math/Vector3Dfwd.h"

#include "Meyer.hh"  
#include <string.h>

#define WAVE_TREE_NAME "waveburst"

ClassImp(netevent)	 // used by THtml doc

using namespace ROOT::Math;

TTree* netevent::Init(TString fName, int n)
{
   iFile = TFile::Open(fName);
   if((iFile==NULL) || (iFile!=NULL && !iFile->IsOpen())) {
     cout << "netevent::Init : Error opening root file " << fName.Data() << endl;
     exit(1);
   }

   TTree* tree = (TTree *) iFile->Get(WAVE_TREE_NAME);
   if(tree) {
     ndim = tree->GetUserInfo()->GetSize(); // get number of detectors
     if(ndim>0) {  
       if(n>0 && ndim!=n) {
         cout << "netevent::Init : number of detectors declared in the constructor (" << n
              << ") are not equals to the one ("<<ndim<<") declared in the root file : "
              << fName.Data() << endl;
         exit(1);
       } 
     } else ndim=n;
   } else {
     cout << "netevent::Init : object tree " << WAVE_TREE_NAME 
          << " not present in the root file " << fName.Data() << endl;
     exit(1);
   }


   if(ndim==0) {
     cout << "netevent::Init : detector number is not declared in the constructor or"
          << " not present in the root file " << fName.Data() << endl;
     exit(1);
   }

    
   return tree;
}

//   Set branch addresses
void netevent::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("ndim",&ndim);
   fChain->SetBranchAddress("run",&run);
   fChain->SetBranchAddress("nevent",&nevent);
   fChain->SetBranchAddress("eventID",eventID);
   fChain->SetBranchAddress("type",type);
   fChain->SetBranchAddress("name",&name);
   fChain->SetBranchAddress("log",&log);
   fChain->SetBranchAddress("rate",rate);

   fChain->SetBranchAddress("volume",volume);
   fChain->SetBranchAddress("size",size);
   fChain->SetBranchAddress("usize",&usize);

   fChain->SetBranchAddress("gap",gap);
   fChain->SetBranchAddress("lag",lag);
   fChain->SetBranchAddress("slag",slag);
   fChain->SetBranchAddress("strain",strain);
   fChain->SetBranchAddress("phi",phi);
   fChain->SetBranchAddress("theta",theta);
   fChain->SetBranchAddress("psi",psi);
   fChain->SetBranchAddress("iota",iota);
   fChain->SetBranchAddress("bp",bp);
   fChain->SetBranchAddress("bx",bx);


   fChain->SetBranchAddress("time",time);
   fChain->SetBranchAddress("gps",gps); 
   fChain->SetBranchAddress("right",right);
   fChain->SetBranchAddress("left",left);
   fChain->SetBranchAddress("duration",duration);
   fChain->SetBranchAddress("start",start);
   fChain->SetBranchAddress("stop",stop);

   fChain->SetBranchAddress("frequency",frequency);
   fChain->SetBranchAddress("low",low);
   fChain->SetBranchAddress("high",high);
   fChain->SetBranchAddress("bandwidth",bandwidth);

   fChain->SetBranchAddress("hrss",hrss);
   fChain->SetBranchAddress("noise",noise);
   fChain->SetBranchAddress("erA",erA);
   if(fChain->GetBranch("Psm")!=NULL) fChain->SetBranchAddress("Psm",&Psm);         
   fChain->SetBranchAddress("null",null);
   fChain->SetBranchAddress("nill",nill);
   fChain->SetBranchAddress("netcc",netcc);
   fChain->SetBranchAddress("neted",neted);
   fChain->SetBranchAddress("rho",rho);

   fChain->SetBranchAddress("gnet",&gnet);
   fChain->SetBranchAddress("anet",&anet);
   fChain->SetBranchAddress("inet",&inet);
   fChain->SetBranchAddress("ecor",&ecor);
   fChain->SetBranchAddress("norm",&norm);
   fChain->SetBranchAddress("ECOR",&ECOR);
   fChain->SetBranchAddress("penalty",&penalty);
   fChain->SetBranchAddress("likelihood",&likelihood);

   fChain->SetBranchAddress("factor",&factor);
   fChain->SetBranchAddress("range",range);
   fChain->SetBranchAddress("chirp",chirp);
   fChain->SetBranchAddress("eBBH",eBBH);
   fChain->SetBranchAddress("Deff",Deff);
   fChain->SetBranchAddress("mass",mass);
   fChain->SetBranchAddress("spin",spin);

   fChain->SetBranchAddress("snr",snr);
   fChain->SetBranchAddress("xSNR",xSNR);
   fChain->SetBranchAddress("sSNR",sSNR);
   fChain->SetBranchAddress("iSNR",iSNR);
   fChain->SetBranchAddress("oSNR",oSNR);
   fChain->SetBranchAddress("ioSNR",ioSNR);

   Notify();
}

// allocate memory
void netevent::allocate()
{
   if(ndim==0) {
     cout << "netevent::allocate : Error - number of detectors must be > 0" << endl;
     exit(1);
   } 

   if (!eventID)     eventID=  (Int_t*)malloc(2*sizeof(Int_t));
   else              eventID=  (Int_t*)realloc(eventID,2*sizeof(Int_t));
   if (!type)        type=     (Int_t*)malloc(2*sizeof(Int_t));
   else              type=     (Int_t*)realloc(type,2*sizeof(Int_t));
   if (!name)        name=     new string();
   else {delete name;name=     new string();}
   if (!log)         log=      new string();
   else {delete log; log=      new string();}
   if (!rate)        rate=     (Int_t*)malloc(ndim*sizeof(Int_t));
   else              rate=     (Int_t*)realloc(rate,ndim*sizeof(Int_t));
   
   if (!volume)      volume=   (Int_t*)malloc(ndim*sizeof(Int_t));
   else              volume=   (Int_t*)realloc(volume,ndim*sizeof(Int_t));
   if (!size)        size=     (Int_t*)malloc(ndim*sizeof(Int_t));
   else              size=     (Int_t*)realloc(size,ndim*sizeof(Int_t));
   
   if (!gap)         gap=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              gap=      (Float_t*)realloc(gap,ndim*sizeof(Float_t));
   if (!lag)         lag=      (Float_t*)malloc((ndim+1)*sizeof(Float_t));
   else              lag=      (Float_t*)realloc(lag,(ndim+1)*sizeof(Float_t));
   if (!slag)        slag=     (Float_t*)malloc((ndim+1)*sizeof(Float_t));
   else              slag=     (Float_t*)realloc(slag,(ndim+1)*sizeof(Float_t));
   if (!gps)         gps=      (Double_t*)malloc((ndim)*sizeof(Double_t));  
   else              gps=      (Double_t*)realloc(gps,(ndim)*sizeof(Double_t));
   if (!strain)      strain=   (Double_t*)malloc(2*sizeof(Double_t));
   else              strain=   (Double_t*)realloc(strain,2*sizeof(Double_t));
   if (!phi)         phi=      (Float_t*)malloc(4*sizeof(Float_t));
   else              phi=      (Float_t*)realloc(phi,4*sizeof(Float_t));
   if (!theta)       theta=    (Float_t*)malloc(4*sizeof(Float_t));
   else              theta=    (Float_t*)realloc(theta,4*sizeof(Float_t));
   if (!psi)         psi=      (Float_t*)malloc(2*sizeof(Float_t));
   else              psi=      (Float_t*)realloc(psi,2*sizeof(Float_t));
   if (!iota)        iota=     (Float_t*)malloc(2*sizeof(Float_t));
   else              iota=     (Float_t*)realloc(iota,2*sizeof(Float_t));
   if (!bp)          bp=       (Float_t*)malloc(ndim*2*sizeof(Float_t));
   else              bp=       (Float_t*)realloc(bp,ndim*2*sizeof(Float_t));
   if (!bx)          bx=       (Float_t*)malloc(ndim*2*sizeof(Float_t));
   else              bx=       (Float_t*)realloc(bx,ndim*2*sizeof(Float_t));
   
   if (!time)        time=     (Double_t*)malloc(ndim*2*sizeof(Double_t));
   else              time=     (Double_t*)realloc(time,ndim*2*sizeof(Double_t));
   if (!right)       right=    (Float_t*)malloc(ndim*sizeof(Float_t));
   else              right=    (Float_t*)realloc(right,ndim*sizeof(Float_t));
   if (!left)        left=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              left=     (Float_t*)realloc(left,ndim*sizeof(Float_t));
   if (!duration)    duration= (Float_t*)malloc(ndim*sizeof(Float_t));
   else              duration= (Float_t*)realloc(duration,ndim*sizeof(Float_t));
   if (!start)       start=    (Double_t*)malloc(ndim*sizeof(Double_t));
   else              start=    (Double_t*)realloc(start,ndim*sizeof(Double_t));
   if (!stop)        stop=     (Double_t*)malloc(ndim*sizeof(Double_t));
   else              stop=     (Double_t*)realloc(stop,ndim*sizeof(Double_t));
   
   if (!frequency)   frequency=(Float_t*)malloc(ndim*sizeof(Float_t));
   else              frequency=(Float_t*)realloc(frequency,ndim*sizeof(Float_t));
   if (!low)         low=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              low=      (Float_t*)realloc(low,ndim*sizeof(Float_t));
   if (!high)        high=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              high=     (Float_t*)realloc(high,ndim*sizeof(Float_t));
   if (!bandwidth)   bandwidth=(Float_t*)malloc(ndim*sizeof(Float_t));
   else              bandwidth=(Float_t*)realloc(bandwidth,ndim*sizeof(Float_t));
   if (!hrss)        hrss=     (Double_t*)malloc(ndim*2*sizeof(Double_t));
   else              hrss=     (Double_t*)realloc(hrss,ndim*2*sizeof(Double_t));
   if (!noise)       noise=    (Double_t*)malloc(ndim*sizeof(Double_t));
   else              noise=    (Double_t*)realloc(noise,ndim*sizeof(Double_t));
   if (!null)        null=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              null=     (Float_t*)realloc(null,ndim*sizeof(Float_t));
   if (!nill)        nill=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              nill=     (Float_t*)realloc(nill,ndim*sizeof(Float_t));
   if (!netcc)       netcc=    (Float_t*)malloc(4*sizeof(Float_t));
   else              netcc=    (Float_t*)realloc(netcc,4*sizeof(Float_t));
   if (!neted)       neted=    (Float_t*)malloc(5*sizeof(Float_t));
   else              neted=    (Float_t*)realloc(neted,5*sizeof(Float_t));
   if (!rho)         rho=      (Float_t*)malloc(2*sizeof(Float_t));
   else              rho=      (Float_t*)realloc(rho,2*sizeof(Float_t));
   if (!erA)         erA=      (Float_t*)malloc(11*sizeof(Float_t));  
   else              erA=      (Float_t*)realloc(erA,11*sizeof(Float_t));
   if (!range)       range=    (Float_t*)malloc(2*sizeof(Float_t));
   else              range=    (Float_t*)realloc(range,2*sizeof(Float_t));
   if (!chirp)       chirp=    (Float_t*)malloc(6*sizeof(Float_t));
   else              chirp=    (Float_t*)realloc(chirp,6*sizeof(Float_t));
   if (!eBBH)        eBBH=     (Float_t*)malloc(4*sizeof(Float_t));
   else              eBBH=     (Float_t*)realloc(eBBH,4*sizeof(Float_t));
   if (!Deff)        Deff=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              Deff=     (Float_t*)realloc(Deff,ndim*sizeof(Float_t));
   if (!mass)        mass=     (Float_t*)malloc(2*sizeof(Float_t));
   else              mass=     (Float_t*)realloc(mass,ndim*sizeof(Float_t));
   if (!spin)        spin=     (Float_t*)malloc(6*sizeof(Float_t));
   else              spin=     (Float_t*)realloc(spin,6*sizeof(Float_t));

   if (!snr)         snr=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              snr=      (Float_t*)realloc(snr,ndim*sizeof(Float_t));   
   if (!xSNR)        xSNR=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              xSNR=     (Float_t*)realloc(xSNR,ndim*sizeof(Float_t));
   if (!sSNR)        sSNR=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              sSNR=     (Float_t*)realloc(sSNR,ndim*sizeof(Float_t));
   if (!iSNR)        iSNR=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              iSNR=     (Float_t*)realloc(iSNR,ndim*sizeof(Float_t));
   if (!oSNR)        oSNR=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              oSNR=     (Float_t*)realloc(oSNR,ndim*sizeof(Float_t));
   if (!ioSNR)       ioSNR=    (Float_t*)malloc(ndim*sizeof(Float_t));
   else              ioSNR=    (Float_t*)realloc(ioSNR,ndim*sizeof(Float_t));

   if (!Psm)         Psm=      WAT::USE_HEALPIX() ? new skymap(int(0)) : new skymap(0.);  
   else {delete Psm; Psm=      WAT::USE_HEALPIX() ? new skymap(int(0)) : new skymap(0.);}

   return;
}

// init array
void netevent::init()
{
   for(int i=0;i<2;i++) eventID[i]=0;
   for(int i=0;i<2;i++) type[i]=0;
   for(int i=0;i<2;i++) strain[i]=0;
   for(int i=0;i<4;i++) phi[i]=0;  
   for(int i=0;i<2;i++) psi[i]=0;  
   for(int i=0;i<4;i++) theta[i]=0;
   for(int i=0;i<2;i++) psi[i]=0; 
   for(int i=0;i<2;i++) iota[i]=0;
   for(int i=0;i<4;i++) netcc[i]=0;
   for(int i=0;i<5;i++) neted[i]=0;
   for(int i=0;i<2;i++) rho[i]=0; 
   for(int i=0;i<11;i++) erA[i]=0;
   for(int i=0;i<2;i++) mass[i]=0;
   for(int i=0;i<2;i++) range[i]=0;
   for(int i=0;i<6;i++) chirp[i]=0;
   for(int i=0;i<4;i++) eBBH[i]=0;
   for(int i=0;i<6;i++) spin[i]=0;

   if(Psm) *Psm=0; 

   for(int i=0;i<ndim+1;i++) lag[i]=0; 
   for(int i=0;i<ndim+1;i++) slag[i]=0; 

   for(int i=0;i<2*ndim;i++) bp[i]=0;  
   for(int i=0;i<2*ndim;i++) bx[i]=0; 
   for(int i=0;i<2*ndim;i++) hrss[i]=0; 
   for(int i=0;i<2*ndim;i++) time[i]=0;

   for(int i=0;i<ndim;i++) {
     rate[i]=0; volume[i]=0; size[i]=0; gap[i]=0; 
     gps[i]=0; snr[i]=0; right[i]=0; left[i]=0; 
     duration[i]=0; start[i]=0; stop[i]=0; frequency[i]=0; low[i]=0;
     high[i]=0; bandwidth[i]=0; noise[i]=0; null[i]=0; nill[i]=0; 
     sSNR[i]=0; Deff[i]=0; iSNR[i]=0; oSNR[i]=0; ioSNR[i]=0; xSNR[i]=0;
   }

   return;
}


Bool_t netevent::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_ndim = fChain->GetBranch("ndim");
   b_run = fChain->GetBranch("run");
   b_nevent = fChain->GetBranch("nevent");
   b_eventID = fChain->GetBranch("eventID");
   b_type = fChain->GetBranch("type");
   b_name = fChain->GetBranch("name");
   b_log = fChain->GetBranch("log");
   b_rate = fChain->GetBranch("rate");

   b_volume = fChain->GetBranch("volume");
   b_size = fChain->GetBranch("size");
   b_usize = fChain->GetBranch("usize");

   b_gap = fChain->GetBranch("gap");
   b_lag = fChain->GetBranch("lag");
   b_slag = fChain->GetBranch("slag");
   b_strain = fChain->GetBranch("strain");
   b_phi = fChain->GetBranch("phi");
   b_theta = fChain->GetBranch("theta");
   b_psi = fChain->GetBranch("psi");
   b_iota = fChain->GetBranch("iota");
   b_bp= fChain->GetBranch("bx");
   b_bx = fChain->GetBranch("bp");

   b_time = fChain->GetBranch("time");
   b_gps = fChain->GetBranch("gps");
   b_right = fChain->GetBranch("right");
   b_left = fChain->GetBranch("left");
   b_start = fChain->GetBranch("start");
   b_stop = fChain->GetBranch("stop");
   b_duration = fChain->GetBranch("duration");

   b_frequency = fChain->GetBranch("frequency");
   b_low = fChain->GetBranch("low");
   b_high = fChain->GetBranch("high");
   b_bandwidth = fChain->GetBranch("bandwidth");

   b_hrss = fChain->GetBranch("hrss");
   b_noise = fChain->GetBranch("noise");
   b_erA = fChain->GetBranch("erA");
   b_Psm = fChain->GetBranch("Psm");                  
   b_null = fChain->GetBranch("null");
   b_nill = fChain->GetBranch("nill");
   b_netcc = fChain->GetBranch("netcc");
   b_neted = fChain->GetBranch("neted");
   b_rho = fChain->GetBranch("rho");

   b_gnet = fChain->GetBranch("gnet");
   b_anet = fChain->GetBranch("anet");
   b_inet = fChain->GetBranch("inet");
   b_ecor = fChain->GetBranch("ecor");
   b_norm = fChain->GetBranch("norm");
   b_ECOR = fChain->GetBranch("ECOR");
   b_penalty = fChain->GetBranch("penalty");
   b_likelihood = fChain->GetBranch("likelihood");

   b_factor = fChain->GetBranch("factor");
   b_range = fChain->GetBranch("range");
   b_chirp = fChain->GetBranch("chirp");
   b_eBBH = fChain->GetBranch("eBBH");
   b_Deff = fChain->GetBranch("Deff");
   b_mass = fChain->GetBranch("mass");
   b_spin = fChain->GetBranch("spin");

   b_snr = fChain->GetBranch("snr");
   b_xSNR = fChain->GetBranch("xSNR");
   b_sSNR = fChain->GetBranch("sSNR");
   b_iSNR = fChain->GetBranch("iSNR");
   b_oSNR = fChain->GetBranch("oSNR");
   b_ioSNR = fChain->GetBranch("ioSNR");

   return kTRUE;
}

Int_t netevent::GetEntries() 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntries(); 
};

Int_t netevent::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void netevent::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}

//++++++++++++++++++++++++++++++++++++++++++++++
// set super lags 
//++++++++++++++++++++++++++++++++++++++++++++++
void netevent::setSLags(float* slag)
{
  for(int n=0;n<=ndim;n++) this->slag[n]=slag[n];
}

//++++++++++++++++++++++++++++++++++++++++++++++
// set single event tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* netevent::setTree()
{
   TTree* waveTree = new TTree(WAVE_TREE_NAME,WAVE_TREE_NAME);

   int i;

   char crate[16];     
   
   char cvolume[16];   
   char csize[16];     
   
   char cgap[16];      
   char clag[16];    
   char cslag[16];
   char cgps[16];      
   char cbp[16];       
   char cbx[16];       
      
   char ctime[16];     
   char cright[16];    
   char cleft[16];     
   char cduration[16]; 
   char cstart[16];    
   char cstop[16];     
   
   char cfrequency[16];
   char clow[16];      
   char chigh[16];     
   char cbandwidth[16];
   char chrss[16];     
   char cnoise[16];    
   char cnull[16];
   char cnill[16];
   char crange[16];
   char cchirp[16];
   char ceBBH[16];
   char cDeff[16];
   char cmass[16];
   char cspin[16];
   char csnr[16];    
   char csSNR[16];
   char cxSNR[16];
   char ciSNR[16];
   char coSNR[16];
   char cioSNR[16];
 

   sprintf(crate,      "rate[%1d]/I",ndim);     
   
   sprintf(cvolume,    "volume[%1d]/I",ndim);   
   sprintf(csize,      "size[%1d]/I",ndim);     
   
   sprintf(cgap,       "gap[%1d]/F",ndim);      
   sprintf(clag,       "lag[%1d]/F",ndim+1);    
   sprintf(cslag,      "slag[%1d]/F",ndim+1);
   sprintf(cgps,       "gps[%1d]/D",ndim); 

   if(ndim<5) sprintf(cbp,        "bp[%1d]/F",ndim*2);       
   else       sprintf(cbp,        "bp[%2d]/F",ndim*2);       
   if(ndim<5) sprintf(cbx,        "bx[%1d]/F",ndim*2);       
   else       sprintf(cbx,        "bx[%2d]/F",ndim*2);       
      
   if(ndim<5) sprintf(ctime,      "time[%1d]/D",ndim*2);     
   else       sprintf(ctime,      "time[%2d]/D",ndim*2);     

   sprintf(cright,     "right[%1d]/F",ndim);    
   sprintf(cleft,      "left[%1d]/F",ndim);     
   sprintf(cduration,  "duration[%1d]/F",ndim); 
   sprintf(cstart,     "start[%1d]/D",ndim);    
   sprintf(cstop,      "stop[%1d]/D",ndim);     
   
   sprintf(cfrequency, "frequency[%1d]/F",ndim);
   sprintf(clow,       "low[%1d]/F",ndim);      
   sprintf(chigh,      "high[%1d]/F",ndim);     
   sprintf(cbandwidth, "bandwidth[%1d]/F",ndim);

   if(ndim<5) sprintf(chrss,      "hrss[%1d]/D",ndim*2);     
   else       sprintf(chrss,      "hrss[%2d]/D",ndim*2);     

   sprintf(cnoise,     "noise[%1d]/D",ndim);     
   sprintf(cnull,      "null[%1d]/F",ndim); 
   sprintf(cnill,      "nill[%1d]/F",ndim); 
   sprintf(cDeff,      "Deff[%1d]/F",ndim); 
   sprintf(crange,     "range[2]/F"); 
   sprintf(ceBBH,      "eBBH[4]/F"); 
   sprintf(cchirp,     "chirp[6]/F");
   sprintf(cmass,      "mass[2]/F"); 
   sprintf(cspin,      "spin[6]/F");
   sprintf(csnr,       "snr[%1d]/F",ndim);    
   sprintf(csSNR,      "sSNR[%1d]/F",ndim); 
   sprintf(cxSNR,      "xSNR[%1d]/F",ndim); 
   sprintf(ciSNR,      "iSNR[%1d]/F",ndim);
   sprintf(coSNR,      "oSNR[%1d]/F",ndim);
   sprintf(cioSNR,     "ioSNR[%1d]/F",ndim);
   
 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("ndim",        &ndim,        "ndim/I");
   waveTree->Branch("run",         &run,         "run/I");
   waveTree->Branch("nevent",      &nevent,      "nevent/I");
   waveTree->Branch("eventID",     eventID,      "eventID[2]/I");
   waveTree->Branch("type",        type,         "type[2]/I");
   waveTree->Branch("name",        name);
   waveTree->Branch("log",         log);
   waveTree->Branch("rate",        rate,          crate);
   
   waveTree->Branch("usize",       &usize,       "usize/I");
   waveTree->Branch("volume",      volume,        cvolume);
   waveTree->Branch("size",        size,          csize);
   
   waveTree->Branch("gap",         gap,           cgap);
   waveTree->Branch("lag",         lag,           clag);
   waveTree->Branch("slag",        slag,          cslag);
   waveTree->Branch("strain",      strain,       "strain[2]/D");
   waveTree->Branch("phi",         phi,          "phi[4]/F");
   waveTree->Branch("theta",       theta,        "theta[4]/F");
   waveTree->Branch("psi",         psi,          "psi[2]/F");
   waveTree->Branch("iota",        iota,         "iota[2]/F");
   waveTree->Branch("bp",          bp,            cbp);
   waveTree->Branch("bx",          bx,            cbx);
    
   waveTree->Branch("time",        time,          ctime);
   waveTree->Branch("gps",         gps,           cgps);  
   waveTree->Branch("left",        left,          cleft);
   waveTree->Branch("right",       right,         cright);
   waveTree->Branch("start",       start,         cstart);
   waveTree->Branch("stop",        stop,          cstop);
   waveTree->Branch("duration",    duration,      cduration);
   
   waveTree->Branch("frequency",   frequency,     cfrequency);
   waveTree->Branch("low",         low,           clow);
   waveTree->Branch("high",        high,          chigh);
   waveTree->Branch("bandwidth",   bandwidth,     cbandwidth);
   
   waveTree->Branch("hrss",        hrss,          chrss);
   waveTree->Branch("noise",       noise,         cnoise);
   //waveTree->Branch("ndm",         ndm,           cndm);
   waveTree->Branch("erA",         erA,          "erA[11]/F");

   waveTree->Branch("Psm","skymap",&Psm,         32000,0);          

   waveTree->Branch("null",        null,          cnull);
   waveTree->Branch("nill",        nill,          cnill);
   waveTree->Branch("netcc",       netcc,        "netcc[4]/F");
   waveTree->Branch("neted",       neted,        "neted[5]/F");
   waveTree->Branch("rho",         rho,          "rho[2]/F");

   waveTree->Branch("gnet",        &gnet,        "gnet/F");
   waveTree->Branch("anet",        &anet,        "anet/F");
   waveTree->Branch("inet",        &inet,        "inet/F");
   waveTree->Branch("ecor",        &ecor,        "ecor/F");
   waveTree->Branch("norm",        &norm,        "norm/F");
   waveTree->Branch("ECOR",        &ECOR,        "ECOR/F");
   waveTree->Branch("penalty",     &penalty,     "penalty/F");
   waveTree->Branch("likelihood",  &likelihood,  "likelihood/F");

   waveTree->Branch("factor",      &factor,      "factor/F");
   waveTree->Branch("chirp",       chirp,         cchirp);
   waveTree->Branch("range",       range,         crange);
   waveTree->Branch("eBBH",        eBBH,          ceBBH);
   waveTree->Branch("Deff",        Deff,          cDeff);
   waveTree->Branch("mass",        mass,          cmass);
   waveTree->Branch("spin",        spin,          cspin);

   waveTree->Branch("snr",         snr,           csnr);
   waveTree->Branch("xSNR",        xSNR,          cxSNR);
   waveTree->Branch("sSNR",        sSNR,          csSNR);
   waveTree->Branch("iSNR",        iSNR,          ciSNR);
   waveTree->Branch("oSNR",        oSNR,          coSNR);
   waveTree->Branch("ioSNR",       ioSNR,         cioSNR);

   return waveTree;
}

netevent& netevent::operator=(const netevent& a)
{
   int i,j;

   if(ndim != a.ndim) { ndim=a.ndim; this->allocate(); }

   ifoList=      a.ifoList;  

   run=          a.run;
   nevent=       a.nevent;
   eventID[0]=   a.eventID[0];
   eventID[1]=   a.eventID[1];
   type[0]=      a.type[0];
   type[1]=      a.type[1];
   *name=        *a.name;
   *log=         *a.log;
   strain[0]=    a.strain[0];
   strain[1]=    a.strain[1];
   usize=        a.usize;
   psi[0]=       a.psi[0];             
   psi[1]=       a.psi[1];             
   iota[0]=      a.iota[0];             
   iota[1]=      a.iota[1];             
   mass[0]=      a.mass[0];             
   mass[1]=      a.mass[1];             
   gnet=         a.gnet;             
   anet=         a.anet;             
   inet=         a.inet;             
   ecor=         a.ecor;             
   norm=         a.norm;             
   ECOR=         a.ECOR;             
   penalty=      a.penalty;             
   likelihood=   a.likelihood;             
   factor=       a.factor;
   Psave=        a.Psave;
   *Psm =        *(a.Psm);

   for(i=0; i<6; i++) spin[i] = a.spin[i];
   for(i=0; i<4; i++) netcc[i] = a.netcc[i];             
   for(i=0; i<5; i++) neted[i] = a.neted[i];             
   for(i=0; i<2; i++) rho[i] = a.rho[i];             
   for(i=0; i<4; i++) theta[i] = a.theta[i];             
   for(i=0; i<4; i++) phi[i] = a.phi[i];             
   for(i=0; i<11; i++) erA[i] = a.erA[i];             
   for(i=0; i<2; i++) range[i] = a.range[i];             
   for(i=0; i<4; i++) eBBH[i] = a.eBBH[i];             
   for(i=0; i<6; i++) chirp[i] = a.chirp[i];             

   lag[ndim]=          a.lag[ndim];
   slag[ndim]=         a.slag[ndim];
   for(i=0; i<ndim; i++){

      gps[i]=          a.gps[i];   
      
      rate[i]=         a.rate[i];
      
      gap[i]=          a.gap[i];
      lag[i]=          a.lag[i];
      slag[i]=         a.slag[i];
      volume[i]=       a.volume[i];
      size[i]=         a.size[i];
      bp[i]=           a.bp[i];             
      bp[i+ndim]=      a.bp[i+ndim];             
      bx[i]=           a.bx[i];             
      bx[i+ndim]=      a.bx[i+ndim];             
            
      time[i]=         a.time[i];
      time[i+ndim]=    a.time[i+ndim];
      right[i]=        a.right[i];
      left[i]=         a.left[i];
      duration[i]=     a.duration[i];
      start[i]=        a.start[i];
      stop[i]=         a.stop[i];
      
      frequency[i]=    a.frequency[i];
      low[i]=          a.low[i];
      high[i]=         a.high[i];
      bandwidth[i]=    a.bandwidth[i];
      
      hrss[i]=         a.hrss[i];
      hrss[i+ndim]=    a.hrss[i+ndim];
      noise[i]=        a.noise[i];
      null[i]=         a.null[i];             
      nill[i]=         a.nill[i];             
      Deff[i]=         a.Deff[i];

      snr[i]=          a.snr[i];
      sSNR[i]=         a.sSNR[i];             
      xSNR[i]=         a.xSNR[i];
      iSNR[i]=         a.iSNR[i];
      oSNR[i]=         a.oSNR[i];
      ioSNR[i]=        a.ioSNR[i];

   }
   return *this;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++
// output network events for 2G analysis
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void netevent::output2G(TTree* waveTree, network* net, size_t ID, int LAG, double factor)
{
// output event ID in lag LAG to root file
   if(!net || !net->nLag) exit(1);

   int i,j,k,ind,kid;
   int n,m,K,M,injID;
   detector* pd;
   netcluster* p;
   int nIFO = (int)net->ifoListSize();   	// number of detectors
   wavecomplex Aa, gC;
   double Pi = 3.14159265358979312;
   this->factor = fabs(factor); 		// used to tag events in root file
   factor = factor<=0 ? 1 : fabs(factor); 	// used to rescale injected events
   double inRate = net->getifo(0)->rate;	// get original rate
   bool pat0 = net->pattern==0 ? true : false;  // pattern flag 

   if(!nIFO) return;

// injections

   injection INJ(nIFO);
   size_t mdcID, id;
   double T, injTime, TAU, pcc;
   double x, ndmLike, Etot;
   skymap* psm;
   netcluster* pwc;
   std::vector<float>* vP;
   std::vector<int>*   vI;

   bool ellips = net->tYPe=='i' || net->tYPe=='I' || 
                 net->tYPe=='g' || net->tYPe=='G' || 
                 net->tYPe=='s' || net->tYPe=='S' ||
                 net->tYPe=='r' || net->tYPe=='R';

   bool burst  = net->tYPe=='b' || net->tYPe=='B';  

 // arrays for cluster parameters
      
   wavearray<double> clusterID_net;
   wavearray<double> vol0_net;
   wavearray<double> vol1_net;
   wavearray<double> size_net;
   wavearray<double> start_net;
   wavearray<double> stop_net;
   wavearray<double> low_net;
   wavearray<double> high_net;
   wavearray<double> noise_net;
   wavearray<double> NOISE_net;
   wavearray<double> cFreq_net;
   wavearray<double> rate_net;
   wavearray<double> duration_net;
   wavearray<double> bandwidth_net;

   // why do we need all this?

   this->ifoList = net->ifoList;  // add detectors to tree user info
   if(waveTree!=NULL) {
     int dsize = waveTree->GetUserInfo()->GetSize();
     if(dsize!=0 && dsize!=nIFO) {
        cout<<"netevent::output2G(): wrong user detector list in header tree"<<endl; exit(1);
     }  
     if(dsize==0) { 
       for(int n=0;n<nIFO;n++) {
         // must be done a copy because detector object 
         // is destroyed when waveTree is destroyed 
         detector* pD = new detector(*net->getifo(n));  
         waveTree->GetUserInfo()->Add(pD);  
       }
     }
   }

   this->run = net->nRun;
   this->nevent = 0;

   pwc = net->getwc(LAG);                           // pointer to netcluster
   if(!pwc->size()) return;

   clusterID_net = pwc->get((char*)"ID",0,'S',0);
   K = clusterID_net.size();
   kid = -1;
   for(k=0; k<K; k++) {                           // find cluster
      id = size_t(clusterID_net.data[k]+0.1);
      if(ID&&(ID==id)) {kid=k; break;} 
   }
   if(kid<0) return;
            
// read cluster parameters
      
   start_net.resize(0);
   stop_net.resize(0);
   noise_net.resize(0);
   NOISE_net.resize(0);

   rate_net      = pwc->get((char*)"rate",0,'R',0);
   vol0_net      = pwc->get((char*)"volume",0,'R',0);     // stored in volume[0]
   vol1_net      = pwc->get((char*)"VOLUME",0,'R',0);     // stored in volume[1]
   size_net      = pwc->get((char*)"size",0,'R',0);       // stored in size[0]
   low_net       = pwc->get((char*)"low",0,'R',0);
   high_net      = pwc->get((char*)"high",0,'R',0);
   cFreq_net     = pwc->get((char*)"freq",0,'L',0,false);
   duration_net  = pwc->get((char*)"duration",0,'L',0,false);
   bandwidth_net = pwc->get((char*)"bandwidth",0,'L',0,false);
   
   for(i=1; i<=int(nIFO); i++) {                          // loop on detectors
      start_net.append(pwc->get((char*)"start",i,'L',0));
      stop_net.append(pwc->get((char*)"stop",i,'L',0));
      noise_net.append(pwc->get((char*)"noise",i,'S',0));
      NOISE_net.append(pwc->get((char*)"NOISE",i,'S',0));
   }	 

   psm = &(net->getifo(0)->tau);
   vI = &(net->wc_List[LAG].p_Ind[ID-1]);
   ind = (*vI)[0];                                         // reconstructed sky index
   
   for(i=0; i<int(nIFO); i++)                              // store gps time of data intervals  
      this->gps[i] = pwc->start+(slag[i]-slag[0]);

   clusterdata* pcd = &(pwc->cData[ID-1]); 

   this->ndim = nIFO;
   psm->gps  = pcd->cTime + this->gps[0];
   this->ecor = pcd->netecor;
   this->nevent += 1;
   this->eventID[0] = ID;
   this->eventID[1] = 0;
   this->iota[0] = pcd->iota;                       // ellipticity : 2*cos(iota)/(1+cos(iota)^2)
   //this->iota[0] = pcd->skyChi2;                    // sky chi2 for all resolutions (temporary)
   this->psi[0] = pcd->psi;
   //this->psi[0] = pcd->Gnoise;                      // estimated Gaussian noise (temporary)  
   this->phi[0] = psm->getPhi(ind);                 // reconstructed phi
   this->phi[2] = psm->getRA(ind);
   this->phi[3] = pcd->phi;                         // detection phi 
   this->theta[0] = psm->getTheta(ind);             // reconstructed theta 
   this->theta[2] = psm->getDEC(ind);
   this->theta[3] = pcd->theta;                     // detection theta
   this->gnet = pcd->gNET;
   this->anet = pcd->aNET;
   this->inet = pcd->iNET;                          // network index
   this->norm = pcd->norm;                          // packet norm
   this->likelihood = pcd->likenet;                 // total likelihood in sky loop
   this->volume[0] = int(vol0_net.data[kid]+0.5);   // event volume
   this->volume[1] = int(vol1_net.data[kid]+0.5);   // selected pixels volume
   this->size[0] = int(size_net.data[kid]+0.5);     // event size
   this->size[1] = pcd->skySize;                    // signal size
   this->chirp[1] = pcd->mchirp;                    // reconstructed chirp mass
   this->chirp[2] = pcd->mchirperr;                 // reconstructed chirp mass error
   this->chirp[3] = pcd->chirpEllip;                // ellipticity parameter
   this->chirp[4] = pcd->chirpPfrac;                // pixel fraction
   this->chirp[5] = pcd->chirpEfrac;                // energy fraction
   //this->chirp[6] = pcd->chi2chirp;                 // chirp chi2/DoF
   //this->chirp[7] = pcd->tmrgr;                     // t merger parameter
   //this->chirp[8] = pcd->tmrgrerr;                  // t merger error
      
   this->range[0] = 0.;                             // reconstructed distance to source
   
   TAU = psm->get(this->theta[0],this->phi[0]);
   
   M = 0; gC = 0.;
   this->strain[0] = 0.;
   this->penalty = 0.;
   
   for(i=0; i<5; i++) { this->neted[i] = 0.; }          // zero neted array
   
   this->lag[nIFO] = pwc->shift;

   net->getMRAwave(ID,LAG,'s',net->optim?1:0);    	// get signal strain

   for(i=0; i<int(nIFO); i++) {                         // loop on detectors
      pd = net->getifo(i);
      Aa = pd->antenna(this->theta[0],this->phi[0],this->psi[0]);    // antenna pattern
      m = i*K+kid;
      
      this->type[0] =         1;
      this->rate[i] =         net->optim ? int(rate_net.data[kid]+0.1) : 0;
      
      this->gap[i] =          0.;
      this->lag[i] =          pd->lagShift.data[LAG];
      
      this->snr[i]  =         pd->enrg;
      this->nill[i] =         pd->xSNR-pd->sSNR;
      this->null[i] =         pd->null;                    // null per detector
      this->xSNR[i] =         pd->xSNR;
      this->sSNR[i] =         pd->sSNR;
      this->time[i] =         pcd->cTime + this->gps[i]; 

      if(i) {                 // take delays into account
         psm = &(net->getifo(i)->tau);
         this->time[i] += psm->get(this->theta[0],this->phi[0])-TAU;
      }
      
      this->left[i] =         start_net.data[m];
      this->right[i] =        pwc->stop-pwc->start-stop_net.data[m];
      this->duration[i] =     stop_net.data[m] - start_net.data[m];
      this->start[i] =        start_net.data[m] + this->gps[i]; 
      this->stop[i] =         stop_net.data[m] + this->gps[i]; 
     
      // take lag shift into account
      double xstart = this->gps[i]+net->Edge;  // start data
      double xstop  = this->gps[i]+pwc->stop-pwc->start-net->Edge;  // end data
      this->time[i] += lag[i];
      if(this->time[i]>xstop) this->time[i] = xstart+(this->time[i]-xstop);  // circular buffer
 
      this->frequency[i] =    cFreq_net.data[k];
      this->low[i] =          low_net.data[k];
      this->high[i] =         high_net.data[k];
      this->bandwidth[i] =    high_net.data[k] - low_net.data[k];
      
      this->hrss[i] =         sqrt(pd->get_SS()/inRate);
      this->noise[i] =        pow(10.,noise_net.data[m])/sqrt(inRate);
      this->bp[i]=            Aa.real();
      this->bx[i]=            Aa.imag();
      this->strain[0] +=      this->hrss[i]*this->hrss[i];

      Aa /= pow(10.,NOISE_net.data[m]);
      gC += Aa*Aa;
	    
       psm->gps  = pcd->cTime+this->gps[0]; 
   }

   // Fix start,stop,duration,left,right when simulation!=0.
   // Due to circular buffer the events detected on the edges 
   // of the segment could have a wrong start,stop values
   // The start,stop of det with lag=0 are used to fix the wrong values of the other detectors
   int mdet=0; for(i=0; i<int(nIFO); i++) if(this->lag[i]==0) mdet=i;
   for(i=0; i<int(nIFO); i++) {
      if(this->duration[i]!=this->duration[mdet]) {
	 double xstart = this->gps[i]+net->Edge;  // start data
	 double xstop  = this->gps[i]+pwc->stop-pwc->start-net->Edge;  // end data
	 
	 this->start[i]    = this->start[mdet]+this->lag[i]+(slag[i]-slag[mdet]);
	 if(this->start[i]>xstop) this->start[i] = xstart+(this->start[i]-xstop);  // circular buffer
	 
	 this->stop[i]    = this->stop[mdet]+this->lag[i]+(slag[i]-slag[mdet]);
	 if(this->stop[i]>xstop) this->stop[i] = xstart+(this->stop[i]-xstop);  // circular buffer
	 
	 this->duration[i] = this->duration[mdet];
	 this->left[i]     = this->start[i]-pwc->start; 
	 this->right[i]    = pwc->stop-this->stop[i]; 
      }
   }
   this->duration[0]  = duration_net.data[kid];
   this->bandwidth[0] = bandwidth_net.data[kid];
   this->frequency[0] = pcd->cFreq;

   ind = pwc->sArea[ID-1].size(); 
   for(i=0; i<11; i++) 
      this->erA[i] = i<ind ? pwc->sArea[ID-1][i] : 0.;

   this->ECOR     = pcd->normcor;                         // normalized coherent energy
   this->netcc[0] = pcd->netcc;                           // MRA or SRA cc statistic 
   this->netcc[1] = pcd->skycc;                           // all-resolution cc statistic
   this->netcc[2] = pcd->subnet;                          // MRA or SRA sub-network statistic 
   this->netcc[3] = pcd->SUBNET;                          // all-resolution sub-network statistic

   this->neted[0] = pcd->netED;                           // network ED
   this->neted[1] = pcd->netnull;                         // total null energy with Gaussian bias correction
   this->neted[2] = pcd->energy;                          // total event energy
   this->neted[3] = pcd->likesky;                         // total likelihood at all resolutions
   this->neted[4] = pcd->enrgsky;                         // total energy at all resolutions
   //this->neted[5] = pcd->Gnoise;                        // estimated contribution of Gaussian noise
   //this->neted[6] = pcd->skyChi2;                       // sky chi2 for all resolutions

   double chrho   = this->chirp[3]*sqrt(this->chirp[5]);  // reduction factor for chirp events
   if(pcd->netRHO>=0) {		// original 2G
     this->rho[0]   = pcd->netRHO;                        // reduced coherent SNR per detector
     this->rho[1]   = pat0 ?pcd->netrho:pcd->netRHO*chrho;// reduced coherent SNR per detector for chirp events
   } else {			// (XGB.rho0) 
     this->rho[0]   = -pcd->netRHO;                       // reduced coherent SNR per detector
     this->rho[1]   = pcd->netrho;                        // reduced coherent SNR per detector //GV original 2G rho, only for tests
   }
   
   this->strain[0] = sqrt(this->strain[0]);
   if(!ellips) this->psi[0] = gC.arg()*180/Pi;
   this->penalty = pcd->netnull/nIFO;
   this->penalty /= pat0 ? this->size[0] : pcd->nDoF;     // cluster chi2/nDoF    
   
   
// set injections if there are any

   M = net->mdc__IDSize();
   if(!LAG) {                                              // only for zero lag
      injTime = 1.e12;
      injID   = -1;
      for(m=0; m<M; m++) {
         mdcID = net->getmdc__ID(m);
         T = fabs(this->time[0] - net->getmdcTime(mdcID));
         if(T<injTime && INJ.fill_in(net,mdcID)) { 
            injTime = T; 
            injID = mdcID; 
         } 
         //	     printf("%d  %12.3f  %12.3f\n",mdcID,net->getmdcTime(mdcID),T);
      }
      
      if(INJ.fill_in(net,injID)) {                   // set injections
         this->range[1]  = INJ.distance/factor;
         this->chirp[0]  = INJ.mchirp;
         this->eBBH[1]   = INJ.e0;
         this->eBBH[2]   = INJ.rp0;
         this->eBBH[3]   = INJ.redshift;
         this->strain[1] = INJ.strain*factor;
         this->type[1]   = INJ.type;
         *this->name     = net->getmdcType(this->type[1]-1);
         *this->log      = net->getmdcList(injID);
         this->log->erase(std::remove(this->log->begin(), this->log->end(), '\n'), this->log->end()); // remove new line
         this->theta[1]  = INJ.theta[0];
         this->phi[1]    = INJ.phi[0];
         this->psi[1]    = INJ.psi[0];
        // this->iota[1]   = 2.*INJ.iota[0]/(1+INJ.iota[0]*INJ.iota[0]);  // ellipticity
         this->iota[1]   = INJ.iota[1];  // cos(iota)
         this->mass[0]   = INJ.mass[0];
         this->mass[1]   = INJ.mass[1];
         
         for(i=0; i<6; i++) this->spin[i] = INJ.spin[i];
         
//	     printf("injection type: %d  %12.3f\n",INJ.type,INJ.time[0]);
         
         wavearray<double>** pwfINJ = new wavearray<double>*[nIFO];  
         wavearray<double>** pwfREC = new wavearray<double>*[nIFO];  
         pd = net->getifo(0);  
         int idSize = pd->RWFID.size();  
         int wfIndex=-1;  
         for (int mm=0; mm<idSize; mm++) if (pd->RWFID[mm]==ID) wfIndex=mm;  
 
         for(j=0; j<nIFO; j++) {
            pd = net->getifo(j);
            Aa = pd->antenna(this->theta[1],this->phi[1],this->psi[1]);    // inj antenna pattern
            this->hrss[j+nIFO] = INJ.hrss[j]*factor;
            this->bp[j+nIFO]   = Aa.real();
            this->bx[j+nIFO]   = Aa.imag();
            this->time[j+nIFO] = INJ.time[j];
            this->Deff[j]   = INJ.Deff[j]/factor;
            
            pwfINJ[j] = INJ.pwf[j];
            if (pwfINJ[j]==NULL) {
               cout << "Error : Injected waveform not saved !!! : detector "
                    << net->ifoName[j] << endl;
               continue;
            }
            if (wfIndex<0) {
               cout << "Error : Reconstructed waveform not saved !!! : ID -> "
                    << ID << " : detector " << net->ifoName[j] << endl;
               continue;
            }
            
            if (wfIndex>=0) pwfREC[j] = pd->RWFP[wfIndex];
            double R = pd->TFmap.rate();
            //double rFactor = log(2.)/log(pd->rate/R);  // rescale waveform
            double rFactor = 1.;
            rFactor *= factor;
            wavearray<double>* wfINJ = pwfINJ[j];
            *wfINJ*=rFactor;
            wavearray<double>* wfREC = pwfREC[j];
            
            double bINJ = wfINJ->start();
            double eINJ = wfINJ->start()+wfINJ->size()/R;
            double bREC = wfREC->start();
            double eREC = wfREC->start()+wfREC->size()/R;
            //cout.precision(14);
            //cout << "bINJ : " << bINJ << " eINJ : " << eINJ << endl;
            //cout << "bREC : " << bREC << " eREC : " << eREC << endl;
            
            int oINJ = bINJ>bREC ? 0 : int((bREC-bINJ)*R+0.5);
            int oREC = bINJ<bREC ? 0 : int((bINJ-bREC)*R+0.5);
            //cout << "oINJ : " << oINJ << " oREC : " << oREC << endl;
            
            double startXCOR = bINJ>bREC ? bINJ : bREC;
            double endXCOR   = eINJ<eREC ? eINJ : eREC;
            int sizeXCOR   = int((endXCOR-startXCOR)*R+0.5);
            //cout << "startXCOR : " << startXCOR << " endXCOR : " << endXCOR << " sizeXCOR :" << sizeXCOR << endl;
            
            if (sizeXCOR<=0) {*wfINJ*=1./rFactor; continue;}
            
            // the enINJ, enREC, xcorINJ_REC are computed in the INJ range
            
            double enINJ=0;
            for (int i=0;i<wfINJ->size();i++) enINJ+=wfINJ->data[i]*wfINJ->data[i];
            //for (int i=0;i<sizeXCOR;i++) enINJ+=wfINJ->data[i+oINJ]*wfINJ->data[i+oINJ];
            
            double enREC=0;
            for (int i=0;i<wfREC->size();i++) enREC+=wfREC->data[i]*wfREC->data[i];
            //for (int i=0;i<sizeXCOR;i++) enREC+=wfREC->data[i+oREC]*wfREC->data[i+oREC];
            
            double xcorINJ_REC=0;
            for (int i=0;i<sizeXCOR;i++) xcorINJ_REC+=wfINJ->data[i+oINJ]*wfREC->data[i+oREC];
            
            WSeries<double> wfREC_SUB_INJ;  
            wfREC_SUB_INJ.resize(sizeXCOR);
            for (int i=0;i<sizeXCOR;i++) wfREC_SUB_INJ.data[i]=wfREC->data[i+oREC]-wfINJ->data[i+oINJ];
            wfREC_SUB_INJ.start(startXCOR);
            wfREC_SUB_INJ.rate(wfREC->rate());
            
            this->iSNR[j]    = enINJ;
            this->oSNR[j]    = enREC;
            this->ioSNR[j]   = xcorINJ_REC;
            
            //double erINJ_REC = enINJ+enREC-2*xcorINJ_REC;
            //cout << "enINJ : " << enINJ << " enREC : " << enREC << " xcorINJ_REC : " << xcorINJ_REC << endl;
            //cout << "erINJ_REC/enINJ : " << erINJ_REC/enINJ << endl;
            
            *wfINJ*=1./rFactor;
         }
         delete [] pwfINJ;
         delete [] pwfREC;
      }
      else {                   // no injections
         this->range[1]  = 0.;
         this->chirp[0]  = 0.;
         this->eBBH[1]   = 0.;
         this->eBBH[2]   = 0.;
         this->eBBH[3]   = 0.;
         this->strain[1] = 0.;
         this->type[1]   = 0;
         this->theta[1]  = 0.;
         this->phi[1]    = 0.;
         this->psi[1]    = 0.;
         this->iota[1]   = 0.;
         this->mass[0]   = 0.;
         this->mass[1]   = 0.;
         
         for(i=0; i<6; i++) this->spin[i] = 0.;
         
         for(j=0; j<nIFO; j++) {
            this->hrss[j+nIFO] = 0.;
            this->bp[j+nIFO]   = 0.;
            this->bx[j+nIFO]   = 0.;
            this->time[j+nIFO] = 0.;
            this->Deff[j]   = 0.;
            this->iSNR[j]   = 0.;
            this->oSNR[j]   = 0.;
            this->ioSNR[j]  = 0.;
         }
      }
   }
   
   if(this->fP!=NULL) { 
      fprintf(fP,"\n# trigger %d in lag %d for \n",int(ID),int(LAG));    	 
      this->Dump("2G");
      vP = &(net->wc_List[LAG].p_Map[ID-1]);
      vI = &(net->wc_List[LAG].p_Ind[ID-1]);
      x = cos(psm->theta_1*PI/180.)-cos(psm->theta_2*PI/180.);
      x*= (psm->phi_2-psm->phi_1)*180/PI/psm->size();
      fprintf(fP,"sky_res:    %f\n",sqrt(fabs(x)));
      fprintf(fP,"map_lenght: %d\n",int(vP->size()));
      fprintf(fP,"#skyID  theta   DEC     step   phi     R.A    step  probability    cumulative\n");
      x = 0;
      for(j=0; j<int(vP->size()); j++) {
         i = (*vI)[j];
	 if(net->mdc__IDSize()) {			// simulation mode
	   x = (j==int(vP->size())-1) ? 0 : x+(*vP)[j];	// last value is the inj sky index (x=0)
         } else x+=(*vP)[j];
         fprintf(fP,"%6d  %5.1f  %5.1f  %6.2f  %5.1f  %5.1f  %6.2f  %e  %e\n",
                 int(i),psm->getTheta(i),psm->getDEC(i),psm->getThetaStep(i),
                 psm->getPhi(i),psm->getRA(i),psm->getPhiStep(i),(*vP)[j],x); 
      }
   }
   
   // save the probability skymap to tree
   if(waveTree!=NULL && net->wc_List[LAG].p_Map.size()) {

      vP = &(net->wc_List[LAG].p_Map[ID-1]);
      vI = &(net->wc_List[LAG].p_Ind[ID-1]);
      if(this->Psave) {

         *Psm = *psm; 
         *Psm = 0.;

         int k;
         double th,ph;
         for(j=0; j<int(vP->size()); j++) {
            i = (*vI)[j];
            th = Psm->getTheta(i);
            ph = Psm->getPhi(i);
            k=Psm->getSkyIndex(th, ph);
            Psm->set(k,(*vP)[j]);
         }
      }
   }
   
   if(waveTree!=NULL) waveTree->Fill();
   
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++
// output network events
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void netevent::output(TTree* waveTree, network* net, double factor, size_t iID, int LAG)
{
  if(!net || !net->nLag) exit(1);

   int i,j,k,ind;
   int n,m,K,M,injID;
   detector* pd;
   netcluster* p;
   int N = (int)net->ifoListSize();   		// number of detectors
   int bLag = LAG<0 ? 0 : LAG;
   int eLag = LAG<0 ? net->nLag : LAG+1;
   wavecomplex Aa, gC;
   double Pi = 3.14159265358979312;
   this->factor = fabs(factor); 		// used to tag events in root file
   factor = factor<=0 ? 1 : fabs(factor); 	// used to rescale injected events
   double inRate = net->getifo(0)->rate;        // get original rate

   if(!N) return;
   double* ndm = (double*)malloc(N*N*sizeof(double));
   int iTYPE = net->MRA ? 0 : 1;

// injections

   injection INJ(N);
   size_t mdcID, ID;
   double T, injTime, TAU, pcc;
   double x, tot_null, ndmLike, Etot;
   skymap* psm;
   std::vector<float>* vP;
   std::vector<int>*   vI;

   bool ellips = net->tYPe=='i' || net->tYPe=='I' || 
                 net->tYPe=='g' || net->tYPe=='G' || 
                 net->tYPe=='s' || net->tYPe=='S' ||
                 net->tYPe=='r' || net->tYPe=='R';

   bool burst  = net->tYPe=='b' || net->tYPe=='B';  

   wavearray<double> skSNR(N);
   wavearray<double> xkSNR(N);

// arrays for cluster parameters
      
   wavearray<double> clusterID_net;
   wavearray<double> volume_net;
   wavearray<double> size_net;
   wavearray<double> start_net;
   wavearray<double> stop_net;
   wavearray<double> time_net;
   wavearray<double> frequency_net;
   wavearray<double> low_net;
   wavearray<double> high_net;
   wavearray<double> LH_net;
   wavearray<double> null_net;
   wavearray<double> rSNR_net;
   wavearray<double> gSNR_net;
   wavearray<double> gSNR_NET;
   wavearray<double> rate_net;
   wavearray<double> hrss_net;
   wavearray<double> hrss_NET;
   wavearray<double> noise_net;
   wavearray<double> NOISE_net;
   wavearray<double> psi_net;
   wavearray<double> ell_net;
   wavearray<double> phi_net;
   wavearray<double> theta_net;
   wavearray<double> energy_net;
   wavearray<double> energy_NET;

   this->ifoList = net->ifoList;
   // add detectors to tree user info
   if(waveTree!=NULL) {
     for(int n=0;n<N;n++) {
       // must be done a copy because detector object 
       // is destroyed when waveTree is destroyed 
       detector* pD = new detector(*net->getifo(n));  
       waveTree->GetUserInfo()->Add(pD);  
     }
   }

   this->run = net->nRun;
   this->nevent = 0;

   for(n=bLag; n<eLag; n++){  // loop on time lags

      p = net->getwc(n);                           // pointer to netcluster
      if(!p->size()) continue;

      clusterID_net = p->get((char*)"ID",0,'S',iTYPE);
      K = clusterID_net.size();

      if(!K) continue;
            
// read cluster parameters
      
      time_net.resize(0);
      start_net.resize(0);
      stop_net.resize(0);
      frequency_net.resize(0);
      energy_net.resize(0);
      energy_NET.resize(0);
      rSNR_net.resize(0);
      gSNR_net.resize(0);
      gSNR_NET.resize(0);
      hrss_net.resize(0);
      hrss_NET.resize(0);
      noise_net.resize(0);
      NOISE_net.resize(0);
      null_net.resize(0);

      LH_net     = p->get((char*)"likelihood",0,'R',iTYPE);
      rate_net   = p->get((char*)"rate",0,'R',iTYPE);
      ell_net    = p->get((char*)"ellipticity",0,'R',iTYPE);
      psi_net    = p->get((char*)"psi",0,'R',iTYPE);
      phi_net    = p->get((char*)"phi",0,'R',iTYPE);
      theta_net  = p->get((char*)"theta",0,'R',iTYPE);
      size_net   = p->get((char*)"size",0,'R',iTYPE);
      volume_net = p->get((char*)"volume",0,'R',iTYPE);
      low_net    = p->get((char*)"low",0,'R',iTYPE);
      high_net   = p->get((char*)"high",0,'R',iTYPE);

      for(i=1; i<int(N+1); i++)                              // loop on detectors
      {                                                 // read cluster parameters	 
	 time_net.append(p->get((char*)"time",i,'L',0));
	 start_net.append(p->get((char*)"start",i,'L',iTYPE));
	 stop_net.append(p->get((char*)"stop",i,'L',iTYPE));
	 frequency_net.append(p->get((char*)"FREQUENCY",i,'L',0));
	 rSNR_net.append(p->get((char*)"SNR",i,'R',iTYPE));
	 gSNR_net.append(p->get((char*)"SNR",i,'S',iTYPE));
	 gSNR_NET.append(p->get((char*)"SNR",i,'P',iTYPE));
	 hrss_net.append(p->get((char*)"hrss",i,'W',iTYPE));
	 hrss_NET.append(p->get((char*)"hrss",i,'U',iTYPE));
	 noise_net.append(p->get((char*)"noise",i,'S',iTYPE));
	 NOISE_net.append(p->get((char*)"NOISE",i,'S',iTYPE));
	 energy_net.append(p->get((char*)"energy",i,'S',iTYPE));
	 energy_NET.append(p->get((char*)"energy",i,'P',iTYPE));
	 null_net.append(p->get((char*)"null",i,'W',iTYPE));
      }	 

      if(ellips) { 
         energy_net += energy_NET; 
         energy_net *= net->MRA ? 1.0 : 0.5; 
      }

      for(k=0; k<K; k++)               // loop on events
      {
 	 ID = size_t(clusterID_net.data[k]+0.1);
         if((iID)&&(ID!=iID)) continue; 
	 if(ellips) { 
	   net->SETNDM(ID,n,true,net->MRA?0:1);
	 }
      	 else if(burst)
	   net->setndm(ID,n,true,1);
//	 else
//           cout<<"netevent::output(): incorrect search option"<<endl; //exit(1);}

	 psm = &(net->getifo(0)->tau);
	 vI = &(net->wc_List[n].p_Ind[ID-1]);
	 ind = (*vI)[0];               // reconstructed sky index

	 this->ndim = N;
         for(i=0; i<N; i++)            // loop on detectors  
           this->gps[i] = net->getifo(i)->getTFmap()->start()+(slag[i]-slag[0]);
	 psm->gps  = time_net.data[k]+this->gps[0];
	 this->ecor = net->eCOR;
	 this->nevent += 1;
	 this->eventID[0] = ID;
	 this->eventID[1] = n;
	 this->psi[0] = psi_net.data[k];
	 this->phi[0] = psm->getPhi(ind);                // reconstructed phi
	 this->phi[2] = psm->getRA(ind);
	 this->phi[3] = phi_net.data[k];                 // detection phi 
	 this->theta[0] = psm->getTheta(ind);            // reconstructed theta 
	 this->theta[2] = psm->getDEC(ind);
	 this->theta[3] = theta_net.data[k];             // detection theta
	 this->gnet = net->gNET;
	 this->anet = net->aNET;
	 this->inet = net->iNET;
	 this->iota[0] = net->norm;
	 this->norm = net->norm;
	 this->likelihood = 0.;

	 TAU = psm->get(this->theta[0],this->phi[0]);

	 M = 0; gC = 0.;
	 this->strain[0] = 0.;
	 this->penalty = 0.;
	 tot_null = 0.;
	 
	 Etot = 0.;
	 for(i=0; i<N; i++) Etot += energy_net.data[i*K+k];

	 for(i=0; i<5; i++) { this->neted[i] = 0.; }

	 this->lag[N] = p->shift;

	 for(i=0; i<N; i++)                              // loop on detectors
	 {
	    pd = net->getifo(i);
	    Aa = pd->antenna(this->theta[0],this->phi[0]);    // antenna pattern
	    m = i*K+k;

	    if(i) {      // take delays into account
	      psm = &(net->getifo(i)->tau);
	      time_net.data[m] += psm->get(this->theta[0],this->phi[0])-TAU;
	    }

	    this->type[0] =         1;
	    this->rate[i] =         int(rate_net.data[k]+0.1);
	    
	    this->gap[i] =          0.;
	    this->lag[i] =          pd->lagShift.data[n];
	    
	    this->volume[i] =       int(volume_net.data[k]+0.5);
	    this->size[i] =         int(size_net.data[k]+0.5);
	    
	    this->snr[i]  =         energy_net.data[m];
	    this->nill[i] =         pd->xSNR-pd->sSNR;
	    this->null[i] =         pd->null;
	    this->xSNR[i] =         pd->xSNR;
	    this->sSNR[i] =         pd->sSNR;
	    this->neted[0] +=       fabs(pd->xSNR-pd->sSNR);
	    this->neted[1] +=       fabs(pd->ED[1]);
	    this->neted[2] +=       fabs(pd->ED[2]);
	    this->neted[3] +=       fabs(pd->ED[3]);
	    this->neted[4] +=       fabs(pd->ED[4]);
	    this->likelihood +=     snr[i] - pd->null;
	    
	    this->time[i] =         time_net.data[m] + this->gps[i]; 
	    this->left[i] =         start_net.data[m];
	    this->right[i] =        p->stop-p->start-stop_net.data[m];
	    this->duration[i] =     stop_net.data[m] - start_net.data[m];
	    this->start[i] =        start_net.data[m] + this->gps[i]; 
	    this->stop[i] =         stop_net.data[m] + this->gps[i]; 
	    
	    this->frequency[i] =    frequency_net.data[m];
	    this->low[i] =          low_net.data[k];
	    this->high[i] =         high_net.data[k];
	    this->bandwidth[i] =    high_net.data[k] - low_net.data[k];
	  
            if(net->MRA) { 
	      this->hrss[i] =       pow(pow(10.,hrss_net.data[m]),2)+pow(pow(10.,hrss_NET.data[m]),2);
              this->hrss[i] =       sqrt(this->hrss[i]/inRate);
            } else {   
              this->hrss[i] =       pow(10.,hrss_net.data[m])/sqrt(inRate);
            }
	    this->noise[i] =        pow(10.,noise_net.data[m])/sqrt(inRate);
	    this->bp[i]=            Aa.real();
	    this->bx[i]=            Aa.imag();
	    this->strain[0]+=       this->hrss[i]*this->hrss[i];

	    tot_null += this->null[i];
	    Aa /= pow(10.,NOISE_net.data[m]);
	    gC += Aa*Aa;
	    
	    skSNR.data[i] = pd->sSNR;
	    xkSNR.data[i] = pd->xSNR;

	    x = 1. - pd->sSNR/(this->snr[i]+net->precision*Etot);
	    if(x<this->penalty) this->penalty = x;

	    psm->gps  = time_net.data[m]+this->gps[0]; 

 	 }

	 ndmLike = 0.;
	 this->ECOR = 0.;
	 this->netcc[2] = 0.;
	 this->netcc[3] = 0.;

	 for(i=0; i<N; i++) {                              // loop on detectors
	   for(j=i; j<int(N); j++) { 
	     ndm[M] = net->getNDM(i,j);
 
	     if(i!=j) { 
	       ndm[M] *= 2.;
	       pcc = 2*sqrt(net->getNDM(i,i)*net->getNDM(j,j));
	       pcc = pcc>0. ? ndm[M]/pcc : 0.;
	       this->ECOR += ndm[M]*fabs(pcc);         // effective correlated energy
	       x = skSNR.data[i] + skSNR.data[j];
	       this->netcc[2] += x*pcc;
	       x = xkSNR.data[i] + xkSNR.data[j];
	       this->netcc[3] += x*pcc;
	     }

	     ndmLike += ndm[M++]; 
	   }
	 }

         if(ndmLike>0.) {                           // temporary 2G condition
            ind = p->sArea[ID-1].size(); 
            for(i=0; i<11; i++) 
               this->erA[i] = i<ind ? p->sArea[ID-1][i] : 0.;

	 x = this->ecor;
	 this->netcc[0]  = x/(tot_null+x);          // network (ecor) correlation

	 x = this->ECOR;
	 this->netcc[1]  = x/(tot_null+x);          // network (ECOR) correlation

	 this->netcc[2] /= (N-1)*ndmLike;           // network Pearson's correlation
	 this->netcc[3] /= (N-1)*ndmLike;           // network Pearson's correlation
         }

	 this->rho[0] = sqrt(ecor*this->netcc[0]/N);
	 this->rho[1] = sqrt(ECOR*this->netcc[0]/N);

	 this->strain[0] = sqrt(this->strain[0]);
	 this->penalty = sqrt(1./(1-this->penalty));
	 if(!ellips) this->psi[0] = gC.arg()*180/Pi;

// set injections if there are any

	 M = net->mdc__IDSize();
	 if(!n) {                                         // only for zero lag
	   injTime = 1.e12;
	   injID   = -1;
	   for(m=0; m<M; m++) {
	     mdcID = net->getmdc__ID(m);
	     T = fabs(this->time[0] - net->getmdcTime(mdcID));
     	     if(T<injTime && INJ.fill_in(net,mdcID)) { 
	       injTime = T; 
	       injID = mdcID; 
	     } 
//	     printf("%d  %12.3f  %12.3f\n",mdcID,net->getmdcTime(mdcID),T);
	   }
	
	   if(INJ.fill_in(net,injID)) {                   // set injections
	     this->range[1]  = INJ.distance/factor;
	     this->chirp[0]  = INJ.mchirp;
	     this->eBBH[1]   = INJ.e0;
	     this->eBBH[2]   = INJ.rp0;
	     this->eBBH[3]   = INJ.redshift;
	     this->strain[1] = INJ.strain*factor;
	     this->type[1]   = INJ.type;
             *this->name     = net->getmdcType(this->type[1]-1);
	     this->theta[1]  = INJ.theta[0];
	     this->phi[1]    = INJ.phi[0];
	     this->psi[1]    = INJ.psi[0];
	     this->iota[1]   = INJ.iota[0];
	     this->mass[0]   = INJ.mass[0];
	     this->mass[1]   = INJ.mass[1];

             for(i=0; i<6; i++) this->spin[i] = INJ.spin[i];
 
//	     printf("injection type: %d  %12.3f\n",INJ.type,INJ.time[0]);

             wavearray<double>** pwfINJ = new wavearray<double>*[N];  
             wavearray<double>** pwfREC = new wavearray<double>*[N];  
             pd = net->getifo(0);  
             int idSize = pd->RWFID.size();  
             int wfIndex=-1;  
             for (int mm=0; mm<idSize; mm++) if (pd->RWFID[mm]==ID) wfIndex=mm;  

	     for(j=0; j<int(N); j++) {
	       pd = net->getifo(j);
	       Aa = pd->antenna(this->theta[1],this->phi[1]);    // inj antenna pattern
	       this->hrss[j+N] = INJ.hrss[j]*factor;
	       this->bp[j+N]   = Aa.real();
	       this->bx[j+N]   = Aa.imag();
	       this->time[j+N] = INJ.time[j];
	       this->Deff[j]   = INJ.Deff[j]/factor;
               
               pwfINJ[j] = INJ.pwf[j];
               if (pwfINJ[j]==NULL) {
                 cout << "Error : Injected waveform not saved !!! : detector "
                      << net->ifoName[j] << endl;
                 continue;
               }
               if (wfIndex<0) {
                 cout << "Error : Reconstructed waveform not saved !!! : ID -> "
                      << ID << " : detector " << net->ifoName[j] << endl;
                 continue;
               }

               if (wfIndex>=0) pwfREC[j] = pd->RWFP[wfIndex];
               double R = pd->TFmap.rate();
               //double rFactor = log(2.)/log(pd->rate/R);  // rescale waveform
               double rFactor = 1.;
               rFactor *= factor;
               wavearray<double>* wfINJ = pwfINJ[j];
               *wfINJ*=rFactor;
               wavearray<double>* wfREC = pwfREC[j];

               double bINJ = wfINJ->start();
               double eINJ = wfINJ->start()+wfINJ->size()/R;
               double bREC = wfREC->start();
               double eREC = wfREC->start()+wfREC->size()/R;
               //cout.precision(14);
               //cout << "bINJ : " << bINJ << " eINJ : " << eINJ << endl;
               //cout << "bREC : " << bREC << " eREC : " << eREC << endl;

               int oINJ = bINJ>bREC ? 0 : int((bREC-bINJ)*R+0.5);
               int oREC = bINJ<bREC ? 0 : int((bINJ-bREC)*R+0.5);
               //cout << "oINJ : " << oINJ << " oREC : " << oREC << endl;

               double startXCOR = bINJ>bREC ? bINJ : bREC;
               double endXCOR   = eINJ<eREC ? eINJ : eREC;
               int sizeXCOR   = int((endXCOR-startXCOR)*R+0.5);
               //cout << "startXCOR : " << startXCOR << " endXCOR : " << endXCOR << " sizeXCOR :" << sizeXCOR << endl;

               if (sizeXCOR<=0) continue;   

               // the enINJ, enREC, xcorINJ_REC are computed in the INJ range

               double enINJ=0;
               for (int i=0;i<wfINJ->size();i++) enINJ+=wfINJ->data[i]*wfINJ->data[i];
               //for (int i=0;i<sizeXCOR;i++) enINJ+=wfINJ->data[i+oINJ]*wfINJ->data[i+oINJ];

               double enREC=0;
               //for (int i=0;i<wfREC->size();i++) enREC+=wfREC->data[i]*wfREC->data[i];
               for (int i=0;i<sizeXCOR;i++) enREC+=wfREC->data[i+oREC]*wfREC->data[i+oREC];

               double xcorINJ_REC=0;
               for (int i=0;i<sizeXCOR;i++) xcorINJ_REC+=wfINJ->data[i+oINJ]*wfREC->data[i+oREC];

               WSeries<double> wfREC_SUB_INJ;  
               wfREC_SUB_INJ.resize(sizeXCOR);
               for (int i=0;i<sizeXCOR;i++) wfREC_SUB_INJ.data[i]=wfREC->data[i+oREC]-wfINJ->data[i+oINJ];
               wfREC_SUB_INJ.start(startXCOR);
               wfREC_SUB_INJ.rate(wfREC->rate());
               this->iSNR[j]    = enINJ;
               this->oSNR[j]    = enREC;
               this->ioSNR[j]   = xcorINJ_REC;

               //double erINJ_REC = enINJ+enREC-2*xcorINJ_REC;
               //cout << "enINJ : " << enINJ << " enREC : " << enREC << " xcorINJ_REC : " << xcorINJ_REC << endl;
               //cout << "erINJ_REC/enINJ : " << erINJ_REC/enINJ << endl;

               *wfINJ*=1./rFactor;
	     }
             delete [] pwfINJ;
             delete [] pwfREC;
	   }
	   else {                   // no injections
	     this->range[1]  = 0.;
	     this->chirp[0]  = 0.;
	     this->eBBH[1]   = 0.;
	     this->eBBH[2]   = 0.;
	     this->eBBH[3]   = 0.;
	     this->strain[1] = 0.;
	     this->type[1]   = 0;
	     this->theta[1]  = 0.;
	     this->phi[1]    = 0.;
	     this->psi[1]    = 0.;
	     this->iota[1]   = 0.;
	     this->mass[0]   = 0.;
	     this->mass[1]   = 0.;

             for(i=0; i<6; i++) this->spin[i] = 0.;
 
	     for(j=0; j<int(N); j++) {
	       this->hrss[j+N] = 0.;
	       this->bp[j+N]   = 0.;
	       this->bx[j+N]   = 0.;
	       this->time[j+N] = 0.;
	       this->Deff[j]   = 0.;
               this->iSNR[j]   = 0.;
               this->oSNR[j]   = 0.;
               this->ioSNR[j]  = 0.;
	     }
	   }
	 }

      	 if(this->fP!=NULL) { 
	   fprintf(fP,"\n# trigger %d in lag %d for \n",int(ID),int(n));    	 
       	   this->Dump("1G");
	   vP = &(net->wc_List[n].p_Map[ID-1]);
	   vI = &(net->wc_List[n].p_Ind[ID-1]);
	   x = cos(psm->theta_1*PI/180.)-cos(psm->theta_2*PI/180.);
	   x*= (psm->phi_2-psm->phi_1)*180/PI/psm->size();
	   fprintf(fP,"sky_res:    %f\n",sqrt(fabs(x)));
	   fprintf(fP,"map_lenght: %d\n",int(vP->size()));
	   fprintf(fP,"#skyID  theta   DEC     step   phi     R.A    step  probability    cumulative\n");
	   x = 0.;
	   for(j=0; j<int(vP->size()); j++) {
	     i = (*vI)[j];
	     if(net->mdc__IDSize()) {				// simulation mode
	       x = (j==int(vP->size())-1) ? 0 : x+(*vP)[j];	// last value is the inj sky index (x=0)
             } else x+=(*vP)[j];
	     fprintf(fP,"%6d  %5.1f  %5.1f  %6.2f  %5.1f  %5.1f  %6.2f  %e  %e\n",
		     int(i),psm->getTheta(i),psm->getDEC(i),psm->getThetaStep(i),
		     psm->getPhi(i),psm->getRA(i),psm->getPhiStep(i),(*vP)[j],x); 
	   }
      	 }

         // save the probability skymap to tree
         if(waveTree!=NULL && net->wc_List[n].p_Map.size()) {

           vP = &(net->wc_List[n].p_Map[ID-1]);
           vI = &(net->wc_List[n].p_Ind[ID-1]);
           if(this->Psave) {

              *Psm = *psm;
              *Psm = 0.;

              int k;
              double th,ra;
              for(j=0; j<int(vP->size()); j++) {
                 i = (*vI)[j];
                 th = Psm->getTheta(i);
                 ra = Psm->getRA(i);
                 k=Psm->getSkyIndex(th, ra);
                 Psm->set(k,(*vP)[j]);
              }
           }
         }

         if(waveTree!=NULL) waveTree->Fill();

      }
   }
   if(ndm) free(ndm);
}


