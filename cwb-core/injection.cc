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


#include <fstream>
#include "injection.hh"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

#define MDC_TREE_NAME "mdc"

ClassImp(injection)	 // used by THtml doc

TTree* injection::Init(TString fName, int n)
{
   iFile = TFile::Open(fName);
   if((iFile==NULL) || (iFile!=NULL && !iFile->IsOpen())) {
     cout << "injection::Init : Error opening root file " << fName.Data() << endl;
     exit(1);                                                                    
   }

   TTree* tree = (TTree *) iFile->Get(MDC_TREE_NAME);
   if(tree) {                                         
     ndim = tree->GetUserInfo()->GetSize(); // get number of detectors
     if(ndim>0) {                                                     
       if(n>0 && ndim!=n) {                                           
         cout << "injection::Init : number of detectors declared in the constructor (" << n
              << ") are not equals to the one ("<<ndim<<") declared in the root file : "
              << fName.Data() << endl;
         exit(1);
       }
     } else ndim=n;
   } else {
     cout << "injection::Init : object tree " << MDC_TREE_NAME
          << " not present in the root file " << fName.Data() << endl;
     exit(1);
   }

   if(ndim==0) {
     cout << "injection::Init : number of detectors is not declared in the constructor or"
          << " not present in the root file : " << endl << fName.Data() << endl;
     exit(1);
   }

   return tree;
}

//   Set branch addresses
void injection::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("run",&run);
   fChain->SetBranchAddress("ndim",&ndim);
   fChain->SetBranchAddress("nevent",&nevent);
   fChain->SetBranchAddress("eventID",&eventID);
   fChain->SetBranchAddress("type",&type);
   fChain->SetBranchAddress("name",&name);
   fChain->SetBranchAddress("log",&log);

   fChain->SetBranchAddress("factor",&factor);
   fChain->SetBranchAddress("distance",&distance);
   fChain->SetBranchAddress("mchirp",&mchirp);
   fChain->SetBranchAddress("rp0",&rp0);
   fChain->SetBranchAddress("e0",&e0);
   fChain->SetBranchAddress("redshift",&redshift);
   fChain->SetBranchAddress("gps",&gps);
   fChain->SetBranchAddress("strain",&strain);
   fChain->SetBranchAddress("psi",psi);
   fChain->SetBranchAddress("iota",iota);
   fChain->SetBranchAddress("phi",phi);
   fChain->SetBranchAddress("theta",theta);
   fChain->SetBranchAddress("bp",bp);
   fChain->SetBranchAddress("bx",bx);

   fChain->SetBranchAddress("time",time);
   fChain->SetBranchAddress("duration",duration);

   fChain->SetBranchAddress("frequency",frequency);
   fChain->SetBranchAddress("bandwidth",bandwidth);
   fChain->SetBranchAddress("hrss",hrss);
   fChain->SetBranchAddress("snr",snr);
   fChain->SetBranchAddress("Deff",Deff);
   fChain->SetBranchAddress("mass",mass);
   fChain->SetBranchAddress("spin",spin);

   Notify();
}

// allocate memory
void injection::allocate()
{
   if (!name)        name=     new string();
   else {delete name;name=     new string();}
   if (!log)         log=      new string();
   else {delete log; log=      new string();}
   if (!psi)         psi=      (Float_t*)malloc(2*sizeof(Float_t));
   else              psi=      (Float_t*)realloc(psi,2*sizeof(Float_t));
   if (!iota)        iota=     (Float_t*)malloc(2*sizeof(Float_t));
   else              iota=     (Float_t*)realloc(iota,2*sizeof(Float_t));
   if (!phi)         phi=      (Float_t*)malloc(2*sizeof(Float_t));
   else              phi=      (Float_t*)realloc(phi,2*sizeof(Float_t));
   if (!theta)       theta=    (Float_t*)malloc(2*sizeof(Float_t));
   else              theta=    (Float_t*)realloc(theta,2*sizeof(Float_t));
   if (!bp)          bp=       (Float_t*)malloc(ndim*sizeof(Float_t));
   else              bp=       (Float_t*)realloc(bp,ndim*sizeof(Float_t));
   if (!bx)          bx=       (Float_t*)malloc(ndim*sizeof(Float_t));
   else              bx=       (Float_t*)realloc(bx,ndim*sizeof(Float_t));
   
   if (!time)        time=     (Double_t*)malloc(ndim*sizeof(Double_t));
   else              time=     (Double_t*)realloc(time,ndim*sizeof(Double_t));
   if (!duration)    duration= (Float_t*)malloc(ndim*sizeof(Float_t));
   else              duration= (Float_t*)realloc(duration,ndim*sizeof(Float_t));
   
   if (!frequency)   frequency=(Float_t*)malloc(ndim*sizeof(Float_t));
   else              frequency=(Float_t*)realloc(frequency,ndim*sizeof(Float_t));
   if (!bandwidth)   bandwidth=(Float_t*)malloc(ndim*sizeof(Float_t));
   else              bandwidth=(Float_t*)realloc(bandwidth,ndim*sizeof(Float_t));
   if (!hrss)        hrss=     (Double_t*)malloc(ndim*sizeof(Double_t));
   else              hrss=     (Double_t*)realloc(hrss,ndim*sizeof(Double_t));
   if (!snr)         snr=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              snr=      (Float_t*)realloc(snr,ndim*sizeof(Float_t));
   if (!Deff)        Deff=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              Deff=     (Float_t*)realloc(Deff,ndim*sizeof(Float_t));
   if (!mass)        mass=     (Float_t*)malloc(2*sizeof(Float_t));
   else              mass=     (Float_t*)realloc(mass,2*sizeof(Float_t));
   if (!spin)        spin=     (Float_t*)malloc(6*sizeof(Float_t));
   else              spin=     (Float_t*)realloc(spin,6*sizeof(Float_t));
   if (!pwf)         pwf=      (wavearray<double>**)malloc(ndim*sizeof(wavearray<double>*));
   else              pwf=      (wavearray<double>**)realloc(pwf,ndim*sizeof(wavearray<double>*));

   for(int n=0; n<ndim; n++) pwf[n] = NULL;  

   return;
}

// init array
void injection::init()
{
//   for(int i=0;i<NIFO_MAX;i++) pD[i]=NULL;
   return;
}

Bool_t injection::Notify()
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
   b_factor = fChain->GetBranch("factor");
   b_distance = fChain->GetBranch("distance");
   b_mchirp = fChain->GetBranch("mchirp");
   b_rp0 = fChain->GetBranch("rp0");
   b_e0 = fChain->GetBranch("e0");
   b_redshift = fChain->GetBranch("redshift");
   b_gps = fChain->GetBranch("gps");
   b_strain = fChain->GetBranch("strain");
   b_psi = fChain->GetBranch("psi");
   b_iota = fChain->GetBranch("iota");
   b_phi = fChain->GetBranch("phi");
   b_theta = fChain->GetBranch("theta");
   b_bp= fChain->GetBranch("bx");
   b_bx = fChain->GetBranch("bp");
   b_time = fChain->GetBranch("time");
   b_duration = fChain->GetBranch("duration");
   b_frequency = fChain->GetBranch("frequency");
   b_bandwidth = fChain->GetBranch("bandwidth");
   b_hrss = fChain->GetBranch("hrss");
   b_snr = fChain->GetBranch("snr");
   b_Deff = fChain->GetBranch("Deff");
   b_mass = fChain->GetBranch("mass");
   b_spin = fChain->GetBranch("spin");

   return kTRUE;
}

Int_t injection::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};
Int_t injection::GetEntries() 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntries(); 
};

void injection::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set single event tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* injection::setTree()
{
   TTree* waveTree = new TTree(MDC_TREE_NAME,MDC_TREE_NAME);

   char cpsi[16];      
   char ciota[16];      
   char cphi[16];      
   char ctheta[16];     
   char cbp[16];       
   char cbx[16];       
   char ctime[16];     
   char cduration[16]; 
   char cfrequency[16];
   char cbandwidth[16];
   char chrss[16];     
   char csnr[16];     
   char cDeff[16];     
   char cmass[16];     
   char cspin[16];     

   sprintf(cpsi,       "psi[2]/F");      
   sprintf(ciota,      "iota[2]/F");      
   sprintf(cphi,       "phi[2]/F");      
   sprintf(ctheta,     "theta[2]/F");     
   sprintf(cbp,        "bp[%1d]/F",ndim);       
   sprintf(cbx,        "bx[%1d]/F",ndim);       
   sprintf(ctime,      "time[%1d]/D",ndim);     
   sprintf(cduration,  "duration[%1d]/F",ndim); 
   sprintf(cfrequency, "frequency[%1d]/F",ndim);
   sprintf(cbandwidth, "bandwidth[%1d]/F",ndim);
   sprintf(chrss,      "hrss[%1d]/D",ndim);     
   sprintf(csnr,       "snr[%1d]/F",ndim);     
   sprintf(cDeff,      "Deff[%1d]/F",ndim);     
   sprintf(cmass,      "mass[2]/F");     
   sprintf(cspin,      "spin[6]/F");     
   
 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("ndim",        &ndim,        "ndim/I");
   waveTree->Branch("run",         &run,         "run/I");
   waveTree->Branch("nevent",      &nevent,      "nevent/I");
   waveTree->Branch("eventID",     &eventID,     "eventID/I");
   waveTree->Branch("type",        &type,        "type/I");
   waveTree->Branch("name",        name);
   waveTree->Branch("log",         log);
   waveTree->Branch("factor",      &factor,      "factor/F");
   waveTree->Branch("distance",    &distance,    "distance/F");
   waveTree->Branch("mchirp",      &mchirp,      "mchirp/F");
   waveTree->Branch("rp0",         &rp0,         "rp0/F");
   waveTree->Branch("e0",          &e0,          "e0/F");
   waveTree->Branch("redshift",    &redshift,    "redshift/F");
   waveTree->Branch("gps",         &gps,         "gps/D");
   waveTree->Branch("strain",      &strain,      "strain/D");
   waveTree->Branch("psi",         psi,           cpsi);
   waveTree->Branch("iota",        iota,          ciota);
   waveTree->Branch("phi",         phi,           cphi);
   waveTree->Branch("theta",       theta,         ctheta);
   waveTree->Branch("bp",          bp,            cbp);
   waveTree->Branch("bx",          bx,            cbx);
   waveTree->Branch("time",        time,          ctime);
   waveTree->Branch("duration",    duration,      cduration);
   waveTree->Branch("frequency",   frequency,     cfrequency);
   waveTree->Branch("bandwidth",   bandwidth,     cbandwidth);
   waveTree->Branch("hrss",        hrss,          chrss);
   waveTree->Branch("snr",         snr,           csnr);
   waveTree->Branch("Deff",        Deff,          cDeff);
   waveTree->Branch("mass",        mass,          cmass);
   waveTree->Branch("spin",        spin,          cspin);

   return waveTree;
}

injection& injection::operator=(const injection& a)
{
   int i;

   if(this->ndim != a.ndim) { this->ndim=a.ndim; this->allocate(); }

   this->run=          a.run;
   this->nevent=       a.nevent;
   this->eventID=      a.eventID;
   this->factor=       a.factor;
   this->distance=     a.distance;
   this->mchirp=       a.mchirp;
   this->rp0=          a.rp0;
   this->e0=           a.e0;
   this->redshift=     a.redshift;
   this->gps=          a.gps;
   this->type=         a.type;
   *this->name=        *a.name;
   *this->log=         *a.log;
   this->strain=       a.strain;
   this->psi[0]=       a.psi[0];             
   this->iota[0]=      a.iota[0];             
   this->phi[0]=       a.phi[0];             
   this->theta[0]=     a.theta[0];             
   this->psi[1]=       a.psi[1];             
   this->phi[1]=       a.phi[1];             
   this->theta[1]=     a.theta[1];             
   this->mass[0]=      a.mass[1];             
   this->mass[1]=      a.mass[1];             

   for(i=0; i<6; i++) this->spin[i] = a.spin[i];

   for(i=0; i<ndim; i++){
      
      this->bp[i]=           a.bp[i];             
      this->bx[i]=           a.bx[i];             
      this->time[i]=         a.time[i];
      this->duration[i]=     a.duration[i];
      this->frequency[i]=    a.frequency[i];
      this->bandwidth[i]=    a.bandwidth[i];
      this->hrss[i]=         a.hrss[i];
      this->snr[i]=          a.snr[i];
      this->Deff[i]=         a.Deff[i];
   }
   return *this;
}



//++++++++++++++++++++++++++++++++++++++++++++++++++++
// fill this with data
//++++++++++++++++++++++++++++++++++++++++++++++++++++
Bool_t injection::fill_in(network* net, int id, bool checkEdges)
{ 
  bool save = true;

  size_t N = net->mdc__IDSize();
  size_t I = net->mdcTypeSize();
  size_t M = net->ifoListSize();

  int ID = id<0 ? abs(id+1) : abs(id);

  if(!N || !M || !I || ID<0) return false;

  size_t i,m,nst;
  string itag;
  char* p;
  char ch[1024];
  double hphp, hxhx, hphx, T0, To, Eo;
  double Pi = 3.14159265358979312;
  detector* pd;
 
  this->gps    = net->getifo(0)->getTFmap()->start();
  size_t K     = net->getifo(0)->getTFmap()->size();
  double R     = net->getifo(0)->getTFmap()->wavearray<double>::rate();
  double sTARt = this->gps + net->Edge + 1.;
  double sTOp  = this->gps + K/R - net->Edge - 1.;

  if((net->getmdcTime(ID)<sTARt || net->getmdcTime(ID)>sTOp) && checkEdges) return false;
  
  this->eventID = ID;
 
  string str(net->getmdcList(ID));
  sprintf(ch,"%s",str.c_str());

  this->log->assign(ch, strlen(ch));
  this->log->erase(std::remove(this->log->begin(), this->log->end(), '\n'), this->log->end()); // remove new line

  if((p = strtok(ch," \t")) == NULL) return false;

  p = strtok(NULL," \t");   // get MDC strain
  this->strain = atof(p);
  
  p = strtok(NULL," \t");   // skip
  p = strtok(NULL," \t");   // skip
  p = strtok(NULL," \t");   // get internal phi
  //this->iota[0] = atof(p);  // get iota[0]
  //p = strtok(NULL," \t");   // get internal phi
  this->iota[1] = atof(p);  // get iota[1]
  this->phi[1] = atof(p);
  p = strtok(NULL," \t");   // get internal psi
  this->psi[1] = atof(p); 
  p = strtok(NULL," \t");   // get external theta
  if(fabs(atof(p))>1) {
    cout<<"injection:fill_in error: external theta not valid, must be [-1,1]\n"<<endl;
    exit(1);
  }
  this->theta[0] = acos(atof(p));
  this->theta[0]*= 180/Pi;
  p = strtok(NULL," \t");   // get external phi
  this->phi[0] = atof(p) > 0 ? atof(p) : 2*Pi+atof(p);
  this->phi[0]*= 180/Pi;
  p = strtok(NULL," \t");   // get external psi
  this->psi[0] = atof(p);
  this->psi[0]*= 180/Pi;
  
  p = strtok(NULL," \t");   // skip
  p = strtok(NULL," \t");   // injection time

  //  printf("%12.3f %12.3f %12.3f \n",this->time[0],sTARt,sTOp);

  p = strtok(NULL," \t");   // injection name

  this->name->assign(p, strlen(p));

  for(i=0; i<I; i++) {
    itag = net->getmdcType(i);
    if(itag.find(p) == string::npos) continue;
    this->type = i+1; break;
  }
  
  p = strtok(NULL," \t");   // h+h+
  hphp = atof(p);
  p = strtok(NULL," \t");   // hxhx
  hxhx = atof(p);
  p = strtok(NULL," \t");   // h+hx
  hphx = atof(p);

  save = true;
  for(m=0; m<M; m++) {                      // loop over detectors
    nst = str.find(net->getifo(m)->Name);
    if(nst >= str.length()) {
      cout<<"injection:fill_in error: no injections for detector "
          << net->getifo(m)->Name <<" was found in the injection list !!!\n\n";
      save = false;
      exit(1);
    }

    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) == NULL) continue;   // detector name

    p = strtok(NULL," \t");       // time
    this->time[m] = atof(p);
//    if(this->time[m]<sTARt || this->time[m]>sTOp) save = false;
    
//    printf("%13.3f %13.3f %13.3f\n",sTARt,time[m],sTOp);

    p = strtok(NULL," \t");       // F+
    this->bp[m] = atof(p); 
    p = strtok(NULL," \t");       // Fx
    this->bx[m] = atof(p); 
    
    nst = str.find("insp") < str.find("ebbh") ? str.find("insp") : str.find("ebbh");
    this->Deff[m] = 0.; 
    if(nst < str.length()) {     // inspiral MDC log 
      p = strtok(NULL," \t");     // effective distance
      this->Deff[m] = atof(p); 
    }
    else {                        // standard burst MDC log
      this->hrss[m] = hphp*bp[m]*bp[m] + hxhx*bx[m]*bx[m] + 2*hphx*bp[m]*bx[m];
      this->hrss[m] = this->hrss[m]>0. ? sqrt(this->hrss[m]) : 0.;
    }
  }

  if(!save) return save;

  nst = str.find("distance");
  this->distance = 0.; 
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL)
      this->distance = atof(strtok(NULL," \t"));
  }

  nst = str.find("mass1");
  this->mass[0] = 0.; 
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) 
      this->mass[0] = atof(strtok(NULL," \t"));
  }

  nst = str.find("mass2");
  this->mass[1] = 0.; 
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) 
      this->mass[1] = atof(strtok(NULL," \t"));
   }

  nst = str.find("mchirp");
  this->mchirp = 0.; 
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) 
      this->mchirp = atof(strtok(NULL," \t"));
  }

  nst = str.find("rp0");
  this->rp0 = 0.; 
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) 
      this->rp0 = atof(strtok(NULL," \t"));
  }

  nst = str.find("e0");
  this->e0 = 0.; 
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) 
      this->e0 = atof(strtok(NULL," \t"));
  }

  nst = str.find("redshift");
  this->redshift = 0.; 
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) 
      this->redshift = atof(strtok(NULL," \t"));
  }

  nst = str.find("spin1");
  this->spin[0] = 0.;
  this->spin[1] = 0.;
  this->spin[2] = 0.;
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) {
      this->spin[0] = atof(strtok(NULL," \t"));
      this->spin[1] = atof(strtok(NULL," \t"));
      this->spin[2] = atof(strtok(NULL," \t"));
    }
  }

  nst = str.find("spin2");
  this->spin[3] = 0.;
  this->spin[4] = 0.;
  this->spin[5] = 0.;
  if(nst < str.length()) {
    itag = str.substr(nst);
    sprintf(ch,"%s",itag.c_str());
    if((p = strtok(ch," \t")) != NULL) {
      this->spin[3] = atof(strtok(NULL," \t"));
      this->spin[4] = atof(strtok(NULL," \t"));
      this->spin[5] = atof(strtok(NULL," \t"));
    }
  }

// read simulation parameters from the detector objects

  To = Eo = 0.;
  for(m=0; m<M; m++) {   // loop over detectors
    pd = net->getifo(m);
    if(!pd->HRSS.size()) continue;
    To += pd->TIME.data[ID]*pd->ISNR.data[ID]*pd->ISNR.data[ID];
    Eo += pd->ISNR.data[ID]*pd->ISNR.data[ID];
    if(pd->TIME.data[ID] < 1.) save = false;
  }

  if(!save || Eo<=0.) return save;

  To /= Eo;              // central injection time
  T0 = this->time[0];
  for(m=0; m<M; m++) {   // loop over detectors
    pd = net->getifo(m);
    if(!pd->HRSS.size()) continue;
    this->frequency[m] = pd->FREQ.data[ID];
    this->bandwidth[m] = pd->BAND.data[ID];
    this->time[m] += To - T0;
    this->duration[m] = pd->TDUR.data[ID];
    this->hrss[m] = pd->HRSS.data[ID];
    this->snr[m] = sqrt(pd->ISNR.data[ID]);

    int idSize = pd->IWFID.size();  
    int wfIndex=-1;  
    for (int mm=0; mm<idSize; mm++) if (pd->IWFID[mm]==id) wfIndex=mm;  
    this->pwf[m] = wfIndex>=0 ? pd->IWFP[wfIndex] : NULL;  
  }

  return save;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++
// output injection metadata
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void injection::output(TTree* waveTree, network* net, double factor, bool checkEdges)
{ 
  size_t N = net->mdc__IDSize();
  size_t I = net->mdcTypeSize();
  size_t M = net->ifoListSize();
  if(!N || !M || !I) return;
  double FACTOR = fabs(factor);                // used to tag events in root file
  factor = factor<=0 ? 1 : fabs(factor);       // used to rescale injected events


  size_t n,m;

  // add detectors to tree user info
  if(waveTree!=NULL) {
    int dsize = waveTree->GetUserInfo()->GetSize();
    if(dsize!=0 && dsize!=M) {
       cout<<"injection::output(): wrong user detector list in header tree"<<endl; exit(1);
    }
    if(dsize==0) {
      for(int n=0;n<M;n++) {
        // must be done a copy because detector object
        // is destroyed when waveTree is destroyed
        detector* pD = new detector(*net->getifo(n));
        waveTree->GetUserInfo()->Add(pD);
      }
    }
  }

  this->run = net->nRun;   
  this->nevent = 0; 

  for(n=0; n<N; n++) {
    this->eventID = net->getmdc__ID(n); 
    if(this->fill_in(net,this->eventID,checkEdges)) {
      this->nevent++; 
      this->factor    = FACTOR;
      this->strain   *= factor;
      this->distance /= factor;
      for(m=0; m<M; m++) {
	this->hrss[m] *= factor;
	this->snr[m]  *= factor;
	this->Deff[m] /= factor;
      }
      waveTree->Fill(); 
    }
  }
}



/*
Int_t injection::LoadTree(Int_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Int_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->IsA() != TChain::Class()) return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

Int_t injection::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void injection::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L injection.C
//      Root > injection t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Int_t nentries = Int_t(fChain->GetEntriesFast());

   Int_t nbytes = 0, nb = 0;
   for (Int_t jentry=0; jentry<nentries;jentry++) {
      Int_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
   }
}
*/








