#define FRDIR   "frames"
#define FRLABEL "SimStrain"
#define FRGPS   931158100
#define FRLEN   600
#define FRNAME  "SimFrame"

#define RUN     1
#define SRATE   16384

void GetNoise(wavearray<double>& x, TString ifo, int run);
void WriteFrameFile(TString frDir, TString frLabel, size_t gps, size_t length, int nIFO, TString* IFOS, int run);

void CreateFramesNOISE() {

  TString IFOS[3] = {"L1",  /* LIGO Livingston                   */
                     "H1",  /* LIGO Hanford                      */
                     "V1"   /* Virgo                             */
                    }; 

  int  nIFO = 3;           // size of network starting with first detector ifo[]

  WriteFrameFile(FRDIR, FRLABEL, FRGPS, FRLEN, nIFO, IFOS, RUN);


  return 0;
}

void 
GetNoise(wavearray<double>& x, TString ifo, int run) {

  CWB::Toolbox TB;

  int seed;
  if(ifo.CompareTo("L1")==0) seed=1000;
  if(ifo.CompareTo("H1")==0) seed=2000;
  if(ifo.CompareTo("V1")==0) seed=3000;
  if(ifo.CompareTo("J1")==0) seed=4000;
  if(ifo.CompareTo("A2")==0) seed=5000;

  TString fName;
  if(ifo.CompareTo("L1")==0) fName="plugins/strains/advLIGO_NSNS_Opt_8khz_one_side.txt";
  if(ifo.CompareTo("H1")==0) fName="plugins/strains/advLIGO_NSNS_Opt_8khz_one_side.txt";
  if(ifo.CompareTo("V1")==0) fName="plugins/strains/advVIRGO_sensitivity_12May09_8khz_one_side.txt";
  if(ifo.CompareTo("J1")==0) fName="plugins/strains/LCGT_sensitivity_8khz_one_side.txt";
  if(ifo.CompareTo("A2")==0) fName="plugins/strains/advLIGO_NSNS_Opt_8khz_one_side.txt";

  int size=x.size();
  double start=x.start();
  TB.getSimNoise(x, fName, seed, run);
  x.resize(size);
  x.start(start);

  return;
}    

//______________________________________________________________________________
void
WriteFrameFile(TString frDir, TString frLabel, size_t gps, size_t length, int nIFO, TString* IFOS, int run) {
//
// Write mdc to frame file
//
//
// Input: frDir     - output directory
//        frLabel   - label used for output file name
//                    file name path = frDir/network-frLabel-(gps/100000)/network-frLabel-gps.gwf
//        gps       - time of frame (sec - integer)     
//        length    - time length of frame (sec - integer)
//


  char ifoLabel[64]="";
  for(int i=0;i<nIFO;i++) sprintf(ifoLabel,"%s%s",ifoLabel,IFOS[i].Data());

  // make sub directory
  char sdir[64];
  sprintf(sdir,"%s-%s-%04d",ifoLabel,frLabel.Data(),int(gps/100000));
  char cmd[128];sprintf(cmd,"mkdir -p %s/%s",frDir.Data(),sdir);
  cout << cmd << endl;
  gSystem->Exec(cmd);

  wavearray<double> x(SRATE*length);
  x.rate(SRATE);
  x.start(gps);
  char chName[64];
  char frFile[512];
  for(int i=0;i<nIFO;i++) {

    sprintf(frFile,"%s/%s/%s-%s-%lu-%lu.gwf",frDir.Data(),sdir,IFOS[i].Data(),frLabel.Data(),gps,length);
    cout << frFile << endl;

    sprintf(chName,"%s:SIM-STRAIN",IFOS[i].Data());

    CWB::frame fr(frFile,chName,"WRITE");
    fr.setFrName(FRNAME);

    GetNoise(x,IFOS[i].Data(),run); 
    x >> fr;
    
    cout << "Size (sec) " << x.size()/x.rate() << endl;
    fr.close();
  }


  return;
}

