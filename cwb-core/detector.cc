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


// ++++++++++++++++++++++++++++++++++++++++++++++
// S. Klimenko, University of Florida
// WAT detector class
// ++++++++++++++++++++++++++++++++++++++++++++++

#define DETECTOR_CC
#include "detector.hh"
#include "Meyer.hh"
#include "Symlet.hh"

#include "TVector3.h"
#include "TRotation.h"
#include "Math/Rotation3D.h"
#include "Math/Vector3Dfwd.h"

using namespace ROOT::Math;
using namespace std;

// All LIGO coordinates are verbatim from LIGO-P000006-D-E:
//   Rev. Sci. Instrum., Vol. 72, No. 7, July 2001

// LIGO Hanford 4k interferometer
extern const double _rH1[3]  = {-2.161414928e6, -3.834695183e6, 4.600350224e6};
extern const double _eH1x[3] = {-0.223891216,  0.799830697,  0.556905359};
extern const double _eH1y[3] = {-0.913978490,  0.026095321, -0.404922650};

// LIGO Hanford 2k interferometer
extern const double _rH2[3]  = {-2.161414928e6, -3.834695183e6, 4.600350224e6};
extern const double _eH2x[3] = {-0.223891216,  0.799830697,  0.556905359};
extern const double _eH2y[3] = {-0.913978490,  0.026095321, -0.404922650};

// LIGO Livingston interferometer
extern const double _rL1[3]  = {-7.427604192e4, -5.496283721e6, 3.224257016e6};
extern const double _eL1x[3] = {-0.954574615, -0.141579994, -0.262187738};
extern const double _eL1y[3] = { 0.297740169, -0.487910627, -0.820544948};

// GEO coordinates on 
// http://www.geo600.uni-hannover.de/geo600/project/location.html
// x any y arms swaped compare to 
// Anderson, Brady, Creighton, and Flanagan, PRD 63 042003, 2001, 

// GEO-600 interferometer
extern const double _rG1[3]  = {3.8563112e6,  6.665978e5, 5.0196406e6};
extern const double _eG1x[3] = {-0.445184239,  0.866534205,  0.225675575}; 
extern const double _eG1y[3] = {-0.626000687, -0.552167273,  0.550667271};

// Virgo interferometer
extern const double _rV1[3]  = {4.5463741e6, 8.429897e5, 4.378577e6};
extern const double _eV1x[3] = {-0.700458309,  0.208487795, 0.682562083};
extern const double _eV1y[3] = {-0.053791331, -0.969082169, 0.240803326}; 

// TAMA interferometer
extern const double _rT1[3]  = {-3.946409e6, 3.366259e6, 3.6991507e6};
extern const double _eT1x[3] = {0.648969405, 0.760814505, 0};
extern const double _eT1y[3] = {-0.443713769, 0.378484715, -0.812322234}; 

// KAGRA interferometer
extern const double _rK1[3]  = {-3.7770908e+06, 3.4846722e+06, 3.7650676e+06};
extern const double _eK1x[3] = {-0.3740476967, -0.8378767376, 0.3975561509}; 
extern const double _eK1y[3] = {0.7142972255, 0.01312009275, 0.6997194701};

// AIGO interferometer
extern const double _rA1[3]  = {-2.3567784e6, 4.8970238e6, -3.3173147e6};
extern const double _eA1x[3] = {-9.01077021322091554e-01, -4.33659084587544319e-01, 0};
extern const double _eA1y[3] = {-2.25940560005277846e-01, 4.69469807139026196e-01, 8.53550797275327455e-01}; 

// AURIGA bar
extern const double _rO1[3]  = {4.392467e6, 0.9295086e6, 4.515029e6};
extern const double _eO1x[3] = {-0.644504130, 0.573655377, 0.50550364};
extern const double _eO1y[3] = {0., 0., 0.};

// NAUTILUS bar
extern const double _rN1[3]  = {4.64410999868e6, 1.04425342477e6, 4.23104713307e6};
extern const double _eN1x[3] = {-0.62792641437, 0.56480832712, 0.53544371484};
extern const double _eN1y[3] = {0., 0., 0.};

// EXPLORER bar
extern const double _rE1[3]  = {4.37645395452e6, 4.75435044067e5, 4.59985274450e6};
extern const double _eE1x[3] = {-0.62792641437, 0.56480832712, 0.53544371484};
extern const double _eE1y[3] = {0., 0., 0.};


// see also detector coordinates used by GravEn
// https://gravity.psu.edu/~s4/sims/BurstMDC/Validation/getifo.m

ClassImp(detector)

// constructors

detector::detector()
{
   for(int i=0; i<3; i++){
     this->Rv[i] = _rL1[i]; 
     this->Ex[i] = _eL1x[i]; 
     this->Ey[i] = _eL1y[i];
   }
   init();
   this->sHIFt = 0.;
   this->null  = 0.;
   this->sSNR  = 0.;
   this->xSNR  = 0.;
   this->ekXk  = 0.;
   this->ifoID = 0;
   this->rate  = 16384.;
   this->TFmap.rate(4096);
   detectorParams null_dP = {"",0.0,0,0,0,0,0,0};
   this->dP    = null_dP; 
   this->polarization = TENSOR; 
   this->wfSAVE = 0;
}

detector::detector(char* name, double t)
{
   const double* pRv;
   const double* pEx;
   const double* pEy;

   bool xifo=false;

   if(strstr(name,"L1"))    {pRv=_rL1; pEx=_eL1x; pEy=_eL1y;xifo=true;}
   if(strstr(name,"H1"))    {pRv=_rH1; pEx=_eH1x; pEy=_eH1y;xifo=true;}
   if(strstr(name,"H2"))    {pRv=_rH2; pEx=_eH2x; pEy=_eH2y;xifo=true;}
   if(strstr(name,"G1"))    {pRv=_rG1; pEx=_eG1x; pEy=_eG1y;xifo=true;}
   if(strstr(name,"T1"))    {pRv=_rT1; pEx=_eT1x; pEy=_eT1y;xifo=true;}
   if(strstr(name,"V1"))    {pRv=_rV1; pEx=_eV1x; pEy=_eV1y;xifo=true;}
   if(strstr(name,"A1"))    {pRv=_rA1; pEx=_eA1x; pEy=_eA1y;xifo=true;}
   if(strstr(name,"A2"))    {pRv=_rA1; pEx=_eA1x; pEy=_eA1y;xifo=true;}
   if(strstr(name,"Virgo")) {pRv=_rV1; pEx=_eV1x; pEy=_eV1y;xifo=true;}
   if(strstr(name,"VIRGO")) {pRv=_rV1; pEx=_eV1x; pEy=_eV1y;xifo=true;}
   if(strstr(name,"GEO"))   {pRv=_rG1; pEx=_eG1x; pEy=_eG1y;xifo=true;}
   if(strstr(name,"TAMA"))  {pRv=_rT1; pEx=_eT1x; pEy=_eT1y;xifo=true;}
   if(strstr(name,"K1"))    {pRv=_rK1; pEx=_eK1x; pEy=_eK1y;xifo=true;}
   if(strstr(name,"O1"))    {pRv=_rO1; pEx=_eO1x; pEy=_eO1y;xifo=true;}
   if(strstr(name,"N1"))    {pRv=_rN1; pEx=_eN1x; pEy=_eN1y;xifo=true;}
   if(strstr(name,"E1"))    {pRv=_rE1; pEx=_eE1x; pEy=_eE1y;xifo=true;}
   
   if(!xifo) {
     cout << "detector::detector - Error : detector " << name 
          << " is not in present in the builtin list" << endl; 
     exit(1);
   }
   
   sprintf(this->Name,"%s",name);
   SetName(name);
   for(int i=0; i<3; i++){
     this->Rv[i] = pRv[i]; 
     this->Ex[i] = pEx[i]; 
     this->Ey[i] = pEy[i]; 
   }
   if(strstr(name,"A2")) this->rotate(45);   // rotate arms for A2
   init();                                   // fill detector tensor.
   this->sHIFt = t;
   this->null  = 0.;
   this->sSNR  = 0.;
   this->xSNR  = 0.;
   this->ekXk  = 0.;
   this->ifoID = 0;
   this->rate  = 16384.;
   this->TFmap.rate(4096);
   detectorParams null_dP = {"",0.0,0,0,0,0,0,0};
   this->dP    = null_dP; 
   this->polarization = TENSOR; 
   this->wfSAVE = 0;
}  

detector::detector(detectorParams dP, double t)
{
   double rad2deg = 180./TMath::Pi();
   double deg2rad = TMath::Pi()/180.;

   double uRv[3];
   double uEx[3];
   double uEy[3];

   GeodeticToGeocentric(dP.latitude*deg2rad,dP.longitude*deg2rad,dP.elevation,uRv[0],uRv[1],uRv[2]);
   GetCartesianComponents(uEx,dP.AltX*deg2rad,dP.AzX*deg2rad, dP.latitude*deg2rad,dP.longitude*deg2rad);
   GetCartesianComponents(uEy,dP.AltY*deg2rad,dP.AzY*deg2rad, dP.latitude*deg2rad,dP.longitude*deg2rad);

   sprintf(this->Name,"%s",dP.name);
   SetName(dP.name);
   for(int i=0; i<3; i++){
     this->Rv[i] = uRv[i];
     this->Ex[i] = uEx[i];
     this->Ey[i] = uEy[i];
   }
   init();                             // fill detector tensor.
   this->sHIFt = t;
   this->null  = 0.;
   this->sSNR  = 0.;
   this->xSNR  = 0.;
   this->ekXk  = 0.;
   this->ifoID = 0;
   this->rate  = 16384.;
   this->TFmap.rate(4096);
   this->dP    = dP; 
   this->polarization = TENSOR; 
   this->wfSAVE = 0;
}

detectorParams detector::getDetectorParams() 
{
  // -----------------------------------------
  // user define detector
  // -----------------------------------------
  if(strlen(this->dP.name)>0) return this->dP; 

  // -----------------------------------------
  // builtin detector
  // -----------------------------------------
  double rad2deg = 180./TMath::Pi();
  double deg2rad = TMath::Pi()/180.;

  XYZVector iRv(this->Rv[0],this->Rv[1],this->Rv[2]);
  TVector3 vRv(iRv.X(),iRv.Y(),iRv.Z());

  // Ex angle respect to east direction
  XYZVector iEx(this->Ex[0],this->Ex[1],this->Ex[2]);
  XYZVector iEy(this->Ey[0],this->Ey[1],this->Ey[2]);
  XYZVector iEZ(0.,0.,1.);  // Zeta cartesian axis

  TVector3  vEx(iEx.X(),iEx.Y(),iEx.Z());
  TVector3  vEy(iEy.X(),iEy.Y(),iEy.Z());
  TVector3  vEZ(iEZ.X(),iEZ.Y(),iEZ.Z());

  vEx*=1./vEx.Mag();
  vEy*=1./vEy.Mag();
  vEZ*=1./vEZ.Mag();
  vRv*=1./vRv.Mag();

  TVector3 vEe=vEZ.Cross(vRv);       // vEe point to east in the local detector frame
  vEe*=1./vEe.Mag();

  double cosExAngle=vEe.Dot(vEx);    // ExAngle is the angle of Ex respect to East counter-clockwise
  if(cosExAngle>1.) cosExAngle=1.;
  if(cosExAngle<-1.) cosExAngle=-1.;
  double cosEyAngle=vEe.Dot(vEy);    // EyAngle is the angle of Ey respect to Eas counter-clockwiset
  if(cosEyAngle>1.) cosEyAngle=1.;
  if(cosEyAngle<-1.) cosEyAngle=-1.;

  double ExAngle=acos(cosExAngle)*rad2deg;  
  double EyAngle=acos(cosEyAngle)*rad2deg;

  // fix sign of ExAngle,EyAngle
  TVector3 vEn;   
  vEn=vEe.Cross(vEx);   
  if(vEn.Dot(vRv)<0) ExAngle*=-1;   
  vEn=vEe.Cross(vEy);   
  if(vEn.Dot(vRv)<0) EyAngle*=-1;   

  // Convert ExAngle to the angle of Ex respect to North clockwise
  ExAngle=fmod(90-ExAngle,360.); 
  // Convert EyAngle to the angle of Ey respect to North clockwise
  EyAngle=fmod(90-EyAngle,360.); 

  double latitude,longitude,elevation;
  GeocentricToGeodetic(iRv.X(),iRv.Y(),iRv.Z(),latitude,longitude,elevation);

  detectorParams idP;

  sprintf(idP.name,"%s",this->Name);
  idP.latitude  = latitude*rad2deg;
  idP.longitude = longitude*rad2deg;
  idP.elevation = elevation;
  idP.AltX      = 0.;
  idP.AzX       = ExAngle;
  idP.AltY      = 0.;
  idP.AzY       = EyAngle;

  return  idP;
}

// rotate arms in the detector plane by angle a in degrees counter-clockwise
void detector::rotate(double a)
{
  double ax[3];
  double ay[3];
  double si = sin(a*PI/180.);
  double co = cos(a*PI/180.);

  for(int i=0; i<3; i++) {
    ax[i] = this->Ex[i]; 
    ay[i] = this->Ey[i];
  }
/*
  for(int i=0; i<3; i++) {
    this->Ex[i] = ax[i]*co+ay[i]*si; 
    this->Ey[i] = ay[i]*co-ax[i]*si; 
  }
*/
  double aww=0.;
  double aw[3];
  double axy=0.; 

  for(int i=0; i<3; i++) axy+=ax[i]*ay[i];

  // compute vector aw in the plane Ex,Ey ortogonal to Ex and rotate Ex 
  aww=0.;
  for(int i=0; i<3; i++) {aw[i]=-ax[i]*axy+ay[i]; aww+=aw[i]*aw[i];}
  for(int i=0; i<3; i++) aw[i]/=sqrt(aww);
  for(int i=0; i<3; i++) this->Ex[i]=ax[i]*co+aw[i]*si; // rotate Ex

  // compute vector aw in the plane Ex,Ey ortogonal to Ey and rotate Ey 
  aww=0.;
  for(int i=0; i<3; i++) {aw[i]=-ax[i]+ay[i]*axy; aww+=aw[i]*aw[i];}
  for(int i=0; i<3; i++) aw[i]/=sqrt(aww);
  for(int i=0; i<3; i++) this->Ey[i]=ay[i]*co+aw[i]*si; // rotate Ey

  init();       // fill detector tensor.

  // update user define detector
  if(strlen(this->dP.name)>0) {
    this->dP.AzX = fmod(this->dP.AzX-a,360.);  
    this->dP.AzY = fmod(this->dP.AzY-a,360.);  
  }  
}


detector::detector(const detector& value)
{
   *this = value;
}

// destructor

detector::~detector(){

  int n;

  n = IWFP.size();
  for (int i=0;i<n;i++) {
    wavearray<double>* wf = (wavearray<double>*)IWFP[i];
    delete wf;
  }
  IWFP.clear();
  IWFID.clear();

  n = RWFP.size();
  for (int i=0;i<n;i++) {
    wavearray<double>* wf = (wavearray<double>*)RWFP[i];
    delete wf;
  }
  RWFP.clear();
  RWFID.clear();

}

//: operator =
//: !!! not fully implemented

detector& detector::operator=(const detector& value)
{
   sprintf(Name,"%s",value.Name);
   SetName(Name);
   for(int i=0; i<3; i++){
     this->Rv[i] = value.Rv[i]; 
     this->Ex[i] = value.Ex[i]; 
     this->Ey[i] = value.Ey[i];
   }
   init();

   tau = value.tau;
   mFp = value.mFp;
   mFx = value.mFx;

   TFmap = value.TFmap;
   waveForm = value.waveForm;
   sHIFt = value.sHIFt;
   null = value.null;
   nRMS = value.nRMS;
   nVAR = value.nVAR;

   dP = value.dP;

   polarization = value.polarization; 

   wfSAVE = value.wfSAVE; 

   return *this;
}


detector& detector::operator=(const WSeries<double>& value)
{
   double tsRate =TFmap.wavearray<double>::rate();
   TFmap = value; 
   waveForm.resize(size_t(tsRate));
   waveForm.rate(tsRate);
   waveForm.setWavelet(*(TFmap.pWavelet));  
   return *this;
}

// copy 'from' injection stuff
detector& detector::operator<<(detector& value)
{
  wfSAVE = value.wfSAVE;
  HRSS   = value.HRSS;
  ISNR   = value.ISNR;
  FREQ   = value.FREQ;
  BAND   = value.BAND;
  TIME   = value.TIME;
  TDUR   = value.TDUR;

  IWFID  = value.IWFID;
   
  for (int i=0;i<IWFP.size();i++) {
    wavearray<double>* wf = (wavearray<double>*)IWFP[i];
    delete wf;
  }
  IWFP.clear();
  for(int i=0;i<value.IWFP.size();i++) {
    wavearray<double>* wf = new wavearray<double>;
    *wf = *value.IWFP[i];
    IWFP.push_back(wf); 
  }

  return *this;
}

// copy 'to' injection stuff
detector& detector::operator>>(detector& value)
{
  value.wfSAVE = wfSAVE;
  value.HRSS   = HRSS;
  value.ISNR   = ISNR;
  value.FREQ   = FREQ;
  value.BAND   = BAND;
  value.TIME   = TIME;
  value.TDUR   = TDUR;

  value.IWFID  = IWFID;
   
  for (int i=0;i<value.IWFP.size();i++) {
    wavearray<double>* wf = (wavearray<double>*)value.IWFP[i];
    delete wf;
  }
  value.IWFP.clear();
  for(int i=0;i<IWFP.size();i++) {
    wavearray<double>* wf = new wavearray<double>;
    *wf = *IWFP[i];
    value.IWFP.push_back(wf); 
  }

  return *this;
}


//**************************************************************************
// initialize detector tenzor
//**************************************************************************
void detector::init()
{
   DT[0] = Ex[0]*Ex[0]-Ey[0]*Ey[0];
   DT[1] = Ex[0]*Ex[1]-Ey[0]*Ey[1];
   DT[2] = Ex[0]*Ex[2]-Ey[0]*Ey[2];

   DT[3] = DT[1];
   DT[4] = Ex[1]*Ex[1]-Ey[1]*Ey[1];
   DT[5] = Ex[1]*Ex[2]-Ey[1]*Ey[2];

   DT[6] = DT[2];
   DT[7] = DT[5];
   DT[8] = Ex[2]*Ex[2]-Ey[2]*Ey[2];

   if (strcmp(Name,"O1")==0) for (int i=0;i<9;i++) DT[i]*=2;  
   if (strcmp(Name,"N1")==0) for (int i=0;i<9;i++) DT[i]*=2;  
   if (strcmp(Name,"E1")==0) for (int i=0;i<9;i++) DT[i]*=2;  
}

//**************************************************************************
// return antenna pattern
//**************************************************************************
wavecomplex detector::antenna(double theta, double phi, double psi)
{
   double a,b;

   theta *= PI/180.; phi *= PI/180.; psi *= PI/180.;

   double cT = cos(theta);
   double sT = sin(theta);
   double cP = cos(phi);
   double sP = sin(phi);
   
   double d11 = DT[0];
   double d12 = DT[1];
   double d13 = DT[2];

   double d21 = DT[3];
   double d22 = DT[4];
   double d23 = DT[5];

   double d31 = DT[6];
   double d32 = DT[7];
   double d33 = DT[8];


   double fp = 0.;
   double fx = 0.;

   if(polarization==TENSOR) {

     fp =  (cT*cP*d11 + cT*sP*d21 - sT*d31)*cT*cP
        +  (cT*cP*d12 + cT*sP*d22 - sT*d32)*cT*sP
        -  (cT*cP*d13 + cT*sP*d23 - sT*d33)*sT
        +  (cP*d21-sP*d11)*sP 
        -  (cP*d22-sP*d12)*cP;

     fx = -(cT*cP*d11 + cT*sP*d21 - sT*d31)*sP
        +  (cT*cP*d12 + cT*sP*d22 - sT*d32)*cP
        +  (cP*d21-sP*d11)*cT*cP 
        +  (cP*d22-sP*d12)*cT*sP
        -  (cP*d23-sP*d13)*sT;

     fp = -fp;             // to follow convention in LIGO-T010110 and Anderson et al.

     if(fabs(psi)>0.) {    // rotate in the waveframe A'=exp(-2*i*psi)A
       a =  fp*cos(2*psi)+fx*sin(2*psi);
       b =  -fp*sin(2*psi)+fx*cos(2*psi);
       fp = a; fx = b;
     }
   }

   if(polarization==SCALAR) {

     fp = -(cT*cP*d11 + cT*sP*d21 - sT*d31)*cT*cP
        -  (cT*cP*d12 + cT*sP*d22 - sT*d32)*cT*sP
        +  (cT*cP*d13 + cT*sP*d23 - sT*d33)*sT
        +  (cP*d21-sP*d11)*sP
        -  (cP*d22-sP*d12)*cP;

     fp = 2*fp;
     fx = 0;
   }

   wavecomplex z(fp/2.,fx/2.);
   return z;
}



//**************************************************************************
// reconstruct wavelet series for a cluster, put it in waveForm  
//**************************************************************************
double detector::getwave(int ID, netcluster& wc, char atype, size_t index) 
{ 
  int    i,j,n,m,k,l;
   double a,b,rms,fl,fh;
   double  R = this->TFmap.rate();
   int L = int(this->TFmap.maxLayer()+1);

   waveForm.setWavelet(*(TFmap.pWavelet));  
   waveForm.rate(R);
   rms = wc.getwave(ID,waveForm,atype,index); 

   if(rms==0.) return rms;

// create bandlimited detector output

   waveBand = waveForm; waveBand = 0.;
   fh = waveBand.gethigh(); 
   fl = waveBand.getlow();
   l =  int(this->TFmap.getLevel()) - int(waveBand.getLevel()); 

// adjust waveBand resolution to match TFmap

   if(l<0) { waveBand.Inverse(-l); }
   else    { waveBand.Forward(l);  }

   a = waveBand.start()*R; 
   i = int(a + ((a>0) ? 0.1 : -0.1));
   k = waveBand.size(); j = 0;
   if(i<0) { k += i; j = -i; i = 0; } 
   if((i/L)*L != i) cout<<"detector::getwave() time mismatch: "<<L<<"  "<<i<<"\n";

   waveBand.cpf(this->TFmap,k,i,j);
   waveNull = waveBand;

   n = int(2*L*fl/R+0.1)-1;  // first layer
   m = int(2*L*fh/R+0.1)+1;  // last layer
   if(n<=0) n=0;
   if(m>=int(L)) m=L;
   if(m<=n) { n=0; m=L; }

   WSeries<double> w = waveBand;
   wavearray<double> x;

//   cout<<i<<" L="<<L<<" Band start="<<waveBand.start()<<endl;

   waveBand = 0;
   for(k=n; k<m; k++) { w.getLayer(x,k); waveBand.putLayer(x,k); }

   waveBand.Inverse(); 
   waveForm.Inverse(); 
   waveNull.Inverse(); 
   waveForm *= atype=='w' ? 1./sqrt(this->rate/R) : 1.;  // rescale waveform
   w = waveForm;

// window the waveforms

   double sTARt= waveForm.start();
   size_t I    = waveForm.size();
   size_t M    = I/2;
   double sum  = waveForm.data[M]*waveForm.data[M];

   a = waveForm.rms();
   double hrss = a*a*I;

   for(i=1; i<int(M); i++) {
      a = waveForm.data[M-i];
      b = waveForm.data[M+i];
      sum += a*a+b*b;
      if(sum/hrss > 0.999 && i/R>0.05) break;
   }
   n = i+int(0.05*R);
   if(n < int(M-1)) i = size_t(n);
   i = M - ((M-i)/L)*L;              // sink with wavelet resolution.

//   cout<<"M="<<M<<"  2i="<<2*i<<"  i/R"<<i/R<<endl;

   waveForm.cpf(w,2*i,M-i);
   waveForm.resize(2*i);
   waveForm.start(sTARt+(M-i)/R);

   w = waveBand;
   waveBand.cpf(w,2*i,M-i);
   waveBand.resize(2*i);
   waveBand.start(sTARt+(M-i)/R);

   return rms;
}


//**************************************************************************
// set time delays  
// time delay convention: t_detector-tau - arrival time at the center of Earth
//**************************************************************************
void detector::setTau(double sms,double t1,double t2,double p1,double p2)
{
   size_t i;
   skymap SM(sms,t1,t2,p1,p2); 
   size_t n = SM.size();
   double x,y,z;

   for(i=0; i<n; i++) {
      x = SM.getTheta(i)*PI/180.;
      y = SM.getPhi(i)*PI/180.;
      z = Rv[0]*sin(x)*cos(y) + Rv[1]*sin(x)*sin(y) + Rv[2]*cos(x);
      SM.set(i,-z/speedlight);
   }

   tau = SM;
   return;
}


//**************************************************************************
// set time delays
// time delay convention: t_detector-tau - arrival time at the center of Earth
//**************************************************************************
void detector::setTau(int order) 
{
   size_t i;
   skymap SM(order);
   size_t n = SM.size();
   double x,y,z;

   for(i=0; i<n; i++) {
      x = SM.getTheta(i)*PI/180.;
      y = SM.getPhi(i)*PI/180.;
      z = Rv[0]*sin(x)*cos(y) + Rv[1]*sin(x)*sin(y) + Rv[2]*cos(x);
      SM.set(i,-z/speedlight);
   }

   tau = SM;
   return;
}

//**************************************************************************
// return detector time delay for specified source location  
//**************************************************************************
double detector::getTau(double theta, double phi)
{
   double x = theta*PI/180.;
   double y = phi*PI/180.;
   double z = Rv[0]*sin(x)*cos(y) + Rv[1]*sin(x)*sin(y) + Rv[2]*cos(x);
   return -z/speedlight;
}

//**************************************************************************
// set antenna patterns
//**************************************************************************
void detector::setFpFx(double sms,double t1,double t2,double p1,double p2)
{
   size_t i;
   skymap Sp(sms,t1,t2,p1,p2); 
   skymap Sx(sms,t1,t2,p1,p2); 
   size_t n = Sp.size();
   double x,y;
   wavecomplex a;

   for(i=0; i<n; i++) {
      x = Sp.getTheta(i);
      y = Sp.getPhi(i);
      a = antenna(x,y);
      Sp.set(i,a.real());
      Sx.set(i,a.imag());
   }

   mFp = Sp;
   mFx = Sx;
   return;
}

//**************************************************************************
// set antenna patterns
//**************************************************************************
void detector::setFpFx(int order)  
{
   size_t i;
   skymap Sp(order);
   skymap Sx(order);
   size_t n = Sp.size();
   double x,y;
   wavecomplex a;

   for(i=0; i<n; i++) {
      x = Sp.getTheta(i);
      y = Sp.getPhi(i);
      a = antenna(x,y);
      Sp.set(i,a.real());
      Sx.set(i,a.imag());
   }

   mFp = Sp;
   mFx = Sx;
   return;
}
   
//: initialize delay filter  
//  delay index n:  0  1  2  3  4  5  6 ... M-3  M-2  M-1  M
//  sample delay:   0 -1 -2 -3 -4 -5 -6       3    2    1  0
size_t detector::setFilter(size_t K, double phase, size_t upL)
{
  if(TFmap.isWDM()) {
     cout<<"wseries::setFilter(): not applicable to WDM TFmaps\n";
     return 0;
  } 
  size_t i,j,k,n,ii,jj;
  size_t M = TFmap.maxLayer()+1;      // number of wavelet layers
  size_t L = TFmap.getLevel();        // wavelet decomposition depth
  size_t m = M*TFmap.pWavelet->m_H;   // buffer length  
  size_t N = M*(1<<upL);              // number of time delays

// K - total length of the delay filter
  std::vector<delayFilter> F;  // delay filter buffer
  delayFilter v; v.index.resize(K); v.value.resize(K);
  for(i=0; i<K; i++) v.value[i] = 0.; 
  for(i=0; i<M; i++) F.push_back(v); 
  slice S; 
  size_t s;
  short inDex;
  float vaLue;
  int offst;
  

// set wavelet buffer

  Wavelet* pW = TFmap.pWavelet->Clone();    // wavelet used to calculate delay filter
  Meyer<double> wM(1024,1);                 // up-sample wavelet
  WSeries<double> w(*pW);  
  WSeries<double> W(wM);                    // up-sampled wavelet series 

  cout<<"w.pWavelet->m_H "<<w.pWavelet->m_H<<"  level "<<w.pWavelet->m_Level<<" ";

  w.resize(1024);
  while(int(L)>w.getMaxLevel() || w.size()<m) w.resize(w.size()*2); 
  w.setLevel(L);
  W.resize(w.size()*N/M);
  W.setLevel(upL);

  cout<<"wsize: "<<w.size()<<endl;

  S = w.getSlice(0);
  j = M*S.size()/2;
  
  double*  pb = (double * )malloc(m*sizeof(double));
  double** pp = (double **)malloc(m*sizeof(double*));
  for(i=0; i<m; i++) pp[i] = w.data + i + (int(j) - int(m/2));
  double* p0 = pp[0];
  double* p;
  double sum;
  wavecomplex Z(cos(phase*PI/180.),sin(phase*PI/180.));        // phase shift
  wavecomplex z;

  filter.clear();
  this->nDFS = N;           // store number of Delay Filter Samples
  this->nDFL = M;           // store number of Delay Filter Layers

  for(n=0; n<N; n++) {      // loop over delays

    for(i=0; i<M; i++) {    // loop over wavelet layers

      w = 0.;
      S = w.getSlice(i);
      p = w.data+S.start()+j;
      s = S.start();
      *p = 1.;
      w.Inverse();

// up-sample

      W = 0.;
      W.putLayer(w,0);
      W.Inverse();

// phase shift
      if(phase != 0.) {
	W.FFTW(1);
	for(k=2; k<W.size(); k+=2) {
	  z.set(W.data[k],W.data[k+1]);  // complex amplitude
	  z *= Z;
	  W.data[k] = z.real();
	  W.data[k+1] = z.imag();
	}
	W.FFTW(-1);
      }

// time shift by integer number of samples

      W.cpf(W,W.size()-n,n);

// down-sample

      W.Forward(upL);
      W.getLayer(w,0);

// get filter coefficients

      w.Forward(L);
      if(n >= N/2) p -= M;   // positive shift

      for(k=0; k<m; k++) { 
	*(pb+k)  = *(p0+k);  // save data in the buffer
	*(p0+k) *= *(p0+k);  // square
      } 

      w.waveSort(pp,0,m-1);

      for(k=m-1; k>=0; k--) {
        offst = pp[k]-p;
	if(abs(offst) >32767) continue;
	inDex = short(offst);
	vaLue = float(pb[pp[k]-p0]);
	if(fabs(vaLue)<1.e-4) break;
	offst += s;
	offst -= (offst/M)*M;
	if(offst<0) offst += M;       // calculate offset
	ii = pW->convertO2F(L,offst); // convert offset into frequency index

	if(offst !=  pW->getOffset(L,pW->convertF2L(L,ii))) cout<<"setFilter error 1\n";
	offst -= inDex;
	offst -= (offst/M)*M;
	if(offst<0) offst += M;       // calculate offset
	if(offst != int(s)) cout<<"setFilter error 2: "<<offst<<" "<<s<<endl;
	
	jj = minDFindex(F[ii])-1;     // index of least significant element in F[ii]

	for(size_t kk=0; kk<K; kk++) 
	   if(fabs(F[ii].value[jj])>fabs(F[ii].value[kk])) cout<<"setFilter error 3:\n";

	if(jj>=K) {cout<<jj<<endl; continue;}
	if(fabs(F[ii].value[jj]) < fabs(vaLue)){
	   F[ii].value[jj] =  vaLue;
	   F[ii].index[jj] = -inDex;
	}
      }

    }

    for(i=0; i<M; i++) {
       filter.push_back(F[i]);

//       if(n==3) {
//       S = w.getSlice(i);
//       printf("%3d %3d %1d %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f\n",
//	      S.start(),i,n,F[i].value[0],F[i].value[1],F[i].value[2],F[i].value[3],F[i].value[4],
//	      F[i].value[5],F[i].value[6],F[i].value[7],F[i].value[8],F[i].value[9]);
//       printf("%3d %3d %1d %7d %7d %7d %7d %7d %7d %7d %7d %7d %7d\n",
//	      S.start(),i,n,F[i].index[0],F[i].index[1],F[i].index[2],F[i].index[3],F[i].index[4],
//	      F[i].index[5],F[i].index[6],F[i].index[7],F[i].index[8],F[i].index[9]);  
//       }

       sum = 0;
       v = filter[n*M+i];
       for(k=0; k<K; k++) { 
	  sum += F[i].value[k]*F[i].value[k];
	  if(n && F[i].value[k] == 0.) printf("%4d %4d %d4 \n",int(n),int(i),int(k));
	  if(v.value[k] !=  F[i].value[k] || 
	     v.index[k] !=  F[i].index[k]) cout<<"setFilter error 4\n";
	  F[i].value[k] = 0.;
       }
       if(sum<0.97) printf("%4d %4d %8.5f \n",int(n),int(i),sum);
	  
    }

  }

  delete pW;
  free(pp);
  free(pb);
  return filter.size();
}


//: initialize delay filter from another detector 
size_t detector::setFilter(detector &d) {  
  size_t K = d.filter.size();
  filter.clear(); 
  std::vector<delayFilter>().swap(filter);
  filter.reserve(K);

  for(size_t k=0; k<K; k++) {
    filter.push_back(d.filter[k]);
  }
  return filter.size();
}


//: Dumps filters to file *fname in binary format.
void detector::writeFilter(const char *fname)
{
  size_t i,j,k;
  FILE *fp;

  if ( (fp=fopen(fname, "wb")) == NULL ) {
     cout << " DumpBinary() error : cannot open file " << fname <<". \n";
     exit(1);
  }

  size_t M = size_t(TFmap.maxLayer()+1);           // number of wavelet layers
  size_t K = size_t(filter[0].index.size());       // delay filter length
  size_t N = this->nDFS;                           // number of delays
  size_t n = K * sizeof(float);
  size_t m = K * sizeof(short);

  wavearray<float> value(K);
  wavearray<short> index(K);

  fwrite(&K, sizeof(size_t), 1, fp);  // write filter length
  fwrite(&M, sizeof(size_t), 1, fp);  // number of layers
  fwrite(&N, sizeof(size_t), 1, fp);  // number of delays
  
  for(i=0; i<N; i++) {         // loop over delays
    for(j=0; j<M; j++) {       // loop over wavelet layers
       for(k=0; k<K; k++) {    // loop over filter coefficients
	  value.data[k] = filter[i*M+j].value[k];
	  index.data[k] = filter[i*M+j].index[k];
       }
       fwrite(value.data, n, 1, fp);
       fwrite(index.data, m, 1, fp);
    }
  }
  fclose(fp);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//: Read filters from file *fname.
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void detector::readFilter(const char *fname)
{
  size_t i,j,k;
  FILE *fp;

  if ( (fp=fopen(fname, "rb")) == NULL ) {
     cout << " DumpBinary() error : cannot open file " << fname <<". \n";
     exit(1);
  }

  size_t M;           // number of wavelet layers
  size_t K;           // delay filter length
  size_t N;           // number of delays

  fread(&K, sizeof(size_t), 1, fp);  // read filter length
  fread(&M, sizeof(size_t), 1, fp);  // read number of layers
  fread(&N, sizeof(size_t), 1, fp);  // read number of delays
  
  size_t n = K * sizeof(float);
  size_t m = K * sizeof(short);
  wavearray<float> value(K);
  wavearray<short> index(K);
  delayFilter v;

  v.value.clear(); v.value.reserve(K);
  v.index.clear(); v.index.reserve(K);
  this->clearFilter(); filter.reserve(N*M);
  this->nDFS = N;         // set number of delay samples
  this->nDFL = M;         // set number of delay layers

  for(k=0; k<K; k++) {    // loop over filter coefficients
     v.value.push_back(0.);
     v.index.push_back(0);
  }

  for(i=0; i<N; i++) {         // loop over delays
    for(j=0; j<M; j++) {       // loop over wavelet layers
       fread(value.data, n, 1, fp);
       fread(index.data, m, 1, fp);
       for(k=0; k<K; k++) {    // loop over filter coefficients
	  v.value[k] = value.data[k];
	  v.index[k] = index.data[k];
       }

//       if(i<3){
//       printf("%6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f \n",
//	      v.value[0],v.value[1],v.value[2],v.value[3],v.value[4],
//	      v.value[5],v.value[6],v.value[7],v.value[8],v.value[9]);
//       printf("%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d \n",
//	      v.index[0],v.index[1],v.index[2],v.index[3],v.index[4],
//	      v.index[5],v.index[6],v.index[7],v.index[8],v.index[9]);
//       }

       filter.push_back(v);
    }
  }
  fclose(fp);
}


//: apply delay filter to input WSeries and put result in TFmap  
void detector::delay(double t, WSeries<double> &w)
{
  int i,j,jb,je;

  int k;
  double* p;
  double* q;

  if(TFmap.isWDM()) {
     cout<<"wseries::delay(): not applicable to WDM TFmaps\n";
     return;
  } 

  slice S; 
  delayFilter v = filter[0];         // delay filter

  int M = this->nDFL;                // number of wavelet layers
  int N = this->nDFS;                // number of delay samples per pixel
  int K = int(v.index.size());       // delay filter length
  int n = int(t*TFmap.wavearray<double>::rate());
  int m = n>0 ? (n+N/2-1)/N : (n-N/2)/N; // delay in wavelet pixels
  int l = TFmap.pWavelet->m_H/4+2;

  n = n - m*N;                       // n - delay in samples
  if(n <= 0) n = -n;                 // filter index for negative delays
  else       n = N-n;                // filter index for positive delays

  cout<<"delay="<<t<<" m="<<m<<" n="<<n<<"  M="<<M<<"  K="<<K<<"  N="<<N<<endl;

  int mM = m*M;
  double* A = (double*)malloc(K*sizeof(double));
     int* I = (int*)malloc(K*sizeof(int));

  for(i=0; i<M; i++) {

    S  = w.getSlice(i);
    v  = filter[n*M+i];
    jb = m>0 ? S.start()+(l+m)*M : S.start()+l*M;
    je = m<0 ? w.size() +(m-l)*M : w.size() -l*M;

    for(k=0; k<K; k++) { A[k]=double(v.value[k]); I[k]=int(v.index[k]); }

    if(K==16){
    for(j=jb; j<je; j+=M) {
      p = w.data+j-mM;
      TFmap.data[j] = A[0]*p[I[0]]   + A[1]*p[I[1]]   + A[2]*p[I[2]]   + A[3]*p[I[3]]   
	            + A[4]*p[I[4]]   + A[5]*p[I[5]]   + A[6]*p[I[6]]   + A[7]*p[I[7]]   
                    + A[8]*p[I[8]]   + A[9]*p[I[9]]   + A[10]*p[I[10]] + A[11]*p[I[11]] 
                    + A[12]*p[I[12]] + A[13]*p[I[13]] + A[14]*p[I[14]] + A[15]*p[I[15]];
    }
    }

    else if(K==32){
    for(j=jb; j<je; j+=M) {
      p = w.data+j-mM;
      TFmap.data[j] = A[0]*p[I[0]]   + A[1]*p[I[1]]   + A[2]*p[I[2]]   + A[3]*p[I[3]]   
	            + A[4]*p[I[4]]   + A[5]*p[I[5]]   + A[6]*p[I[6]]   + A[7]*p[I[7]]   
                    + A[8]*p[I[8]]   + A[9]*p[I[9]]   + A[10]*p[I[10]] + A[11]*p[I[11]] 
                    + A[12]*p[I[12]] + A[13]*p[I[13]] + A[14]*p[I[14]] + A[15]*p[I[15]]
                    + A[16]*p[I[16]] + A[17]*p[I[17]] + A[18]*p[I[18]] + A[19]*p[I[19]]
                    + A[20]*p[I[20]] + A[21]*p[I[21]] + A[22]*p[I[22]] + A[23]*p[I[23]]
                    + A[24]*p[I[24]] + A[25]*p[I[25]] + A[26]*p[I[26]] + A[27]*p[I[27]]
                    + A[28]*p[I[28]] + A[29]*p[I[29]] + A[30]*p[I[30]] + A[31]*p[I[31]];
    }
    }

    else {
    for(j=jb; j<je; j+=M) {
      p = w.data+j-mM;
      q = TFmap.data+j;
      k = K; *q = 0.;
      while(k-- > 0) { *q += *(A++) * p[*(I++)]; }
      A -= K; I -= K;
    }
    }

  }
  free(A);
  free(I);
}

//: return noise variance for selected wavelet layer or pixel
double detector::getNoise(size_t I, int J)
{
  if(!nRMS.size()) return 0.;
  if(int(I)>TFmap.maxLayer()) return 0.;
  
  size_t l,j;
  double x, g;
  size_t N = nRMS.maxLayer()+1;          // number of layers in nRMS
  size_t M = TFmap.maxLayer()+1;         // number of layers in TFmap
  size_t n = I*N/M;                      // layer low index in nRMS
  size_t m = (I+1)*(N/M);                // layer high index in nRMS
  double t = TFmap.start();
  double T = nRMS.start();
  slice  S = TFmap.getSlice(I);
  double rATe = TFmap.wrate();           // map rate

  double rESn = nRMS.frequency(1)-nRMS.frequency(0);       // noise F resolution
  
  //   printf("%4d %4d %16.3f %16.3f\n",M,N,t,T);

  if(M>N || t>T) {
    cout<<"detector::getNoise(): invalid noise rms array nRMS\n"; 
    return 0.;
  }
  
  size_t L = size_t(TFmap.getlow()/rESn+0.1);   // low F boundary
  size_t H = size_t(TFmap.gethigh()/rESn+0.1);  // high F boundary
  if(n>=H || m<=L) return 0.;                   // out of boundaries 

  //   printf("%4d %4d\n",n,m);

  if(n >= m) {
    cout<<"detector::getNoise():: invalid noise rms array nRMS\n"; 
    return 0.;
  }

  S = nRMS.getSlice(n);
  size_t K = S.size();         // size of noise vector in each layer
  
  if(!K) return 0.;
  
  if(J < 0) {                  // get noise rms for specified layer
    wavearray<double> rms(K);
    rms = 0.;
    
    for(l=n; l<m; l++) {      
      S = nRMS.getSlice(l);
      g = (n<L || m>H) ? 1.e10 : 1.;  // supress layers below HP cut-off frequency
      for(j=0; j<rms.size(); j++) {
	x = nRMS.data[S.start()+j*S.stride()]*g;
	rms.data[j] += 1./x/x;
      }
    }

    for(j=0; j<rms.size(); j++) {
      rms.data[j] = sqrt((double(m)-double(n))/rms.data[j]);
    }
    
    return rms.mean();
  }
  
  else {                       // get noise rms for specified pixel
    
    double RMS = 0;
    
    t += J/rATe;   // pixel GPS time
    
    int inRMS = int((t-nRMS.start())*nRMS.rate());
    int inVAR = nVAR.size() ? int((t-nVAR.start())*nVAR.rate()) : 0;
    
    if(inRMS < 0) inRMS = 0; 
    else if(size_t(inRMS)>=K && K) inRMS = K-1; 
    
    if(inVAR <= 0) inVAR = 0; 
    else if(size_t(inVAR)>=nVAR.size()) inVAR = nVAR.size()-1; 
    
    for(l=n; l<m; l++) {         // get noise vector for specified fl-fh
      S = nRMS.getSlice(l);
      g = (n<L || m>H) ? 1.e10 : 1.;    // supress layers below HP cut-off
      x = nRMS.data[S.start()+inRMS*S.stride()]*g;
      RMS += 1./x/x;
    }
    
    RMS /= double(m)-double(n);
    RMS  = sqrt(1./RMS);
    
    if(!nVAR.size() || rESn*n<nVAR.getlow() || rESn*m>nVAR.gethigh()) return RMS;
    
    return RMS*double(nVAR.data[inVAR]);
  }
}


//**************************************************************************
// set noise rms in pixel data array in cluster structure
//**************************************************************************
bool detector::setrms(netcluster* wc, size_t I)
{
   size_t i,j,n,m;
   size_t M = wc->size();

   if(!M) return false;

   if(!nRMS.size()) return false;

   size_t max_layer = nRMS.maxLayer();
   netpixel* p = NULL;                           // pointer to pixel structure
   slice S;

   int k;
   int K     = nRMS.size()/(max_layer+1);        // number of RMS measurements per layer
   double To = nRMS.start();
   double Ro = nRMS.wrate();
   double Fo = nRMS.frequency(0);                // central frequency of zero layer
   double dF = nRMS.frequency(1)-Fo;             // nRMS frequency resolution
   double fl = wc->getlow()-0.1;
   double fh = wc->gethigh()+0.1;

   double x,f,t,r,F,g;
   Fo = Fo==0. ? 0.5 : 0.;                       // WDM : wavelet frequency correction

   for(i=0; i<M; i++){
      p = wc->getPixel(0,i);

      if(p->frequency > max_layer ||
	 int(p->rate/Ro+0.01) < 1 || 
	 p->frequency == 0) {                    // never use zero layer
         cout<<"detector::setrms() - illegal pixel from zero level\n"; 
         exit(0);
      } 

      x = p->frequency-Fo;                       // fractional frequency index for wavelet and WDM
      f = x*p->rate/2.;
      n = size_t(f/dF+0.6);                      // first layer in nRMS
      F = (x+1)*p->rate/2.;
      m = size_t(F/dF+0.6);                      // last layer in nRMS
      if(m>max_layer) m=max_layer+1;
      t = p->getdata('I',I)/p->rate/p->layers;   // takes into account time lag
      t+= wc->start;                             // gps time
      k = int((t-To)*Ro);                        // time index in the noise array

      if(k>=K) k -= k ? 1 : 0; 
      if(k<0 || n>=m || k>=K) {
	 cout<<"detector::setrms() - invalid input: ";
	 cout<<k<<" "<<n<<" "<<m<<" "<<f<<" "<<F<<" "<<t<<endl;
	 cout<<p->frequency<<" "<<p->rate/2.<<" "<<dF<<" "<<fl<<endl;
	 exit(0);
      }

// get noise rms for specified pixel

      r = 0.;
      for(j=n; j<m; j++) {
	 S = nRMS.getSlice(j);
	 g = (f<fl || F>fh) ? 1.e10 : 1.;        // supress layers below HP cut-off
	 x = nRMS.data[S.start()+k*S.stride()];
	 r += 1./x/x;
      }

      if(nVAR.size()) {                          // mitigation of PSD variability added on June 2019
	 double ff,FF;                           // does not affect analysis if nVAR is not set
	 ff = f<nVAR.getlow()  ? nVAR.getlow() : f;
	 if(ff>=nVAR.gethigh()) ff=nVAR.gethigh();
	 FF = F>nVAR.gethigh() ? nVAR.gethigh() : F;
	 if(FF<=nVAR.getlow())  FF=nVAR.getlow();
	 ff = 2*(FF-ff)/p->rate;                 // band fraction affected by variability
	 FF = nVAR.get(t,0.5/p->rate);           // variability
	 r *= 1-ff+ff*FF*FF;                     // corrected RMS
      }

      r  = (m-n)/r;
      if(r <= 0) cout<<"detector:setrms() error!\n";
      p->setdata(sqrt(r),'N',I);
      
   }
   return true;
}

//**************************************************************************
// apply band pass filter with cut-offs specified by parameters (used by 1G)
//**************************************************************************
void detector::bandPass1G(double f1, double f2)
{
   int i;
   double dF = TFmap.frequency(1)-TFmap.frequency(0);              // frequency resolution
   double fl = fabs(f1)>0. ? fabs(f1) : this->TFmap.getlow();
   double fh = fabs(f2)>0. ? fabs(f2) : this->TFmap.gethigh();
   size_t n  = TFmap.pWavelet->m_WaveType==WDMT ? size_t((fl+dF/2.)/dF+0.1) : size_t(fl/dF+0.1);
   size_t m  = TFmap.pWavelet->m_WaveType==WDMT ? size_t((fh+dF/2.)/dF+0.1)-1 : size_t(fh/dF+0.1)-1;
   size_t M  = this->TFmap.maxLayer()+1;
   wavearray<double> w;

   if(n>m) return;

   for(i=0; i<int(M); i++) {                            // ......f1......f2......

     if((f1>=0 && i>=n) && (f2>=0 && i<=m)) continue;   // zzzzzz..........zzzzzz       band pass
     if((f1<0 && i<n) || (f2<0 && i>m))     continue;   // ......zzzzzzzzzz......       band cut
     if((f1<0 && f2>=0 && i<n))             continue;   // ......zzzzzzzzzzzzzzzz       low  pass
     if((f1>=0 && f2<0 && i>=m))            continue;   // zzzzzzzzzzzzzzzz......       high pass

     this->TFmap.getLayer(w,i+0.01); w=0.; this->TFmap.putLayer(w,i+0.01);
     this->TFmap.getLayer(w,-i-0.01); w=0.; this->TFmap.putLayer(w,-i-0.01);
   }
   return;
}

//**************************************************************************
// apply band pass filter with cut-offs specified by parameters
//**************************************************************************
/*
void detector::bandPass(double f1, double f2, double a)
{
// assign constant value a to wseries layer coefficients 
// 0utside of the band defined by frequencies f1 and f2
// f1>0, f2>0 -  zzzzzz..........zzzzzz 	band pass
// f1<0, f2<0 -  ......zzzzzzzzzz...... 	band cut 
// f1<0, f2>0 -  ......zzzzzzzzzzzzzzzz 	low  pass
// f1>0, f2<0 -  zzzzzzzzzzzzzzzz...... 	high pass
   int i;
   double dF = TFmap.frequency(1)-TFmap.frequency(0);              // frequency resolution
   double fl = fabs(f1)>0. ? fabs(f1) : this->TFmap.getlow();
   double fh = fabs(f2)>0. ? fabs(f2) : this->TFmap.gethigh();
   size_t n  = TFmap.pWavelet->m_WaveType==WDMT ? size_t((fl+dF/2.)/dF+0.1) : size_t(fl/dF+0.1);
   size_t m  = TFmap.pWavelet->m_WaveType==WDMT ? size_t((fh+dF/2.)/dF+0.1)-1 : size_t(fh/dF+0.1)-1;
   size_t M  = this->TFmap.maxLayer()+1;
   wavearray<double> w;

   if(n>m) return;

   for(i=0; i<int(M); i++) {                          	// ......f1......f2......

     if((f1>=0 && i>=n) && (f2>=0 && i<=m)) continue; 	// zzzzzz..........zzzzzz 	band pass
     if((f1<0 && i<n) || (f2<0 && i>m))     continue; 	// ......zzzzzzzzzz...... 	band cut 
     if((f1<0 && f2>=0 && i<n))             continue; 	// ......zzzzzzzzzzzzzzzz 	low  pass
     if((f1>=0 && f2<0 && i>=m))            continue; 	// zzzzzzzzzzzzzzzz...... 	high pass

     this->TFmap.getLayer(w,i+0.01); w=a; this->TFmap.putLayer(w,i+0.01);
     this->TFmap.getLayer(w,-i-0.01); w=a; this->TFmap.putLayer(w,-i-0.01);
   }
   return;
}
*/
//**************************************************************************
// calculate hrss of injected responses
// returns number of eligible injections
// update MDC series with whitened time series
//**************************************************************************
size_t detector::setsim(WSeries<double> &wi, std::vector<double>* pT, double dT, double offset, bool saveWF)  
{
   int j,nstop,nstrt,n,m,J;
   size_t i,k;
   double a,b,T,E,D,H,f;
   double R  = this->rate;                         // original data rate
   size_t K  = pT->size();
   size_t I  = wi.maxLayer()+1;
   size_t M  = 0;
   bool pOUT = dT>0. ? false : true;               // printout flag

   dT = fabs(dT);

   if(wi.size() != TFmap.size()) {
     cout<<"setsim(): mismatch between MDC size "
	 <<wi.size()<<" and data size "<<TFmap.size()<<"\n";
     exit(1);
   } 

   if(!K) return K;
   if(!this->nRMS.size() || !this->TFmap.size()) return 0;

   if(this->HRSS.size() != K) {
     this->HRSS.resize(K);
     this->ISNR.resize(K);
     this->FREQ.resize(K);
     this->BAND.resize(K);
     this->TIME.resize(K);
     this->TDUR.resize(K);
   }
   this->HRSS = 0.;
   this->ISNR = 0.;
   this->FREQ = 0.;
   this->BAND = 0.;
   this->TIME = 0.;
   this->TDUR = 0.;

   WSeries<double> W; W=wi;
   WSeries<double> w;
   WSeries<double> hot; hot=wi; hot.Inverse();
   wavearray<double> x;
   std::slice S;

// whiten injection if noise array is filled in

   if(W.pWavelet->m_WaveType==WDMT) {	// WDM type
     if(!W.white(this->nRMS,1)) {	// whiten  0 phase WSeries
       cout<<"detector::setsim error: invalid noise array\n"; exit(1);
     }
     if(!W.white(this->nRMS,-1)) {	// whiten 90 phase WSeries
       cout<<"detector::setsim error: invalid noise array\n"; exit(1);
     }
     w = W;				// whitened WS
     WSeries<double> wtmp = w;
     w.Inverse();
     wtmp.Inverse(-2);
     w += wtmp;
     w *= 0.5;
   } else {				// wavelet type
     if(!W.white(this->nRMS)) {
       cout<<"detector::setsim error: invalid noise array\n"; exit(1);
     }
     w = W;				// whitened WS
     w.Inverse();			// whitened TS
   }


// isolate injections in time series w

   size_t N    = w.size();                        // MDC size
   double rate = w.wavearray<double>::rate();     // simulation rate
   double bgps = w.start()+offset+1.;             // analysed start time
   double sgps = w.start()-offset+N/rate-1.;      // analysed stop time

   for(k=0; k<K; k++) {

     T = (*pT)[k] - w.start();

     nstrt = int((T - dT)*rate); 
     nstop = int((T + dT)*rate);  
     if(nstrt<=0) nstrt = 0; 
     if(nstop>=int(N)) nstop = N; 
     if(nstop<=0) continue;                     // outside of the segment 
     if(nstrt>=int(N)) continue;                // outside of the segment

     E = T = 0.;
     for(j=nstrt; j<nstop; j++) {            
       a = w.data[j]; 
       T += a*a*j/rate;                         // central time
       E += a*a;                                // SNR
     }
     T /= E;                                    // central time for k-th injection
     
// zoom in

     nstrt = int((T - dT)*rate); 
     nstop = int((T + dT)*rate);  
     if(nstrt<=0) nstrt = 0; 
     if(nstop>=int(N)) nstop = N; 
     if(nstop<=0) continue;                     // outside of the segment 
     if(nstrt>=int(N)) continue;                // outside of the segment

     E = T = D = H = 0.;
     for(j=nstrt; j<nstop; j++) {            
       a = w.data[j]; 
       T += a*a*j/rate;                         // central time
       E += a*a;                                // SNR
     }
     T /= E;
     
     m = int((T - dT)*W.wrate()); 
     n = int((T + dT)*W.wrate());  
     S = W.getSlice(0);                        // zero layer
     if(m<=0) m = 0;                           // check left
     if(n>=int(S.size())) n = S.size()-1;      // check right 
 
     for(j=nstrt; j<nstop; j++) {            
       a = w.data[j]*(j/rate-T); 
       D += a*a;                                // duration
       a = hot.data[j];
       H += a*a;                                // hrss
     }
     
     D = sqrt(D/E); 
     H = sqrt(H/R);
     T += w.start();
     if(T<bgps || T>sgps) continue;             // outside of the segment
     
     this->ISNR.data[k] = E;
     this->TIME.data[k] = T;
     this->TDUR.data[k] = D;
     this->HRSS.data[k] = H;

     E = 0.;
     for(i=0; i<I; i++) {
       S = W.getSlice(i+0.001);
       J = S.stride();
       a = W.rms(slice(S.start()+m*J,n-m,J));   // get rms
       f = W.frequency(i);                      // layer frequency
       this->FREQ.data[k] += f*a*a;             // central frequency
       this->BAND.data[k] += f*f*a*a;           // bandwidth
       E += a*a;
     }
       
     this->FREQ.data[k] /= E;
     a = this->FREQ.data[k];
     b = this->BAND.data[k]/E - a*a;
     this->BAND.data[k]  = b>0. ? sqrt(b) : 0.; 
     
// save waveform 

     double time = this->TIME.data[k];
     double tdur = this->TDUR.data[k];
     double tDur = 6*tdur;
     if (tDur > dT) tDur = dT;
     int nStrt = int((time-tDur-w.start())*rate);
     int nStop = int((time+tDur-w.start())*rate);
     if(nStrt<0) nStrt = 0;
     if(nStop>int(N)) nStop = N;

// window the injected waveforms  

     size_t I    = int(2.*dT*rate);
     int    OS   = int((time - dT - w.start())*rate);
     double ms = 0;

     for (j=0;j<I;j++) ms += ((OS+j>=0)&&(OS+j<N)) ? w.data[OS+j]*w.data[OS+j] : 0.;

     OS += I/2;       
     double a,b;
     double sum  = ((OS>=0)&&(OS<N)) ? w.data[OS]*w.data[OS] : 0.;
     for(j=1; j<int(I/2); j++) {
        a = ((OS-j>=0)&&(OS-j<N)) ? w.data[OS-j] : 0.;
        b = ((OS+j>=0)&&(OS+j<N)) ? w.data[OS+j] : 0.;
        sum += a*a+b*b;
        if(sum/ms > 0.999) break;
     }

     nStrt = int((time-w.start())*rate)-j;
     nStop = int((time-w.start())*rate)+j;
     if(nStrt<0) nStrt = 0;
     if(nStop>int(N)) nStop = N;

     wavearray<double>* wf = new wavearray<double>;
     wf->rate(rate);
     wf->start(w.start()+nStrt/rate);
     wf->resize(nStop-nStrt);
     for(j=nStrt; j<nStop; j++) {
       wf->data[j-nStrt] = w.data[j];
     }

// apply freq cuts 

     double f_low  = this->TFmap.getlow();
     double f_high = this->TFmap.gethigh();
     //cout << "f_low : " << f_low << " f_high : " << f_high << endl;
     wf->FFTW(1);
     double Fs=((double)wf->rate()/(double)wf->size())/2.;
     for (j=0;j<wf->size()/2;j+=2) {
       double f=j*Fs;
       if((f<f_low)||(f>f_high)) {wf->data[j]=0.;wf->data[j+1]=0.;}
     }
     wf->FFTW(-1);

// compute SNR,TIME,DUR within the search frequency band

     E = T = D = 0.;
     for(j=0;j<wf->size();j++) {
       a = wf->data[j]; 
       T += a*a*j/rate;                         // central time
       E += a*a;                                // SNR
     }
     T /= E;
     
     for(j=0;j<wf->size();j++) {
       a = wf->data[j]*(j/rate-T); 
       D += a*a;                                // duration
     }
     
     D = sqrt(D/E); 
     T += wf->start();
    
     this->ISNR.data[k] = E;
     this->TIME.data[k] = T;
     this->TDUR.data[k] = D;

     if (saveWF) {
       wf->resetFFTW();
       IWFID.push_back(k);
       IWFP.push_back(wf);
     } else {
       delete wf;
     }

     // save strain waveform
     if (saveWF) {

       // compute central time
       T = E = 0.;
       for(j=nstrt; j<nstop; j++) {
         a = hot.data[j]; 
         T += a*a*j/rate;                         // central time
         E += a*a;                                // energy
       }
       T /= E;

       // compute the time range containing the 0.999 of the total energy
       I    = int(2.*dT*rate);
       OS   = int((T - dT)*rate);
       ms = 0;

       for (j=0;j<I;j++) ms += ((OS+j>=0)&&(OS+j<N)) ? hot.data[OS+j]*hot.data[OS+j] : 0.;

       OS += I/2;       
       sum  = ((OS>=0)&&(OS<N)) ? hot.data[OS]*hot.data[OS] : 0.;
       for(j=1; j<int(I/2); j++) {
          a = ((OS-j>=0)&&(OS-j<N)) ? hot.data[OS-j] : 0.;
          b = ((OS+j>=0)&&(OS+j<N)) ? hot.data[OS+j] : 0.;
          sum += a*a+b*b;
          if(sum/ms > 0.999) break;
       }

       nStrt = int(T*rate)-j;
       nStop = int(T*rate)+j;
       if(nStrt<0) nStrt = 0;
       if(nStop>int(N)) nStop = N;

       // select strain mdc data 
       wavearray<double>* WF = new wavearray<double>;
       WF->rate(rate);
       WF->start(hot.start()+nStrt/rate);
       WF->resize(nStop-nStrt);
       for(j=nStrt; j<nStop; j++) WF->data[j-nStrt] = hot.data[j];

       // store strain data (use ID=-(k+1))
       IWFID.push_back(-(k+1));
       IWFP.push_back(WF);
     }

     if(pOUT) 
       printf("%3d T+-dT: %8.3f +-%5.3f, F+-dF: %4.0f +-%4.0f, SNR: %6.1e, hrss: %6.1e\n",
	      int(M), T-bgps, D, FREQ.data[k], BAND.data[k], E, H);

//     (*pT)[k] = T;
     M++;
   }
   wi = w;
   return M;
}

//**************************************************************************
// modify input signals (wi) at times pT according the factor pF
//**************************************************************************
size_t detector::setsnr(wavearray<double> &wi, std::vector<double>* pT, std::vector<double>* pF, double dT, double offset) 
{
   int j,nstop,nstrt;
   size_t k;
   double F,T;
   size_t K    = pT->size();
   size_t N    = wi.size();
   size_t M    = 0;
   double rate = wi.rate();                        // simulation rate

   dT = fabs(dT);

   wavearray<double> w; w=wi;
// isolate injections

   for(k=0; k<K; k++) {

     F = (*pF)[k];
     T = (*pT)[k] - w.start();

     nstrt = int((T - dT)*rate);
     nstop = int((T + dT)*rate);
     if(nstrt<=0) nstrt = 0;
     if(nstop>=int(N)) nstop = N;
     if(nstop<=0) continue;                     // outside of the segment
     if(nstrt>=int(N)) continue;                // outside of the segment

     for(j=nstrt; j<nstop; j++) {
       w.data[j]*=F;
     }

     M++;
   }
   wi = w;
   return M;
}


//**************************************************************************    
// apply sample shift to time series in TFmap                 
//**************************************************************************    
void detector::delay(double theta, double phi)                                   
{                                                                               
  if(!TFmap.size()) return;
  double R = this->TFmap.wavearray<double>::rate();
  double T = this->getTau(theta,phi);   // time delay: +/- increase/decrease gps  
  size_t n = size_t(fabs(T)*R);         // shift in samples
  size_t m = this->TFmap.size(); 
  wavearray<double> w; 
  w = this->TFmap;
  TFmap = 0.;

  if(T<0) TFmap.cpf(w,m-n,0,n);       // shift forward
  else    TFmap.cpf(w,m-n,n,0);       // shift backward
  return;                                                                      
}                                                                               

//**************************************************************************    
// apply sample shift to input time series                 
//**************************************************************************    
void detector::delay(wavearray<double> &x, double theta, double phi)                                   
{                                                                               
  if(!x.size()) return;
  double R = x.rate();
  double T = this->getTau(theta,phi);   // time delay: +/- increase/decrease gps  
  size_t n = size_t(fabs(T)*R);         // shift in samples
  size_t m = x.size(); 
  wavearray<double> w; 
  w = x;
  x = 0.;

  if(T<0) x.cpf(w,m-n,0,n);       // shift forward
  else    x.cpf(w,m-n,n,0);       // shift backward
  return;                                                                      
}                                                                               

//**************************************************************************    
// apply time shift T to input time series                 
//**************************************************************************    
void detector::delay(wavearray<double> &x, double T)                                   
{                                                                               
  if(!x.size()) return;
  double R = x.rate();
  size_t n = size_t(fabs(T)*R+0.5);     // shift in samples
  size_t m = x.size(); 
  wavearray<double> w; 
  w = x;
  x = 0.;

  if(T<0) x.cpf(w,m-n,0,n);       // shift forward
  else    x.cpf(w,m-n,n,0);       // shift backward
  return;                                                                      
}                                                                               

double detector::getWFtime(char atype) 
{
// returns central time of reconstructed waveform
   double e;
   double T = 0.;
   double E = 0.;
   WSeries<double>* wf = atype=='S' ? &this->waveForm : &this->waveBand;
   for(size_t i=0; i<wf->size(); i++) {
      e  = wf->data[i]*wf->data[i]; 
      T += e*i;
      E += e;
   }
   return E>0. ? wf->start()+T/E/wf->rate() : 0.;
}

double detector::getWFfreq(char atype) 
{
// returns central frequency of reconstructed waveform
   double e;
   double F = 0.;
   double E = 0.;
   WSeries<double>* wf = atype=='S' ? &this->waveForm : &this->waveBand;
   wf->FFTW(1);
   for(size_t i=0; i<wf->size(); i+=2) {
      e  = wf->data[i]*wf->data[i]; 
      e += wf->data[i+1]*wf->data[i+1]; 
      F += e*i;
      E += e;
   }
   return E>0. ? 0.5*F*wf->rate()/E/wf->size() : 0.;
}

//______________________________________________________________________________
void detector::print()
{
  detectorParams xdP = getDetectorParams();

  char LAT;
  double theta_t=xdP.latitude;
  if(theta_t>0) LAT='N'; else {LAT='S';theta_t=-theta_t;}
  int theta_d = int(theta_t);
  int theta_m = int((theta_t-theta_d)*60);
  float theta_s = (theta_t-theta_d-theta_m/60.)*3600.;

  char LON;
  double phi_t=xdP.longitude;
  if(phi_t>0) LON='E'; else {LON='W';phi_t=-phi_t;}
  int phi_d = int(phi_t);
  int phi_m = int((phi_t-phi_d)*60);
  float phi_s = (phi_t-phi_d-phi_m/60.)*3600.;

  cout << endl;
  cout << "----------------------------------------------" << endl;
  cout << "IFO : " << xdP.name << " (Geographic Coordinates) " << endl;
  cout << "----------------------------------------------" << endl;
  cout << endl;
  cout << "latitude : " << xdP.latitude << " longitude : " << xdP.longitude << endl;
  cout << endl;
  cout << "LAT : " << LAT << " " << theta_d << ", " << theta_m << ", " << theta_s << endl;
  cout << "LON : " << LON << " " << phi_d   << ", " << phi_m   << ", " << phi_s   << endl;
  cout << endl;
  int precision=cout.precision(12);
  // radius vector to beam splitter
  cout << "Rv  : " << Rv[0] << " " << Rv[1] << " " << Rv[2] << " " << endl;
  // vector along x-arm
  cout << "Ex  : " << Ex[0] << " " << Ex[1] << " " << Ex[2] << " " << endl;
  // vector along y-arm
  cout << "Ey  : " << Ey[0] << " " << Ey[1] << " " << Ey[2] << " " << endl;
  cout << endl;
  cout.precision(precision);
  cout << "Ex-North Angle Clockwise (deg) : " << xdP.AzX << endl;
  cout << "Ey-North Angle Clockwise (deg) : " << xdP.AzY << endl;
  cout << endl;
  if(polarization==TENSOR) cout << "GW Polarization = TENSOR" << endl;
  if(polarization==SCALAR) cout << "GW Polarization = SCALAR" << endl;
  cout << endl;

  return;
}

//______________________________________________________________________________
void detector::Streamer(TBuffer &R__b)
{
   // Stream an object of class detector.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      TNamed::Streamer(R__b);
      R__b.ReadStaticArray((char*)Name);
      R__b.StreamObject(&(dP),typeid(detectorParams));
      R__b.ReadStaticArray((double*)Rv);
      R__b.ReadStaticArray((double*)Ex);
      R__b.ReadStaticArray((double*)Ey);
      R__b.ReadStaticArray((double*)DT);
      R__b.ReadStaticArray((double*)ED);
      if(R__v > 2) R__b >> ifoID;
      R__b >> sHIFt;
      if(R__v > 1) {
        void *ptr_polarization = (void*)&polarization;
        R__b >> *reinterpret_cast<Int_t*>(ptr_polarization);
      }

      if(R__v > 3) {
        R__b >> wfSAVE;
        if(wfSAVE) {
          HRSS.Streamer(R__b);
          ISNR.Streamer(R__b);
          FREQ.Streamer(R__b);
          BAND.Streamer(R__b);
          TIME.Streamer(R__b);
          TDUR.Streamer(R__b);
          if(wfSAVE==1||wfSAVE==3) {
            {
              vector<int> &R__stl =  IWFID;
              R__stl.clear();
              int R__i, R__n;
              R__b >> R__n;
              R__stl.reserve(R__n);
              for (R__i = 0; R__i < R__n; R__i++) {
                int R__t;
                R__b >> R__t;
                R__stl.push_back(R__t);
              }
            }
            {  
              vector<wavearray<double>*> &R__stl =  IWFP;
              R__stl.clear();
              int R__i, R__n;
              R__b >> R__n;
              R__stl.reserve(R__n);
              for (R__i = 0; R__i < R__n; R__i++) {
                wavearray<double>* R__t = new wavearray<double>;
                R__t->Streamer(R__b);
                R__stl.push_back(R__t);
              }
            }
          }
          if(wfSAVE==2||wfSAVE==3) {
            { 
              vector<int> &R__stl =  RWFID;
              R__stl.clear();
              int R__i, R__n;
              R__b >> R__n;
              R__stl.reserve(R__n);
              for (R__i = 0; R__i < R__n; R__i++) {
                int R__t;
                R__b >> R__t;
                R__stl.push_back(R__t);
              }
            }
            {
              vector<wavearray<double>*> &R__stl =  RWFP;
              R__stl.clear();
              int R__i, R__n;
              R__b >> R__n;
              R__stl.reserve(R__n);
              for (R__i = 0; R__i < R__n; R__i++) {
                wavearray<double>* R__t = new wavearray<double>;
                R__t->Streamer(R__b);
                R__stl.push_back(R__t);
              }
            }
          }
        }
      }

      R__b.CheckByteCount(R__s, R__c, detector::IsA());
   } else {
      R__c = R__b.WriteVersion(detector::IsA(), kTRUE);
      TNamed::Streamer(R__b);
      R__b.WriteArray(Name, 16);
      R__b.StreamObject(&(dP),typeid(detectorParams));
      R__b.WriteArray(Rv, 3);
      R__b.WriteArray(Ex, 3);
      R__b.WriteArray(Ey, 3);
      R__b.WriteArray(DT, 9);
      R__b.WriteArray(ED, 5);
      R__b << ifoID;
      R__b << sHIFt;
      R__b << (Int_t)polarization;

      R__b << wfSAVE;
      if(wfSAVE) {
        HRSS.Streamer(R__b);
        ISNR.Streamer(R__b);
        FREQ.Streamer(R__b);
        BAND.Streamer(R__b);
        TIME.Streamer(R__b);
        TDUR.Streamer(R__b);
        if(wfSAVE==1||wfSAVE==3) {
          { 
            vector<int> &R__stl =  IWFID;
            int R__n=(&R__stl) ? int(R__stl.size()) : 0;
            R__b << R__n;
            if(R__n) {
              vector<int>::iterator R__k;
              for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
              R__b << (*R__k);
              }
            }
          }
          { 
            vector<wavearray<double> >  IWF(IWFP.size());
            for(int i=0;i<IWFP.size();i++) IWF[i] = *IWFP[i];
            vector<wavearray<double> > &R__stl =  IWF;
            int R__n=(&R__stl) ? int(R__stl.size()) : 0;
            R__b << R__n;
            if(R__n) {
              vector<wavearray<double> >::iterator R__k;
              for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
                ((wavearray<double>&)(*R__k)).Streamer(R__b);  
              }
            }
          } 
        }
        if(wfSAVE==2||wfSAVE==3) {
          { 
            vector<int> &R__stl =  RWFID;
            int R__n=(&R__stl) ? int(R__stl.size()) : 0;
            R__b << R__n;
            if(R__n) {
              vector<int>::iterator R__k;
              for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
              R__b << (*R__k);
              }
            }
          }
          {
            vector<wavearray<double> >  RWF(RWFP.size());
            for(int i=0;i<RWFP.size();i++) RWF[i] = *RWFP[i];
            vector<wavearray<double> > &R__stl =  RWF;
            int R__n=(&R__stl) ? int(R__stl.size()) : 0;
            R__b << R__n;
            if(R__n) {
              vector<wavearray<double> >::iterator R__k;
              for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
                ((wavearray<double>&)(*R__k)).Streamer(R__b);  
              }
            }
          }
        }
      }

      R__b.SetByteCount(R__c, kTRUE);
   }
}

