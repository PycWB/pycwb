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
// WAT sky map class
// ++++++++++++++++++++++++++++++++++++++++++++++

#define SKYMAP_CC
#include "skymap.hh"
#include "wavearray.hh"
#include "TObjArray.h"
#include "TObjString.h"
#include "TSystem.h"
#include "time.hh"

using namespace std;

#ifdef _USE_HEALPIX
unsigned int healpix_repcount (tsize npix)
  {
  if (npix<1024) return 1;
  if ((npix%1024)==0) return 1024;
  return isqrt (npix/12);
  }
#endif

ClassImp(skymap)

// constructors

skymap::skymap() { 

   deg2rad = PI/180.;
   rad2deg = 180./PI;
   healpix = NULL;
   healpix_order = 0;
}

skymap::skymap(double sms,double t1,double t2,double p1,double p2)
{
   int i,j;
   int ntheta, nphi;
   double c,s,p,d;
   double a = PI/180.;

   if(t1<0 || t1>180. || t2<0 || t2>180. || t2<t1) {
      cout<<"skymap(): invalid theta parameters"<<endl;
      return;
   }

   if(p1<0 || p1>360. || p2<0 || p2>360. || p2<p1) {
      cout<<"skymap(): invalid phi parameters"<<endl;
      return;
   }

   this->sms = sms;             // step on phi and theta
   theta_1 = t1;                // theta range
   theta_2 = t2;                // set theta range
   phi_1   = p1;                // phi range
   phi_2   = p2;                // phi range

   ntheta  = 2*size_t((t2-t1)/sms/2.)+1;       // divisions on theta
   d = ntheta>1 ? (t2-t1)/(ntheta-1) : sms; 

   this->index.reserve(ntheta);                // reserve index array
   this->value.reserve(ntheta);                // reserve value array

   for(i=-ntheta/2; i<=ntheta/2; i++) {
     c = cos(((t2+t1)/2.+i*d)*a);
     s = sin(((t2+t1)/2.+i*d)*a);
     c = s>0. ? (cos(sms*a)-c*c)/s/s : 2.;
     p = c>0&&c<1. ? acos(c)/a : 361.;         // step on phi
     nphi = 2*size_t((p2-p1)/p/2.)+1;          // divisions on phi

     std::vector<double> v;

     v.reserve(nphi);
     for(j=0; j<nphi; j++) v.push_back(0.);
     index.push_back(nphi);                    // initialize index array
     value.push_back(v);                       // initialize value array
   }
   mIndex = mTheta = mPhi = -1;
   gps = -1.;

   deg2rad = PI/180.;
   rad2deg = 180./PI;
   healpix = NULL;
   healpix_order = 0;

}

skymap::skymap(char* ifile)  
{
#ifdef _USE_HEALPIX
   int i,j;
   int ntheta, nphi;

   sms     = 0.0;                // step on phi and theta
   theta_1 = 0.0;                // theta range
   theta_2 = 180.0;              // set theta range
   phi_1   = 0.0;                // phi range
   phi_2   = 360.0;              // phi range

   healpix = new Healpix_Map<double>();  //Healpix MAP
   read_Healpix_map_from_fits(ifile,*healpix,1,2);
   this->healpix_order = healpix->Order();

   ntheta  = 1;       // divisions on theta
   for(i=0;i<=healpix->Order();i++) ntheta=2*ntheta+1;

   this->index.reserve(ntheta);                // reserve index array
   this->value.reserve(ntheta);                // reserve value array
   for(i=1; i<=ntheta; i++) {

     int ring=i; int startpix; int ringpix; double costheta; double sintheta; bool shifted=false;
     healpix->get_ring_info(ring, startpix, ringpix, costheta, sintheta, shifted);
     nphi=ringpix; // divisions on phi

     std::vector<double> v;

     v.reserve(nphi);
     for(j=0; j<nphi; j++) v.push_back(0.);
     index.push_back(nphi);                    // initialize index array
     value.push_back(v);                       // initialize value array
   }
   mIndex = mTheta = mPhi = -1;
   gps = -1.;

   // fill skymap
   int L = healpix->Npix();
   for(int l=0;l<L;l++) this->set(l,(*healpix)[l]);

   deg2rad = PI/180.;
   rad2deg = 180./PI;
#else
   cout << "skymap::skymap(char* ifile) : not available - healpix not enabled" << endl;
   exit(1);
#endif
}

skymap::skymap(int healpix_order)  
{
#ifdef _USE_HEALPIX
   int i,j;
   int ntheta, nphi;

   sms     = 0.0;                // step on phi and theta
   theta_1 = 0.0;                // theta range
   theta_2 = 180.0;              // set theta range
   phi_1   = 0.0;                // phi range
   phi_2   = 360.0;              // phi range

   ntheta  = 1;       // divisions on theta
   for(i=0;i<=healpix_order;i++) ntheta=2*ntheta+1;

   healpix = new Healpix_Map<double>(healpix_order,RING);  //Healpix MAP
   int L = healpix->Npix();
   this->healpix_order = healpix->Order();

// TEST
/*
  char pixDir[8][3] = {"SW", "W", "NW", "N", "NE", "E", "SE", "S"};
  int ring=0; int startpix; int ringpix; double costheta; double sintheta; bool shifted=false;
  for(int i=1;i<=ntheta;i++) {
    ring=i;
    healpix->get_ring_info(ring, startpix, ringpix, costheta, sintheta, shifted);
    cout << "ring : " << ring << " startpix : " << startpix << " ringpix : " << ringpix << endl;
    fix_arr< int, 8 > result;
    for(int j=startpix;j<startpix+ringpix;j++) {
      healpix->neighbors (j, result);
      for(int k=0;k<8;k++) cout << "pixel : " << j << " pixDir : " << pixDir[k] << " " << result[k] << endl;
      cout << endl;
    }
  }
*/
// END TEST

   this->index.reserve(ntheta);                // reserve index array
   this->value.reserve(ntheta);                // reserve value array
   for(i=1; i<=ntheta; i++) {

     int ring=i; int startpix; int ringpix; double costheta; double sintheta; bool shifted=false;
     healpix->get_ring_info(ring, startpix, ringpix, costheta, sintheta, shifted);
     nphi=ringpix; // divisions on phi

     std::vector<double> v;

     v.reserve(nphi);
     for(j=0; j<nphi; j++) v.push_back(0.);
     index.push_back(nphi);                    // initialize index array
     value.push_back(v);                       // initialize value array
   }
   mIndex = mTheta = mPhi = -1;
   gps = -1.;

   deg2rad = PI/180.;
   rad2deg = 180./PI;
#else
   cout << "skymap::skymap(int healpix_order) : not available - healpix not enabled" << endl;
   exit(1);
#endif
}

skymap::skymap(TString ifile, TString name)  
{
  healpix = NULL;
  healpix_order = 0;

  TFile *rfile = TFile::Open(ifile);
  if (rfile==NULL) {
    cout << "skymap::skymap - Error : No " << ifile.Data() << " file !!!" << endl;
    exit(1);
  }
  // read skymap object
  if(rfile->Get(name)!=NULL) {
    *this = *(skymap*)rfile->Get(name);
  } else {
    cout << "skymap::skymap - Error : skymap is not contained in root file " << ifile.Data() << endl;
    exit(1);
  }
  rfile->Close();
}

skymap::skymap(const skymap& value)
{
   healpix = NULL;
   healpix_order = 0;

   *this = value;
}

// destructor

skymap::~skymap(){
  if(healpix!=NULL) delete healpix; 
}

//: operators 

skymap& skymap::operator=(const skymap& a)
{
   if(healpix!=NULL) {delete healpix;healpix=NULL;}

   if(a.healpix!=NULL) {  
#ifdef _USE_HEALPIX
     healpix = new Healpix_Map<double>(a.healpix->Order(),a.healpix->Scheme());
     this->healpix_order = healpix->Order();
#endif
   } else {
     healpix = NULL;
     this->healpix_order = 0;
   }

   sms     = a.sms;       // step on phi and theta
   theta_1 = a.theta_1;   // theta range
   theta_2 = a.theta_2;   // set theta range
   phi_1   = a.phi_1;     // phi range
   phi_2   = a.phi_2;     // phi range
   value   = a.value;
   index   = a.index;
   mTheta  = a.mTheta;
   mPhi    = a.mPhi;
   mIndex  = a.mIndex;
   gps     = a.gps;

   deg2rad = PI/180.;
   rad2deg = 180./PI;

   return *this;
}

skymap& skymap::operator+=(const skymap& a)
{
   size_t i,j;
   size_t n = value.size();
   size_t m;

   if( a.theta_1      != theta_1 ||
       a.theta_2      != theta_2 ||
       a.phi_1        != phi_1   ||
       a.phi_2        != phi_2   ||
       a.sms          != sms     ||
       a.value.size()    != n )
   {
      cout<<"skymap::operator+ - incompatible skymaps"<<endl;
   }

   
   for(i=0; i<n; i++) {
     m = value[i].size(); 
     if(m != a.value[i].size()) {
       cout<<"skymap::operator+ - incompatible skymaps"<<endl;
       break;
     }
     for(j=0; j<m; j++) { value[i][j] += a.value[i][j]; }
   }

   return *this;
}

skymap& skymap::operator-=(const skymap& a)
{
   size_t i,j;
   size_t n = value.size();
   size_t m;

   if( a.theta_1      != theta_1 ||
       a.theta_2      != theta_2 ||
       a.phi_1        != phi_1   ||
       a.phi_2        != phi_2   ||
       a.sms          != sms     ||
       a.value.size()    != n )
   {
      cout<<"skymap::operator- - incompatible skymaps"<<endl;
   }

   
   for(i=0; i<n; i++) {
     m = value[i].size(); 
     if(m != a.value[i].size()) {
       cout<<"skymap::operator- - incompatible skymaps"<<endl;
       break;
     }
     for(j=0; j<m; j++) { value[i][j] -= a.value[i][j]; }
   }

   return *this;
}

skymap& skymap::operator*=(const skymap& a)
{
   size_t i,j,m;
   size_t n = value.size();

   if( a.theta_1      != theta_1 ||
       a.theta_2      != theta_2 ||
       a.phi_1        != phi_1   ||
       a.phi_2        != phi_2   ||
       a.sms          != sms     ||
       a.value.size()    != n )
   {
      cout<<"skymap::operator* - incompatible skymaps"<<endl;
   }

   
   for(i=0; i<n; i++) {
     m = value[i].size(); 
     if(m != a.value[i].size()) {
       cout<<"skymap::operator* - incompatible skymaps"<<endl;
       break;
     }
     for(j=0; j<m; j++) { value[i][j] *= a.value[i][j]; }
   }
   return *this;
}

skymap& skymap::operator/=(const skymap& a)
{
   size_t i,j,m;
   size_t n = value.size();

   if( a.theta_1      != theta_1 ||
       a.theta_2      != theta_2 ||
       a.phi_1        != phi_1   ||
       a.phi_2        != phi_2   ||
       a.sms          != sms     ||
       a.value.size()    != n )
   {
      cout<<"skymap::operator/ - incompatible skymaps"<<endl;
   }

   
   for(i=0; i<n; i++) {
     m = value[i].size(); 
     if(m != a.value[i].size()) {
       cout<<"skymap::operator/ - incompatible skymaps"<<endl;
       break;
     }
     for(j=0; j<m; j++) { value[i][j] /= a.value[i][j]!=0. ? a.value[i][j] : 1.; }
   }
   return *this;
}

skymap& skymap::operator=(const double a)
{
   size_t i,j;
   size_t n =  value.size();
   size_t m;

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { this->value[i][j] = a; }
   }
   return *this;
}

skymap& skymap::operator*=(const double a)
{
   size_t i,j;
   size_t n =  value.size();
   size_t m;

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { this->value[i][j] *= a; }
   }
   return *this;
}

skymap& skymap::operator+=(const double a)
{
   size_t i,j,m;
   size_t n =  value.size();

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { this->value[i][j] += a; }
   }
   return *this;
}

double skymap::max()
{
   size_t i,j,m;
   size_t k = 0;
   size_t n = value.size();
   double a = -1.e100;
   double x;

   mTheta = mPhi = mIndex = -1;

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { 
       x = this->value[i][j];
       if(x>a) { a=x; mTheta=i; mPhi=j; mIndex=k; }
       k++;
     }
   }
   return a;
}

double skymap::min()
{
   size_t i,j,m;
   size_t k = 0;
   size_t n = value.size();
   double a = 1.e100;
   double x;

   mTheta = mPhi = mIndex = -1;

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { 
       x = this->value[i][j];
       if(x<a) { a=x; mTheta=i; mPhi=j; mIndex=k; }
       k++;
     }
   }
   return a;
}

double skymap::mean()
{
   size_t i,j,m;
   size_t n = value.size();
   size_t k = 0;
   double a = 0.;

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     k += m;
     for(j=0; j<m; j++) { 
       a += this->value[i][j];
     }
   }
   return a/double(k);
}

double skymap::fraction(double t)
{
   size_t i,j,m;
   size_t n = value.size();
   size_t k = 0;
   double a = 0.;

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     k += m;
     for(j=0; j<m; j++) { 
       if(this->value[i][j]>t) a += 1.;
     }
   }
   return a/double(k);
}

double skymap::norm(double a)
{
   size_t i,j,m;
   size_t n = value.size();
   double s = 0;
   double x;

   (*this) += 1.-this->max(); 
   if(a==0.) return a; 
     
   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { 
       x  = exp((this->value[i][j]-1.)*fabs(a));
       s += x;
       if(a>0.) this->value[i][j] = x;
     }
   }
   if(a>0.) (*this) *= 1./s; 
   return s;
}

void skymap::downsample(wavearray<short>& index, size_t k)
{
   size_t i,j;
   size_t n = value.size();
   size_t m = this->size();
   size_t l = 0;

   if(index.size() != m) index.resize(m);
   index = 0;

   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { 
       if(i&1) index.data[l] = 1;
       if(j&1 && k==4) index.data[l] = 1;
       l++;
     }
   }
   return;
}

#ifdef _USE_HEALPIX
//: dump skymap into fits file
void skymap::Dump2fits(const char* file, double gps_obs, const char* configur, 
                       const char* TTYPE1, const char* TUNIT1, char coordsys)  
{

   if(healpix!=NULL) {  
     int L = healpix->Npix();
     for(int l=0;l<L;l++) (*healpix)[l]=this->get(l);
     try {

       TString fName = file;
       if(fName.EndsWith(".gz")) fName.Resize(fName.Sizeof()-4);

       if(!fName.EndsWith(".fits")) {
         cout << "skymap::Dump2fits Error : wrong file extension" << endl;
         cout << "fits file must ends with '.fits' or '.fits.gz'" << endl;
         exit(1); 
       }

       // if already exist the delete file 
       Long_t id,size,flags,mt;
       int estat = gSystem->GetPathInfo(fName.Data(),&id,&size,&flags,&mt);
       if (estat==0) {
         char cmd[1024];
         sprintf(cmd,"rm %s",fName.Data());
         cout << cmd << endl;
         int err=gSystem->Exec(cmd);
         if(err) { 
           cout << "skymap::Dump2fits Error : failed to remove file" << endl;
           exit(1); 
         }
       }

       // write_Healpix_map_to_fits(file,*healpix,PLANCK_FLOAT32);
       fitshandle out;
       out.create(fName.Data());
       PDT datatype = PLANCK_FLOAT32;

       // prepare_Healpix_fitsmap (out, *healpix, datatype, colname);
       arr<string> colname(1);
       colname[0] = (TString(TTYPE1)!="") ? TTYPE1 : "unknown ";
       string tunit1 = (TString(TUNIT1)!="") ? TUNIT1 : "unknown ";
       vector<fitscolumn> cols;
       int repcount = healpix_repcount (healpix->Npix());
       for (tsize m=0; m<colname.size(); ++m)
         cols.push_back (fitscolumn (colname[m],tunit1,repcount, datatype));
       out.insert_bintab(cols);
       out.set_key ("PIXTYPE",string("HEALPIX"),"HEALPIX pixelisation");
       string ordering = (healpix->Scheme()==RING) ? "RING" : "NESTED";
       out.set_key ("ORDERING",ordering, "Pixel ordering scheme, either RING or NESTED");
       out.set_key ("NSIDE",healpix->Nside(),"Resolution parameter for HEALPIX");
       out.set_key ("FIRSTPIX",0,"First pixel # (0 based)");
       out.set_key ("LASTPIX",healpix->Npix()-1,"Last pixel # (0 based)");
       out.set_key ("INDXSCHM",string("IMPLICIT"), "Indexing: IMPLICIT or EXPLICIT");
       if(coordsys=='c' || coordsys=='C')
         out.set_key ("COORDSYS",string("C       "),"Pixelisation coordinate system");  
       out.set_key ("CREATOR",string("CWB     "),"Program that created this file");  
       if(TString(configur)!="")  
         out.set_key ("CONFIGUR",string(configur),"software configuration used to process the data");  

       // convert gps_obs to (YYYY-MM-DDThh:mm:ss.xx UT)
       if(gps_obs>0) {
         wat::Time date(gps_obs);
         TString sdate = date.GetDateString();
         sdate.Resize(19);
         sdate.ReplaceAll(" ","T"); 
         char date_obs[32];
         sprintf(date_obs,"%s.%d",sdate.Data(),int(100*(date.GetNSec()/1000000000.)));
         out.set_key ("DATE-OBS",string(date_obs),"UTC date of the observation");  
         char mjd_obs[32];
         sprintf(mjd_obs,"%.9f",date.GetModJulianDate());
         out.set_key ("MJD-OBS",string(mjd_obs),"modified Julian date of the observation");  
       }

       // UTC date of file creation
       wat::Time date("now");
       TString sdate = date.GetDateString();
       sdate.Resize(19);
       sdate.ReplaceAll(" ","T"); 
       char date_obs[32];
       sprintf(date_obs,"%s.%d",sdate.Data(),int(100*(date.GetNSec()/1000000000.)));
       out.set_key ("DATE",string(date_obs),"UTC date of file creation");  

       out.write_column(1,healpix->Map());

       out.close();

       // if file ends with .gz then the file zipped
       if(TString(file).EndsWith(".gz")) {
         char cmd[1024];
         sprintf(cmd,"gzip -f %s",fName.Data());
         cout << cmd << endl;
         int err=gSystem->Exec(cmd);
         if(err) { 
           cout << "skymap::Dump2fits Error : gzip error" << endl;
           exit(1); 
         }
       }

     } catch(...) {}
     return;
   } else {
     cout << "skymap::Dump2fits Error : healpix not initialized" << endl;
     exit(1); 
   }
}
#endif

//: dump skymap into root file
void skymap::DumpObject(char* file)  
{
  TFile* rfile = new TFile(file, "RECREATE");
  this->Write("skymap"); // write this object
  rfile->Close();
}

//: dump skymap into binary file
void skymap::DumpBinary(char* file, int mode)
{
   size_t i,j,m;
   size_t n = value.size();
   size_t k = 0;

   for(i=0; i<n; i++) k += value[i].size(); 
   wavearray<float> x(k);

   k = 0;
   for(i=0; i<n; i++) {
     m = value[i].size(); 
     for(j=0; j<m; j++) { 
       x.data[k++] = this->value[i][j];
     }
   }

   x.DumpBinary(file,mode);
   return;
}

//: get skymap value at index l (access as a linear array) 
//  and fill in mTheta, mPhi and mIndex fields
//!param: sky index
double skymap::get(size_t l) { 
  size_t i,m;
  size_t n = value.size();
  size_t k = 0;
 
  mIndex = l;
  for(i=0; i<n; i++) {
    m = value[i].size();
    k += m;
    if(k <= l) continue;
    mTheta = i;
    mPhi = m-int(k-l);
    return this->value[mTheta][mPhi];
  }
  mTheta = mPhi = mIndex = -1;
  return 0.;
}

//: get sky index at theta,phi  
//!param: theta 
//!param: phi
size_t skymap::getSkyIndex(double th, double ph) {
  size_t k;

  if(healpix!=NULL) {  
#ifdef _USE_HEALPIX
    pointing P(th*deg2rad,ph*deg2rad);
    k = healpix->ang2pix(P);
    this->get(k); 
    return mIndex;
#endif
  }

  double g; 
  g = (value.size()-1)/(theta_2-theta_1);
  if(g>0.) {
    if(th<theta_1) th = theta_1;
    if(th>theta_2) th = theta_2;
    mTheta = int(g*(th-theta_1)+0.5);
  }
  else { mTheta = 0; }

  g = value[mTheta].size()/(phi_2-phi_1);
  if(ph< phi_1) ph = phi_2 - 0.5/g;
  if(ph>=phi_2) ph = phi_1 + 0.5/g;
  mPhi = int(g*(ph-phi_1));

  k = mPhi;
  for(int i=0; i<mTheta; i++) k += value[i].size();
  mIndex = k;
  return k;
}

#ifdef _USE_HEALPIX
//:
wavearray<int> skymap::neighbors(int index)
{
/*! Returns the neighboring pixels of pixel index in neighbors.
    On exit, neighbors contains (in this order)
    the pixel numbers of the SW, W, NW, N, NE, E, SE and S neighbor
    of the pixel index. If a neighbor does not exist (this can only be the case
    for the W, N, E and S neighbors), its entry is set to -1.
*/

   wavearray<int> neighbors(8);

   if(healpix!=NULL) {
     fix_arr< int, 8 > result;
     healpix->neighbors (index, result);
     for(int k=0;k<8;k++) neighbors[k]=result[k];
   }

   return neighbors;
}
#endif

#ifdef _USE_HEALPIX
void skymap::median(double radius) { 
//
// Applies a median filter with the desired radius
//
// Input: radius     - the radius (in degrees) of the disc 

  if(healpix==NULL) {  
     cout << "skymap::median Error : healpix not initialized" << endl;
     exit(1); 
  }
 
  radius*=deg2rad;

  // fill healpix map with skymap values
  int L = healpix->Npix();
  for(int l=0;l<L;l++) (*healpix)[l]=this->get(l);

  // This method works in both RING and NEST schemes, but is
  // considerably faster in the RING scheme.
  if (healpix->Scheme()==NEST) healpix->swap_scheme();
 
  vector<int> list;
  vector<float> list2;
  for (int l=0; l<L; ++l) {
    // Returns the numbers of all pixels that lie at least partially within
    // radius of healpix->pix2ang(l) in list. It may also return a few pixels
    // which do not lie in the disk at all
    healpix->query_disc(healpix->pix2ang(l),radius,list);
    list2.resize(list.size());
    for (tsize i=0; i<list.size(); ++i) list2[i] = (*healpix)[list[i]];
    double x=::median(list2.begin(),list2.end());
    // fill skymap with healpix map values
    this->set(l,x);
  }

  return;
}
#endif
 
#ifdef _USE_HEALPIX
void skymap::smoothing(double fwhm, int nlmax, int num_iter) {
//
// Applies a convolution with a Gaussian beam with an FWHM of fwhm_arcmin arcmin to alm.
//
// Input: fwhm     - Gaussian beam with an FWHM (degrees) 
//                   note : If fwhm<0, a deconvolution with -fwhm is performed
//        nlmax    - l max of spherical harmonic coefficients.
//        num_iter - the number of iterations (default=0 is identical to map2alm())

  if (fwhm<0)
    cout << "NOTE: negative FWHM supplied, doing a deconvolution..." << endl;

  if(healpix==NULL) {
     cout << "skymap::smoothing Error : healpix not initialized" << endl;
     exit(1);
  }

  int L = healpix->Npix();

  // fill healpix map with skymap values
  for(int l=0;l<L;l++) (*healpix)[l]=this->get(l);
  double avg=healpix->average();
  healpix->Add(double(-avg));

  // get alm coefficients
  wat::Alm alm = getAlm(nlmax, num_iter);
  // applies gaussian smoothing to alm
  alm.smoothWithGauss(fwhm);
  // converts a set of alm to a HEALPix map.
  setAlm(alm);

  // fill skymap with healpix map values
  healpix->Add(double(avg));
  for(int l=0;l<L;l++) this->set(l,(*healpix)[l]);

  return;
}
#endif

#ifdef _USE_HEALPIX
void skymap::rotate(double psi, double theta, double phi, int nlmax, int num_iter) {
//
// Rotates alm through the Euler angles psi, theta and phi.
// The Euler angle convention  is right handed, rotations are active.
// - psi is the first rotation about the z-axis (vertical)
// - then theta about the ORIGINAL (unrotated) y-axis
// - then phi  about the ORIGINAL (unrotated) z-axis (vertical)
//   relates Alm */

  if(healpix==NULL) {
     cout << "skymap::smoothing Error : healpix not initialized" << endl;
     exit(1);
  }

  // get alm coefficients
  wat::Alm alm = getAlm(nlmax, num_iter);
  // applies gaussian smoothing to alm
  alm.rotate(psi, theta, phi);
  // converts a set of alm to a HEALPix map.
  setAlm(alm);

  return;
}
#endif
 
#ifdef _USE_HEALPIX
void skymap::setAlm(wat::Alm alm) { 
//
// Converts a set of a_lm to a HEALPix map.
//
// Input : alm - spherical harmonic coefficients.

  if(healpix==NULL) {  
     cout << "skymap::setAlm Error : healpix not initialized" << endl;
     exit(1); 
  }

  // comvert wat::alm to alm
  Alm<xcomplex<double> > _alm(alm.Lmax(),alm.Mmax());
  for(int l=0;l<=_alm.Lmax();l++) {
    int limit = TMath::Min(l,_alm.Mmax());
    for (int m=0; m<=limit; m++) 
      _alm(l,m)=complex<double>(alm(l,m).real(),alm(l,m).imag());
  }
  // converts a set of alm to a HEALPix map.
  alm2map(_alm,*healpix);

  // fill skymap with healpix map values
  int L = healpix->Npix();
  for(int l=0;l<L;l++) this->set(l,(*healpix)[l]);

  return;
}
#endif
 
#ifdef _USE_HEALPIX
wat::Alm  
skymap::getAlm(int nlmax, int num_iter) { 
//
// Converts a Healpix map to a set of a_lms, using an iterative scheme
// which is more accurate than plain map2alm().
//
// Input: nlmax    - l max of spherical harmonic coefficients.
//        num_iter - the number of iterations (default=0 is identical to map2alm())


  if(healpix==NULL) {  
     cout << "skymap::getAlm Error : healpix not initialized" << endl;
     exit(1); 
  }

  if(nlmax<0) nlmax=0;

  // fill healpix map with skymap values
  int L = healpix->Npix();
  for(int l=0;l<L;l++) (*healpix)[l]=this->get(l);

  // create weight array and set to 1
  // weight array containing the weights for the individual rings of the map
  arr<double> weight(2*healpix->Nside());
  for(int i=0;i<2*healpix->Nside();i++) weight[i]=1.0;

  // get alm coefficients
  Alm<xcomplex<double> > alm(nlmax,nlmax);
  if (healpix->Scheme()==NEST) healpix->swap_scheme();
  // Converts a Healpix map to a set of a_lms, using an iterative scheme
  map2alm_iter(*healpix,alm,num_iter,weight);

  // comverts alm to wat::alm
  wat::Alm _alm = alm;

  return _alm;
}
#endif
 
#ifdef _USE_HEALPIX
void  
skymap::resample(int order) { 
//
// Resample the Healpix map to the healpix order
// the resampling is done using the spherical harmonic coefficients
//
// Input: order    - healpix order of the resampled skymap

  if(healpix==NULL) {  
     cout << "skymap::resample Error : healpix not initialized" << endl;
     exit(1); 
  }

  if(order == getOrder()) return;  // nothing to do

  // get alm coefficients
  int nlmax = 2*healpix->Nside();
  wat::Alm alm = getAlm(nlmax);

  // skymap with new healpix order
  *this = skymap(order);

  int _nlmax = 2*healpix->Nside();
  wat::Alm _alm(_nlmax,_nlmax);

  // fill new _alm with original alm
  int min_nlmax = TMath::Min(nlmax,_nlmax);  
  for(int l=0;l<=min_nlmax;l++) {
    int limit = TMath::Min(l,alm.Mmax());
    for (int m=0; m<=limit; m++) _alm(l,m)=alm(l,m);
  }

  // set new _alm coefficients
  setAlm(_alm);

  return;
}
#endif
 
#ifdef _USE_HEALPIX
int  
skymap::getRings() { 
//
// return the number of division in theta

  if(healpix==NULL) {  
     cout << "skymap::getRings Error : healpix not initialized" << endl;
     exit(1); 
  }

  int nrings  = 1;       // divisions on theta
  for(int i=0;i<=healpix->Order();i++) nrings=2*nrings+1;

  return nrings;
}
#endif

#ifdef _USE_HEALPIX
int 
skymap::getRingPixels(int ring) {
//
// return the number of pixels in the ring

  if(healpix==NULL) {  
     cout << "skymap::getRingPixels Error : healpix not initialized" << endl;
     exit(1); 
  }

  if(ring<1 || ring>getRings()) {
     cout << "skymap::getRingPixels Error : ring index " << ring 
          << " not allowed  [1:" << getRings() << "]" << endl;
     exit(1); 
  }

  int startpix; int ringpix; double costheta; double sintheta; bool shifted=false;
  healpix->get_ring_info(ring, startpix, ringpix, costheta, sintheta, shifted);
  return ringpix;  
}
#endif

#ifdef _USE_HEALPIX
int 
skymap::getStartRingPixel(int ring) {
//
// return the start pixel index in the ring

  if(healpix==NULL) {  
     cout << "skymap::getStartRingPixel Error : healpix not initialized" << endl;
     exit(1); 
  }

  if(ring<1 || ring>getRings()) {
     cout << "skymap::getRingPixels Error : ring index " << ring 
          << " not allowed  [1:" << getRings() << "]" << endl;
     exit(1); 
  }

  int startpix; int ringpix; double costheta; double sintheta; bool shifted=false;
  healpix->get_ring_info(ring, startpix, ringpix, costheta, sintheta, shifted);
  return startpix; 
}
#endif

#ifdef _USE_HEALPIX
int 
skymap::getEulerCharacteristic(double threshold) {
//
// return the get the Euler characteristic for pixels value > threshold

  if(healpix==NULL) {  
     cout << "skymap::getEulerCharacteristic Error : healpix not initialized" << endl;
     exit(1); 
  }


  int nV=0;	// number of vertices (= number of pixels)
  int nE=0;	// number of edges
  int nF=0;	// number of faces

  int F1=0;
  int F2=0;
  int F3=0;
  int F4=0;

  wavearray<int> index;

  double a;
  for(int l=0;l<size();l++) {
    a=get(l);
    if(a>threshold){
      nV++;
      index = neighbors(l);
      a=get(index[0]);
      if(a>threshold) {nE++;F1++;F4++;}
      a=get(index[1]);
      if(a>threshold) {F1++;}
      a=get(index[2]);
      if(a>threshold) {nE++;F1++;F2++;}
      a=get(index[3]);
      if(a>threshold) {F2++;}
      a=get(index[4]);
      if(a>threshold) {nE++;F2++;F3++;}
      a=get(index[5]);
      if(a>threshold) {F3++;}
      a=get(index[6]);
      if(a>threshold) {nE++;F3++;F4++;}
      a=get(index[7]);
      if(a>threshold) {F4++;}
      nF+=F1/3+F2/3+F3/3+F4/3;
      F1=0; F2=0; F3=0; F4=0;
    }
  }
  nE/=2;nF/=4;

  return nV-nE+nF; 
}
#endif

char* skymap::operator >> (char* fname) {

  TString name = fname;
  name.ReplaceAll(" ","");
  TObjArray* token = TString(fname).Tokenize(TString("."));
  TObjString* ext_tok = (TObjString*)token->At(token->GetEntries()-1);
  TString ext = ext_tok->GetString();
  if(ext=="dat") {
    //: dump skymap into binary file
    DumpBinary(fname,0);
  } else 
#ifdef _USE_HEALPIX
  if(ext=="fits") {
    //: dump skymap into fits file
    Dump2fits(fname);
  } else 
#endif
  if(ext=="root") {
    //: dump skymap object into root file
    DumpObject(fname);
  } else { 
    cout << "skymap operator (>>) : file type " << ext.Data() << " not supported" << endl; 
  }
  return fname;
}

//______________________________________________________________________________
void skymap::Streamer(TBuffer &R__b)
{
   // Stream an object of class skymap.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      TNamed::Streamer(R__b);
      {
         vector<vectorD> &R__stl =  value;
         R__stl.clear();
         TClass *R__tcl1 = TBuffer::GetClass(typeid(vector<double,allocator<double> >));
         if (R__tcl1==0) {
            Error("value streamer","Missing the TClass object for vector<double,allocator<double> >!");
            return;
         }
         int R__i, R__n;
         R__b >> R__n;
         R__stl.reserve(R__n);
         for (R__i = 0; R__i < R__n; R__i++) {
            vector<double,allocator<double> > R__t;
            R__b.StreamObject(&R__t,R__tcl1);
            R__stl.push_back(R__t);
         }
      }
      {
         vector<int> &R__stl =  index;
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
      if(R__v > 2) R__b >> sms;
      R__b >> theta_1;
      R__b >> theta_2;
      R__b >> phi_1;
      R__b >> phi_2;
      R__b >> gps;
      R__b >> mTheta;
      R__b >> mPhi;
      R__b >> mIndex;
      if(R__v > 1) {
        R__b >> healpix_order;
        if(healpix_order > 0) {
#ifdef _USE_HEALPIX
          healpix = new Healpix_Map<double>(healpix_order,RING);
#endif
        } else {
          healpix = NULL;
          healpix_order = 0;
        }
      }
      R__b.CheckByteCount(R__s, R__c, skymap::IsA());
   } else {
      R__c = R__b.WriteVersion(skymap::IsA(), kTRUE);
      TNamed::Streamer(R__b);
      {
         vector<vectorD> &R__stl =  value;
         int R__n=(&R__stl) ? int(R__stl.size()) : 0;
         R__b << R__n;
         if(R__n) {
         TClass *R__tcl1 = TBuffer::GetClass(typeid(vector<double,allocator<double> >));
         if (R__tcl1==0) {
            Error("value streamer","Missing the TClass object for vector<double,allocator<double> >!");
            return;
         }
            vector<vectorD>::iterator R__k;
            for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
            R__b.StreamObject((vector<double,allocator<double> >*)&(*R__k),R__tcl1);
            }
         }
      }
      {
         vector<int> &R__stl =  index;
         int R__n=(&R__stl) ? int(R__stl.size()) : 0;
         R__b << R__n;
         if(R__n) {
            vector<int>::iterator R__k;
            for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
            R__b << (*R__k);
            }
         }
      }
      R__b << sms;
      R__b << theta_1;
      R__b << theta_2;
      R__b << phi_1;
      R__b << phi_2;
      R__b << gps;
      R__b << mTheta;
      R__b << mPhi;
      R__b << mIndex;
      R__b << healpix_order;
      R__b.SetByteCount(R__c, kTRUE);
   }
}
 
