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


//**********************************************************
// Wavelet Analysis Tool  
// Sergey Klimenko, University of Florida
// sky map data container for network analysis
// used with DMT and ROOT
//**********************************************************

#ifndef SKYMAP_HH
#define SKYMAP_HH

#include <iostream>
#include <vector>
#include "complex"
#include "wavearray.hh"
#include "watfun.hh"
#include "TNamed.h"
#include "TFile.h"
#include "TString.h"
#ifdef _USE_HEALPIX
#include "alm.hh"
#endif

#ifdef _USE_HEALPIX
#include "xcomplex.h"
//#include "paramfile.h"
#include "healpix_data_io.h"
#include "alm.h"
#include "healpix_map.h"
#include "healpix_map_fitsio.h"
#include "alm_healpix_tools.h"
#include "alm_powspec_tools.h"
#include "fitshandle.h"
//#include "levels_facilities.h"
#include "lsconstants.h"
#endif

typedef std::vector<double> vectorD;

// GPS seconds of the J2000.0 epoch (2000 JAN 1 12h UTC).           
#define EPOCH_J2000_0_GPS 630763213 


class skymap : public TNamed
{
  public:
      
      // constructors
      
      //: Default constructor
      skymap();

      //: skymap constructor
      //!param - step on phi and theta
      //!param - theta begin
      //!param - theta end
      //!param - phi begin
      //!param - phi end
      skymap(double,double=0.,double=180.,double=0.,double=360.);

      //: skymap constructor
      //!param - healpix order
      skymap(int); 

      //: skymap constructor
      //!param - fits file
      skymap(char*); 

      //: skymap constructor
      //!param ifile - root file name
      //!param name  - object name
      skymap(TString ifile, TString name="skymap"); 
    
      //: Copy constructor
      //!param: value - object to copy from 
      skymap(const skymap&);
      
      //: destructor
      virtual ~skymap();
    
      // operators

      skymap& operator  = (const skymap&);
      skymap& operator += (const skymap&);
      skymap& operator -= (const skymap&);
      skymap& operator *= (const skymap&);
      skymap& operator /= (const skymap&);
      skymap& operator  = (const double);
      skymap& operator *= (const double);
      skymap& operator += (const double);
      char*   operator >> (char* fname);
    
      // accessors

      //: get skymap value at index i  
      //: sets index nTheta and mPhi
      //!param: sky index
      double get(size_t i); 

      //: set skymap value at index i,j  
      //!param: sky index
      //!param: value to set
      inline void set(size_t i, double a) { 
	if(int(i) != mIndex) this->get(i);              // fill in sky index	
	if(mTheta>=0) value[mTheta][mPhi] = a;
      }

      //: add skymap value at index i,j  
      //!param: sky index
      //!param: value to add
      inline void add(size_t i ,double a) { 
	if(int(i) != mIndex) this->get(i);              // fill in sky index	
	if(mTheta>=0) value[mTheta][mPhi] += a;
      }

      //: get skymap size  
      inline size_t size() {
	size_t n = value.size();
	size_t m = 0;
	for(size_t i=0; i<n; i++) m += value[i].size(); 
	return m;
      } 

      //: skymap sizes  
      inline size_t size(size_t k) {
	if(!k) return value.size();
	else if(k<=value.size()) return value[k-1].size();
	else return 0;
      } 

      //: get sky index  at theta,phi  
      //!param: theta 
      //!param: phi
      size_t getSkyIndex(double th, double ph);

      //: get skymap value at theta,phi  
      //!param: theta 
      //!param: phi
      inline double get(double th, double ph) {
	mIndex = getSkyIndex(th,ph);
	return get(mIndex);
      }

      //: get phi value
      inline double getPhi(size_t i) {
        if(healpix!=NULL) { 
#ifdef _USE_HEALPIX
          pointing P = healpix->pix2ang(i);
          return rad2deg*P.phi;
#else     
          return 0.;
#endif
        } else {
          if(int(i) != mIndex) this->get(i);              // fill in sky index
          size_t n = this->value[mTheta].size();
          if(mTheta>=0 && n>1)
            return phi_1+(mPhi+0.5)*(phi_2-phi_1)/double(n);
          else return (phi_2-phi_1)/2.;
        }
      }

      //: get phi step  
      inline double getPhiStep(size_t i) { 
	if(int(i) != mIndex) this->get(i);              // fill in sky index
	size_t n = this->value[mTheta].size();
	if(mTheta>=0 && n)
	  return (phi_2-phi_1)/double(n);
	else return 0.;
      }

      // get RA angle from EFEC phi angle in degrees
      // Earth angular velocity defines duration of the siderial day
      // 1 siderial day = 23h 56m 04.0905s 
      // (from http://www.maa.mhn.de/Scholar/times.html)
      // phiRA function is implemented by Gabriele Vedovato and compared
      // with LAL - agrees within 0.01 degrees

      static inline double phiRA(double ph,  double gps, bool inverse=false) {
	double sidereal_time; // sidereal time in sidereal seconds.  (magic)
	double t = (gps-EPOCH_J2000_0_GPS)/86400./36525.;
	
	sidereal_time = (-6.2e-6 * t + 0.093104) * t * t + 67310.54841;
	sidereal_time += 8640184.812866*t + 3155760000.*t;

	double gmst = 360.*sidereal_time/86400.;
	if(inverse) gmst=-gmst;

	double ra = fmod( ph + gmst, 360. );
	return ra<0. ? ra+360 : ra;

      }

      inline double phi2RA(double ph,  double gps) { return phiRA(ph,gps,false); }
      inline double RA2phi(double ph,  double gps) { return phiRA(ph,gps,true); }

      inline double getRA(size_t i) {
	if(this->gps<=0.) return 0.;
//	double omega = 7.292115090e-5;  // http://hypertextbook.com/facts/2002/JasonAtkins.shtml
//	double gps2000 = 630720013;     // GPS time at 01/01/2000 00:00:00 UTC
//	double GMST2000 = 24110.54841;  // GMST at UT1=0 on Jan 1, 2000
	return phi2RA(getPhi(i),this->gps);
      }

      //: get theta value
      inline double getTheta(size_t i) {
        if(healpix!=NULL) {
#ifdef _USE_HEALPIX
          pointing P = healpix->pix2ang(i);
          return rad2deg*P.theta;
#else     
          return 0.;
#endif
        } else {
          size_t n = this->value.size()-1;
          if(int(i) != mIndex) this->get(i);              // fill in sky index
          if(mTheta >= 0 && n)
            return theta_1+mTheta*(theta_2-theta_1)/double(n);
          else return (theta_2-theta_1)/2.;
        }
      }

      //: get theta step  
      inline double getThetaStep(size_t i) {
	size_t n = this->value.size()-1;
	if(int(i) != mIndex) this->get(i);              // fill in sky index
	if(mTheta >= 0 && n)
	  return (theta_2-theta_1)/double(n);
	else return 0.;
      }

      // get declination angle from EFEC theta angle in degrees
      inline double getDEC(size_t i) { return -(getTheta(i)-90.); }

     //: find and return maximum value in skymap
     //: set mTheta and mPhi to be theta and phi index for the maximum value  
      double max();

     //: find and return minimum value in skymap
     //: set mTheta and mPhi to be theta and phi index for the minimum value  
      double min();

     //: find and return mean value over entire skymap
      double mean();

     //: find for what fraction of the sky the statistic > threshold t
      double fraction(double);

     //: normalize skymap
      double norm(double=0.);

     //: downsample sky index array
     //: input sky index array
     //: down-sampling option: 2 or 4 
      void downsample(wavearray<short>&, size_t=4);

     //: dump skymap into binary file
     //: parameter: file name   
     //: parameter: append mode   
      void DumpBinary(char*, int);

#ifdef _USE_HEALPIX
     //: dump skymap into fits file
     //: file     : output fits file name [ext : .fits or .fits.gz]
     //: gps_obs  : gps time of the observation 
     //: configur : software configuration used to process the data
     //: TTYPE1   : label for field 1
     //: TUNIT1   : physical unit of field 1
     //: coordsys : Pixelisation coordinate system [c/C : select celestial] 
     void Dump2fits(const char* file, double gps_obs=0, const char configur[]="", 
                    const char TTYPE1[]="", const char TUNIT1[]="", char coordsys='x');	// *MENU*
#endif

     //: dump skymap object into root file
     void DumpObject(char*);  

     // return 1 if healpix else 0
     int getType() {if(healpix==NULL) return 0; else return 1;}   

     // works only with the healpix scheme  !!!
#ifdef _USE_HEALPIX
     wavearray<int> neighbors(int index);
     void median(double radius);
     void smoothing(double fwhm, int nlmax=256, int num_iter=0);
     void rotate(double psi, double theta, double phi, int nlmax=256, int num_iter=0);
     void setAlm(wat::Alm alm);
     wat::Alm getAlm(int nlmax, int num_iter=0);
     void resample(int order);
     int  getRings();
     int  getRingPixels(int ring);
     int  getStartRingPixel(int ring);
     int  getEulerCharacteristic(double threshold);
#endif

     // return the healpix order (if=0 healpix skygrid is disabled)
     int  getOrder() {return healpix_order;}
 
// data members

      std::vector<vectorD> value;  // skymap map array
      std::vector<int> index;      // sample index array
      double sms;                  // step on phi and theta
      double theta_1;              // theta range begin
      double theta_2;              // theta range end
      double phi_1;                // phi range begin
      double phi_2;                // phi range end
      double gps;                  // gps time 
      int    mTheta;               // theta index
      int    mPhi;                 // phi index
      int    mIndex;               // sky index

private:
      int    healpix_order;	   // healpix order (if=0 healpix is disabled)

#ifdef _USE_HEALPIX
      Healpix_Map<double>* healpix;//!
#else
      bool* healpix;
#endif
      double deg2rad;
      double rad2deg;

      ClassDef(skymap,5)

}; // class skymap

using namespace std;

namespace {

template<typename Iterator> typename iterator_traits<Iterator>::value_type
  median(Iterator first, Iterator last)
  {
  Iterator mid = first+(last-first-1)/2;
  nth_element(first,mid,last);
  if ((last-first)&1) return *mid;
  return typename iterator_traits<Iterator>::value_type
    (0.5*((*mid)+(*min_element(mid+1,last))));
  }

} // unnamed namespace

#endif // SKYMAP_HH

