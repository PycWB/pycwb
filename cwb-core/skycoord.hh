/*
# Copyright (C) 2019 Gabriele Vedovato
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


/**********************************************************
 * Package:      sky coordinates (extract from the lal library)
 * File name:    skycoord.hh
 * Author:       Gabriele Vedovato (vedovato@lnl.infn.it)
 * Release Date: 01/27/2012
 * Version:      1.00
 **********************************************************/

#ifndef SKYCOORD_HH
#define SKYCOORD_HH

#define LAL_DELTAGAL (0.473477302)
#define LAL_ALPHAGAL (3.366032942)
#define LAL_LGAL     (0.576)
#define LAL_IEARTH    0.409092600600582871467239393761915655     // Earth inclination (2000),radians : LALConstants.h 

#define LAL_REARTH_SI 6378136.6       // Earth equatorial radius : LALConstants.h 
#define LAL_EARTHFLAT (0.00335281)
#define LAL_HSERIES (0.0001)          // value H below which we expand sqrt(1-H) 

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "skymap.hh"

using namespace std;

inline void 
GalacticToEquatorial(double ilongitude, double ilatitude, double& olongitude,  double& olatitude)
{ // see LAL CelestialCoordinates.c
  double deg2rad = TMath::Pi()/180.;
  ilongitude*=deg2rad;
  ilatitude*=deg2rad;

  double sinDGal = sin( LAL_DELTAGAL ); /* sin(delta_NGP) */
  double cosDGal = cos( LAL_DELTAGAL ); /* cos(delta_NGP) */
  double l = -LAL_LGAL;          	/* will be l-l(ascend) */
  double sinB, cosB, sinL, cosL; 	/* sin and cos of b and l */
  double sinD, sinA, cosA;       	/* sin and cos of delta and alpha */

  /* Compute intermediates. */
  l += ilongitude;
  sinB = sin( ilatitude );
  cosB = cos( ilatitude );
  sinL = sin( l );
  cosL = cos( l );

  /* Compute components. */
  sinD = cosB*cosDGal*sinL + sinB*sinDGal;
  sinA = cosB*cosL;
  cosA = sinB*cosDGal - cosB*sinL*sinDGal;

  /* Compute final results. */
  olatitude = asin( sinD );
  l = atan2( sinA, cosA ) + LAL_ALPHAGAL;

  /* Optional phase correction. */
  if ( l < 0.0 )
    l += TMath::TwoPi();
  if ( l > 360. )
    l -= TMath::TwoPi();
  olongitude = l;

  double rad2deg = 180./TMath::Pi();
  olongitude*=rad2deg;
  olatitude*=rad2deg;
 
  return;
}

inline void 
EquatorialToGalactic(double ilongitude, double ilatitude, double& olongitude,  double& olatitude)
{ // see LAL CelestialCoordinates.c
  double deg2rad = TMath::Pi()/180.;
  ilongitude*=deg2rad;
  ilatitude*=deg2rad;

  double sinDGal = sin( LAL_DELTAGAL ); /* sin(delta_NGP) */
  double cosDGal = cos( LAL_DELTAGAL ); /* cos(delta_NGP) */
  double a = -LAL_ALPHAGAL;      	/* will be alpha-alpha_NGP */
  double sinD, cosD, sinA, cosA; 	/* sin and cos of delta and alpha */
  double sinB, sinL, cosL;       	/* sin and cos of b and l */

  /* Compute intermediates. */
  a += ilongitude;
  sinD = sin( ilatitude );
  cosD = cos( ilatitude );
  sinA = sin( a );
  cosA = cos( a );

  /* Compute components. */
  sinB = cosD*cosDGal*cosA + sinD*sinDGal;
  sinL = sinD*cosDGal - cosD*cosA*sinDGal;
  cosL = cosD*sinA;

  /* Compute final results. */
  olatitude = asin( sinB );
  a = atan2( sinL, cosL ) + LAL_LGAL;

  /* Optional phase correction. */
  if ( a < 0.0 )
    a += TMath::TwoPi();
  if ( a > 360. )
    a -= TMath::TwoPi();
  olongitude = a;

  double rad2deg = 180./TMath::Pi();
  olongitude*=rad2deg;
  olatitude*=rad2deg;

  return;
}

inline void 
EclipticToEquatorial(double ilongitude, double ilatitude, double& olongitude,  double& olatitude)
{ // see LAL CelestialCoordinates.c
  double deg2rad = TMath::Pi()/180.;
  ilongitude*=deg2rad;
  ilatitude*=deg2rad;

  double sinE = sin( LAL_IEARTH ); /* sin(epsilon) */
  double cosE = cos( LAL_IEARTH ); /* cos(epsilon) */
  double sinB, cosB, sinL, cosL;   /* sin and cos of b and l */
  double sinD, sinA, cosA;         /* sin and cos of delta and alpha */

  /* Compute intermediates. */
  sinB = sin( ilatitude );
  cosB = cos( ilatitude );
  sinL = sin( ilongitude );
  cosL = cos( ilongitude );

  /* Compute components. */
  sinD = cosB*sinL*sinE + sinB*cosE;
  sinA = cosB*sinL*cosE - sinB*sinE;
  cosA = cosB*cosL;

  /* Compute final results. */
  olatitude = asin( sinD );
  olongitude = atan2( sinA, cosA );

  /* Optional phase correction. */
  if ( olongitude < 0.0 )
    olongitude += TMath::TwoPi();
  if ( olongitude > 360. )
    olongitude -= TMath::TwoPi();

  double rad2deg = 180./TMath::Pi();
  olongitude*=rad2deg;
  olatitude*=rad2deg;
  olatitude=-olatitude;

  return;
}

inline void 
EquatorialToEcliptic(double ilongitude, double ilatitude, double& olongitude,  double& olatitude)
{ // see LAL CelestialCoordinates.c
  double deg2rad = TMath::Pi()/180.;
  ilongitude*=deg2rad;
  ilatitude*=deg2rad;

  double sinE = sin( LAL_IEARTH ); /* sin(epsilon) */
  double cosE = cos( LAL_IEARTH ); /* cos(epsilon) */
  double sinD, cosD, sinA, cosA;   /* sin and cos of delta and alpha */
  double sinB, sinL, cosL;         /* sin and cos of b and l */

  /* Compute intermediates. */
  sinD = sin( ilatitude );
  cosD = cos( ilatitude );
  sinA = sin( ilongitude );
  cosA = cos( ilongitude );

  /* Compute components. */
  sinB = sinD*cosE - cosD*sinA*sinE;
  sinL = cosD*sinA*cosE + sinD*sinE;
  cosL = cosD*cosA;

  /* Compute final results. */
  olatitude = asin( sinB );
  olongitude = atan2( sinL, cosL );

  /* Optional phase correction. */
  if ( olongitude < 0.0 )
    olongitude += TMath::TwoPi();
  if ( olongitude > 360. )
    olongitude -= TMath::TwoPi();

  double rad2deg = 180./TMath::Pi();
  olongitude*=rad2deg;
  olatitude*=rad2deg;
  olatitude=-olatitude;

  return;
}

inline void
GeodeticToGeocentric(double latitude, double longitude, double elevation, double& X, double& Y, double& Z) {

  double c, s; /* position components in and orthogonal to the equator */
  double cosP, sinP; /* cosine and sine of latitude */
  double fFac;       /* ( 1 - f )^2 */
  double x, y;       /* Cartesian coordinates */
  double maxComp, r; /* max{x,y,z}, and sqrt(x^2+y^2+z^2) */


  /* Compute intermediates. */
  fFac = 1.0 - LAL_EARTHFLAT;
  fFac *= fFac;
  cosP = cos( latitude );
  sinP = sin( latitude );
  c = sqrt( 1.0 / ( cosP*cosP + fFac*sinP*sinP ) );
  s = fFac*c;
  c = ( LAL_REARTH_SI*c + elevation )*cosP;
  s = ( LAL_REARTH_SI*s + elevation )*sinP;

  /* Compute Cartesian coordinates. */
  X = x = c*cos( longitude );
  Y = y = c*sin( longitude );
  Z = s;

  /* Compute the radius. */
  maxComp = x;
  if ( y > maxComp )
    maxComp = y;
  if ( s > maxComp )
    maxComp = s;
  x /= maxComp;
  y /= maxComp;
  s /= maxComp;
  r = sqrt( x*x + y*y + s*s );
/*
  // Compute the spherical coordinates, and exit.
  location->radius = maxComp*r;
  location->geocentric.longitude = longitude;
  location->geocentric.latitude = asin( s / r );
*/
}

inline void 
HeapSort( double* data, double length) {

  int i;
  int j;
  int k;
  int n;
  double temp;

  n=length;

  /* A vector of length 0 or 1 is already sorted. */
  if(n<2)
  {
    cout << "A vector of length 0 or 1 is already sorted" << endl;
  }

  /* Here is the heapsort algorithm. */
  j=n-1;
  n >>= 1;

  while(1){
    if(n>0)
      temp=data[--n];
    else{
      temp=data[j];
      data[j]=data[0];
      if(--j==0){
        data[0]=temp;
        break;
      }
    }
    i=n;
    k=(n << 1)+1;
    while(k<=j){
      if(k<j && data[k]<data[k+1])
        k++;
      if(temp<data[k]){
        data[i]=data[k];
        i=k;
        k<<=1;
        k++;
      }else
        k=j+1;
    }
    data[i]=temp;
  }
}

inline void
GeocentricToGeodetic(double X, double Y, double Z, double& latitude, double& longitude, double& elevation) {

  double x, y, z;   /* normalized geocentric coordinates */
  double pi;        /* axial distance */

  /* Declare some local constants. */
  const double rInv = 1.0 / LAL_REARTH_SI;
  const double f1 = 1.0 - LAL_EARTHFLAT;
  const double f2 = LAL_EARTHFLAT*( 2.0 - LAL_EARTHFLAT );

  /* See if we've been given a special set of coordinates. */
  x = rInv*X;
  y = rInv*Y;
  z = rInv*Z;
  pi = sqrt( x*x + y*y );
  if ( pi == 0.0 ) {
    longitude = atan2( y, x );
    if ( z >= 0.0 ) {
      latitude = TMath::PiOver2();
      elevation = z - f1;
    } else {
      latitude = -TMath::PiOver2();
      elevation = f1 - z;
    }
    elevation *= LAL_REARTH_SI;
  }

  /* Do the general transformation even if z=0. */
  else {
    double za, e, f, p, q, d, v, w, g, h, t, phi, tanP;

    /* Compute intermediates variables. */
    za = f1*fabs( z );
    e = za - f2;
    f = za + f2;
    p = ( 4.0/3.0 )*( pi*pi + za*za - f2*f2 );
    q = 8.0*f2*za;
    d = p*p*p + pi*pi*q*q;
    if ( d >= 0.0 ) {
      v = pow( sqrt( d ) + pi*q, 1.0/3.0 );
      v -= pow( sqrt( d ) - pi*q, 1.0/3.0 );
    } else {
      v = 2.0*sqrt( -p )*cos( acos( pi*q/( p*sqrt( -p ) ) )/3.0 );
    }
    w = sqrt( e*e + v*pi );
    g = 0.5*( e + w );
    h = pi*( f*pi - v*g )/( g*g*w );

    /* Compute t, expanding the square root if necessary. */
    if ( fabs( h ) < LAL_HSERIES )
      t = g*( 0.5*h + 0.375*h*h + 0.3125*h*h*h );
    else
      t = g*( sqrt( 1.0 + h ) - 1.0 );

    /* Compute and sort the factors in the arctangent. */
    {
      double tanPFac[4];    /* factors of tanP */
      tanPFac[0] = pi - t;
      tanPFac[1] = pi + t;
      tanPFac[2] = 1.0/pi;
      tanPFac[3] = 1.0/t;
      double length = 4;
      HeapSort( tanPFac, length );
      tanP = tanPFac[0]*tanPFac[3];
      tanP *= tanPFac[1]*tanPFac[2];
      tanP /= 2.0*f1;
    }

    /* Compute latitude, longitude, and elevation. */
    phi = atan( tanP );
    latitude = phi;
    if ( z < 0.0 ) latitude *= -1.0;
    longitude = atan2( y, x );
    elevation = ( pi - t/pi )*cos( phi );
    elevation += ( fabs( z ) - f1 )*sin( phi );
    elevation *= LAL_REARTH_SI;
  }
}

inline void 
GetCartesianComponents( double u[3], double Alt, double Az, double Lat, double Lon) {

  double cosAlt=cos(Alt); double sinAlt=sin(Alt);
  double cosAz=cos(Az);   double sinAz=sin(Az);
  double cosLat=cos(Lat); double sinLat=sin(Lat);
  double cosLon=cos(Lon); double sinLon=sin(Lon);

  double uNorth = cosAlt * cosAz;
  double uEast = cosAlt * sinAz;
  /* uUp == sinAlt */
  double uRho = - sinLat * uNorth + cosLat * sinAlt;
  /* uLambda == uEast */

  u[0] = cosLon * uRho - sinLon * uEast;
  u[1] = sinLon * uRho + cosLon * uEast;
  u[2] = cosLat * uNorth + sinLat * sinAlt;

  return;
}

// ************************************
// cWB Coordinate System
// theta=0 : North Pole
// phi=0   : Greenwich meridian
// ************************************

inline void
CwbToGeographic(double ilongitude, double ilatitude, double& olongitude,  double& olatitude)
{
  olongitude = ilongitude>180 ? ilongitude-360 : ilongitude;
  olatitude=-(ilatitude-90);
}

inline void
GeographicToCwb(double ilongitude, double ilatitude, double& olongitude,  double& olatitude)
{
  olongitude = ilongitude<0 ? ilongitude+360 : ilongitude;
  olatitude=-(ilatitude-90);
}

inline void
CwbToCelestial(double ilongitude, double ilatitude, double& olongitude,  double& olatitude, double gps=0)
{
  skymap sm;
  olongitude = gps>0 ? sm.phi2RA(ilongitude, gps) : ilongitude;
  olongitude = 360-olongitude;
  olatitude=-(ilatitude-90);
}

inline void
CelestialToCwb(double ilongitude, double ilatitude, double& olongitude,  double& olatitude, double gps=0)
{
  skymap sm;
  olongitude = gps>0 ? sm.RA2phi(ilongitude, gps) : ilongitude;
  olongitude = 360-olongitude;
  olatitude=-(ilatitude-90);
}

#endif
