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


#ifndef TIME_H
#define TIME_H

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string>
#include <iostream>
#include <fstream>
#include "TROOT.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"

#define INT_4S	int	
#define INT_4U	unsigned int

#define UTC_UNIX_SPAN 315964800
#define TAI_GPS_LEAPS_SPAN  19 

#define GPS_LEAPS_TABLE_SIZE  19   

using namespace std;

struct gps_leap {  
  int    gps;       /* GPS time when leap sec was introduced */
  int    gps_utc;   /* GPS-UTC at GPS time gps */
};

// UPDATE-ME -- to update, add the new (GPS, GPS - UTC) entry at the end
// see http://hpiers.obspm.fr/eoppc/bul/bulc/bulletinc.dat */
// Table of GPS-UTC */
static const gps_leap gps_leaps_table[GPS_LEAPS_TABLE_SIZE] =   
{
  {0,          0},  /* 1980-Jan-06 */
  {46828801,   1},  /* 1981-Jul-01 */
  {78364802,   2},  /* 1982-Jul-01 */
  {109900803,  3},  /* 1983-Jul-01 */
  {173059204,  4},  /* 1985-Jul-01 */
  {252028805,  5},  /* 1988-Jan-01 */
  {315187206,  6},  /* 1990-Jan-01 */
  {346723207,  7},  /* 1991-Jan-01 */
  {393984008,  8},  /* 1992-Jul-01 */
  {425520009,  9},  /* 1993-Jul-01 */
  {457056010, 10},  /* 1994-Jul-01 */
  {504489611, 11},  /* 1996-Jan-01 */
  {551750412, 12},  /* 1997-Jul-01 */
  {599184013, 13},  /* 1999-Jan-01 */
  {820108814, 14},  /* 2006-Jan-01 */
  {914803215, 15},  /* 2009-Jan-01 */
  {1025136016,16},  /* 2012-Jul-01 */
  {1119744017,17},  /* 2015-Jul-01 */
  {1167264018,18}   /* 2017-Jan-01 */
};

namespace wat {

//-----------------------------------------------------------------------------
//
// A class storing seconds & nanoseconds.
// This is a very simple class storing time as integer seconds and nanoseconds.
// A time type storing seconds and nanoseconds as integers with full precision.

class Time {

public:

  // Constructor 
  explicit Time(INT_4S sec = 0, INT_4U nsec = 0);
  
  Time(TString date) {tzset();setenv("TZ", ":UTC", 1);SetDateString(date);}
 
  // Copy constructor 
  Time(Time& time);
   
  // Double constructor 
  Time(double dtime);

  // = Overload 
  Time& operator=(Time& time);

  // Addition & assignment.
  // param: Time& time -
  // return: Time& time -
    
  Time& operator+=(Time& time);
  Time& operator-=(Time& time);

    
  // Multiplication and assignment.
  // param: double& d -
  // return: Time& time -
  // todo: What happens if d is negative?
    
  Time& operator*=(double& d);
  Time& operator/=(double& d);
  double operator/(Time& time );
  bool operator==(Time& time);
  bool operator<=(Time& time);
  bool operator>=(Time& time);
  bool operator!=(Time& time);
  bool operator<(Time& time);
  bool operator>(Time& time);

  // Access gps seconds 
  INT_4S GetGPS() {return GetSec();}

  // Access seconds 
  INT_4S GetSec();
  
  // Access nanoseconds 
  INT_4U GetNSec();

  // Mutators 
   
  //Set seconds 
  INT_4S SetSec(INT_4S s) {return mSec = s;}
  
  // Set nanosecond residual 
  INT_4U SetNSec(INT_4U nsec) {return mNSec = nsec;}

  //set Double conversion method 
  void SetDouble(double dt);
  
  // get Double conversion method 
  double GetDouble();

  // set Date conversion method 
  void SetDate(int ss, int mm, int hh, int DD, int MM, int YY, int nsec = 0);
   
  // set Date from String conversion method 
  void SetDateString(TString date);
  void SetString(char* date, int nsec = 0);
  void SetYear(unsigned int year) {
         TString str;str.Form("%04d",year);
         SetDateString(GetDateString().Replace(0,4,str,4)(0,19));}
  void SetMonth(unsigned int month) {
         TString str;str.Form("%02d",month);
         SetDateString(GetDateString().Replace(5,2,str,2)(0,19));}
  void SetDay(unsigned int day) {
         TString str;str.Form("%02d",day);
         SetDateString(GetDateString().Replace(8,2,str,2)(0,19));}
  void SetHour(unsigned int hour) {
         TString str;str.Form("%02d",hour);
         SetDateString(GetDateString().Replace(11,2,str,2)(0,19));}
  void SetMinute(unsigned int min) {
         TString str;str.Form("%02d",min);
         SetDateString(GetDateString().Replace(14,2,str,2)(0,19));}
  void SetSecond(unsigned int sec) {
         TString str;str.Form("%02d",sec);
         SetDateString(GetDateString().Replace(17,2,str,2)(0,19));}

  // get Date String conversion method 
  TString GetDateString();
  int GetDayOfYear() {
        TString date = GetDateString();
        TString begin_of_year = date(0,4)+TString("-01-01 00:00:00");
        return 1+(GetSec()-Time(begin_of_year).GetSec())/86400.;}
  int GetYear() {
        TString date = GetDateString();
        TString year = date(0,4);
        return year.Atoi();}
  int GetMonth() {
        TString date = GetDateString();
        TString month = date(5,2);
        return month.Atoi();}
  int GetDay() {
        TString date = GetDateString();
        TString day = date(8,2);
        return day.Atoi();}
  int GetHour() {
        TString date = GetDateString();
        TString hour = date(11,2);
        return hour.Atoi();}
  int GetMinute() {
        TString date = GetDateString();
        TString minute = date(14,2);
        return minute.Atoi();}
  int GetSecond() {
        TString date = GetDateString(); 
        TString  second = date(17,2);
        return second.Atoi();}

  int GetLeapSecs() {return GpsToGpsLeaps(mSec);}  
  void PrintLeapSecs() {  
         for(int i=0;i<GPS_LEAPS_TABLE_SIZE;i++)
           cout << Time(gps_leaps_table[i].gps).GetDateString().Data() << "  " 
                << gps_leaps_table[i].gps << " "
                << gps_leaps_table[i].gps_utc << endl;}
 
  // Dumps date 
  void Print();

  int  GetJulianDay() {return floor(GetJulianDate());}
  int  GetModJulianDay() {return floor(GetModJulianDate());}
  double  GetJulianDate();
  double  GetModJulianDate();

  int GpsToGpsLeaps(int gps); 
  int UnixToGpsLeaps(int unix_time); 
  int GpsToGpsLeaps() {return GpsToGpsLeaps(mSec);} 
  int UnixToGpsLeaps() {return UnixToGpsLeaps(mSec);} 

  int GpsToTaiLeaps(int gps) {return GpsToGpsLeaps(gps)+TAI_GPS_LEAPS_SPAN;} 
  int UnixToTaiLeaps(int unix_time) {return UnixToGpsLeaps(unix_time)+TAI_GPS_LEAPS_SPAN;} 
  int GpsToTaiLeaps() {return GpsToGpsLeaps()+TAI_GPS_LEAPS_SPAN;} 
  int UnixToTaiLeaps() {return UnixToGpsLeaps()+TAI_GPS_LEAPS_SPAN;} 

  // UNIX --> GPS (frame) time conversion
  // UNIX = UTC from 1/1/1970 (unix time()) ; GPS = GPS from 6/1/1980
   
  void UnixToGps() {
    mSec = mSec - UTC_UNIX_SPAN + UnixToGpsLeaps(); } 
   
  // GPS (frame) --> UNIX time conversion
  // UNIX = UTC from 1/1/1970 (unix time()) ; GPS = GPS from 6/1/1980
  
  void GpsToUnix() {
    mSec = mSec + UTC_UNIX_SPAN - GpsToGpsLeaps(); } 
   
  // return unix time from GPS 
  int GpsToUnixTime() {
    return mSec + UTC_UNIX_SPAN - GpsToGpsLeaps(); } 

  // cout adapter 
  friend istream& operator>>( istream&, Time& );

private:

  void Error(char* msg) {cout << "CWB::Time:Error " << msg << endl;exit(1);}

  INT_4S mSec;
  INT_4U mNSec;

  // used by THtml doc
  ClassDef(Time,1)       
};

static Time Time_MAX(0x7FFFFFFF,999999999);

Time operator+(Time& t1, Time& t2);
Time operator-(Time& t1, Time& t2);
Time operator*(Time& t, double& d);
Time operator/(Time& t, double& d);
Time operator*(double& d, Time& t);

istream& operator>>(istream& in, Time& time);
ostream& operator<<(ostream& out, Time& time);

} // end namespace

#endif
