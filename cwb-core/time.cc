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


#include "time.hh"
#include "TMath.h"

#define CWB_MJD_REF 2400000.5   // Reference Julian Day for Mean Julian Day 

ClassImp(wat::Time)       // used by THtml doc

using namespace wat;

/* :TODO: this Time class is used in many place where seconds are INT_4U.
   Perhaps I should rename this class OffsetTime and write another called Time,
   where seconds are unsigned */

//-----------------------------------------------------------------------------
// Constructors
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// Constructor.
//
// param: INT_4S sec - number of seconds. Default: 0
// param: INT_4U nsec - number of nanoseconds. Default: 0

wat::Time::Time(INT_4S sec, INT_4U nsec) : mSec(sec), mNSec(nsec) {

  tzset();
  setenv("TZ", ":UTC", 1);

  if ( mNSec >= 1000000000 ) {
    mSec += INT_4S( mNSec / 1000000000 );
    mNSec = mNSec % 1000000000;
  }
}

//: uct from Double conversion AC
wat::Time::Time(double dtime) : mSec(0), mNSec(0) {

  tzset();
  setenv("TZ", ":UTC", 1);

  SetDouble(dtime);
}

//-----------------------------------------------------------------------------
// Copy Constructor.
// param: Time& time -

wat::Time::Time(Time& time) : mSec(time.GetSec()), mNSec(time.GetNSec()) {
  tzset();
  setenv("TZ", ":UTC", 1);
}

//-----------------------------------------------------------------------------
// Operator Overloads
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Assignment Operator.
// param: Time& time -
// return: Time&
// exc: bad_alloc - Memory allocation failed.

Time& wat::Time::operator=( Time& time ) {

  if ( this != &time ) {
    mSec = time.GetSec();
    mNSec = time.GetNSec();
  }
  return *this;
}


//-----------------------------------------------------------------------------
// Addition & assignment.
// param: Time& time -
// return: Time& time -

Time& wat::Time::operator+=(Time& time) {

  mSec += time.GetSec();
  mNSec += time.GetNSec();
  if ( mNSec >= 1000000000 ) {
    mSec += INT_4S( mNSec / 1000000000 );
    mNSec = mNSec % 1000000000;
  }
  return *this;
}


//-----------------------------------------------------------------------------
// Subtraction and assignment.
// param: Time& time -
// return: Time& time -

Time& wat::Time::operator-=(Time& time) {

  mSec -= time.GetSec();
  if ( mNSec >= time.GetNSec() ) {
    mNSec -= time.GetNSec();
  } else {
    --mSec;
    mNSec += 1000000000 - time.GetNSec();
  }
  return *this;
}

//-----------------------------------------------------------------------------
// Multiplication and assignment.
// param: double& d -
// return: Time& time -
// todo: What happens if d is negative?

Time& wat::Time::operator*=(double& d) {
		
  unsigned long long tmp = d*1000000000;
  unsigned long long lsec1 = d;
  unsigned long long lnsec1 = (tmp%1000000000)/100;
  unsigned long long lsec2 = mSec;
  unsigned long long lnsec2 = mNSec/100;
	
  tmp = lsec1*lsec2*1000000000+(lsec1*lnsec2+lnsec1*lsec2)*100+(lnsec1*lnsec2)/100000;
		
  mSec = tmp/1000000000;
  mNSec = tmp%1000000000;
		
  return *this;
}

//-----------------------------------------------------------------------------
// Division and assignment.
// param: double& d -
// return: Time& time -
// todo: What happens if d is negative?

Time& wat::Time::operator/=(double& d) {

  double ns = mNSec / d;
  double s  = mSec / d;
  mSec = INT_4S( mSec / d );
  mNSec = (INT_4U)( ns+(s-mSec)*1000000000 );
  return *this;
}

//-----------------------------------------------------------------------------
// Division.
// param: Time& time -
// return: double -

double wat::Time::operator/(Time& time) {

  return ((1000000000*mSec+mNSec)
         /(1000000000*time.GetSec()+time.GetNSec()));
}

//-----------------------------------------------------------------------------
// Equal comparison.
// Determines whether two Time objects are equal.
// param: Time& time - The object to compare with.
// return: bool - true if the objects are equal.

bool wat::Time::operator==(Time& time) {
  return (mSec==time.GetSec())&&(mNSec==time.GetNSec());
}

//-----------------------------------------------------------------------------
// Less than or equal to comparison.
// param: Time& time - The object to compare with.
// return: bool -

bool wat::Time::operator<=(Time& time) {
  return !( *this > time );
}

//-----------------------------------------------------------------------------
// Greater than or equal to comparison.
// param: Time& time - The object to compare with.
// return: bool -

bool wat::Time::operator>=(Time& time) {
  return !( *this < time );
}

//-----------------------------------------------------------------------------
// Not equal comparison.
// param: Time& time - The object to compare with.
// return: bool -

bool wat::Time::operator!=(Time& time) {
  return !( *this == time );
}

//-----------------------------------------------------------------------------
// Less than comparison.
// param: Time& time - The object to compare with.
// return: bool -

bool wat::Time::operator<(Time& time) {

  if ( mSec != time.GetSec() ) {
    return ( mSec < time.GetSec() );
  } else {
    return ( mNSec < time.GetNSec() );
  }
}

//-----------------------------------------------------------------------------
// Greater than comparison.
// param: Time& time - The object to compare with.
// return: bool -

bool wat::Time::operator>(Time& time) {

  if (mSec != time.GetSec()) {
    return ( mSec > time.GetSec() );
  } else {
    return ( mNSec > time.GetNSec() );
  }
}

//-----------------------------------------------------------------------------
// Addition.
// param: Time& t1
// param: Time& t2
// return: Time -

Time wat::operator+(Time& t1, Time& t2) {

  Time t3( t1 );
  return ( t3 += t2 );
}

//-----------------------------------------------------------------------------
// Subtraction.
// param: Time& t1
// param: Time& t2
// return: Time -

Time wat::operator-(Time& t1, Time& t2) {
  Time t3( t1 );
  return ( t3 -= t2 );
}

//-----------------------------------------------------------------------------
// Multiplication.
// param: Time& t
// param: double& d
// return: Time -

Time wat::operator*(Time& t, double& d) {
  Time t3( t );
  return ( t3 *= d );
}

//-----------------------------------------------------------------------------
// Division.
// param: Time& t
// param: double& d
// return: Time -

Time wat::operator/(Time& t, double& d) {
  Time t3( t );
  return ( t3 /= d );
}

//-----------------------------------------------------------------------------
// Multiplication.
// param: double& d
// param: Time& t
// return: Time -

Time wat::operator*(double& d, Time& t) {
  Time t3( t );
  return ( t3 *= d );
}

//-----------------------------------------------------------------------------
// Input extraction operator.
// param: Input& in
// param: Time& time
// return: Input&
// exc: read_failure

istream& wat::operator>>(istream& in, Time& time) {
  in >> time.mSec >> time.mNSec;
  return in;
}

//-----------------------------------------------------------------------------
// Double conversion operators (AC)

void wat::Time::SetDouble(double dt) {

  mSec  = dt;
  unsigned long long tmp = dt*1000000000;
  mNSec = (tmp%1000000000);
}

//-----------------------------------------------------------------------------

double wat::Time::GetDouble() {

  double dt;
  dt = mSec + 1.e-9 * mNSec;
  return dt;
}	

//-----------------------------------------------------------------------------
// Set Date String
// Date Format : XXYY-MM-DD hh:mm:ss

void wat::Time::SetDateString(TString date) {

  date.ToLower();

  if (date.CompareTo("now")==0) {  // set current date
    time_t ticks = time(NULL);
    this->SetSec(mktime(gmtime(&ticks)));
    this->SetNSec(0);
    this->UnixToGps();
    return;
  }

  date.Resize(19);

  // if date string is compatible with a integer it is converted to date
  if (date.IsDigit()) {this->SetSec(date.Atoi());return;}  

  TString idate;
  if (date.Sizeof()==20) {  // "XXYY-MM-DD hh:mm:ss"  ->  "ss:mm:hh-DD:MM:YY"
    int xx = TString(date(0,2)).Atoi();
    int yy = TString(date(2,2)).Atoi();
    if (yy>=80) if(xx!=19) {
      char msg[256];
      sprintf(msg,"error in data format : %s [Year >= 1980 && <2079]",date.Data());
      Error(msg);
    }
    if (yy<80)  if(xx!=20) {
      char msg[256];
      sprintf(msg,"error in data format : %s [Year >= 1980 && <2079]",date.Data());
      Error(msg);
    }
    idate = date(17,2)+":"+date(14,2)+":"+date(11,2)+"-"+date(8,2)+":"+date(5,2)+":"+date(2,2);
  } else {
    idate = date;
  }
  SetString(const_cast<char*>(idate.Data()));
}

//-----------------------------------------------------------------------------
// Set Date String
// Date Format :  ss:mm:hh:DD:MM:YY

void wat::Time::SetString(char* date, int nsec) {

  char	DD_s[4],MM_s[4],YY_s[4],hh_s[4],mm_s[4],ss_s[4];
  int	DD,MM,YY,hh,mm,ss;
  if (strlen(date) != 17) Error(const_cast<char*>("date length not valid"));
  strncpy(ss_s,date,2);
  
  if(!isdigit(ss_s[0])) Error(const_cast<char*>("sec not valid format"));
  if(!isdigit(ss_s[1])) Error(const_cast<char*>("sec not valid format"));
  ss_s[2]=0;  
  ss=atoi(ss_s);

  strncpy(mm_s,date+3,2);
  if(!isdigit(mm_s[0])) Error(const_cast<char*>("minutes not valid format"));
  if(!isdigit(mm_s[1])) Error(const_cast<char*>("minutes not valid format"));
  mm_s[2]=0;  
  mm=atoi(mm_s);

  strncpy(hh_s,date+6,2);
  if(!isdigit(hh_s[0])) Error(const_cast<char*>("hour not valid format"));
  if(!isdigit(hh_s[1])) Error(const_cast<char*>("hour not valid format"));
  hh_s[2]=0;  
  hh=atoi(hh_s);

  strncpy(DD_s,date+9,2);
  if(!isdigit(DD_s[0])) Error(const_cast<char*>("day not valid format"));
  if(!isdigit(DD_s[1])) Error(const_cast<char*>("day not valid format"));
  DD_s[2]=0;  
  DD=atoi(DD_s);

  strncpy(MM_s,date+12,2);
  if(!isdigit(MM_s[0])) Error(const_cast<char*>("month not valid format"));
  if(!isdigit(MM_s[1])) Error(const_cast<char*>("month not valid format"));
  MM_s[2]=0;  
  MM=atoi(MM_s);

  strncpy(YY_s,date+15,2);
  if(!isdigit(YY_s[0])) Error(const_cast<char*>("year not valid format"));
  if(!isdigit(YY_s[1])) Error(const_cast<char*>("year not valid format"));
  YY_s[2]=0;  
  YY=atoi(YY_s);
  if (YY<70) YY+=100;

  SetDate(ss,mm,hh,DD,MM,YY,nsec);
}

void wat::Time::SetDate(int ss, int mm, int hh, int DD, int MM, int YY, int nsec) {

  // the values must be checked !!!!!!!!! -> to be done

  if(YY>1900) YY-=1900;

  //	extern char *tzname[2]; 
  struct  tm in_tp;

  in_tp.tm_sec	= ss;       // seconds 0:59
  in_tp.tm_min	= mm;       // minutes 0:59
  in_tp.tm_hour	= hh;       // hours 0:23
  in_tp.tm_mday	= DD;       // day of the month 1:31
  in_tp.tm_mon	= MM-1;     // month 0:11
  in_tp.tm_year = YY;       // year since 1900
  in_tp.tm_isdst= 0;

  
  
  setenv("TZ", "", 1);
  tzset();
   			
  time_t in_utc_sec = mktime(&in_tp);
  
  struct tm* out_tp = gmtime(&in_utc_sec);
 
  time_t out_utc_sec = mktime(out_tp);
 	
  
  // setenv("TZ", *tzname, 1);
	
  if (in_tp.tm_sec !=	ss)   Error(const_cast<char*>("sec not valid format"));
  if (in_tp.tm_min !=	mm)   Error(const_cast<char*>("minutes not valid format"));
  if (in_tp.tm_hour != hh)    Error(const_cast<char*>("hour not valid format"));
  if (in_tp.tm_mday != DD)    Error(const_cast<char*>("day not valid format"));
  if (in_tp.tm_mon !=	MM-1) Error(const_cast<char*>("month not valid format"));
  if (in_tp.tm_year != YY)    Error(const_cast<char*>("year not valid format"));
	
  if(in_utc_sec != out_utc_sec) Error(const_cast<char*>("Date not valid format"));

  mSec = in_utc_sec; // - UTC_UNIX_SPAN - UTC_LEAP_SECONDS;
  mNSec = nsec;

  UnixToGps();
};

TString wat::Time::GetDateString() {
					
  Time tempTime(*this);
  tempTime.GpsToUnix();
  time_t time = tempTime.GetSec();

  bool leap=false;
  for(int i=0;i<GPS_LEAPS_TABLE_SIZE;i++) 
    if(gps_leaps_table[i].gps==GetSec()+1) {time-=1;leap=true;break;}

  // Convert Data Format
  // From
  // Thu Feb  5 05:30:34 1981
  // To
  // 1981-02-05 05:30:34 UTC Thu

  TObjArray* token = TString(ctime(&time)).Tokenize(TString(' '));
  TObjString* week_tok = (TObjString*)token->At(0);
  TString week = week_tok->GetString();
  TObjString* month_tok = (TObjString*)token->At(1);
  TString month = month_tok->GetString();
  TObjString* day_tok = (TObjString*)token->At(2);
  TString day = day_tok->GetString();
  TObjString* hhmmss_tok = (TObjString*)token->At(3);
  TString hhmmss = hhmmss_tok->GetString();
  if(leap) {hhmmss[6]='6';hhmmss[7]='0';}
  TObjString* year_tok = (TObjString*)token->At(4);
  TString year = year_tok->GetString();
  year.Resize(year.Sizeof()-2);

  if(month.CompareTo("Jan")==0) month="01";
  if(month.CompareTo("Feb")==0) month="02";
  if(month.CompareTo("Mar")==0) month="03";
  if(month.CompareTo("Apr")==0) month="04";
  if(month.CompareTo("May")==0) month="05";
  if(month.CompareTo("Jun")==0) month="06";
  if(month.CompareTo("Jul")==0) month="07";
  if(month.CompareTo("Aug")==0) month="08";
  if(month.CompareTo("Sep")==0) month="09";
  if(month.CompareTo("Oct")==0) month="10";
  if(month.CompareTo("Nov")==0) month="11";
  if(month.CompareTo("Dec")==0) month="12";

  char date[256];
  // 1981-02-05 05:30:34 UTC Thu
  sprintf(date,"%s-%s-%02d %s UTC %s",year.Data(),month.Data(),day.Atoi(),hhmmss.Data(),week.Data());

  return date;
}

void wat::Time::Print() {
/*					
  char	time_str[26];
  Time  tempTime(*this);
  tempTime.GpsToUnix();
  time_t  time = tempTime.GetSec();
  strcpy(time_str, ctime(&time));
  cout << time_str << endl;
*/
  cout << GetDateString().Data() << endl;
}

double wat::Time::GetJulianDate() {
//
// From LAL XLALCivilTime.c
//
// Returns the Julian Date (JD) 
//                                                                         
// See ref esaa1992 and ref green1985 for details.  First, some          
// definitions:                                                            
//                                                                         
// Mean Julian Year = 365.25 days                                          
// Julian Epoch = 1 Jan 4713BCE, 12:00 GMT (4713 BC Jan 01d.5 GMT)         
// Fundamental Epoch J2000.0 = 2001-01-01.5 TDB                            
//                                                                         
// Julian Date is the amount of time elapsed since the Julian Epoch,       
// measured in days and fractions of a day.  There are a couple of         
// complications arising from the length of a year:  the Tropical Year is  
// 365.2422 days.  First, the Gregorian correction where 10 days           
// (1582-10-05 through 1582-10-14) were eliminated.  Second, leap years:   
// years ending with two zeroes (e.g., 1700, 1800) are leap only if        
// divisible by 400;  so, 400 civil years contain 400 * 365.25 - 3 = 146097
// days.  So, the Julian Date of J2000.0 is JD 2451545.0, and thus the     
// Julian Epoch = J2000.0 + (JD - 2451545) / 365.25, i.e., number of years 
// elapsed since J2000.0.                                                  
//                                                                         
// One algorithm for computing the Julian Day is from ref vfp1979 based   
// on a formula in ref esaa1992 where the algorithm is due to             
// fvf1968 and ``compactified'' by P. M. Muller and R. N. Wimberly.   
// The formula is                                                          
//                                                                         
// \f[                                                                     
// jd = 367 \times y - 7 \times (y + (m + 9)/12)/4 - 3 \times ((y + (m -   
// 9)/7)/100 + 1)/4 + 275 \times m/9 + d + 1721029                         
// \f]                                                                     
//                                                                         
// where jd is the Julian day number, y is the year, m is the month (1-12),
// and d is the day (1-31).  This formula is valid only for JD > 0, i.e.,  
// after -4713 Nov 23 = 4712 BCE Nov 23.                                   
//                                                                         
// A shorter formula from the same reference, but which only works for     
// dates since 1900 March is:                                              
//                                                                         
// \f[                                                                     
// jd = 367 \times y - 7 \times (y + (m + 9)/12)/4 + 275 \times m/9 + d +  
// 1721014                                                                 
// \f]                                                                     
//                                                                         
// We will use this shorter formula since there is unlikely to be any      
// analyzable data from before 1900 March.                                 
//

  const int sec_per_day = 60 * 60 * 24; 	// seconds in a day                                     
  int year, month, day, sec;                                                                      
  double jd;                                                                                       

  // this routine only works for dates after 1900 
  if(GetYear()<=0) Error(const_cast<char*>("Year must be after 1900"));

  year  = GetYear();
  month = GetMonth();     	// month is in range 1-12 
  day   = GetDay();        	// day is in range 1-31   
  sec   = GetSecond() + 60*(GetMinute() + 60*GetHour()); // seconds since midnight 

  jd = 367*year - 7*(year + (month + 9)/12)/4 + 275*month/9 + day + 1721014;
  // note: Julian days start at noon: subtract half a day                 
  jd += (double)sec/(double)sec_per_day - 0.5;                                
  return jd;
}

double wat::Time::GetModJulianDate() {
//
// From LAL XLALCivilTime.c
//
// Returns the Modified Julian Day (MJD) 
//                                                                       
// Note:                                                                 
//   - By convention, MJD is an integer.                                 
//   - MJD number starts at midnight rather than noon.                   
//                                                                       
// If you want a Modified Julian Day that has a fractional part, simply use
//                                                                         
                                                                        
  double jd = GetJulianDate();
  if(TMath::IsNaN(jd)) Error(const_cast<char*>("julian day is a NaN"));
  double mjd = jd - CWB_MJD_REF;
  return mjd;
}

int wat::Time::GpsToGpsLeaps(int gps) { 

  int i = 1;
  while (gps >= gps_leaps_table[i].gps) {++i;if(i>=GPS_LEAPS_TABLE_SIZE) break;}
  return gps_leaps_table[i-1].gps_utc;
}

int wat::Time::UnixToGpsLeaps(int unix_time) { 

  int i = 1;
  while (unix_time >= (gps_leaps_table[i].gps + UTC_UNIX_SPAN - GpsToGpsLeaps(gps_leaps_table[i].gps))) 
    {++i;if(i>=GPS_LEAPS_TABLE_SIZE) break;}
  return gps_leaps_table[i-1].gps_utc;
}

//-----------------------------------------------------------------------------
// Output insertion operator.
// param: Output& out
// param: Time& time
// return: Output&
// exc: write_failure

ostream& wat::operator<<(ostream& out, Time& time) {

  out << time.GetSec() << ":";
  out.fill('0');
  out.width(9);
  out << time.GetNSec();
  out << endl;
  return out;
}

//-----------------------------------------------------------------------------
// Get the number of seconds.
// return: INT_4S

INT_4S wat::Time::GetSec() {
  return mSec;
}

//-----------------------------------------------------------------------------
// Get the number of nanoseconds.
// return: INT_4U

INT_4U wat::Time::GetNSec() {
  return mNSec;
}
