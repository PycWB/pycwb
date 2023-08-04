/*
# Copyright (C) 2019 Sergey Klimenko
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


#ifndef LINEFILTER_HH
#define LINEFILTER_HH

#include <iosfwd>
#include <list>
#include <vector>
#include "Pipe.hh"
#include "Time.hh"
#include "Interval.hh"
#include "TSeries.hh"
#include "wseries.hh"

#include <complex>

typedef std::complex<float> f_complex; 
typedef double wavereal; 
typedef wavearray<wavereal> WaveData; 

struct lineData {
      Time                T_current;
      float               frequency;
      float               intensity;
      unsigned int        first;
      std::vector<f_complex>   amplitude;
      std::vector<float>       line;
      std::vector<float>       noise;
      std::vector<float>       filter;
};

/**  The LineFilter class containes methods to track and remove quasi-
  *  monochromatic lines. a TSeries by 2^N. The TSeries 
  *  data are filtered before decimation to remove aliasing.
  *  @memo Line Removal.
  *  @version 1.2 ; Modified November 01, 2000
  *  @version 1.3 ; Modified November 17, 2000
  *  @author Sergey Klimenko
  */
class LineFilter : public Pipe {
public:
  /**  Build an empty LineFilter.
    *  @memo Default constructor.
    */
  LineFilter(void);

  /**  @memo LineFilter constructor
    *  filter type (default fid = 1) and time interval T to estimate and 
    *  remove interference (T=0 - the whole input TS is used).
    *  @memo Constructor.
   *@memo   Set parameters of the line filter   
   *@param   f - line base frequency 
   *@param   T - time interval to estimate interference 
   *             (T=0 - the whole TS is used) 
   *@param fid - filter ID: 
   *         1 - use resampling and FFT with noise floor estimation
   *         0 - use resampling and FFT, no noise floor estimation
   *        -1 - heterodyne estimation of amplitude and phase, frequency=const
   *        -2 - -1 + Hann window
   *        -3 - the same as -2, but sin(), cos() and Hann() are tabulated 
   *@param  nT - number of interval T sub0divisions
   */
  LineFilter(double f, double T = 0., int fid = 1, int nT = 1);

  /**  Build a LineFilter identical to an existing filter.
    *  @memo Copy constructor.
    */
  LineFilter(const LineFilter& x);

  /**  Destroy the LineFilter object and release the function storage.
    *  @memo Virtual destructor.
    */
  ~LineFilter(void);


  /**  Clone a LineFilter
   */
  LineFilter* clone(void) const;

  /**  The argument time series is filtered to remove lines, and 
    *  the argument TSeries ts is left unchanged.
    *  @memo Return cleaned TSeries.
    */
  TSeries apply(const TSeries& ts);
  TSeries operator()(const TSeries& ts);
  FilterIO& operator()(const FilterIO& in) {
     return Pipe::operator() (in); }
     
  /**  Operate on wavearray object
    */
  void apply(WaveData& ts);
  FilterIO& apply(const FilterIO& in) {
     return Pipe::apply (in); }
     
  /**  Check the data for validity. If the data are not applicable for 
    *  line removal, an exception is thrown.
    */
  void dataCheck(const TSeries& ts) const;
  void dataCheck(const FilterIO& in) const {
     Pipe::dataCheck (in); }

  /**  Check the data for validity. Performs the same data checks as 
    *  dataCheck() but returns a boolean status instead f throwing an
    *  exception.
    */
  bool isDataValid(const TSeries& ts) const;
  bool isDataValid(const FilterIO& in) const {
     return Pipe::isDataValid (in); }


  /***setFilter*********************************************************
   *@memo   Set parameters of the line filter   
   *@param  nF - first harmonic
   *@param  nL - last harmonic
   *@param  nS - skip nS-1 harmonics (take nF, nF+nS, nF+2nS,....)
   *@param  nD - wavelet decimation factor
   *@param  nB - nB/T is a frequency band to estimate noise 
   *@param  nR - order of Resample Interpolating Filter
   *@param  nW - order of the decimating lifting wavelet
   *********************************************************************/
  void setFilter(int nF = 1,
		 int nL = 0,
		 int nS = 1,
		 int nD = -1,
		 int nB = 5,
		 int nR = 6,
		 int nW = 8);

  /***setFScan*********************************************************
   *@memo   Set parameters for getFrequency()   
   *@param   f - base frequency: if f<=0 - don't scan frequency,
   *@            f=0 - don't change frequency specified (by LineFilter)
   *@param  sn - limit on signal to noise ratio 
   *@param  fS - initial range of frequency scan in units of fft bin
   *@param  nS - number of steps during frequency scan 
   *********************************************************************/
  void setFScan(double  f = 0., 
		double sn = 2.,
		double fS = 0.45,
		int nS = 20);


  /**  Clear/release the internal History vector and reset the current 
    *  time.
    */
  void reset();
  void resize(size_t=0);

  inline Time getStartTime(void) const;
  inline Time getCurrentTime(void) const;
  inline bool inUse(void) const;
  
//private:

  int          FilterID;  
  double       Frequency;          // fundamental line frequency
  double       Window;
  double       Stride;
  unsigned int nFirst;             // first line harmonic
  unsigned int nLast;              // last line harmonic
  int          nStep;              // skip harmonics (take nF, nF+nS, nF+2nS,....) 
  int          nScan;              // # of frequency steps to scan frequency  
  unsigned int nBand;              // frequency band in fft bins to average noise  
  int          nSubs;              // number of data subsets to estimate signal PSD 
  double       fBand;              // frequency step in fft bins to scan frequency
  int          nLPF;               // decimation factor
  int          nWave;              // order of the interpolating wavelet
  bool         clean;              // true if to clean data  
  bool         badData;            // false if valid data  
  bool         noScan;             // true if Frequency is fixed 
  int          nRIF;               // order of Resample Interpolating Filter
  double       SNR;                // limit on SNR used by makeFilter
  bool         reFine;             // refine frequency if true  (set by SNR<0)
  size_t       dumpStart;          // first lineList index used to dump data
  int          FilterState;
  double       SeedFrequency;

  Time     CurrentTime;
  Time     StartTime;
  Interval Sample;

  wavearray<double> ct;            // tabulated cos() 
  wavearray<double> st;            // tabulated cos() 
  wavearray<double> wt;            // tabulated window

  std::list<lineData> lineList;

  WaveData NoiseSD;
  WaveData LineSD;
  WaveData Filter;

  WaveData getPSD(const WaveData &, int = 1);
    double makeFilter(const WaveData &, int = 0); 
  lineData getLine(WaveData &);

  /***getHeteroLine****************************************************
   *@memo   reconstruct line amplitude and phase using heterodyne method   
   *@param  input time series
   *********************************************************************/
  lineData getHeteroLine(WaveData &);

    double getOmega(const WaveData &, int = 2);
    double fScan(const WaveData &);
    double Interference(WaveData &, double);

  wavearray<float> getTrend(int, char);
  bool DumpTrend(const char*, int = 0);
  bool LoadTrend(const char*);

  inline double newRate(double);
  unsigned int maxLine(int);
  inline double axb(double, double);
  inline double wrap(double);
  inline long intw(double);

  // used by THtml doc
  ClassDef(linefilter,1)	 
};

inline double LineFilter::newRate(double rate) 
{
  double f = rate/Frequency;
  f *= (nLPF >= 0) ? 1 : 2;
  return  (int(f)+1)*Frequency;
}

inline double LineFilter::axb(double a, double b) 
{return  (a-long(a))*long(b) + (b-long(b))*long(a) + (a-long(a))*(b-long(b));}

inline long LineFilter::intw(double a) 
{return  (a>0) ? long(a+0.5) : long(a-0.5);}

inline double LineFilter::wrap(double a) 
{ 
  long l = a>0 ? long(a/PI/2. + 0.5) : long(a/PI/2. - 0.5);
  return a - 2*PI*l; 
}

inline Time LineFilter::getStartTime(void) const {
    return StartTime;
}

inline Time LineFilter::getCurrentTime(void) const {
    return CurrentTime;
}

inline bool LineFilter::inUse(void) const {
    return (StartTime != Time(0));
}

#endif  // LineFilter_HH
