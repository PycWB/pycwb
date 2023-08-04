#ifdef __CLING__ 

#pragma link off all globals; 
#pragma link off all classes; 
#pragma link off all functions; 

//#pragma link C++ global gROOT;
//#pragma link C++ global gEnv;

#pragma link C++ enum POLARIZATION;
#pragma link C++ enum BORDER;
#pragma link C++ enum WAVETYPE;

#pragma link C++ class slice+; 
#pragma link C++ class wavepixel; 
#pragma link C++ class netpixel+; 
#pragma link C++ class wavecomplex; 

#ifdef _USE_DMT
#pragma link C++ class LineFilter;
#endif

#ifdef _USE_HEALPIX
#pragma link C++ class wat::Alm_Base;
#pragma link C++ class wat::Alm_Template<complex<double>>;
#pragma link C++ class wat::Alm;
#endif

#pragma link C++ class Wavelet+; 
#pragma link C++ class wavearray<Long64_t>-;
#pragma link C++ class wavearray<int>-;
#pragma link C++ class wavearray<unsigned int>-;
#pragma link C++ class wavearray<long long>-;
#pragma link C++ class wavearray<short>-;
#pragma link C++ class wavearray<long>-;
#pragma link C++ class wavearray<float>-;
#pragma link C++ class wavearray<double>-;
#pragma link C++ function wavearray<int>::wavearray<int>(const int*,unsigned int,double);
#pragma link C++ function wavearray<unsigned int>::wavearray<unsigned int>(const unsigned int*,unsigned int,double);
#pragma link C++ function wavearray<long long>::wavearray<long long>(const long long*,unsigned int,double);
#pragma link C++ function wavearray<short>::wavearray<short>(const short*,unsigned int,double);
#pragma link C++ function wavearray<long>::wavearray<long>(const long*,unsigned int,double);
#pragma link C++ function wavearray<float>::wavearray<float>(const float*,unsigned int,double);
#pragma link C++ function wavearray<double>::wavearray<double>(const double*,unsigned int,double);
#pragma link C++ class WaveDWT<float>-;
#pragma link C++ class WaveDWT<double>-;
#pragma link C++ class Haar<float>+;
#pragma link C++ class Haar<double>+;
#pragma link C++ class Biorthogonal<float>+;
#pragma link C++ class Biorthogonal<double>+;
#pragma link C++ class Daubechies<float>+;
#pragma link C++ class Daubechies<double>+;
#pragma link C++ class Symlet<float>+;
#pragma link C++ class Symlet<double>+;
#pragma link C++ class Meyer<float>+;
#pragma link C++ class Meyer<double>+;
#pragma link C++ class WDM<float>+;
#pragma link C++ class WDM<double>+;
//#pragma link C++ class WDMOverlap<double>;
#pragma link C++ class monster+;
#pragma link C++ class SymmArray<int>+;
#pragma link C++ class SymmArray<float>+;
#pragma link C++ class SymmArray<double>+;
#pragma link C++ class SymmArraySSE<int>+;
#pragma link C++ class SymmArraySSE<float>+;
#pragma link C++ class SymmArraySSE<double>+;
#pragma link C++ class SymmObjArray<SymmArray<int> >+;
#pragma link C++ class SymmObjArray<SymmArray<float> >+;
#pragma link C++ class SymmObjArray<SymmArray<double> >+;
#pragma link C++ class SymmObjArray<SymmArraySSE<int> >+;
#pragma link C++ class SymmObjArray<SymmArraySSE<float> >+;
#pragma link C++ class SymmObjArray<SymmArraySSE<double> >+;
#pragma link C++ class WSeries<float>-;
#pragma link C++ class WSeries<double>-;
#pragma link C++ class WaveRDC+;
#pragma link C++ class wavecluster;
#pragma link C++ class netcluster+;
#pragma link C++ class clusterdata+;
#pragma link C++ class wavecor;
#pragma link C++ class linefilter;
#pragma link C++ class skymap-;
#pragma link C++ class detector-;
#pragma link C++ class network+;
#pragma link C++ class netevent;
#pragma link C++ class injection;
#pragma link C++ class watplot;
#pragma link C++ class regression;
#pragma link C++ class SSeries<float>-;
#pragma link C++ class SSeries<double>-;

#pragma link C++ TClass char*;
#pragma link C++ TClass size_t;
#pragma link C++ TClass double;
#pragma link C++ TClass vectorD;

#pragma link C++ struct detectorParams+;
#pragma link C++ struct delayFilter+;
#pragma link C++ struct waveSegment+;
#pragma link C++ struct pixdata+;
#pragma link C++ struct xtalk+;

#pragma link C++ nestedtypedef;
#pragma link C++ typedef vectorD;

#pragma link C++ class vector<waveSegment>+;
#pragma link C++ class vector<detector*>+;
#pragma link C++ class vector<netcluster>+;
#pragma link C++ class vector<clusterdata>+;
#pragma link C++ class vector<std::string>+;
#pragma link C++ class vector<vectorD>+;
#pragma link C++ class vector<delayFilter>+;
#pragma link C++ class vector<int>+;
#pragma link C++ class vector<netpixel>+;
#pragma link C++ class vector<pixdata>+;
#pragma link C++ class vector<vector_int>+;
#pragma link C++ class vector<vector_float>+;
#pragma link C++ class vector<wavearray<float>>+;
#pragma link C++ class vector<WDM<double>*>+;
#pragma link C++ class vector<SymmArraySSE<float>>+;
#pragma link C++ class vector<WSeries<double>*>+;
#pragma link C++ class vector<SSeries<double>>+;
#pragma link C++ class vector<TGraph*>+;

#pragma link C++ class watconstants;
#pragma link C++ class wat::Time;

//#pragma link C++ function unCompress(int*, wavearray<double>&);

#ifdef _USE_FR
#pragma link C++ function ReadFrFile;
#pragma link C++ function ReadFrame;
#endif

#pragma link C++ class WAT+; 
#pragma link C++ global	NIFO_MAX;
#pragma link C++ global	NRES_MAX;

#pragma link C++ function watversion;
#pragma link C++ function waveAssign(wavearray<int> &, wavearray<float> &);
#pragma link C++ function waveAssign(wavearray<int> &, wavearray<double> &);
#pragma link C++ function waveAssign(wavearray<float> &, wavearray<int> &);
#pragma link C++ function waveAssign(wavearray<float> &, wavearray<double> &);
#pragma link C++ function waveAssign(wavearray<double> &, wavearray<float> &);
#pragma link C++ function waveAssign(wavearray<double> &, wavearray<short> &);

#pragma link C++ function Lagrange;
#pragma link C++ function fLagrange;
#pragma link C++ function Nevill(const double, int, double*, double*);
#pragma link C++ function Nevill(const double, int, float*, double*);
#pragma link C++ function signPDF;
#pragma link C++ function gammaCL;
#pragma link C++ function gammaCLa;
#pragma link C++ function logNormCL;
#pragma link C++ function logNormArg;
#pragma link C++ function Gamma(double,double);
#pragma link C++ function iGamma(double,double);
#pragma link C++ function Gamma(double);

#pragma link C++ function CwbToGeographic;
#pragma link C++ function GeographicToCwb;
#pragma link C++ function CwbToCelestial;
#pragma link C++ function CelestialToCwb;
#pragma link C++ function GalacticToEquatorial;
#pragma link C++ function EquatorialToGalactic;
#pragma link C++ function EclipticToEquatorial;
#pragma link C++ function EquatorialToEcliptic;
#pragma link C++ function GeodeticToGeocentric;
#pragma link C++ function GeocentricToGeodetic;
#pragma link C++ function GetCartesianComponents;

#pragma link C++ function operator>>(watplot&,wavearray<double>&);
#pragma link C++ function operator>>(wavearray<double>&,watplot&);
#pragma link C++ function operator>>(watplot&, TString&);
#pragma link C++ function operator>>(watplot&, char*&);

#pragma link C++ class WAT;

#endif // __CLING__
