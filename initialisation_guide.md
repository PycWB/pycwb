# Initialisation


## Load all required libraries

### Reference from original bash + C
ROOT require a `.root` file to point to the `logon.c`

>  `.root`: ${SOURCE}/tools/cwb/cwb.rootrc
 
```bash
 #Browser.Name:            TRootBrowser
Browser.Name:            TRootBrowserLite
#Rint.Logon:              rootlogon.C
Rint.Logon:              $CWB_ROOTLOGON_FILE
```

> `logon.c`: ${INSTALL}/etc/cwb/macros/cwb_rootlogon.C

```bash
_USE_ROOT6 = 
_USE_ICC = 
_USE_HEALPIX = 
HOME_CFITSIO = # leave empty if installed with conda
HOME_HEALPIX = # leave empty if installed with conda

_USE_LAL = 
HOME_LAL = # leave empty if installed with conda

_USE_EBBH = # optional
HOME_CVODE = 

HOME_WAT = 
HOME_WAT_INSTALL = 

HOME_FRLIB = 
CWB_STFT = 
CWB_GWAT = 
CWB_TOOLBOX = 
CWB_HISTORY = 
CWB_BICO = 
CWB_FRAME = 
CWB_FILTER = 
HOME_CWB = 
CWB_MACROS = 
```

### Python code replacement
remove `.so` for all relative import in `cwb_rootlogon` to enable import both `.so` and `.dylib`

```python
_USE_ROOT6 = True
_USE_HEALPIX = True
_USE_LAL = True

HOME_CFITSIO = "" # leave empty if installed with conda
HOME_HEALPIX = "" # leave empty if installed with conda
HOME_LAL = "" # leave empty if installed with conda

_USE_ICC = False # optional

_USE_EBBH = False # optional
HOME_CVODE = ""

# HOME_WAT = "~/Project/Physics/cwb/cwb_source"
# HOME_WAT_INSTALL = "~/Project/Physics/cwb/cwb_source/tools/install"
CWB_INSTALL = "~/Project/Physics/cwb/cwb_source/tools/install"
CWB_SOURCE = "~/Project/Physics/cwb/cwb_source" # For loading macros
HOME_FRLIB = "" # leave empty if installed with conda
# CWB_STFT = 
# CWB_GWAT = 
# CWB_TOOLBOX = 
# CWB_HISTORY = 
# CWB_BICO = 
# CWB_FRAME = 
# CWB_FILTER = 
HOME_CWB = ""
CWB_MACROS = ""

import platform
import os

os.environ['LD_LIBRARY_PATH'] = f"{CWB_INSTALL}/lib" # TODO: add previous LD path

from ROOT import gROOT, gSystem, gStyle
import ROOT

# include paths
include_path = [f"{CWB_INSTALL}/include"]
if _USE_EBBH: include_path += [f"{HOME_CVODE}/include"]

for inc in include_path:
	# TODO: error process
	gROOT.ProcessLine(f".include {inc}")

# load libraries
root_lib = ['libPhysics', 'libFFTW', 'libHtml', 'libTreeViewer', 'libpng', 'libFITSIO'] #libFITSIO mac only?
lal_lib = ['liblal', 'liblalsupport', 'liblalframe', 'liblalmetaio', 'liblalsimulation', 'liblalinspiral', 'liblalburst']
healpix_lib = ['libhealpix_cxx', 'libgomp']
wat_lib = ['wavelet']
ebbh = ['eBBH']# TODO

libs = root_lib
if _USE_LAL: libs += lal_lib
if _USE_HEALPIX: libs += healpix_lib
libs += wat_lib
if _USE_EBBH: libs += ebbh

for lib in libs:
	# TODO: error process and version check (healpix >= 3.00)
	# use user path if can't find lal healpix etc. in default
	gSystem.Load(lib)

# Loading Macros
wat_macros = ["Histogram.C", "AddPulse.C", "readAscii.C", "readtxt.C"]

for macro in wat_macros:
	gROOT.LoadMacro(f"{CWB_SOURCE}/wat/macro/{macro}")

# Loading tools
tools_lib = ['STFT', 'gwat', 'Toolbox', 'History', 'Bicoherence', 'Filter', 'frame', 'cwb', 'wavegraph']

for lib in tools_lib:
	# TODO: error process
	gSystem.Load(f"{CWB_INSTALL}/lib/{lib}")


# declare ACLiC includes environment 
for inc in include_path:
	gSystem.AddIncludePath(f"-I{inc}")

# declare ACLiCFlag options 
flag = "-D_USE_ROOT -fPIC -Wno-deprecated -mavx -Wall -Wno-unknown-pragmas -fexceptions -O2 -D__STDC_CONSTANT_MACROS"
if _USE_HEALPIX: flag += " -D_USE_HEALPIX"
if _USE_LAL: flag += " -D_USE_LAL"
if _USE_EBBH: flag += " -D_USE_EBBH"
if _USE_ROOT6: flag += " -D_USE_ROOT6"
if platform.system() == "Darwin": flag += " -fno-common -dynamiclib -undefined dynamic_lookup"
flag += " -fopenmp"
if _USE_ICC: flag += " -diag-disable=2196"


gSystem.SetFlagsOpt(flag);

# set the offset for TimeDisplay, the seconds declared in xaxis
# are refered to "1980-01-06 00:00:00 UTC Sun" -> GPS = 0
gStyle.SetTimeOffset(315964790); 

gStyle.SetPalette(1,0);
gStyle.SetNumberContours(256);

gROOT.ForceStyle(0);
```


If some error happened, try restart first.

## Setup the environment and initialise the parameters

Convert `watenv.sh`

> what is the use of `/home/waveburst/SOFT/cWB/tags/config/cWB-OfflineO3-v9-O3b/XTALKS`

Below should be replaced by yaml file in the future

```python
envs = {
	"HOME_WAT_FILTERS": "path",
	"HOME_BAUDLINE": "baudline", # independent, path of software
	"HOME_ALADIN": "aladin", # independent, path of software
	"HOME_SKYMAP_LIB": "path", # required
	"CWB_USER_URL": "url", # report url
	"WWW_PUBLIC_DIR": "~/Downloads/tmp/public_html/reports",
	"CONDOR_LOG_DIR": "~/Downloads/tmp/condor",
	"NODE_DATA_DIR": "~/Downloads/tmp/node",
	"CWB_ANALYSIS": '2G',
	"CWB_PARAMETERS_FILE": f"{CWB_INSTALL}/etc/cwb/macros/cwb2G_parameters.C",
	"CWB_ROOTLOGON_FILE": "",
	"CWB_UPARAMETERS_FILE": "",
	"CWB_EPARAMETERS_FILE": "",
	}

envs['CWB_PARMS_FILES'] = f"{CWB_ROOTLOGON_FILE} {envs['CWB_PARAMETERS_FILE']} {envs['CWB_UPARAMETERS_FILE']} {envs['CWB_EPARAMETERS_FILE']}"

for key in envs.keys():
	os.environ[key] = envs[key]

gROOT.LoadMacro(envs['CWB_PARMS_FILES'])
```

## Run examples

Go to the installation directory
```bash
cd tools/install/etc/cwb
```


```python
# gROOT.LoadMacro(f"{CWB_SOURCE}/tools/cwb/examples/ADV_SIM_NSNS_L1H1V1_MultiStages2G/config/user_parameters.C")
gROOT.LoadMacro(f"{CWB_INSTALL}/etc/cwb/macros/cwb_xnet.C")
ROOT.cwb_xnet(f"{CWB_SOURCE}/tools/cwb/examples/ADV_SIM_NSNS_L1H1V1_MultiStages2G/config/user_parameters.C")
```

Errors:

```c++
cfg.Import("$CWB_PARAMETERS_FILE");


//______________________________________________________________________________
void
CWB::config::Import(TString umacro) {
//
// Import from macro or CINT the configuration parameters
//
// Input: umacro  - unnamed root macro
//                  if umacro="" then parameters are imported from CINT
//
// NOTE: macro must be unnamed, parameters in macro must be declared with type
//
// WARNING: if umacro!="" all CINT global variables with the same name are overwritten !!!
//

#ifdef _USE_ROOT6
  // The interpreter of root6 if full c++ compliant
  // When unamed macros are loaded the symbols which have been redeclared with the same name are not allowed
  // The following code check if umacro has been already loaded from the root command line
  // if already loaded the macro umacro is not reloaded
  // This patch fix the job running with condor
  for(int i=0;i<gApplication->Argc();i++) {
    bool check=true;
    if(TString(gApplication->Argv(i)).EndsWith(".C")) {

      char* file1 = CWB::Toolbox::readFile(gApplication->Argv(i));
      if(file1==NULL) {check=false;continue;}
      char* file2 = CWB::Toolbox::readFile(umacro);
      if(file2==NULL) {delete [] file1;check=false;continue;}

      //cout << "file1 : " << gApplication->Argv(i) << " " << strlen(file2) << endl;
      //cout << "file2 : " << umacro << " " << strlen(file2) << endl;
      if(strlen(file1)==strlen(file2)) {
        for(int i=0;i<strlen(file1);i++) {if(file1[i]!=file2[i]) check=false;break;}
      } else check=false;
      delete [] file1; 
      delete [] file2; 
    } else check=false;
    if(check==true) return;
  } 
#endif

  int err=0;
  if(umacro!="") {
    gROOT->ProcessLine("#include \"xroot.hh\"");	// define macros SEARCH,GAMMA,XROOT
    gROOT->Macro(umacro,&err);
    if(err!=0) {
      cout << "CWB::config::Import : Error Loading Macro " << umacro.Data() << endl;
      exit(1);
    }
  }

  SetVar(0);
}
```