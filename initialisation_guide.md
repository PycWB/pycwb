# Initialisation

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
	gSystem.AddIncludePath(inc)

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
