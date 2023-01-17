# _USE_ROOT6 = True
# _USE_HEALPIX = True
# _USE_LAL = True

# HOME_CFITSIO = "" # leave empty if installed with conda
# HOME_HEALPIX = "" # leave empty if installed with conda
# HOME_LAL = "" # leave empty if installed with conda

# _USE_ICC = False # optional

# _USE_EBBH = False # optional
# HOME_CVODE = ""


import platform
import os
import configparser

def ROOT_logon(gROOT, gSystem, gStyle, config):
	################
	# set include paths for compiling
	################
	include_path = [f"{config['CWB']['CWB_INSTALL']}/include"]
	if config.getboolean('LIB','_USE_EBBH'): include_path += [f"{config['LIB']['HOME_CVODE']}/include"]

	for inc in include_path:
		# TODO: error process
		gROOT.ProcessLine(f".include {inc}")

	################
	# load cwb libraries
	################
	root_lib = ['libPhysics', 'libFFTW', 'libHtml', 'libTreeViewer', 'libpng', 'libFITSIO'] #libFITSIO mac only?
	lal_lib = ['liblal', 'liblalsupport', 'liblalframe', 'liblalmetaio', 'liblalsimulation', 'liblalinspiral', 'liblalburst']
	healpix_lib = ['libhealpix_cxx', 'libgomp']
	wat_lib = ['wavelet']
	ebbh = ['eBBH']# TODO

	libs = root_lib
	if config.getboolean('LIB','_USE_LAL'): libs += lal_lib
	if config.getboolean('LIB','_USE_HEALPIX'): libs += healpix_lib
	libs += wat_lib
	if config.getboolean('LIB','_USE_EBBH'): libs += ebbh

	for lib in libs:
		# TODO: error process and version check (healpix >= 3.00)
		# use user path if can't find lal healpix etc. in default
		gSystem.Load(lib)

	################
	# Loading Macros
	################
	wat_macros = ["Histogram.C", "AddPulse.C", "readAscii.C", "readtxt.C"]

	for macro in wat_macros:
		gROOT.LoadMacro(f"{config['CWB']['CWB_SOURCE']}/wat/macro/{macro}")

	# Loading tools
	tools_lib = ['STFT', 'gwat', 'Toolbox', 'History', 'Bicoherence', 'Filter', 'frame', 'cwb', 'wavegraph']

	for lib in tools_lib:
		# TODO: error process
		gSystem.Load(f"{config['CWB']['CWB_INSTALL']}/lib/{lib}")

	################
	# declare ACLiC includes environment 
	################
	for inc in include_path:
		gSystem.AddIncludePath(f"-I\"{inc}\"")

	################
	# declare ACLiCFlag options 
	################
	flag = "-D_USE_ROOT -fPIC -Wno-deprecated -mavx -Wall -Wno-unknown-pragmas -fexceptions -O2 -D__STDC_CONSTANT_MACROS"
	if config.getboolean('LIB','_USE_HEALPIX'): flag += " -D_USE_HEALPIX"
	if config.getboolean('LIB','_USE_LAL'): flag += " -D_USE_LAL"
	if config.getboolean('LIB','_USE_EBBH'): flag += " -D_USE_EBBH"
	if config.getboolean('LIB','_USE_ROOT6'): flag += " -D_USE_ROOT6"
	if platform.system() == "Darwin": flag += " -fno-common -dynamiclib -undefined dynamic_lookup"
	flag += " -fopenmp"
	if config.getboolean('LIB','_USE_ICC'): flag += " -diag-disable=2196"


	################
	# additional setup
	################
	gSystem.SetFlagsOpt(flag)

	# set the offset for TimeDisplay, the seconds declared in xaxis
	# are refered to "1980-01-06 00:00:00 UTC Sun" -> GPS = 0
	gStyle.SetTimeOffset(315964790)

	gStyle.SetPalette(1,0)
	gStyle.SetNumberContours(256)

	gROOT.ForceStyle(0)


def load_envs(config):
	envs = {
		'LALINSPINJ_EXEC': 'lalapps_inspinj'
	}
	for key in config['CWB'].keys():
		envs[key] = config['CWB'][key]

	for key in config['TOOLS'].keys():
		envs[key] = config['TOOLS'][key]

	for key in config['DUMB'].keys():
		envs[key] = config['DUMB'][key]

	for key in config['DERIVED'].keys():
		envs[key] = config['DERIVED'][key]
		
	for key in config['MACROS'].keys():
		envs[key] = config['MACROS'][key]

	for key in config['PROJECT'].keys():
		envs[key] = config['PROJECT'][key]

	# envs['CWB_PARMS_FILES'] = f"{envs['CWB_ROOTLOGON_FILE']} {envs['CWB_PARAMETERS_FILE']} {envs['CWB_UPARAMETERS_FILE']} {envs['CWB_EPARAMETERS_FILE']}"

	envs['CWB_PARMS_FILES'] = envs['CWB_PARAMETERS_FILE']

	for key in envs.keys():
		os.environ[key] = envs[key]

	return envs


def initialize_parameters(gROOT, envs):
	gROOT.LoadMacro(envs['CWB_PARMS_FILES'])


def root_setup(config):
	# set LD search path before loading ROOT
	os.environ['LD_LIBRARY_PATH'] = f"{config['CWB']['CWB_INSTALL']}/lib" # TODO: add previous LD path

	# load ROOT
	from ROOT import gROOT, gSystem, gStyle
	import ROOT

	# load ROOT libraries and macros
	ROOT_logon(gROOT, gSystem, gStyle, config)

	# load environment variables
	envs = load_envs(config)

	# declare types for c user parameters
	initialize_parameters(gROOT, envs)

	return ROOT, gROOT


