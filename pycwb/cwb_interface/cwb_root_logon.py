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
from pycwb.config import CWBConfig
import pycwb
import configparser


def ROOT_logon(gROOT, gSystem, gStyle, config: CWBConfig):
    ################
    # set include paths for compiling
    ################
    include_path = [f"{config.cwb_install}/include"]
    if config.use_ebbh: include_path += [f"{config.home_cvode}/include"]

    for inc in include_path:
        # TODO: error process
        gROOT.ProcessLine(f".include {inc}")

    ################
    # load cwb libraries
    ################
    root_lib = ['libPhysics', 'libFFTW', 'libHtml', 'libTreeViewer', 'libpng', 'libFITSIO']  # libFITSIO mac only?
    lal_lib = ['liblal', 'liblalsupport', 'liblalframe', 'liblalmetaio', 'liblalsimulation', 'liblalinspiral',
               'liblalburst']
    healpix_lib = ['libhealpix_cxx', 'libgomp']
    wat_lib = ['wavelet']
    ebbh = ['eBBH']  # TODO

    libs = root_lib
    if config.use_lal: libs += lal_lib
    if config.use_healpix: libs += healpix_lib
    libs += wat_lib
    if config.use_ebbh: libs += ebbh

    for lib in libs:
        # TODO: error process and version check (healpix >= 3.00)
        # use user path if can't find lal healpix etc. in default
        gSystem.Load(lib)

    ################
    # Loading Macros
    ################
    wat_macros = ["Histogram.C", "AddPulse.C", "readAscii.C", "readtxt.C"]

    for macro in wat_macros:
        gROOT.LoadMacro(f"{config.cwb_source}/wat/macro/{macro}")

    # Loading tools
    tools_lib = ['STFT', 'gwat', 'Toolbox', 'History', 'Bicoherence', 'Filter', 'frame', 'cwb', 'wavegraph']

    for lib in tools_lib:
        # TODO: error process
        gSystem.Load(f"{config.cwb_install}/lib/{lib}")

    ################
    # declare ACLiC includes environment
    ################
    for inc in include_path:
        gSystem.AddIncludePath(f"-I\"{inc}\"")

    ################
    # declare ACLiCFlag options
    ################
    flag = "-D_USE_ROOT -fPIC -Wno-deprecated -mavx -Wall " \
           "-Wno-unknown-pragmas -fexceptions -O2 -D__STDC_CONSTANT_MACROS"
    if config.use_healpix: flag += " -D_USE_HEALPIX"
    if config.use_lal: flag += " -D_USE_LAL"
    if config.use_ebbh: flag += " -D_USE_EBBH"
    if config.use_root6: flag += " -D_USE_ROOT6"
    if platform.system() == "Darwin": flag += " -fno-common -dynamiclib -undefined dynamic_lookup"
    flag += " -fopenmp"
    if config.use_icc: flag += " -diag-disable=2196"

    ################
    # additional setup
    ################
    gSystem.SetFlagsOpt(flag)

    # set the offset for TimeDisplay, the seconds declared in xaxis
    # are refered to "1980-01-06 00:00:00 UTC Sun" -> GPS = 0
    gStyle.SetTimeOffset(315964790)

    gStyle.SetPalette(1, 0)
    gStyle.SetNumberContours(256)

    gROOT.ForceStyle(0)


def load_envs(config, config_object):
    pycwb_path = os.path.dirname(os.path.abspath(pycwb.__file__))

    envs = {
        'LALINSPINJ_EXEC': 'lalapps_inspinj'
    }

    if config_object.use_lal:
        envs['_USE_LAL'] = "yes"
    if config_object.use_healpix:
        envs['_USE_HEALPIX'] = "yes"
    if config_object.use_root6:
        envs['_USE_ROOT6'] = "yes"
    if config_object.use_ebbh:
        envs['_USE_EBBH'] = "yes"

    for section in ['CWB', 'TOOLS', 'DERIVED', 'MACROS', 'PROJECT']:
        for key in config[section].keys():
            envs[key] = config[section][key]

    # envs['CWB_PARMS_FILES'] = f"{envs['CWB_ROOTLOGON_FILE']} {envs['CWB_PARAMETERS_FILE']} {envs['CWB_UPARAMETERS_FILE']} {envs['CWB_EPARAMETERS_FILE']}"
    envs['HOME_WAT'] = config_object.cwb_source
    envs['HOME_FRDISPLAY'] = f"{config_object.cwb_install}/bin"

    envs['CWB_ROOTLOGON_FILE'] = f"{pycwb_path}/shared/dumb.c"
    envs['CWB_PARMS_FILES'] = config_object.cwb_parameters_file

    for key in envs.keys():
        os.environ[key] = envs[key]

    return envs


def cwb_root_logon(config: CWBConfig):
    # set LD search path before loading ROOT
    os.environ['LD_LIBRARY_PATH'] = f"{config.cwb_install}/lib"  # TODO: add previous LD path

    # load ROOT
    from ROOT import gROOT, gSystem, gStyle
    import ROOT

    # load ROOT libraries and macros
    ROOT_logon(gROOT, gSystem, gStyle, config)

    # load environment variables
    load_envs(config.config, config)

    # declare types for c user parameters
    gROOT.LoadMacro(config.cwb_parameters_file)

    return ROOT, gROOT
