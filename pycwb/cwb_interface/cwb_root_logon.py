import platform
import os
from pycwb.config import CWBConfig
import pycwb
import logging

logger = logging.getLogger(__name__)


def ROOT_logon(gROOT, gSystem, gStyle, config: CWBConfig):
    ################
    # set include paths for compiling
    ################
    logger.info("Setting ROOT include paths")
    include_path = [f"{config.cwb_install}/include"]
    if config.use_ebbh: include_path += [f"{config.home_cvode}/include"]

    for inc in include_path:
        # TODO: error process
        gROOT.ProcessLine(f".include {inc}")

    ################
    # load cwb libraries
    ################
    logger.info("Loading CWB libraries")

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
    logger.info("Loading CWB macros")
    wat_macros = ["Histogram.C", "AddPulse.C", "readAscii.C", "readtxt.C"]

    for macro in wat_macros:
        logger.debug(f"Loading macro {macro}")
        gROOT.LoadMacro(f"{config.cwb_source}/wat/macro/{macro}")

    # Loading tools
    logger.info("Loading CWB tools")
    tools_lib = ['STFT', 'gwat', 'Toolbox', 'History', 'Bicoherence', 'Filter', 'frame', 'cwb', 'wavegraph']

    for lib in tools_lib:
        logger.debug(f"Loading tool {lib}")
        # TODO: error process
        # gSystem.Load(f"{config.cwb_install}/lib/{lib}")
        gSystem.Load(lib)
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
    logger.info("ROOT logon finished")


def cwb_root_logon(config: CWBConfig):
    # set LD search path before loading ROOT
    os.environ['LD_LIBRARY_PATH'] = f"{config.cwb_install}/lib"  # TODO: add previous LD path

    # load ROOT
    from ROOT import gROOT, gSystem, gStyle
    import ROOT

    # load ROOT libraries and macros
    ROOT_logon(gROOT, gSystem, gStyle, config)

    # declare types for c user parameters
    logger.info("Declaring types for user parameters")
    gROOT.LoadMacro(config.cwb_parameters_file)

    return ROOT, gROOT
