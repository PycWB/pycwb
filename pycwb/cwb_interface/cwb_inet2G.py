import os
from pycwb.config import CWBConfig
import logging
from ROOT import gROOT
import ROOT

logger = logging.getLogger(__name__)

def cwb_inet2G_old(config, run_id, j_stage, f_name="", u_name="", eced=False, inet_option=None):
    gROOT.LoadMacro(f"{config['CWB']['CWB_INSTALL']}/etc/cwb/macros/cwb_inet2G.C")

    # TODO: parameters check

    os.environ['CWB_JOBID'] = str(run_id)
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
    ROOT.cwb_inet2G(run_id, f_name, j_stage, u_name, eced)


def cwb_inet2G(config: CWBConfig, run_id, j_stage, inet_option=None, file_name=""):
    logger.info("Starting cwb_inet2G")

    os.environ['CWB_JOBID'] = str(run_id)
    logger.info(f"Setting CWB_JOBID to {run_id}")

    # TODO: parameters check
    # TODO: handle inet_option in python
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
        logger.info(f"inet_option: {inet_option}")

    # TODO: figure out ecem

    # initialize CWB config object
    logger.info("Initializing CWB config object")
    icfg = ROOT.CWB.config()
    if file_name:
        icfg.Import(file_name)
        logger.info(f"Imported config file {file_name}")
    else:
        icfg.Import()
        logger.info("Imported config from global config")
    icfg.Export()

    # initialize CWB job object
    logger.info("Initializing CWB job object")
    CWB = ROOT.cwb2G(icfg, j_stage)

    # TODO: why is this needed? why can't we just use icfg?
    cfg = CWB.GetConfig()

    if CWB.GetLagBuffer().GetSize():
        # read lags from job file (used in multistage analysis)
        lagBuffer = CWB.GetLagBuffer()
        lagFile = lagBuffer.GetArray()
        lagMode = CWB.GetLagMode()

        # FIXME: segmentation error with updating lagFile
        cfg.lagFile = lagFile
        cfg.lagMode = lagMode
        cfg.Export()

    # initialize CWB inet
    logger.info("Initializing CWB inet")
    cfg.Import(f"{config.cwb_install}/etc/cwb/macros/cwb_inet.C")

    # TODO: why setup stage again?
    # setup jobfOptions in global variables
    logger.info("Setting up job stage")
    CWB.SetupStage(j_stage)

    logger.info("Running CWB")
    CWB.run(run_id)
    
    logger.info("Finished cwb_inet2G")
    return 0
