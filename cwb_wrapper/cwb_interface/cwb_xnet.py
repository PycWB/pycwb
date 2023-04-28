import os
from pycwb.config import CWBConfig
import logging
from pycwb.cwb_interface import CWB
from ROOT import gROOT
import ROOT

logger = logging.getLogger(__name__)


def cwb_xnet(config: CWBConfig, run_id: int, j_stage: int,
             batch=False, eced=False, inet_option=None, file_name=""):
    logger.info("Starting cwb_xnet")

    os.environ['CWB_JOBID'] = str(run_id)
    logger.info(f"Setting CWB_JOBID to {run_id}")

    # TODO: parameters check
    # TODO: handle inet_option in python
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
        logger.info(f"inet_option: {inet_option}")

    get_job_from_gps()

    if batch:
        # get process pid
        pid = os.getpid()
        # TODO: implement
        pass

    if config.cwb_analysis == "1G":
        CWB = ROOT.cwb1G(file_name, "", j_stage)
    elif config.cwb_analysis == "2G":
        CWB = ROOT.cwb2G(file_name, "", j_stage)
    elif config.cwb_analysis == "XP":
        CWB = ROOT.cwbXP(file_name, "", j_stage)
    else:
        raise ValueError("Invalid cwb_analysis value in config file")

    cfg = CWB.GetConfig()
    if not batch:
        if eced:
            cfg.Import(f"{config.cwb_macros}/cwb_eced.C")
        logger.info("Initializing CWB inet")
        cfg.Import(f"{config.cwb_macros}/cwb_inet.C")

    logger.info("Setting up job stage")
    CWB.SetupStage(j_stage)

    logger.info("Running CWB")
    CWB.run(run_id)

    logger.info("Finished cwb_xnet2G")


def cwb_xnet_new(config: CWBConfig, run_id: int, j_stage: int,
             batch=False, eced=False, inet_option=None, file_name=""):
    logger.info("Starting cwb_xnet")

    os.environ['CWB_JOBID'] = str(run_id)
    logger.info(f"Setting CWB_JOBID to {run_id}")

    # TODO: parameters check
    # TODO: handle inet_option in python
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
        logger.info(f"inet_option: {inet_option}")

    get_job_from_gps()

    if batch:
        # get process pid
        pid = os.getpid()
        # TODO: implement
        pass

    if config.cwb_analysis == "2G":
        cwb = CWB(ROOT, gROOT, file_name, j_stage, pipeline=ROOT.cwb2G)
    elif config.cwb_analysis == "XP":
        cwb = CWB(ROOT, gROOT, file_name, j_stage, pipeline=ROOT.cwbXP)
    else:
        raise ValueError("Invalid cwb_analysis value in config file")

    cfg = cwb.GetConfig()
    if not batch:
        if eced:
            cfg.Import(f"{config.cwb_macros}/cwb_eced.C")
        logger.info("Initializing CWB inet")
        cfg.Import(f"{config.cwb_macros}/cwb_inet.C")

    logger.info("Setting up job stage")
    cwb.SetupStage(j_stage)

    logger.info("Running CWB")
    cwb.run(run_id)

    logger.info("Finished cwb_xnet2G")

    return cwb

def get_job_from_gps():
    # TODO: implement
    pass
