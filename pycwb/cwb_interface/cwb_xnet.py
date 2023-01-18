import os
from pycwb.config import CWBConfig
import logging
logger = logging.getLogger(__name__)


def cwb_xnet(ROOT, gROOT, config: CWBConfig, run_id, j_stage, inet_option=None, file_name=""):
    if config.cwb_analysis == "1G":
        CWB = ROOT.cwb1G()
    elif config.cwb_analysis == "2G":
        CWB = ROOT.cwb2G("", "", j_stage)


def get_job_from_gps():
    pass