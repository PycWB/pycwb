# cwb.py is a module to convert cwb functionality to python functions

import os
from pycWB.config.constants import CWB_STAGE


def cwb_inet2G(ROOT, gROOT, config, run_id, f_name, j_stage, u_name="", eced=False, inet_option=None):
    gROOT.LoadMacro(f"{config['CWB']['CWB_INSTALL']}/etc/cwb/macros/cwb_inet2G.C")

    # TODO: parameters check

    os.environ['CWB_JOBID'] = str(run_id)
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
    ROOT.cwb_inet2G(run_id, f_name, CWB_STAGE[j_stage], u_name, eced)
