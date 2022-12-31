import os


def cwb_inet2G_old(ROOT, gROOT, config, run_id, f_name, j_stage, u_name="", eced=False, inet_option=None):
    gROOT.LoadMacro(f"{config['CWB']['CWB_INSTALL']}/etc/cwb/macros/cwb_inet2G.C")

    # TODO: parameters check

    os.environ['CWB_JOBID'] = str(run_id)
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
    ROOT.cwb_inet2G(run_id, f_name, j_stage, u_name, eced)


def cwb_inet2G(ROOT, run_id, j_stage, inet_option=None):
    os.environ['CWB_JOBID'] = str(run_id)
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
    # TODO: figure out ecem
    icfg = ROOT.CWB.config()
    icfg.Import()

    CWB = ROOT.cwb2G(icfg, j_stage)

    # TODO: read lags from job file (used in multistage analysis)

    CWB.SetupStage(j_stage)
    CWB.run(run_id)
    pass


def run(ROOT, run_id):
    # The method used to start the analysis
    #
    # runID : is the job ID number, this is used in InitJob method to identify the
    #         the time range to be analyzed
    #
    # These are the main actions performed by this method
    #
    # - LoadPlugin
    # - InitNetwork
    # - InitHistory
    # - InitJob
    # - Loop over Factors
    #   - ReadData
    #   - DataConditioning
    #   - Coherence
    #   - SuperCluster
    #   - Likelihood
    #   - Save Recontructed Parameters
    # - Save Job File (only for multi stage analysis)

    lags = 0
    factor = 1.0  # strain factor
    ioffset = 0  # ifactor offset
    ROOT.watchJob.Start()  # start job benchmark
    ROOT.watchStage.Start()  # start stage benchmark
