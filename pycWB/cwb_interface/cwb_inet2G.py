import os


def cwb_inet2G_old(ROOT, gROOT, config, run_id, j_stage, f_name="", u_name="", eced=False, inet_option=None):
    gROOT.LoadMacro(f"{config['CWB']['CWB_INSTALL']}/etc/cwb/macros/cwb_inet2G.C")

    # TODO: parameters check

    os.environ['CWB_JOBID'] = str(run_id)
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
    ROOT.cwb_inet2G(run_id, f_name, j_stage, u_name, eced)


def cwb_inet2G(ROOT, gROOT, config, run_id, j_stage, inet_option=None):
    os.environ['CWB_JOBID'] = str(run_id)
    if inet_option:
        os.environ['CWB_INET_OPTIONS'] = inet_option
    # TODO: figure out ecem
    icfg = ROOT.CWB.config()
    icfg.Import()
    icfg.analysis = "2G"
    icfg.Export()

    CWB = ROOT.cwb2G(icfg, j_stage)

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

    cfg.Import(f"{config['CWB']['CWB_INSTALL']}/etc/cwb/macros/cwb_inet.C")

    CWB.SetupStage(j_stage)
    CWB.run(run_id)