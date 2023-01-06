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


def init():
    pass


def cwb_InitNetwork():
    pass


def cwb_PrintAnalysis(ROOT, stageInfos):
    ROOT.CWB.PrintAnalysis(stageInfos)


def cwb_InitHistory(ROOT):
    ROOT.CWB.InitHistory()


def cwb_ReadData(ROOT, mdcShift, ifactor):
    #  Read Noise & MDC data from frame file or "On The Fly" from plugin
    #
    #  Loop over detectors
    #  - Read noise from frames or "On The Fly" from plugin
    #    - Resampling data
    #  - Read injections from frames or "On The Fly" from plugin (config::simulation>0)
    #    - Resampling data
    #  - if(simulation==2) MDC are rescaled (detector::setsnr) with a fixed
    #                      network SNR according to the config::factors
    #  - Store noise & MDC to job file
    #

    ROOT.cwb2G.ReadData(mdcShift, ifactor)


def cwb_DataConditioning(ROOT, ifile, jname, ifactor, fname=None):
    #  Apply regression to remove lines & whiten data
    #
    #  Loop over detectors
    #  - read ifo strain from job file
    #  - read MDC data from temporary job file (config::simulation>0)
    #  - if(config::simulation==1) MDC are rescaled according to the config::factors
    #  - Add MDC to noise
    #  - Apply regression to remove lines
    #  - Use detector::white to estimate noise (detector::nRMS)
    #  - Use the estimated noise to whiten data (WSeries<double>::white)
    #  - Store injected waveforms (SaveWaveforms)
    #  - Store whitened data (detector::HoT) to job file (jfile)
    #  - Store estimated noise to job file (detector::nRMS)
    #

    ROOT.ifile = ifile
    ROOT.jname = jname

    if fname:
        ROOT.cwb2G.DataConditioning(fname, ifactor)
    else:
        ROOT.cwb2G.DataConditioning(ifactor)

    # TODO: creat a new jname
    return jname


def cwb_Coherence(ROOT, ifile, jname, ifactor):
    #  Select the significant pixels
    #
    #  Loop over resolution levels (nRES)
    #  - Loop over detectors (cwb::nIFO)
    #    - Compute the maximum energy of TF pixels (WSeries<double>::maxEnergy)
    #  - Set pixel energy selection threshold (network::THRESHOLD)
    #  - Loop over time lags (network::nLag)
    #    - Select the significant pixels (network::getNetworkPixels)
    #    - Single resolution clustering (network::cluster)
    #    - Store selected pixels to job file (netcluster::write)
    #

    ROOT.ifile = ifile
    ROOT.jname = jname
    ROOT.cwb2G.Coherence(ifactor)


def cwb_SuperCluster(ROOT, ifile, jname, ifactor):
    # Multi resolution clustering & Rejection of the sub-threshold clusters
    # Loop over time lags
    # - Read clusters from job file (netcluster::read)
    # - Multi resolution clustering (netcluster::supercluster)
    # - Compute for each pixel the time delay amplitudes (netcluster::loadTDampSSE)
    # - Rejection of the sub-threshold clusters (network::subNetCut)
    # - Defragment clusters (netcluster::defragment)
    # - Store superclusters to job file (netcluster::write)
    # Build & Write to job file the sparse TF maps (WriteSparseTFmap)
    #

    ROOT.ifile = ifile
    ROOT.jname = jname
    ROOT.cwb2G.SuperCluster(ifactor)
    pass


def cwb_Likelihood():
    pass
