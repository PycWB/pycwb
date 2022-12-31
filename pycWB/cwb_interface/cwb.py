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