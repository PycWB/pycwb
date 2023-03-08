import logging, os
import time
from pathlib import Path
from pycwb.cwb_interface import update_global_var

logger = logging.getLogger(__name__)


class CWB:
    def __init__(self, ROOT, gROOT, file_name, j_stage, pipeline):
        self.ROOT = ROOT
        self.gROOT = gROOT
        self.cwb = pipeline(file_name, "", j_stage)
        pass

    def run(self, run_id):
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
        job_start = time.time()

        # ROOT.watchJob.Start()  # start job benchmark
        # ROOT.watchStage.Start()  # start stage benchmark

        bplugin = self.cwb.cfg.plugin.GetName() != ""
        data_rate = 0.0
        pid = os.getpid()

        # Setup variables
        self.cwb.nIFO = self.cwb.cfg.nIFO

        self.ROOT.gInterpreter.Declare("""
        void update_ifo(cwb2G cwb) { 
            for(int n=0;n<cwb.nIFO;n++) {
                if(strlen(cwb.cfg.ifo[n])>0) strcpy(cwb.ifo[n],cwb.cfg.ifo[n]);
                else strcpy(cwb.ifo[n],cwb.cfg.detParms[n].name);
            }
        }
        """)
        self.ROOT.update_ifo(self.cwb)

        if self.cwb.runID == 0:
            self.cwb.runID = int(run_id)

        logger.info("Job INFO: \n"
                    "Job ID           : %d\n"
                    "Ouput            : %s\n"
                    "Label            : %s\n"
                    "nodedir          : %s\n"
                    "Pid              : %d\n"
                    "working directory: %s" %
                    (self.cwb.runID, self.cwb.cfg.output_dir, self.cwb.cfg.data_label, self.cwb.cfg.nodedir,
                     pid, self.cwb.cfg.work_dir))

        # create log, nodedir & output directories
        # this step is necessary in multi stage analysis when pipeline
        # start from an intermediate stage from a non structured working dir
        Path(self.cwb.cfg.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cwb.cfg.nodedir).mkdir(parents=True, exist_ok=True)
        Path(self.cwb.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("directories created")

        # export to CINT istage,jstage (used by plugins)

        # self.ROOT.gISTAGE = self.cwb.istage
        # self.ROOT.gJSTAGE = self.cwb.jstage
        # self.ROOT.gIFACTOR = -1
        logger.debug("gISTAGE = %d, gJSTAGE = %d" % (self.cwb.istage, self.cwb.jstage))
        update_global_var(self.gROOT, 'int', 'gISTAGE', f"gISTAGE = (CWB_STAGE)%d;" % self.cwb.istage)
        update_global_var(self.gROOT, 'int', 'gJSTAGE', f"gJSTAGE = (CWB_STAGE)%d;" % self.cwb.jstage)
        update_global_var(self.gROOT, 'int', 'gIFACTOR', "gIFACTOR = -1;")

        # Load plugin
        if bplugin:
            self.cwb.LoadPlugin(self.cwb.cfg.plugin, self.cwb.cfg.configPlugin)
            print("type:", type(self.cwb.cfg))
            self.ROOT.CWB_Plugin(self.ROOT.nullptr, self.cwb.cfg, self.cwb.NET,
                                 self.ROOT.nullptr, "", self.ROOT.CWB_PLUGIN_CONFIG)
            self.cwb.SetupStage(self.cwb.jstage)

        self.InitNetwork()
        self.PrintAnalysis()
        self.InitHistory()
        mdc_shift = True
        self.InitJob()

        if not self.cwb.cfg.simulation:
            self.cwb.cfg.nfactor = 1
        self.cwb.ioffset = int(self.cwb.cfg.factors[0]) if self.cwb.cfg.simulation == 4 else 0

        nfactor = 1 if self.cwb.cfg.simulation == 5 else self.cwb.cfg.nfactor

        self.cwb.jname = f"{self.cwb.cfg.nodedir}/job_{int(self.cwb.Tb)}_{self.cwb.cfg.data_label}_{self.cwb.runID}_{pid}.root"
        logger.info("job file: %s" % self.cwb.jname)
        # self.ReadData()
        # self.DataConditioning()
        # self.Coherence()
        # self.SuperCluster()
        # self.Likelihood()

        # JOB_END
        update_global_var(self.gROOT, 'int', 'gIFACTOR', "gIFACTOR = -1;")

        # TODO

        job_data_size_sec = self.cwb.dT
        job_end = time.time()
        job_speed_factor = job_data_size_sec / (job_end - job_start)
        # print speed factor format to 2 digits
        logger.info("Job Speed Factor - {:.2f}X".format(job_speed_factor))

    def GetConfig(self):
        return self.cwb.GetConfig()

    def SetupStage(self, j_stage):
        return self.cwb.SetupStage(j_stage)

    def load_job(self, job_id):
        pass

    def save_job(self, job_id):
        pass

    def InitNetwork(self, file_name=None):
        if file_name:
            self.cwb.InitNetwork(file_name)
        else:
            self.cwb.InitNetwork()

    def InitHistory(self):
        self.cwb.InitHistory()

    def InitJob(self, file_name=None):
        if file_name:
            self.cwb.InitJob(file_name)
        else:
            self.cwb.InitJob()

    def ReadData(self, mdcShift, ifactor):
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

        # Load variables

        # run
        self.cwb.ReadData(mdcShift, ifactor)

        # save variables

    def DataConditioning(self, ifile, jname, ifactor, fname=None):
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

        self.cwb.ifile = ifile
        self.cwb.jname = jname

        if fname:
            self.cwb.DataConditioning(fname, ifactor)
        else:
            self.cwb.DataConditioning(ifactor)

        # TODO: creat a new jname
        return jname

    def Coherence(self, ifile, jname, ifactor):
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

        self.cwb.ifile = ifile
        self.cwb.jname = jname
        self.cwb.Coherence(ifactor)

    def SuperCluster(self, ifile, jname, ifactor):
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

        self.cwb.ifile = ifile
        self.cwb.jname = jname
        self.cwb.SuperCluster(ifactor)
        pass

    def Likelihood(self):
        pass

    def PrintAnalysis(self, stageInfos=False):
        self.cwb.PrintAnalysis(stageInfos)

    def SaveJob(self):
        pass
