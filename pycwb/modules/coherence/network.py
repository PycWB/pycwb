import ROOT
import logging
from pycwb.config import Config
from pycwb.constants import WDM_BETAORDER, WDM_PRECISION
import numpy as np

logger = logging.getLogger(__name__)


def create_network(run_id, config: Config,
                   strain_list: list):
    net = ROOT.network()
    load_MRA(config, net)
    wdm_list = create_wdm(config, net)
    check_layers_with_MRAcatalog(config, net)
    # check_lagStep(config)
    net = init_network(config, net, strain_list, run_id)
    lag_buffer, lag_mode = get_lag_buffer(config)
    net = set_liv_time(config, net, lag_buffer, lag_mode)
    return net, wdm_list


def load_MRA(config: Config, net: ROOT.network):
    logger.info("Loading MRA")
    net.setMRAcatalog(config.MRAcatalog)


def create_wdm(config: Config, net: ROOT.network):
    beta_order = WDM_BETAORDER  # beta function order for Meyer
    precision = WDM_PRECISION  # wavelet precision

    if net.wdmMRA.tag != 0:
        beta_order = net.wdmMRA.BetaOrder
        precision = net.wdmMRA.precision

    wdm_list = []
    for i in range(config.l_low, config.l_high + 1):
        level = config.l_high + config.l_low - i
        layers = 2 ** level if level > 0 else 0
        wdm = ROOT.WDM(np.double)(layers, layers, beta_order, precision)
        wdmFLen = wdm.m_H / config.rateANA

        if wdmFLen > config.segEdge + 0.001:
            logger.error("Filter length must be <= segEdge !!!")
            logger.error("filter length : %s sec", wdmFLen)
            logger.error("cwb   scratch : %s sec", config.segEdge)
            raise ValueError("Filter length must be <= segEdge !!!")
        else:
            logger.info("Filter length = %s (sec)", wdmFLen)

        # check if the length for time delay amplitudes is less than cwb scratch length
        # the factor 1.5 is used to avoid to use pixels on the border which could be distorted
        rate = config.rateANA >> level

        if config.segEdge < int(1.5 * (config.TDSize / rate) + 0.5):
            logger.error("segEdge must be > 1.5x the length for time delay amplitudes!!!")
            logger.error("TD length : %s sec", config.TDSize / rate)
            logger.error("segEdge   : %s sec", config.segEdge)
            logger.error("Select segEdge > %s", int(1.5 * (config.TDSize / rate) + 0.5))
            raise ValueError("segEdge must be > 1.5x the length for time delay amplitudes!!!")

        wdm_list.append(wdm)
        net.add(wdm)

    return wdm_list


def check_layers_with_MRAcatalog(config: Config, net: ROOT.network):
    """
    check if analysis layers are contained in the MRAcatalog

    level : is the decomposition level
    layers : are the number of layers along the frequency axis rateANA/(rateANA>>level)
    :param config:
    :param net:
    :return:
    """
    check_layers = 0
    for i in range(config.l_low, config.l_high + 1):
        level = config.l_high + config.l_low - i
        layers = 2 ** level if level > 0 else 0
        for j in range(net.wdmMRA.nRes):
            if layers == net.wdmMRA.layers[j]:
                check_layers += 1

    if check_layers != config.nRES:
        logger.error("analysis layers do not match the MRA catalog")
        logger.error("analysis layers : ")
        for level in range(config.l_high, config.l_low - 1, -1):
            layers = 1 << level if level > 0 else 0
            logger.error("level : %s layers : %s", level, layers)

        logger.error("MRA catalog layers : ")
        for i in range(net.wdmMRA.nRes):
            logger.error("layers : %s", net.wdmMRA.layers[i])
        raise ValueError("analysis layers do not match the MRA catalog")


def init_network(config: Config, net: ROOT.network,
                 strain_list: list,
                 run_id):
    logger.info("Initializing network")

    for i, ifo in enumerate(config.ifo):
        logger.info("Adding ifo %s", ifo)
        det = ROOT.detector(ifo)

        det.rate = config.inRate if not config.fResample else config.fResample
        det.HoT = strain_list[i]['TFmap']
        det.TFmap = strain_list[i]['TFmap']
        det.nRMS = strain_list[i]['nRMS']
        net.add(det)

    # set network skymaps
    logger.info("Setting skymaps")
    net.setSkyMaps(int(config.healpix))
    net.setAntenna()

    # restore network parameters
    logger.info("Restoring network parameters")
    net.constraint(config.delta, config.gamma)
    net.setDelay(config.refIFO)
    net.Edge = config.segEdge
    net.netCC = config.netCC
    net.netRHO = config.netRHO
    net.EFEC = config.EFEC
    net.precision = config.precision
    net.nSky = config.nSky
    net.setRunID(run_id)
    net.setAcore(config.Acore)
    net.optim = config.optim
    net.pattern = config.pattern

    # set sky mask
    logger.info("Setting sky mask")
    tmp_cfg = ROOT.CWB.config()
    tmp_cfg.healpix = config.healpix
    tmp_cfg.Theta1 = config.Theta1
    tmp_cfg.Theta2 = config.Theta2
    tmp_cfg.Phi1 = config.Phi1
    tmp_cfg.Phi2 = config.Phi2

    if len(config.skyMaskFile) > 0:
        ROOT.SetSkyMask(net, tmp_cfg, config.skyMaskFile, 'e')

    if len(config.skyMaskCCFile) > 0:
        ROOT.SetSkyMask(net, tmp_cfg, config.skyMaskCCFile, 'c')

    return net


def set_liv_time(config: Config, net: ROOT.network, lagBuffer: str, lagMode: str):
    if lagBuffer:
        lags = net.setTimeShifts(config.lagSize, config.lagStep, config.lagOff, config.lagMax,
                                 lagBuffer,
                                 lagMode,
                                 config.lagSite)
    else:
        lags = net.setTimeShifts(config.lagSize, config.lagStep, config.lagOff, config.lagMax)
    logger.info("lag step: %s", config.lagStep)
    logger.info("number of time lags: %s", lags)

    return net


def get_lag_buffer(config: Config):
    if config.lagMode == "r":
        with open(config.lagFile, "r") as f:
            lagBuffer = f.read()
        lagMode = 's'
    else:
        lagBuffer = config.lagFile
        lagMode = 'w'

    return lagBuffer, lagMode
