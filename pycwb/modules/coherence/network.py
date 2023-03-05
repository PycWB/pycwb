import ROOT
import logging
from pycwb.config import Config
from pycwb.constants import WDM_BETAORDER, WDM_PRECISION
import numpy as np
import argparse, shlex

logger = logging.getLogger(__name__)


def create_network(run_id, config: Config,
                   tf_maps: list, nRMS_list: list):
    net = ROOT.network()

    # load MRA catalog
    load_MRA(config, net)

    # create WDM
    wdm_list = create_wdm(config, net)

    # check layers
    check_layers_with_MRAcatalog(config, net)

    # Note: check_lagStep(config) is moved to Config

    net = init_network(config, net, tf_maps, nRMS_list, run_id)
    lag_buffer, lag_mode = get_lag_buffer(config)
    net = set_liv_time(config, net, lag_buffer, lag_mode)
    return net, wdm_list


def load_MRA(config: Config, net: ROOT.network):
    logger.info("Loading catalog of WDM cross-talk coefficients")
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
                 tf_maps: list, nRMS_list: list,
                 run_id):
    logger.info("Initializing network")

    for i, ifo in enumerate(config.ifo):
        logger.info("Adding ifo %s", ifo)
        det = ROOT.detector(ifo)

        det.rate = config.inRate if not config.fResample else config.fResample
        det.HoT = tf_maps[i]
        det.TFmap = tf_maps[i]
        det.nRMS = nRMS_list[i]
        net.add(det)

    # set network skymaps
    update_sky_map(config, net)

    # restore network parameters
    logger.info("Restoring network parameters")
    net.constraint(config.delta, config.gamma)
    # net.setDelay(config.refIFO)
    net.Edge = config.segEdge
    net.netCC = config.netCC
    net.netRHO = config.netRHO
    net.EFEC = config.EFEC
    net.precision = config.precision
    net.nSky = config.nSky
    # net.eDisbalance = config.eDisbalance
    net.setRunID(run_id)
    net.setAcore(config.Acore)
    net.optim = config.optim
    net.pattern = config.pattern

    # set sky mask
    update_sky_mask(config, net)

    # mdc = new injection(nIFO);
    # netburst = new netevent(nIFO,cfg.Psave);

    return net


def restore_skymap(config: Config, net: ROOT.network, skyres):
    if skyres:
        if config.healpix:
            net.setSkyMaps(int(config.healpix))
        else:
            net.setSkyMaps(config.angle, config.Theta1, config.Theta2, config.Phi1, config.Phi2)
    net.setAntenna()
    net.setDelay(config.refIFO)
    if len(config.skyMaskFile) > 0:
        set_sky_mask(net, config, config.skyMaskFile, 'e')
    if len(config.skyMaskCCFile) > 0:
        set_sky_mask(net, config, config.skyMaskCCFile, 'c')


def update_sky_map(config: Config, net: ROOT.network, skyres: int = None):
    logger.info("Setting skymaps")
    if skyres:
        net.setSkyMaps(int(skyres))
    else:
        if config.healpix:
            net.setSkyMaps(int(config.healpix))
        else:
            net.setSkyMaps(config.angle, config.Theta1, config.Theta2, config.Phi1, config.Phi2)

    net.setAntenna()
    net.setDelay(config.refIFO)


def update_sky_mask(config: Config, net: ROOT.network, skyres: int = None):
    logger.info("Setting sky mask")

    if skyres:
        if len(config.skyMaskFile) > 0:
            set_sky_mask(net, config, config.skyMaskFile, 'e', skyres)

        if len(config.skyMaskCCFile) > 0:
            set_sky_mask(net, config, config.skyMaskCCFile, 'c', skyres)

    else:
        if len(config.skyMaskFile) > 0:
            set_sky_mask(net, config, config.skyMaskFile, 'e')

        if len(config.skyMaskCCFile) > 0:
            set_sky_mask(net, config, config.skyMaskCCFile, 'c')


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


def set_sky_mask(config: Config, net: ROOT.network, options: str, skycoord: str, skyres: float = None):
    """
    set earth/celestial sky mask is used to define what are the sky locations
    that are analyzed by default all sky is used
    :param config: Config object
    :param net: network object
    :param options: used to define the earth/celestial SkyMap
                    option A :
                    skycoord='e'
                    --theta THETA --phi PHI --radius RADIUS
                    define a circle centered in (THETA,PHI) and radius=RADIUS
                    THETA : [-90,90], PHI : [0,360], RADIUS : degrees
                    skycoord='c'
                    --theta DEC --phi RA --radius RADIUS
                    define a circle centered in (DEC,RA) and radius=RADIUS
                    DEC : [-90,90], RA : [0,360], RADIUS : degrees
                    option B:
                    file name
                    format : two columns ascii file -> [sky_index	value]
                    sky_index : is the sky grid index
                    value     : if !=0 the index sky location is used for the analysis
    :param skycoord: sky coordinates : 'e'=earth, 'c'=celestial
    :param skyres: sky resolution : def=-1 -> use the value defined in parameters (angle,healpix)
    :return:
    """
    if skycoord not in ['e', 'c']:
        raise ValueError("cwb::SetSkyMask - Error : wrong input sky coordinates "
                         " must be 'e'/'c' earth/celestial")

    if options:
        if "--" not in options:  # input parameter is the skyMask file
            if skyres >= 0:
                return 1
            ret = net.setSkyMask(options, skycoord)
            if ret == 0:
                raise ValueError("cwb::SetSkyMask - Error : skyMask file"
                                 " not exist or it has a wrong format"
                                 " format : two columns ascii file -> [sky_index        value]"
                                 " sky_index : is the sky grid index"
                                 " value     : if !=0 the index sky location is used for the analysis")
            if skycoord == 'e':
                net.setIndexMode(0)
            return 0

        # parse options with python for TB.getParameter(options, "--theta")

        parser = argparse.ArgumentParser(description='Example with long option names')
        parser.add_argument('--theta', default=-1000, type=float)
        parser.add_argument('--phi', default=-1000, type=float)
        parser.add_argument('--radius', default=-1000, type=float)
        args = parser.parse_args(shlex.split('--theta 1 --phi 2 --radius 3'))

        theta = args.theta
        phi = args.phi
        radius = args.radius

        if theta == -1000 or phi == -1000 or radius == -1000:  # input parameter are the skyMask params
            raise ValueError("cwb::SetSkyMask - Error : wrong input skyMask params"
                             "wrong input options : " + options +
                             "options must be : --theta THETA --phi PHI --radius RADIUS"
                             "options must be : --theta DEC --phi AR --radius RADIUS"
                             "theta must be in the range [-90,90]"
                             "phi must be in the range [0,360]"
                             "radius must be > 0")
        else:  # create & set SkyMask
            if skyres < 0:
                skyres = config.healpix if config.healpix else config.angle
            if config.healpix:
                SkyMask = ROOT.skymap(int(skyres))
            else:
                SkyMask = ROOT.skymap(skyres, config.Theta1, config.Theta2, config.Phi1, config.Phi2)
            make_sky_mask(SkyMask, theta, phi, radius)
            net.setSkyMask(SkyMask, skycoord)
            if skycoord == 'e':
                net.setIndexMode(0)
    return 0


def make_sky_mask(sky_mask: ROOT.skymap, theta: float, phi: float, radius: float):
    """
    make a sky mask
    is used to define what are the sky locations that are analyzed

    theta,phi,radius : used to define the Celestial SkyMap
                       define a circle centered in (theta,phi) and radius=radius
                       theta : [-90,90], phi : [0,360], radius : degrees

    SkyMask          : output sky celestial mask
                       inside the circle is filled with 1 otherwise with 0
    :param
    :return:
    """
    l = sky_mask.size()
    healpix = sky_mask.getOrder()
    # check input parameters
    if abs(theta) > 90 or (phi < 0 or phi > 360) or radius <= 0 or l <= 0:
        raise ValueError("cwb::MakeSkyMask : wrong input parameters !!! "
                         "if(fabs(theta)>90)   cout << theta << \" theta must be in the range [-90,90]\" << endl;"
                         "if(phi<0 || phi>360) cout << phi << \" phi must be in the range [0,360]\" << endl;"
                         "if(radius<=0)        cout << radius << \" radius must be > 0\" << endl;"
                         "if(L<=0)             cout << L << \" SkyMask size must be > 0\" << endl;"
                         "EXIT(1);")

    if not ROOT.gROOT.GetClass("Polar3DVector"):
        ROOT.gSystem.Load("libMathCore")

    # compute the minimun available radius
    # must be greater than the side of a pixel
    if healpix:
        npix = 12 * (int)(4. ** healpix)
        sphere_solid_angle = 4 * np.pi * np.pow(180. / np.pi, 2.)
        skyres = sphere_solid_angle / npix
        if radius < np.sqrt(skyres):
            radius = np.sqrt(skyres)
    else:
        if radius < sky_mask.sms:
            radius = sky_mask.sms
    ph, th = ROOT.GeographicToCwb(phi, theta)
    ov1 = ROOT.Polar3DVector(1, th * np.pi / 180, ph * np.pi / 180)
    nset = 0
    for l in range(l):
        phi = sky_mask.getPhi(l)
        theta = sky_mask.getTheta(l)
        ov2 = ROOT.Polar3DVector(1, theta * np.pi / 180, phi * np.pi / 180)
        dot = ov1.Dot(ov2)
        d_omega = 180 * np.arccos(dot) / np.pi
        if d_omega <= radius:
            sky_mask.set(l, 1)
            nset += 1
        else:
            sky_mask.set(l, 0)
