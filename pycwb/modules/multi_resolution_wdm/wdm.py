import logging

from pycwb.types.wdm import WDM

logger = logging.getLogger(__name__)


def create_wdm_set(config):
    """
    Create a list of WDM objects for all levels
    :param config: user configuration
    :type config: Config
    :return: list of WDM objects
    :rtype: list[WDM]
    """

    # explicitly list all parameters used from config
    rate_ANA, seg_edge, td_size, l_high, l_low  = config.rateANA, config.segEdge, config.TDSize, \
        config.l_high, config.l_low

    # create WDM
    wdm_list = []
    for i in range(l_low, l_high + 1):
        level = l_high + l_low - i
        wdm_list.append(
            create_wdm_for_level(config, level)
        )

    return wdm_list


def create_wdm_for_level(config, level):
    """
    Create a WDM object for a given level
    :param rate_ANA: analysis rate
    :type rate_ANA: int
    :param seg_edge: cwb scratch length
    :type seg_edge: float
    :param td_size: time delay size
    :type td_size: int
    :param level: level
    :type level: int
    :param beta_order: beta function order for Meyer
    :type beta_order: int
    :param precision: wavelet precision
    :type precision: int
    :return: WDM object
    :rtype: WDM
    """
    # explicitly list all parameters used from config
    rate_ANA, seg_edge, td_size, l_high, l_low  = config.rateANA, config.segEdge, config.TDSize, \
        config.l_high, config.l_low

    # get beta order and precision
    beta_order, precision = config.WDM_beta_order, config.WDM_precision

    layers = 2 ** level if level > 0 else 0
    wdm = WDM(layers, layers, beta_order, precision)
    wdmFLen = wdm.m_H / rate_ANA

    if wdmFLen > seg_edge + 0.001:
        logger.error("Filter length must be <= segEdge !!!")
        logger.error("filter length : %s sec", wdmFLen)
        logger.error("cwb   scratch : %s sec", seg_edge)
        raise ValueError("Filter length must be <= segEdge !!!")
    # else:
    #     logger.info("Filter length = %s (sec)", wdmFLen)

    # check if the length for time delay amplitudes is less than cwb scratch length
    # the factor 1.5 is used to avoid to use pixels on the border which could be distorted
    rate = rate_ANA >> level

    if seg_edge < int(1.5 * (td_size / rate) + 0.5):
        logger.error("segEdge must be > 1.5x the length for time delay amplitudes!!!")
        logger.error("TD length : %s sec", td_size / rate)
        logger.error("segEdge   : %s sec", seg_edge)
        logger.error("Select segEdge > %s", int(1.5 * (td_size / rate) + 0.5))
        raise ValueError("segEdge must be > 1.5x the length for time delay amplitudes!!!")

    return wdm
