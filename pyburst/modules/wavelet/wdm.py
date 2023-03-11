import logging
from pyburst.types import WDM

logger = logging.getLogger(__name__)


def create_wdm_set(config, beta_order, precision):
    """
    Create a list of WDM objects for all levels
    :param config: user configuration
    :type config: Config
    :param beta_order: beta function order for Meyer
    :type beta_order: int
    :param precision: wavelet precision
    :type precision: int
    :return: list of WDM objects
    :rtype: list[WDM]
    """
    wdm_list = []
    for i in range(config.l_low, config.l_high + 1):
        level = config.l_high + config.l_low - i
        wdm_list.append(create_wdm_for_level(config, level, beta_order, precision))

    return wdm_list


def create_wdm_for_level(config, level, beta_order, precision):
    """
    Create a WDM object for a given level
    :param config: user configuration
    :type config: Config
    :param level: level
    :type level: int
    :param beta_order: beta function order for Meyer
    :type beta_order: int
    :param precision: wavelet precision
    :type precision: int
    :return: WDM object
    :rtype: WDM
    """
    layers = 2 ** level if level > 0 else 0
    wdm = WDM(layers, layers, beta_order, precision)
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

    return wdm