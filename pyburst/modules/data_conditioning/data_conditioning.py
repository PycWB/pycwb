import time
import logging
from .regression import regression
from .whitening import whitening
from pyburst.config import Config
from pyburst.constants import WDM_BETAORDER, WDM_PRECISION
from pyburst.types import TimeFrequencySeries, WDM
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def data_conditioning(config, strains):
    """
    Performs data conditioning on the given strain data, including regression and whitening

    :param config: config object
    :type config: Config
    :param strains: list of strain data
    :type strains: list[pycbc.types.timeseries.TimeSeries | gwpy.timeseries.TimeSeries | ROOT.wavearray(np.double)]
    :return: (conditioned_strains, nRMS_list)
    :rtype: tuple[list[TimeFrequencySeries], list[TimeFrequencySeries]]
    """
    # timer
    timer_start = time.perf_counter()

    # initialize WDM
    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm_white = WDM(layers_white, layers_white, WDM_BETAORDER, WDM_PRECISION)

    layers = int(config.rateANA / 8)
    wdm = WDM(layers, layers, WDM_BETAORDER, WDM_PRECISION)

    with Pool(processes=min(config.nproc, config.nIFO)) as p:
        res = p.starmap(_wrapper, [(config, strains[i], wdm, wdm_white) for i in range(len(config.ifo))])

    conditioned_strains, nRMS_list = zip(*res)

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info(f"Data Conditioning Time: {timer_end - timer_start:.2f} seconds")
    logger.info("-------------------------------------------------------")

    return conditioned_strains, nRMS_list


def _wrapper(config, strain, wdm, wdm_white):
    # regression and whitening
    data_reg = regression(config, wdm, strain)
    tf_map, nRMS = whitening(config, wdm_white, data_reg)

    return tf_map, nRMS
