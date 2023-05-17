import time
import logging
from .regression import regression
from .whitening import whitening
from pycwb.types.wdm import WDM
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def data_conditioning(config, strains, parallel=True):
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
    wdm_white = WDM(layers_white, layers_white, config.WDM_beta_order, config.WDM_precision)

    layers = int(config.rateANA / 8)
    wdm = WDM(layers, layers, config.WDM_beta_order, config.WDM_precision)

    if parallel:
        logger.info("Start data conditioning in parallel")
        with Pool(processes=min(config.nproc, config.nIFO)) as p:
            res = p.map(_wrapper, [(config, strains[i], wdm, wdm_white) for i in range(len(config.ifo))])
    else:
        res = list(map(_wrapper, [(config, strains[i], wdm, wdm_white) for i in range(len(config.ifo))]))

    conditioned_strains, nRMS_list = zip(*res)

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info(f"Data Conditioning Time: {timer_end - timer_start:.2f} seconds")
    logger.info("-------------------------------------------------------")

    return conditioned_strains, nRMS_list


def _wrapper(args):
    config, strain, wdm, wdm_white = args
    # regression and whitening
    data_reg = regression(config, wdm, strain)
    tf_map, nRMS = whitening(config, wdm_white, data_reg)

    return tf_map, nRMS
