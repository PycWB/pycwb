import time
import logging
from .regression import regression
from .whitening import whitening
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

    if parallel:
        logger.info("Start data conditioning in parallel")
        with Pool(processes=min(config.nproc, config.nIFO)) as p:
            data_regressions = p.starmap(regression, [(config, h) for h in strains])
            res = p.starmap(whitening, [(config, d) for d in data_regressions])
    else:
        data_regressions = [regression(config, h) for h in strains]
        res = [whitening(config, h) for h in data_regressions]

    conditioned_strains, nRMS_list = zip(*res)

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info(f"Data Conditioning Time: {timer_end - timer_start:.2f} seconds")
    logger.info("-------------------------------------------------------")

    return conditioned_strains, nRMS_list
