import time
import logging
from .regression import regression
from .whitening import whitening
from pycwb.config import Config
from pycbc.types.timeseries import TimeSeries as pycbcTimeSeries
from pycwb.utils.cwb_convert import convert_pycbc_timeseries_to_wavearray
from pycwb.constants import WDM_BETAORDER, WDM_PRECISION
import ROOT
import numpy as np
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def data_conditioning(config: Config, strains):
    # timer

    # convert to wavearray
    # if isinstance(strains[0], pycbcTimeSeries):
    #     timer_start = time.perf_counter()
    #     with Pool(config.nIFO) as p:
    #         wave_array = p.map(convert_pycbc_timeseries_to_wavearray, strains)
    #     # wave_array = [convert_pycbc_timeseries_to_wavearray(d) for d in strains]
    #     timer_end = time.perf_counter()
    #     print(f"Data Conversion Time: {timer_end - timer_start:.2f} seconds")

    timer_start = time.perf_counter()

    # initialize WDM
    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm_white = ROOT.WDM(np.double)(layers_white,
                                    layers_white, WDM_BETAORDER, WDM_PRECISION)

    layers = int(config.rateANA / 8)
    wdm = ROOT.WDM(np.double)(layers, layers, WDM_BETAORDER, WDM_PRECISION)

    with Pool(config.nIFO) as p:
        res = p.starmap(_wrapper, [(config, strains[i], wdm, wdm_white) for i in range(len(config.ifo))])

    tf_maps, nRMS_list = zip(*res)

    # with Pool(config.nIFO) as p:
    #     data_reg = p.starmap(regression, [(config, wdm, wave_array[i]) for i in range(len(config.ifo))])

    # with Pool(config.nIFO) as p:
    #     tf_maps, nRMS_list = p.starmap(whitening, [(config, wdm_white, data_reg[i]) for i in range(len(config.ifo))])

    # data_reg = [regression(config, wdm, wave_array[i]) for i in range(len(config.ifo))]
    # tf_maps, nRMS_list = [whitening(config, wdm_white, data_reg[i]) for i in range(len(config.ifo))]

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info(f"Data Conditioning Time: {timer_end - timer_start:.2f} seconds")
    logger.info("-------------------------------------------------------")

    return tf_maps, nRMS_list


def _wrapper(config, strain, wdm, wdm_white):
    if isinstance(strain, pycbcTimeSeries):
        wave_array = convert_pycbc_timeseries_to_wavearray(strain)
    else:
        wave_array = strain

    # regression and whitening
    data_reg = regression(config, wdm, wave_array)
    tf_map, nRMS = whitening(config, wdm_white, data_reg)

    return tf_map, nRMS