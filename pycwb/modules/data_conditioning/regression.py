from gwpy.timeseries import TimeSeries
from pycwb import utils as ut
import ROOT


def regression(h: TimeSeries, f_min: int, f_max: int, scratch: float):
    """
        Clean data with cWB regression method.
    Input
    ------
    
    h: (wavearray) data to clean
    f_min: (int) minimum frequency
    f_max: (int) maximum frequency
    scratch: (float) extra data to avoid artifacts
    
    Output
    ------
    hh: (ROOT wavearray) cleaned data 
    
    """

    tfmap = ut.data_to_TFmap(h)

    # define regression
    r = ROOT.regression()
    r.add(tfmap, "hchannel")
    r.mask(0)
    r.unmask(0, f_min, f_max)

    # add original channel as aux
    r.add(h, "hchannel")

    # Calculate prediction
    r.setFilter(8)  # length of filter
    r.setMatrix(scratch, .95)  # totalscracht and % of data excluded
    r.solve(0.2, 0, 'h')  # 0.2, 0, 'h'
    r.apply(0.2)  # 0.2

    # get clean channel -> should be converted to timeseries or whatever interested.
    # amplitude is stored in data (array with size hh.size())

    # cleaned data
    hh = r.getClean()

    return hh
