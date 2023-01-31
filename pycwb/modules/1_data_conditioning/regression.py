from gwpy.timeseries import TimeSeries
from pycwb.utils import utils as ut
import ROOT

ROOT.gSystem.Load("cwb.so")


def regression(h: TimeSeries, F1: int, F2: int, scratch: float):
    """
        Clean data with cWB regression method.
    Input
    ------
    
    h: (wavearray) data to clean
    F1: (int) minimum frequency
    F2: (int) maximum frequency
    scratch: (float) extra data to avoid artifacts
    
    Output
    ------
    hh: (ROOT wavearray) cleaned data 
    
    """

    tfmap = ut.data_to_TFmap(h)

    #define regression
    r = ROOT.regression()
    r.add(tfmap, "hchannel")
    r.mask(0)
    r.unmask(0, F1, F2)

    #add original channel as aux
    r.add(h, "hchannel")

    #Calculate prediction
    r.setFilter(8) # length of filter
    r.setMatrix(scratch, .95) # totalscracht and % of data excluded
    r.solve(0.2, 0, 'h') # 0.2, 0, 'h'
    r.apply(0.2) # 0.2

    #get clean channel -> should be converted to timeseries or whatever interested.
    #amplitude is stored in data (array with size hh.size())

    # cleaned data
    hh = r.getClean()
    
    return  hh

