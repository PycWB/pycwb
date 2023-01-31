from gwpy.timeseries import TimeSeries
import ROOT

ROOT.gSystem.Load("cwb.so")


def whitening(data: TimeSeries):
    return data
