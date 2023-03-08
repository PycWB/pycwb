from gwpy.timeseries import TimeSeries


def inject(data: TimeSeries, mdc: TimeSeries):
    # TODO: a lot of checking and shifting

    # inject the mdc into the data
    data = data + mdc

    return data

