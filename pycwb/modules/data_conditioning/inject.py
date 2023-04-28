def inject(data, mdc):
    """
    Injects the MDC into the data, not finished yet

    :param data: the data to inject the MDC into
    :type data: gwpy.timeseries.TimeSeries
    :param mdc: the MDC to inject
    :type mdc: gwpy.timeseries.TimeSeries
    :return: the data with the MDC injected
    :rtype: gwpy.timeseries.TimeSeries
    """
    # TODO: a lot of checking and shifting

    # inject the mdc into the data
    data = data + mdc

    return data
