import numpy as np


def likelihoodWP(netRHO):
    netEC = netRHO * netRHO * 2  # netEC/netRHO threshold
    pass


def set_plus_regulator(acor, gamma, delta, nIFO):
    netwoek_energy_threshold = 2 * acor * acor * nIFO  # network energy threshold in the sky loop
    gamma_regulator = gamma * gamma * 2. / 3.  # gamma regulator for x componet
    delta_regulator = abs(delta)
    if delta_regulator > 1:
        delta_regulator = 1

    return delta_regulator * np.sqrt(2)


def set_cross_regulator(acor, gamma, delta, nIFO):
    pass
