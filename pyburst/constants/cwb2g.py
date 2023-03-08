"""
Default parameters for CWB2G (Should be moved to config file)
"""

# minimun skymap resolution used for subNetCut
MIN_SKYRES_HEALPIX = 4
"""minimun skymap resolution used for subNetCut"""
MIN_SKYRES_ANGLE = 3
"""minimun skymap resolution used for subNetCut"""

# regression parameters
REGRESSION_FILTER_LENGTH = 8
"""regression parameters"""
REGRESSION_MATRIX_FRACTION = 0.95
"""regression parameters"""
REGRESSION_SOLVE_EIGEN_THR = 0.
"""regression parameters"""
REGRESSION_SOLVE_EIGEN_NUM = 10
"""regression parameters"""
REGRESSION_SOLVE_REGULATOR = 'h'
"""regression parameters"""
REGRESSION_APPLY_THR = 0.8
"""regression parameters"""

# WDM default parameters
WDM_BETAORDER = 6  # beta function order for Meyer
"""WDM default parameters: beta function order for Meyer"""
WDM_PRECISION = 10  # wavelet precision
"""WDM default parameters: wavelet precision"""
