import numpy as np
from dataclasses import dataclass


@dataclass
class SkyStatistics:
    Gn: np.float32  # gaussian noise correction
    Ec: np.float32  # core coherent energy in TF domain
    Dc: np.float32  # signal-core coherent energy in TF domain
    Rc: np.float32  # EC normalization
    Eh: np.float32  # satellite energy in TF domain
    Es: np.float32  # sideband energy in TF domain
    Np: np.float32  # number of pixels
    Lm: np.float32  # likelihood map
    norm: np.float32  # normalization factor
    cc: np.float32  # cross-correlation
    rho: np.float32 # cWB SNR
    xrho: np.float32 # new cWB SNR used for XGBoost
    Lo: np.float32  
    Eo: np.float32  
    v00: np.ndarray  # time delayed data slice at sky location l for pol00
    v90: np.ndarray  # time delayed data slice at sky location l for pol90
    p00_POL: np.ndarray  # polar angle component of the signal packet for pol00
    p90_POL: np.ndarray  # polar angle component of the signal packet for pol90
    r00_POL: np.ndarray  # radius component of the residual packet for pol00
    r90_POL: np.ndarray  # radius component of the residual packet for pol90
