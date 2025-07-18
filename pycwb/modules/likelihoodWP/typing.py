import numpy as np
from dataclasses import dataclass


@dataclass
class SkyStatistics:
    """
    Dataclass that encapsulates various sky-related statistical measurements derived from time-frequency data.

    Attributes:
        Gn (np.float32): Gaussian noise correction factor.
        Ec (np.float32): Core coherent energy in the time-frequency (TF) domain.
        Dc (np.float32): Signal core coherent energy in the TF domain.
        Rc (np.float32): Normalization factor for the EC measurements.
        Eh (np.float32): Satellite energy in the TF domain.
        Es (np.float32): Sideband energy in the TF domain.
        Np (np.float32): Number of pixels utilized in the analysis.
        Lm (np.float32): Likelihood map value.
        norm (np.float32): Overall normalization factor.
        cc (np.float32): Cross-correlation value.
        rho (np.float32): cWB Signal-to-Noise Ratio (SNR).
        xrho (np.float32): New cWB SNR used for XGBoost analysis.
        Lo (np.float32): Likelihood offset or related parameter.
        Eo (np.float32): Energy offset or related parameter.
        energy_array_plus (np.ndarray): Energy array for plus polarization.
        energy_array_cross (np.ndarray): Energy array for cross polarization.
        v00 (np.ndarray): Time-delayed data slice at a specific sky location for polarization 00.
        v90 (np.ndarray): Time-delayed data slice at a specific sky location for polarization 90.
        p00_POL (np.ndarray): Polar angle component of the signal packet for polarization 00.
        p90_POL (np.ndarray): Polar angle component of the signal packet for polarization 90.
        r00_POL (np.ndarray): Radius component of the residual packet for polarization 00.
        r90_POL (np.ndarray): Radius component of the residual packet for polarization 90.
    """
    Gn: np.float32    # gaussian noise correction
    Ec: np.float32    # core coherent energy in TF domain
    Dc: np.float32    # signal-core coherent energy in TF domain
    Rc: np.float32    # EC normalization
    Eh: np.float32    # satellite energy in TF domain
    Es: np.float32    # sideband energy in TF domain
    Np: np.float32    # number of pixels
    Em: np.float32    # energy map
    Lm: np.float32    # likelihood map
    norm: np.float32  # normalization factor
    cc: np.float32    # cross-correlation
    rho: np.float32   # cWB SNR
    xrho: np.float32  # new cWB SNR used for XGBoost
    Lo: np.float32
    Eo: np.float32
    N_pix_effective: np.float32  # effective number of pixels
    energy_array_plus: np.ndarray  # energy array for plus polarization
    energy_array_cross: np.ndarray  # energy array for cross polarization
    v00: np.ndarray  # time delayed data slice at sky location l for pol00
    v90: np.ndarray  # time delayed data slice at sky location l for pol90
    pd: np.ndarray  # 00 whitened data
    pD: np.ndarray  # 90 whitened data
    ps: np.ndarray  # 00 reconstructed whitened response
    pS: np.ndarray  # 90 reconstructed whitened response
    pixel_mask: np.ndarray  # mask for pixels
    gaussian_noise_correction: np.ndarray  # gaussian noise correction for pixels (Gn)
    noise_amplitude_00: np.ndarray  # noise amplitude for 00 (pn)
    noise_amplitude_90: np.ndarray  # noise amplitude for 90 (pN)
    coherent_energy: np.ndarray # coherent energy for pixels (ec)
    p00_POL: np.ndarray  # polar angle component of the signal packet for pol00
    p90_POL: np.ndarray  # polar angle component of the signal packet for pol90
    r00_POL: np.ndarray  # radius component of the residual packet for pol00
    r90_POL: np.ndarray  # radius component of the residual packet for pol90
    S_snr: np.ndarray  # SNR for the signal packet
    f: np.ndarray  
    F: np.ndarray  


@dataclass
class SkyMapStatistics:
    """
    Dataclass that encapsulates the sky map statistics for a specific sky location.

    Attributes:
        l_max (int): The maximum likelihood value for the sky location.
    """
    l_max: int
    nAntennaPrior: np.array  # sqrt(ff + FF)
    nAlignment: np.array  # sqrt(FF / ff) if ff > 0 else 0
    nLikelihood: np.array  # Eo - No
    nNullEnergy: np.array  # No
    nCorrEnergy: np.array  # Ec
    nCorrelation: np.array  # Co
    nSkyStat: np.array  # AA
    nDisbalance: np.array  # CH
    nNetIndex: np.array  # cc
    nEllipticity: np.array  # Cr
    nPolarisation: np.array  # Mp

    @classmethod
    def from_tuple(cls, t):
        return cls(*t)