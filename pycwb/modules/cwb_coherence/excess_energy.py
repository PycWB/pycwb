import numpy as np
import logging
from typing import List, Tuple, Dict
from pycwb.types.network_pixel import Pixel
from pycwb.types.time_frequency_series import TimeFrequencySeries

logger = logging.getLogger(__name__)


def get_network_pixels(tf_maps: List[TimeFrequencySeries], nRMS: List[TimeFrequencySeries],
                       veto: np.ndarray, edge: float, energy_threshold: float,
                       norm: float, lag_shifts: List[float], xp: np) -> List[Pixel]:
    pixel_layer_rate = tf_maps[0].w_rate
    n_edge = int(edge * pixel_layer_rate + 0.001)  # number of samples at the edge

    if n_edge & 1:
        logger.error("WDM parity violation: edge must be even, got %d", n_edge)
        raise ValueError("WDM parity violation: edge must be even")
    
    if n_edge < 3:
        logger.error("WDM edge length too small: %d", n_edge)
        raise ValueError("WDM edge length too small")
    
    



def get_network_pixels_sample(tf_maps: List[np.ndarray], nRMS: List[np.ndarray], 
                      veto: np.ndarray, params: Dict, xp=np) -> Dict:
    """
    Main function to identify excess energy pixels with NumPy/CuPy support
    """
    # Validate inputs
    _validate_inputs(tf_maps, nRMS, veto, params)

    # Combine energies from all detectors
    combined_energy = _combine_energies(tf_maps, veto, params, xp)

    # Find significant pixels using core/halo criteria
    pixels = _find_core_halo_pixels(combined_energy, tf_maps, params, xp)

    # Calculate noise RMS for pixels
    pixels = _calculate_rms(pixels, nRMS, params, xp)

    return pixels

def _validate_inputs(tf_maps, nRMS, veto, params):
    """Input validation checks"""
    if tf_maps[0].ndim != 2:
        raise ValueError("Invalid TF map dimensions")
    if veto.shape[0] != tf_maps[0].shape[0]:
        raise ValueError("Veto array size mismatch")

def _combine_energies(tf_maps: List[np.ndarray], veto: np.ndarray,
                     params: Dict, xp) -> np.ndarray:
    """
    Combine energies from all detectors with veto mask application
    """
    n_ifo = len(tf_maps)
    edge = params['edge']
    layer_size = tf_maps[0].shape[1]
    
    # Create combined energy map
    combined = xp.zeros_like(tf_maps[0])
    
    # Calculate valid time indices considering edges
    valid_times = slice(int(edge), tf_maps[0].shape[0] - int(edge))
    
    # Combine energies with vectorized operations
    for det_map in tf_maps:
        combined[valid_times] += det_map[valid_times]
    
    # Apply veto mask
    veto_mask = xp.expand_dims(veto[valid_times], 1)
    combined[valid_times] *= veto_mask
    
    # Apply energy thresholds
    combined[valid_times] = xp.where(
        (combined[valid_times] < params['Eo']) |
        (xp.arange(layer_size) < params['ib']),
        0, xp.minimum(combined[valid_times], 2*params['Eo'])
    )
    
    return combined

def _find_core_halo_pixels(energy_map: np.ndarray, tf_maps: List[np.ndarray],
                          params: Dict, xp) -> Dict:
    """
    Identify pixels satisfying core/halo energy criteria using vectorized operations
    """
    ib, ie = params['ib'], params['ie']
    Em, Eh = 2*params['Eo'], (2*params['Eo'])**2
    
    # Create padded array for neighbor calculations
    padded = xp.pad(energy_map, ((2, 2), (2, 2)), mode='constant')
    
    # Calculate core and halo energies using shifted arrays
    core_top = padded[2:-2, 3:-1] + padded[3:-1, 2:-2] + padded[3:-1, 3:-1]
    core_bottom = padded[2:-2, 1:-3] + padded[1:-1, 2:-2] + padded[1:-1, 1:-3]
    halo_top = padded[4:, 3:-1] + padded[4:, 4:]
    halo_bottom = padded[:-4, 1:-3] + padded[:-4, :-4]

    # Create masks for significant pixels
    energy_center = energy_map[ib:ie, 2:-2]
    mask = (
        (energy_center >= params['Eo']) &
        ((core_top + core_bottom) * energy_center >= Eh) |
        ((core_top + halo_top) * energy_center >= Eh) |
        ((core_bottom + halo_bottom) * energy_center >= Eh)
    )

    # Extract significant pixel indices
    times, freqs = xp.where(mask)
    freqs += ib  # Adjust frequency indices
    
    # Create pixel dictionary
    return {
        'time': times,
        'frequency': freqs,
        'energy': energy_map[times, freqs],
        'asnr': [xp.sqrt(tf_map[times, freqs]) for tf_map in tf_maps]
    }

def _calculate_rms(pixels: Dict, nRMS: List[np.ndarray],
                  params: Dict, xp) -> Dict:
    """
    Calculate noise RMS values for identified pixels
    """
    rms_values = []
    for det_idx, det_rms in enumerate(nRMS):
        # Vectorized RMS calculation using broadcasting
        t_indices = (pixels['time'] / params['rate']).astype(int)
        f_indices = (pixels['frequency'] * params['dF']).astype(int)
        
        valid = (f_indices < det_rms.shape[1]) & (t_indices < det_rms.shape[0])
        rms = xp.zeros_like(pixels['energy'])
        rms[valid] = 1 / xp.sqrt(xp.sum(1 / det_rms[t_indices[valid], f_indices[valid]]**2))
        
        rms_values.append(rms)
    
    return {**pixels, 'rms': rms_values}

def energy_threshold(tf_maps: List[np.ndarray], params: Dict, xp=np) -> float:
    """
    Calculate energy threshold using Gamma distribution statistics
    """
    combined = xp.sum([tf_map for tf_map in tf_maps], axis=0)
    valid = combined[params['nL']:-params['nR']]
    
    clipped = xp.clip(valid, 0, 100*len(tf_maps))
    mask = clipped > 0.001
    
    avg_energy = xp.mean(clipped[mask])
    log_mean = xp.log(avg_energy) - xp.mean(xp.log(clipped[mask]))
    
    shape = (3 - log_mean + xp.sqrt((log_mean-3)**2 + 24*log_mean)) / (12*log_mean)
    bpp = params['p'] * shape / params['shape']
    
    return avg_energy * _igamma(shape, bpp) / (shape * 2)

def _igamma(shape: float, bpp: float) -> float:
    """Inverse gamma function implementation (requires SciPy)"""
    from scipy.special import gammaincinv
    return gammaincinv(shape, bpp)