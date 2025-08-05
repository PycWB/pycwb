import random
def get_injection_parameters():
    """
    Function to return the injection parameters for the test.
    """
    return [{
        "mass1": 36.0,
        "mass2": 29.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "distance": 2000,
        "pol": 0,
        "t_start": -2.0, 
        "t_end": 1.0,
        "polarization": 0.0,
        "approximant": "IMRPhenomTPHM",
        "f_lower": 20.0,
    }]