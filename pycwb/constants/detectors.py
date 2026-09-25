DETECTORS = {
    "H1": {
        "name": "LHO_4k",
        "lat": 0.81079526383,
        "lon": -2.08405676917,
        "elevation": 142.554,
        "x": {
            "alt": -0.00061950000,
            "az": 5.65487724844,
            "midpoint": 1997.54200000000
        },
        "y": {
            "alt": 0.00001250000,
            "az": 4.08408092164,
            "midpoint": 1997.52200000000
        }
    },
    "L1": {
        "name": "LLO_4k",
        "lat": 0.53342313506,
        "lon": -1.58430937078,
        "elevation": -6.574,
        "x": {
            "alt": -0.00031210000,
            "az": 4.40317772346,
            "midpoint": 1997.57500000000
        },
        "y": {
            "alt": -0.00061070000,
            "az": 2.83238139666,
            "midpoint": 1997.57500000000
        }
    },
    "V1": {
        "name": "VIRGO",
        "lat": 0.76151183984,
        "lon": 0.18333805213,
        "elevation": 51.884,
        "x": {
            "alt": 0.00000000000,
            "az": 0.33916285222,
            "midpoint": 1500.00000000000
        },
        "y": {
            "alt": 0.00000000000,
            "az": 5.05155183261,
            "midpoint": 1500.00000000000
        }
    },
    "I1": {
        "name": "LIO_4k",
        "lat": 0.24841853020,
        "lon": 1.33401332494,
        "elevation": 0.0,
        "x": {
            "alt": 0.00000000000,
            "az": 1.57079637051,
            "midpoint": 2000.00000000000
        },
        "y": {
            "alt": 0.00000000000,
            "az": 0.00000000000,
            "midpoint": 2000.00000000000
        }
    },
    "G1": {
        "name": "GEO_600",
        "lat": 0.91184982752,
        "lon": 0.17116780435,
        "elevation": 114.425,
        "x": {
            "alt": 0.00000000000,
            "az": 1.19360100484,
            "midpoint": 300.00000000000
        },
        "y": {
            "alt": 0.00000000000,
            "az": 5.83039279401,
            "midpoint": 300.00000000000
        }
    },
    "E1": {
        "name": "ET1_T1400308",
        "lat": 0.76151183984,
        "lon": 0.18333805213,
        "elevation": 51.884,
        "x": {
            "alt": 0.00000000000,
            "az": 0.33916285222,
            "midpoint": 5000.00000000000
        },
        "y": {
            "alt": 0.00000000000,
            "az": 5.57515060820,
            "midpoint": 5000.00000000000
        }
    },
    "E2": {
        "name": "ET2_T1400308",
        "lat": 0.76299307990,
        "lon": 0.18405858870,
        "elevation": 59.735,
        "x": {
            "alt": 0.00000000000,
            "az": 4.52795305701,
            "midpoint": 5000.00000000000
        },
        "y": {
            "alt": 0.00000000000,
            "az": 3.48075550581,
            "midpoint": 5000.00000000000
        }
    },
    "E3": {
        "name": "ET3_T1400308",
        "lat": 0.76270463257,
        "lon": 0.18192996730,
        "elevation": 59.727,
        "x": {
            "alt": 0.00000000000,
            "az": 2.43355795462,
            "midpoint": 5000.00000000000
        },
        "y": {
            "alt": 0.00000000000,
            "az": 1.38636040342,
            "midpoint": 5000.00000000000
        }
    },
    "E0": {
        "name": "ET0_T1400308",
        "lat": 0.76270463257,
        "lon": 0.18192996730,
        "elevation": 59.727,
        "x": {
            "alt": 0.00000000000,
            "az": 0.00000000000,
            "midpoint": 0.00000000000
        },
        "y": {
            "alt": 0.00000000000,
            "az": 0.00000000000,
            "midpoint": 0.00000000000
        }
    },
    "K1": {
        "name": "KAGRA",
        "lat": 0.6355068497,
        "lon": 2.396441015,
        "elevation": 414.181,
        "x": {
            "alt": 0.0031414,
            "az": 1.054113,
            "midpoint": 1513.2535
        },
        "y": {
            "alt": -0.0036270,
            "az": -0.5166798,
            "midpoint": 1511.611
        }
    }
}


# Literal vectors from cWB 6.4.6.9, wat/detector.cc, commit e03cf7f.
CWB_VECTORS = {
    "H1": (
        (-2161414.928, -3834695.183, 4600350.224),
        (-0.223891216, 0.799830697, 0.556905359),
        (-0.913978490, 0.026095321, -0.404922650),
    ),
    "L1": (
        (-74276.04192, -5496283.721, 3224257.016),
        (-0.954574615, -0.141579994, -0.262187738),
        (0.297740169, -0.487910627, -0.820544948),
    ),
}


# "pycwb-1" versions the bundled LAL-derived geographic table above. It is
# deliberately not a claim that these constants track an installed LAL version.
DETECTOR_GEOMETRIES = {
    f"{name}:lal@pycwb-1": {
        "detector": name, "source": "lal", "version": "pycwb-1",
        "parameters": parameters, "vectors": None,
    }
    for name, parameters in DETECTORS.items()
}
DETECTOR_GEOMETRIES.update({
    f"{name}:cwb": {
        "detector": name, "source": "cwb",
        "parameters": DETECTORS[name], "vectors": vectors,
    }
    for name, vectors in CWB_VECTORS.items()
})
DETECTOR_GEOMETRY_ALIASES = {
    f"{entry['detector']}:{entry['source']}": key
    for key, entry in DETECTOR_GEOMETRIES.items() if entry["source"] == "lal"
}


def resolve_detector_geometry(name, selection=None):
    """Return a canonical, pinned registry ID for one instrument.

    A mapping selects each detector independently. Missing entries use the
    bundled LAL-derived geometry. Qualified names can also be used directly.
    """
    base = name.split(":", 1)[0]
    if isinstance(selection, dict):
        selection = selection.get(base)
    if ":" in name:
        if selection is not None:
            raise ValueError("Specify geometry in the detector name or selection, not both")
        selection = name
    if selection is None or selection == "lal":
        selection = f"{base}:lal@pycwb-1"
    selection = DETECTOR_GEOMETRY_ALIASES.get(selection, selection)
    if selection not in DETECTOR_GEOMETRIES:
        raise ValueError(f"Unknown detector geometry: {selection}")
    if DETECTOR_GEOMETRIES[selection]["detector"] != base:
        raise ValueError(f"Geometry {selection} does not belong to {base}")
    return selection


def resolve_detector_geometries(ifos, selections):
    """Fill defaults and pin aliases before recording a run configuration."""
    if not isinstance(selections, dict):
        raise ValueError("detector_geometry must map detector names to geometry IDs")
    unknown = set(selections) - set(ifos)
    if unknown:
        raise ValueError(f"Geometry specified for inactive detectors: {sorted(unknown)}")
    return {name: resolve_detector_geometry(name, selections) for name in ifos}
