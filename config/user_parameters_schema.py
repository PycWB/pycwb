schema = {
    "type": "object",
    "properties": {
        "analysis": {
            "enum": ["1G", "2G"],
            "description": "true/false -> online/offline",
            "default": "2G"
        },
        "cfg_search": {
            "enum": ["r", "i", "p", "l", "s", "c", "g", "e", "b",
                     "R", "I", "P", "L", "S", "C", "G", "E", "B"],
            "description": "see description below",
            "default": "r"
        },
        "online": {
            "type": "boolean",
            "description": "true/false -> online/offline",
            "default": False
        },
        "optim": {
            "type": "boolean",
            "description": "true -> optimal resolution likelihood analysis",
            "default": False
        },
        "fLow": {
            "type": "number",
            "description": "low frequency of the search",
            "default": 64.
        },
        "fHigh": {
            "type": "number",
            "description": "high frequency of the search",
            "default": 2048.
        },
        "ifo": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "uniqueItems": True,
            "description": "ifo[] can be redefined by user",
            "default": ["L1", "H1", "V1", "I1", "J1", "G1"]
        },
        "nIFO": {
            "type": "integer"
        },
        "refIFO": {
            "type": "string",
            "description": "reference IFO",
            "default": "L1"
        },
        "select_subrho": {
            "type": "number",
            "description": "subrho netcluster select function threshold (coherence)",
            "default": 5.0
        },
        "select_subnet": {
            "type": "number",
            "description": "subnet netcluster select function threshold (coherence)",
            "default": 0.1
        },
        "bpp": {
            "type": "number",
            "description": "probability for black pixel selection (netpixel)",
            "default": 0.001
        },
        "subnet": {
            "type": "number",
            "description": "[0,0.7] sub network threshold (supercluster)",
            "default": 0.7,
            "minimum": 0,
            "maximum": 0.7,
        },
        "subcut": {
            "type": "number",
            "description": "[0,1]   sub network threshold in the skyloop (supercluster)",
            "default": 0.33,
            "minimum": 0,
            "maximum": 1,
        },
        "netRHO": {
            "type": "number",
            "description": "[>4.0] coherent network SNR (supercluster, likelihood)",
            "default": 4.0,
            "minimum": 4.0,
        },
        "netCC": {
            "type": "number",
            "description": "network correlation (supercluster, likelihood)",
            "default": 0.5
        },
        "Acore": {
            "type": "number",
            "description": "threshold of core pixels (supercluster, likelihood)",
            "default": 2**0.5
        },
        "Tgap": {
            "type": "number",
            "description": "defragmentation time gap between clusters (sec)",
            "default": 3.0
        },
        "Fgap": {
            "type": "number",
            "description": "defragmentation frequency gap between clusters (Hz)",
            "default": 130.
        },
        "TFgap": {
            "type": "number",
            "description": "threshold on the time-frequency separation between two pixels",
            "default": 6.
        },
        "delta": {
            "type": "number",
            "description": "[-1:1] regulate 2 Detector sky locations. delta=0 : regulator is disabled, delta<0 : select Lo as skystat instead of Lr",
            "default": 0.5,
            "minimum": -1,
            "maximum": 1,
        },
        "cfg_gamma": {
            "type": "number",
            "description": "[-1:1] regulate |fx|<<|f+| and |f+|<<1 sky locations. gamma=0 : regulator is disabled, gamma<0 : sky prior is applied",
            "default": 0.5,
            "minimum": -1,
            "maximum": 1,
        },
        "segLen": {
            "type": "number",
            "description": "Segment length [sec]",
            "default": 600.
        },
        "segMLS": {
            "type": "number",
            "description": "Minimum Segment Length after DQ_CAT1 [sec]",
            "default": 300.
        },
        "segTHR": {
            "type": "number",
            "description": "Minimum Segment Length after DQ_CAT2 [sec] (to disable put segTHR=0)",
            "default": 30.
        },
        "segEdge": {
            "type": "number",
            "description": "wavelet boundary offset [sec]",
            "default": 8.
        },
        "segOverlap": {
            "type": "number",
            "description": "overlap between job segments [sec]",
            "default": 0.
        },
        "lagSize": {
            "type": "integer",
            "description": "number of lags (simulation:1)",
            "default": 1
        },
        "lagStep": {
            "type": "number",
            "description": "[sec] time interval between lags",
            "default": 1.
        },
        "lagOff": {
            "type": "integer",
            "description": "first lag id (lagOff=0 - include zero lag )",
            "default": 6.
        },
        "lagMax": {
            "type": "number",
            "description": "0/>0 -  standard/extended lags",
            "default": 150
        },
        "channelNamesRaw": {
            "type": "array"
        },
        "channelNamesMDC": {
            "type": "array"
        },
        "frFiles": {
            "type": "array"
        },
        "DQF": {
            "type": "array",
            "c_type": "dqfile"
        },
        "nDQF": {
            "type": "integer"
        },
        "gap": {
            "type": "number"
        },
        "simulation": {
            "type": "number"
        },
        "nfactor": {
            "type": "number"
        },
        "factors": {
            "type": "array",
            "items": {
                "type": "number"
            }
        },
        "levelR": {
            "type": "integer"
        },
        "healpix": {
            "type": "number"
        },
        "plugin": {
            "type": "string",
            "c_type": "TMacro"
        },
        "configPlugin": {
            "type": "string",
            "c_type": "TMacro"
        }
    },
    "required": ["analysis", "ifo", "refIFO"],
    "additionalProperties": False
}
