##########################
# Event configurations
##########################
config = {
    "watched_dir": ["./data"],
    "filewatcher": [
        {
            "name": "strain_*",
            "trigger": {
                "event": "DATA_RETRIEVED"
            }
        },
        {
            "name": "coherence_*",
            "trigger": {
                "event": "COHERENCE_DONE"
            }
        },
        {
            "name": "cluster_*",
            "trigger": {
                "event": "CLUSTERED"
            }
        },
        {
            "name": "likelihood_*",
            "trigger": {
                "event": "LIKELIHOOD_CALCULATED"
            }
        }
    ],
    "middleware": [
        {
            "key": "CWB_CONFIG",
            "inject": ["CWB_2G"],
            "handler": "_middlewares.cwb_config"
        }
    ],
    "periodic": [
        {
            "event": "ONLINE",
            "handler": "_handlers.online_periodic_trigger",
            "interval": 5,
            "repeat": 20,
            "trigger": ["RETRIEVE_DATA"]
        }
    ],
    "events": [
        {
            "event": "RETRIEVE_DATA",
            "handler": "_handlers.retrieve_data",
            "trigger": ["DATA_RETRIEVED"]
        },
        {
            "event": "DATA_RETRIEVED",
            "handler": "_handlers.coherence",
            "trigger": ["COHERENCE_DONE"]
        },
        {
            "event": "COHERENCE_DONE",
            "handler": "_handlers.cluster",
            "trigger": ["CLUSTERED"]
        },
        {
            "event": "COHERENCE_DONE",
            "handler": "_handlers.cluster2",
        },
        {
            "event": "CLUSTERED",
            "handler": "_handlers.likelihood",
            "trigger": ["LIKELIHOOD_CALCULATED"]
        },
        {
            "event": "LIKELIHOOD_CALCULATED",
            "handler": "_handlers.plot"
        },
        {
            "event": "CLUSTERED",
            "handler": "_handlers.plot"
        },
        {
            "event": "PIXELATED",
            "handler": "_handlers.plot"
        }
    ]
}
