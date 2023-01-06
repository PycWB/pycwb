##########################
# Event configurations
##########################
config = {
    "middlewares": [
        {
            "key": "CWB_CONFIG",
            "inject": ["CWB_2G", "ONLINE"],
            "handler": "_handlers.cwb_config"
        }
    ],
    "events": [
        {
            "event": "ONLINE",
            "handler": "_handlers.online_periodic_trigger",
            "periodic": {
                "interval": 5,
                # "repeat": 20
            },
            "trigger": ["RETRIEVE_DATA"]
        },
        {
            "event": "CWB_2G",
            "handler": "_handlers.cwb_init",
            "trigger": ["CWB_INITED"]
        },
        {
            "event": "CWB_INITED",
            "handler": "pycwb.handlers.cwb_read_data",
            "trigger": ["DATA_RETRIEVED"]
        },
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
            "trigger": ["CLUSTERED"]
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
