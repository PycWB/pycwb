from ligo.gracedb.rest import GraceDb
from ligo.gracedb.exceptions import HTTPError


def get_superevent(event_name):
    """
    Get the super event for a given event name.

    :param event_name: The name of the event.
    :type event_name: str
    :return: The super event.
    :rtype: dict
    """
    client = GraceDb()
    try:
        response = client.superevent(event_name).json()
    except HTTPError as e:
        if e.response.status_code == 404:
            print('No super event found for event {}'.format(event_name))
            return None
        else:
            raise

    return response


def get_superevent_t0(event_name):
    """
    Get the GPS time of the super event for a given event name.

    :param event_name: The name of the event.
    :type event_name: str
    :return: The GPS time of the super event.
    :rtype: float
    """
    super_event = get_superevent(event_name)
    if super_event is None:
        return None

    return super_event['t_0']


# --------------------------------------------------------------------------
# Online search extensions
# --------------------------------------------------------------------------

def upload_online_event(event, group="Burst", pipeline="CWB",
                        search="AllSky"):
    """Create a new GraceDB event from a PyCWB online trigger.

    Parameters
    ----------
    event : Event
        PyCWB ``Event`` object carrying trigger parameters.
    group, pipeline, search : str
        GraceDB classification metadata.

    Returns
    -------
    str
        The ``graceid`` assigned by GraceDB.
    """
    client = GraceDb()

    # Build a minimal JSON payload for the event
    payload = {
        "gpstime": getattr(event, "gps_time", 0.0),
        "instruments": getattr(event, "ifo", ""),
        "extra_attributes": {
            "CoincInspiral": {
                "snr": getattr(event, "rho", 0.0),
            },
        },
    }
    import json
    filecontents = json.dumps(payload).encode("utf-8")

    response = client.createEvent(
        group=group,
        pipeline=pipeline,
        search=search,
        filename="pycwb_event.json",
        filecontents=filecontents,
    )
    return response.json()["graceid"]


def upload_skymap(graceid, skymap_data, filename="skymap.fits.gz"):
    """Upload a HEALPix skymap to an existing GraceDB event.

    Parameters
    ----------
    graceid : str
        GraceDB event identifier.
    skymap_data : bytes or object
        Skymap file contents.  If an object with a ``to_fits`` method
        is passed, it is serialised first.
    filename : str
        Filename for the upload.
    """
    client = GraceDb()

    if hasattr(skymap_data, "to_fits"):
        import io
        buf = io.BytesIO()
        skymap_data.to_fits(buf)
        skymap_data = buf.getvalue()

    client.writeLog(
        graceid,
        "PyCWB sky localisation",
        filename=filename,
        filecontents=skymap_data,
        tag_name="sky_loc",
    )


def write_log(graceid, message, tag_name=None):
    """Attach a log message to a GraceDB event.

    Parameters
    ----------
    graceid : str
        GraceDB event identifier.
    message : str
        Log message text.
    tag_name : str or None
        Optional tag for the log entry.
    """
    client = GraceDb()
    kwargs = {}
    if tag_name is not None:
        kwargs["tag_name"] = tag_name
    client.writeLog(graceid, message, **kwargs)
