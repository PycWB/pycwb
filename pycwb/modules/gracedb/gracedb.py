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
