from messenger import Messenger
from _config import config

##########################
# Entry point event
##########################
starter = {"key": "ONLINE", "cwb": "", "data": {
    "job_id": 1,
}}

if __name__ == '__main__':
    app = Messenger(config)
    app.run(starter)
