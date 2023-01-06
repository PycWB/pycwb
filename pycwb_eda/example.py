from messenger import Messenger
from _config import config
##########################
# Entry point event
##########################
starter = {"key": "CWB_2G", "data": {
    "job_id": 1,
    "working_dir": "/Users/yumengxu/Project/Physics/cwb/MultiStages2G_yaml",
    "config": "/Users/yumengxu/Project/Physics/cwb/MultiStages2G_yaml/config.ini",
    "user_parameters": "/Users/yumengxu/Project/Physics/cwb/MultiStages2G_yaml/user_parameters.yaml"
}}

if __name__ == '__main__':
    messenger = Messenger(config, starter)
    messenger.run()
