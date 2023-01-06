from pycwb import pycWB
import os


##########################
# middlewares
##########################
def cwb_config(event):
    cwb_user_config = event['data']['config']
    working_dir = event['data']['working_dir']
    os.chdir(working_dir)
    print(f"Current working dir {os.getcwd()}")
    print(f"Loading cwb config: {cwb_user_config}")
    cwb = pycWB(cwb_user_config)  # load envs and create dirs
    print(f"cwb initialized")
    event['cwb'] = cwb  # inject cwb instance
    return event
