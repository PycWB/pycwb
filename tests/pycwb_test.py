import os
# set parent directory to the first in python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
install_path = "/Users/yumengxu/Project/Physics/cwb/cwb_source/tools/install/lib"
os.environ['LD_LIBRARY_PATH'] = install_path

from pycwb import pycWB

# remove files in data
for file in os.listdir('./data'):
    os.remove(os.path.join('./data', file))

cwb = pycWB('./config.ini')

job_id = 1
job_stage = 'FULL'
job_file = './user_parameters_online.yaml'

cwb.cwb_inet2G(job_id, job_file, job_stage)