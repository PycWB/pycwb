install_path = "/Users/yumengxu/Project/Physics/cwb/cwb_source/tools/install/lib"
import os
os.environ['LD_LIBRARY_PATH'] = install_path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pycwb.modules import cwb_2g

cwb_2g(config='./config.ini', user_parameters='./user_parameters_online.yaml')