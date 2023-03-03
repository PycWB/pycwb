# load user parameters
import sys
# add the parent directory to the first position of the path
sys.path.insert(0, '..')

from pycwb.modules.cwb_2g import cwb_2g

# cwb_2g('./config.ini', './user_parameters_mdc.yaml', 1264130816, 1264131610)
# cwb_2g('./config.ini', './user_parameters_mdc.yaml', 1264130816, 1264131592)
cwb_2g('./config.ini', './user_parameters_mdc.yaml')