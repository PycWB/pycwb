# load user parameters
import sys
# add the parent directory to the first position of the path
sys.path.insert(0, '..')

from pycwb.modules.cwb_2g import cwb_2g

cwb_2g('./user_parameters_mdc.yaml')