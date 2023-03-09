# load user parameters
import sys
# add the parent directory to the first position of the path
sys.path.insert(0, '..')
from pyburst.search import search


search('./user_parameters_mdc.yaml', no_subprocess=True)