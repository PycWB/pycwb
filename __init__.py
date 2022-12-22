from . import setup
from . import cwb

class pycWB:
	def __init__(self, config_file):
		ROOT, gROOT, gSystem, gStyle, config = setup.init(config_file)
		self.ROOT = ROOT
		self.gROOT = gROOT
		self.gSystem = gSystem
		self.gStyle = gStyle
		self.config = config

	def cwb_inet2G(self, run_id, f_name, j_stage, u_name = "", eced = False, inet_option=None):
		cwb.cwb_inet2G(self.ROOT, self.gROOT, self.config, run_id, f_name, j_stage, u_name, eced, inet_option)

