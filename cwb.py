import os

def cwb_inet2G(ROOT, gROOT, config, run_id, f_name, j_stage, u_name = "", eced = False, inet_option=None):
	gROOT.LoadMacro(f"{config['CWB']['CWB_INSTALL']}/etc/cwb/macros/cwb_inet2G.C")

	# TODO: parameters check
	CWB_STAGE = {
		"FULL": 0,
		"INIT": 1,
		"STRAIN": 2,
		"CSTRAIN": 3,
		"COHERENCE": 4,
		"SUPERCLUSTER": 5,
		"LIKELIHOOD": 6,
		"SAVE": 7,
		"FINISH": 8,
	}

	if inet_option:
		os.environ['CWB_INET_OPTIONS'] = inet_option
	ROOT.cwb_inet2G(run_id, f_name, CWB_STAGE[j_stage], u_name, eced)