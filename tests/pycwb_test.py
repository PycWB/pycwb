import unittest
from pycwb import pycWB, sim


class MyTestCase(unittest.TestCase):
    def test_something(self):
        cwb = pycWB('./config.ini')  # config file path
        ROOT = cwb.ROOT
        gROOT = cwb.gROOT

        # sim.create_frame_noise(gROOT, ROOT)
        # sim.setup_sim_data(['H1','L1','V1'])

        # run full `cwb_inet2G` analysis

        job_id = 1
        job_stage = 'INIT'
        job_file = './user_parameters.yaml'
        cwb.cwb_inet2G(job_id, job_file, job_stage)

        if cwb is not None:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)  # add assertion here



if __name__ == '__main__':
    unittest.main()
