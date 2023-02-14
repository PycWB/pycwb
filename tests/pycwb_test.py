import unittest
from pycwb import pycWB, sim


class MyTestCase(unittest.TestCase):
    def test_0_init(self):
        cwb = pycWB('./config.ini')  # config file path


        # sim.create_frame_noise(gROOT, ROOT)
        # sim.setup_sim_data(['H1','L1','V1'])

        # run full `cwb_inet2G` analysis

        if cwb is not None:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)  # add assertion here

    def test_1_cwb_inet2G(self):
        cwb = pycWB('./config.ini')

        job_id = 1
        job_stage = 'INIT'
        job_file = './user_parameters.yaml'
        try:
            cwb.cwb_inet2G(job_id, job_file, job_stage)
            self.assertEqual(True, True)
        except Exception as e:
            self.assertEqual(True, False, msg=e)


if __name__ == '__main__':
    unittest.main()
