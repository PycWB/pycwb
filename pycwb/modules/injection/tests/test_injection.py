from ..injection import distribute_inj_in_gps_time
import unittest


# unittest for distribute_inj_in_gps_time
class TestDistributeInjInGpsTime(unittest.TestCase):
    def test_distribute_inj_in_gps_time(self):
        
        injections = [{'mass1': 50, 'mass2': 20, 'spin1z': 0, 'spin2z': 0, 'distance': 200, 'inclination': 0, 'polarization': 0, 'coa_phase': 0},
                       {'mass1': 70, 'mass2': 20, 'spin1z': 0, 'spin2z': 0, 'distance': 200, 'inclination': 0, 'polarization': 0, 'coa_phase': 0},
                       {'mass1': 90, 'mass2': 20, 'spin1z': 0, 'spin2z': 0, 'distance': 200, 'inclination': 0, 'polarization': 0, 'coa_phase': 0},
                       {'mass1': 110, 'mass2': 20, 'spin1z': 0, 'spin2z': 0, 'distance': 200, 'inclination': 0, 'polarization': 0, 'coa_phase': 0}]
        
        # test case 1: too large jitter
        rate = 1/200
        jitter = 200
        start_gps_time = 0
        end_gps_time = start_gps_time + 300
        
        # should raise ValueError
        with self.assertRaises(ValueError):
            distribute_inj_in_gps_time(injections, rate, jitter, start_gps_time, end_gps_time)

        # test case 2: too small data
        rate = 1/200
        jitter = 20
        start_gps_time = 0
        end_gps_time = start_gps_time + 10

        # should raise ValueError
        with self.assertRaises(ValueError):
            distribute_inj_in_gps_time(injections, rate, jitter, start_gps_time, end_gps_time)

        # test case 3: normal case
        rate = 1/200
        jitter = 20
        start_gps_time = 0
        end_gps_time = start_gps_time + 300

        # should return the distributed injections
        distributed_injections = distribute_inj_in_gps_time(injections, rate, jitter, start_gps_time, end_gps_time)
        self.assertEqual(len(distributed_injections), 4)
        # the gps time of the first injection should be in the range of [80, 120]
        self.assertTrue(80 <= distributed_injections[0]['gps_time'] <= 120)
        # the trail number of the first injection should be 0
        self.assertEqual(distributed_injections[0]['trail'], 0)
        # the gps time of the second injection should be in the range of [80, 120]
        self.assertTrue(80 <= distributed_injections[1]['gps_time'] <= 120)
        # the trail number of the second injection should be 1
        self.assertEqual(distributed_injections[1]['trail'], 1)