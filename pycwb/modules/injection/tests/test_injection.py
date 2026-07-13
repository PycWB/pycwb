import unittest

from ..injection import distribute_inj_in_gps_time_by_rate


# unittest for distribute_inj_in_gps_time_by_rate
class TestDistributeInjInGpsTimeByRate(unittest.TestCase):
    def test_distribute_inj_in_gps_time_by_rate(self):
        
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
            distribute_inj_in_gps_time_by_rate(
                injections, rate, jitter, start_gps_time, end_gps_time
            )

        # test case 2: too small data
        rate = 1/200
        jitter = 20
        start_gps_time = 0
        end_gps_time = start_gps_time + 10

        # should raise ValueError
        with self.assertRaises(ValueError):
            distribute_inj_in_gps_time_by_rate(
                injections, rate, jitter, start_gps_time, end_gps_time
            )

        # test case 3: normal case
        rate = 1/200
        jitter = 20
        start_gps_time = 0
        end_gps_time = start_gps_time + 300

        # should return the distributed injections
        distributed_injections, n_trials = distribute_inj_in_gps_time_by_rate(
            injections,
            rate,
            jitter,
            start_gps_time,
            end_gps_time,
            shuffle=False,
        )
        self.assertEqual(len(distributed_injections), 4)
        self.assertEqual(n_trials, 4)
        self.assertTrue(
            all(80 <= injection['gps_time'] <= 120
                for injection in distributed_injections)
        )
        self.assertEqual(
            [injection['trial_idx'] for injection in distributed_injections],
            [0, 1, 2, 3],
        )
