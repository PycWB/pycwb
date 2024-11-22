import unittest
from .job_segment import read_seg_list, get_job_list

class TestJobSegment(unittest.TestCase):
    def test_read_seg_list(self):
        periods = ([100000], [100000 + 2400])
        expected_output = ([100000], [102400])
        cat1_list = read_seg_list(periods=periods)

        self.assertEqual(cat1_list, expected_output)

    def test_get_job_list(self):
        periods = ([100000], [100000 + 2400])
        ifos = ['H1', 'L1']
        seg_len = 1200
        seg_mls = 600
        seg_edge = 10
        sample_rate = 4096
        cat1_list = read_seg_list(periods=periods)

        job_segments = get_job_list(ifos, cat1_list, seg_len, seg_mls, seg_edge, sample_rate)
        self.assertEqual(len(job_segments), 2)
        self.assertEqual(job_segments[0].start_time, 100010)
        self.assertEqual(job_segments[0].end_time, 101200)
        self.assertEqual(job_segments[1].start_time, 101200)
        self.assertEqual(job_segments[1].end_time, 102390)


if __name__ == '__main__':
    unittest.main()