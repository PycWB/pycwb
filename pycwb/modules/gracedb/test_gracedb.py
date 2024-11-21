import unittest
from .gracedb import get_superevent_t0, get_superevent

class TestGraceDB(unittest.TestCase):

    def test_get_superevent_t0(self):
        event_name = 'S241011k'
        t0 = get_superevent_t0(event_name)
        self.assertAlmostEqual(t0, 1412725132.96, places=1)

    def test_get_superevent_t0_no_event(self):
        event_name = 'S000000a'
        t0 = get_superevent_t0(event_name)
        self.assertIsNone(t0)

    def test_get_superevent(self):
        event_name = 'S241011k'
        super_event = get_superevent(event_name)
        self.assertEqual(super_event['superevent_id'], event_name)

    def test_get_superevent_no_event(self):
        event_name = 'S000000a'
        super_event = get_superevent(event_name)
        self.assertIsNone(super_event)

if __name__ == '__main__':
    unittest.main()