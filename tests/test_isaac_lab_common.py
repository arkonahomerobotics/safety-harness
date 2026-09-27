"""The Isaac Lab adapters' quaternion conversion -- Isaac Lab 3.x stores (x, y, z, w)."""

import math
import unittest

from safety_harness.adapters._isaac_lab_common import quat_xyzw_to_wxyz


class QuatConversionTest(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(quat_xyzw_to_wxyz((0.0, 0.0, 0.0, 1.0)), (1.0, 0.0, 0.0, 0.0))

    def test_g1_rest_wrist_yaw_90(self):
        # the G1 left wrist at rest: 90 degrees about +Z, as Isaac Lab 3.x reports it
        s = math.sqrt(0.5)
        self.assertEqual(quat_xyzw_to_wxyz((0.0, 0.0, s, s)), (s, 0.0, 0.0, s))

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            quat_xyzw_to_wxyz((1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
