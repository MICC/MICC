__author__ = 'Matt'

import unittest
import numpy as np
from micc.utils import cycle_to_ladder, ladder_to_cycle, relabel


class UtilsTests(unittest.TestCase):

    def test_cycle_to_ladder1(self):
        cycle = '4+2-3-5-1+'
        true_ladder = [[1, 2, 3, 2, 4], [5, 3, 4, 1, 5]]
        test_ladder = cycle_to_ladder(cycle)
        self.assertEqual(true_ladder, relabel(test_ladder))

    def test_cycle_to_ladder2(self):
        cycle = '1-6+4-2+5-7+3+'
        true_ladder = relabel([[7, 4, 7, 2, 4, 2, 6], [1, 3, 6, 3, 5, 1, 5]])
        test_ladder = cycle_to_ladder(cycle)
        self.assertEqual(true_ladder, relabel(test_ladder))

    def test_cycle_to_ladder3(self):
        cycle = '2+5-7+3+1-6+4-'
        true_ladder = relabel([[4, 1, 4, 6, 1, 6, 3], [5, 7, 3, 7, 2, 5, 2]])
        test_ladder = cycle_to_ladder(cycle)
        self.assertEqual(true_ladder, relabel(test_ladder))
    """
    TODO
    Please note the ladder to cycle is untested currently. For the time being,
    there is no relabel(...) for cycles. Will be updated at a later date.
    """

if __name__ == '__main__':
    unittest.main()