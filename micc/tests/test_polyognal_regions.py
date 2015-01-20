import unittest
from micc.curves import RigidGraph
from sys import stderr
from micc.curves import  CurvePair


class PolygonalRegionTests(unittest.TestCase):
    """
    Verify that correctness of the new polygonal boundary algorithm. Using
    polygons determined by hand to validate correctness
    """

    def test_1_polygons(self):
        true_polygons = [[0, 4, 5], [7, 3, 2], [6, 11, 10], [9, 8, 1], [5, 11],
                         [3, 4], [6, 1], [10, 9], [0, 7], [8, 2]]
        ladder = [[12, 5, 6, 9, 10, 12, 11, 1, 2, 3, 4, 11],
                  [1, 2, 7, 8, 9, 10, 5, 6, 8, 7, 3, 4]]
        curvepair = CurvePair(ladder, compute=False)
        test_polygons = curvepair.concise_boundaries.values()
        correct = True
        for poly in test_polygons:
            poly = set(poly)
            match_found = False
            for true_poly in true_polygons:
                if poly == set(true_poly):
                    match_found = True

            correct &= match_found
        self.assertTrue(correct)

    def test_2_polygons(self):
        true_polygons = [[0, 5, 4, 5], [1, 2, 3, 2], [0, 3], [1, 4]]
        ladder = [[1, 4, 3, 4, 1, 6], [6, 5, 2, 3, 2, 5]]
        curvepair = CurvePair(ladder, compute=False)
        test_polygons = curvepair.concise_boundaries.values()
        correct = True
        for poly in test_polygons:
            poly = set(poly)
            match_found = False
            for true_poly in true_polygons:
                if poly == set(true_poly):
                    match_found = True
            correct &= match_found
        self.assertTrue(correct)

    def test_3_polygons(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()