import unittest
from micc.curves import RigidGraph
from sys import stderr
from micc.curves import  CurvePair


class PolygonalRegionTests(unittest.TestCase):
    """
    Verify that correctness of the new polygonal boundary algorithm. Using
    polygons determined by hand to validate correctness
    """
    def check_valid(self, true_polygons, test_polygons):
        correct = True
        for poly in test_polygons:
            poly = set(poly)
            match_found = False
            for true_poly in true_polygons:
                if poly == set(true_poly):
                    match_found = True
            correct &= match_found
        return correct

    def test_1_polygons(self):
        true_polygons = [[0, 4, 5], [7, 3, 2], [6, 11, 10], [9, 8, 1], [5, 11],
                         [3, 4], [6, 1], [10, 9], [0, 7], [8, 2]]

        ladder = [[12, 5, 6, 9, 10, 12, 11, 1, 2, 3, 4, 11],
                  [1, 2, 7, 8, 9, 10, 5, 6, 8, 7, 3, 4]]
        curvepair = CurvePair(ladder, compute=False)
        test_polygons = curvepair.concise_boundaries.values()

        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    def test_2_polygons(self):
        true_polygons = [[0, 5, 4, 5], [1, 2, 3, 2], [0, 3], [1, 4]]

        ladder = [[1, 4, 3, 4, 1, 6], [6, 5, 2, 3, 2, 5]]
        curvepair = CurvePair(ladder, compute=False)
        test_polygons = curvepair.concise_boundaries.values()

        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    def test_3_polygons(self):
        true_polygons = [[1, 12, 14], [2, 13, 5], [6, 9, 15], [14, 16, 10],
                         [3, 6], [5, 8], [3, 11], [4, 12], [1, 9], [2, 10],
                         [4, 7], [13, 15], [0, 11], [7, 16], [0, 8]]

        ladder =[[6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 1, 2, 3, 4, 17, 5],
                 [1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4]]
        curvepair = CurvePair(ladder, compute=False)
        test_polygons = curvepair.concise_boundaries.values()

        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    def test_hempel_polygons(self):
        true_polygons = [[0, 6, 15, 11, 5, 21], [2, 11], [3, 12], [4, 13],
                         [5, 14], [5, 14], [1, 7], [2, 8], [12, 16], [13, 17],
                         [14, 18], [15, 19], [16, 20], [17, 21], [18, 22],
                         [19, 23], [20, 24], [22, 6], [23, 7], [24, 8], [3, 9],
                         [4, 10], [1, 10], [0, 9]]

        ladder = [[1, 7, 18, 24, 5, 16, 12, 8, 19, 25, 6, 17, 23, 4, 15, 11, 22, 3, 14, 10, 21, 2, 13, 9, 20],
                  [25, 6, 17, 23, 4, 15, 11, 7, 18, 24, 5, 16, 22, 3, 14, 10, 21, 2, 13, 9, 20, 1, 12, 8, 19]]
        curvepair = CurvePair(ladder, compute=False)
        test_polygons = curvepair.concise_boundaries.values()

        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    # TODO vat least two more curves by hand

if __name__ == '__main__':
    unittest.main()