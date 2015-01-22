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
            if not match_found:
                stderr.write('failed on: '+str(poly)+'\n')
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

        #old = [[(0.0, 3), (9.0, 1)], [(21.0, 3), (0.0, 1)],
        #  [(1.0, 1), (7.0, 3), (16.0, 1), (12.0, 3), (6.0, 1), (22.0, 3)],
        #  [(10.0, 1), (1.0, 3)], [(2.0, 1), (8.0, 3)], [(11.0, 1), (2.0, 3)],
        #  [(3.0, 1), (9.0, 3)], [(12.0, 1), (3.0, 3)], [(4.0, 1), (10.0, 3)],
        #  [(13.0, 1), (4.0, 3)], [(5.0, 1), (11.0, 3)], [(14.0, 1), (5.0, 3)],
        #  [(15.0, 1), (6.0, 3)], [(7.0, 1), (23.0, 3)], [(8.0, 1), (24.0, 3)],
        #[(17.0, 1), (13.0, 3)], [(18.0, 1), (14.0, 3)], [(19.0, 1), (15.0, 3)],
        #[(20.0, 1), (16.0, 3)], [(21.0, 1), (17.0, 3)], [(22.0, 1), (18.0, 3)],
        #  [(23.0, 1), (19.0, 3)], [(24.0, 1), (20.0, 3)]]
        #old = [[(int(val[0])-1)%25 for val in region] for region in old]
        #for r in old:
        #    stderr.write(str(r)+'\n')
        ladder = [[1, 7, 18, 24, 5, 16, 12, 8, 19, 25, 6, 17, 23, 4, 15, 11, 22, 3, 14, 10, 21, 2, 13, 9, 20],
                  [25, 6, 17, 23, 4, 15, 11, 7, 18, 24, 5, 16, 22, 3, 14, 10, 21, 2, 13, 9, 20, 1, 12, 8, 19]]
        curvepair = CurvePair(ladder, compute=False)
        test_polygons = curvepair.concise_boundaries.values()

        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    '''
    These tests are based on MICC Beta edges() output, and not done by hand
    '''

    def test_4_polygons(self):
        # polygonal boundaries (reference curve only)
        old =  [[(0.0, 3), (15.0, 1), (11.0, 1)], [(8.0, 3), (0.0, 1)],
                [(1.0, 1), (9.0, 3)], [(12.0, 1), (1.0, 3)],
                [(2.0, 1), (10.0, 3)], [(13.0, 1), (15.0, 3), (2.0, 3)],
                [(3.0, 1), (11.0, 3)], [(14.0, 3), (6.0, 1), (3.0, 3)],
                [(4.0, 1), (12.0, 3)], [(7.0, 1), (4.0, 3)],
                [(5.0, 1), (13.0, 3)], [(8.0, 1), (5.0, 3)],
                [(9.0, 1), (6.0, 3)], [(10.0, 1), (16.0, 1), (7.0, 3)],
                [(14.0, 1), (16.0, 3)]]
        true_polygons = [[(int(val[0])-1)%17 for val in region] for region in old]
        test = CurvePair(
            [[6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 1, 2, 3, 4, 17, 5],
             [1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4]],
            compute=False)
        test_polygons = test.concise_boundaries.values()
        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    def test_5_polygons(self):
        # polygonal boundaries (reference curve only)
        old = [[(0.0, 3), (5.0, 1), (1.0, 1)], [(6.0, 1), (7.0, 3), (0.0, 1)],
               [(2.0, 1), (1.0, 3)], [(3.0, 1), (15.0, 3), (2.0, 3)],
               [(14.0, 3), (13.0, 1), (3.0, 3)], [(4.0, 1), (16.0, 3)],
               [(14.0, 1), (4.0, 3)], [(15.0, 1), (5.0, 3)],
               [(16.0, 1), (6.0, 3)], [(7.0, 1), (8.0, 3)],
               [(8.0, 1), (9.0, 3)], [(9.0, 1), (10.0, 3)],
               [(10.0, 1), (11.0, 3)], [(11.0, 1), (12.0, 3)],
               [(12.0, 1), (13.0, 3)]]
        test = CurvePair(
            [[17, 1, 2, 3, 4, 17, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16],
            [1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4]],
            compute=False)
        true_polygons = [[(int(val[0])-1)%17 for val in region] for region in old]
        test_polygons = test.concise_boundaries.values()
        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    def test_6_polygons(self):
        # polygonal boundaries (reference curve only)
        old =  [[(0.0, 3), (7.0, 1), (2.0, 1)], [(9.0, 1), (0.0, 1)],
                [(1.0, 1), (8.0, 1)], [(3.0, 1), (1.0, 3)],
                [(4.0, 1), (2.0, 3)], [(5.0, 1), (3.0, 3)],
                [(6.0, 1), (23.0, 3), (4.0, 3)],
                [(22.0, 3), (5.0, 3)], [(21.0, 3), (6.0, 3)],
                [(20.0, 3), (23.0, 1), (10.0, 1), (7.0, 3)],
                [(11.0, 1), (8.0, 3)], [(12.0, 1), (9.0, 3)],
                [(13.0, 1), (10.0, 3)], [(14.0, 1), (11.0, 3)],
                [(15.0, 1), (12.0, 3)], [(16.0, 1), (13.0, 3)],
                [(17.0, 1), (14.0, 3)], [(18.0, 1), (15.0, 3)],
                [(19.0, 1), (16.0, 3)], [(20.0, 1), (17.0, 3)],
                [(21.0, 1), (18.0, 3)], [(22.0, 1), (19.0, 3)]]
        test = CurvePair(
            [[23, 22, 1, 2, 3, 4, 5, 22, 23, 24, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24],
             [1, 2, 3, 4, 19, 20, 21, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 20, 19, 5]],
            compute=False)
        true_polygons = [[(int(val[0])-1)%24 for val in region] for region in old]
        test_polygons = test.concise_boundaries.values()
        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)
    def test_7_polygons(self):
        # polygonal boundaries (reference curve only)
        old =  [[(0.0, 3), (13.0, 1), (4.0, 1)], [(6.0, 3), (0.0, 1)],
                [(1.0, 1), (16.0, 1), (7.0, 3)], [(5.0, 1), (1.0, 3)],
                [(2.0, 1), (15.0, 1)], [(6.0, 1), (14.0, 3), (2.0, 3)],
                [(3.0, 1), (14.0, 1)], [(13.0, 3), (3.0, 3)],
                [(12.0, 3), (4.0, 3)], [(11.0, 3), (20.0, 1), (5.0, 3)],
                [(7.0, 1), (15.0, 3)], [(8.0, 1), (16.0, 3)],
                [(17.0, 1), (8.0, 3)], [(9.0, 1), (17.0, 3)],
                [(18.0, 1), (9.0, 3)], [(10.0, 1), (18.0, 3)],
                [(19.0, 1), (10.0, 3)], [(11.0, 1), (19.0, 3)],
                [(12.0, 1), (20.0, 3)]]

        test = CurvePair([[18, 21, 20, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 10, 11, 12, 13, 17],
                          [1, 2, 14, 15, 16, 17, 18, 10, 11, 12, 13, 16, 15, 14, 3, 4, 5, 6, 7, 8, 9]],
                         compute=False)
        true_polygons = [[(int(val[0])-1)%21 for val in region] for region in old]
        test_polygons = test.concise_boundaries.values()
        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    def test_8_polygons(self):
        # polygonal boundaries (reference curve only)
        old = [[(0.0, 3), (9.0, 1), (22.0, 1), (13.0, 3)],
               [(15.0, 3), (0.0, 1)], [(1.0, 1), (16.0, 3)],
               [(12.0, 3), (1.0, 3)], [(2.0, 1), (17.0, 3)],
               [(11.0, 3), (2.0, 3)], [(3.0, 1), (18.0, 3)],
               [(10.0, 3), (13.0, 1), (3.0, 3)], [(4.0, 1), (19.0, 3)],
               [(14.0, 1), (4.0, 3)], [(5.0, 1), (20.0, 3)],
               [(15.0, 1), (5.0, 3)], [(6.0, 1), (21.0, 3)],
               [(16.0, 1), (6.0, 3)], [(7.0, 1), (22.0, 3)],
               [(17.0, 1), (7.0, 3)], [(8.0, 1), (23.0, 3)],
               [(18.0, 1), (8.0, 3)], [(19.0, 1), (12.0, 1), (9.0, 3)],
               [(10.0, 1), (21.0, 1)], [(11.0, 1), (20.0, 1)],
               [(23.0, 1), (14.0, 3)]]

        test = CurvePair([[3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 24, 12, 16, 17, 18, 19, 20, 21, 24, 23, 22, 1, 2],
                          [13, 14, 15, 16, 17, 18, 19, 20, 21, 12, 15, 14, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
                         compute=False)

        true_polygons = [[(int(val[0])-1)%24 for val in region] for region in old]
        test_polygons = test.concise_boundaries.values()
        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)

    def test_9_polygons(self):
        # polygonal boundaries (reference curve only)
        old = [[(0.0, 3), (11.0, 1), (2.0, 1)], [(13.0, 1), (0.0, 1)],
               [(1.0, 1), (12.0, 1)], [(3.0, 1), (1.0, 3)],
               [(4.0, 1), (2.0, 3)], [(5.0, 1), (3.0, 3)],
               [(6.0, 1), (4.0, 3)], [(7.0, 1), (5.0, 3)],
               [(8.0, 1), (6.0, 3)], [(9.0, 1), (7.0, 3)],
               [(10.0, 1), (20.0, 3), (8.0, 3)], [(19.0, 3), (9.0, 3)],
               [(18.0, 3), (10.0, 3)], [(17.0, 3), (11.0, 3)],
               [(16.0, 3), (19.0, 1), (15.0, 1), (12.0, 3)],
               [(16.0, 1), (13.0, 3)], [(14.0, 1), (20.0, 1)],
               [(17.0, 1), (14.0, 3)], [(18.0, 1), (15.0, 3)]]
        test = CurvePair([[19, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19, 20, 21, 14, 15, 16, 17, 21, 20],
                         [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 13, 12, 11, 10, 9]],
                         compute=False)
        true_polygons = [[(int(val[0])-1)%21 for val in region] for region in old]
        test_polygons = test.concise_boundaries.values()
        valid = self.check_valid(true_polygons, test_polygons)
        self.assertTrue(valid)
if __name__ == '__main__':
    unittest.main()