import unittest
from micc.curves import RigidGraph
from sys import stderr
from micc.curves import  CurvePair


class DualGraphTests(unittest.TestCase):
    """
    Verify that correctness of the new polygonal boundary algorithm. Using
    polygons determined by hand to validate correctness
    """
    def check_valid(self, true_dual_graph, test_dual_graph):
        correct = True

        if len(true_dual_graph) != len(test_dual_graph):
            return False
        if set(true_dual_graph.keys()) != set(test_dual_graph.keys()):
            return False

        for vertex in test_dual_graph.keys():
            true_adj_list = true_dual_graph[vertex]
            test_adj_list = test_dual_graph[vertex]
            equal_adj_lists = set(true_adj_list) == set(test_adj_list)
            if not equal_adj_lists:
                stderr.write('broke it: '+str(vertex)+'\n')
            correct &= equal_adj_lists
        return correct

    def test_1_dual_graph(self):
        true_polygons = [[0, 4, 5], [7, 3, 2], [6, 11, 10], [9, 8, 1], [5, 11],
                         [3, 4], [6, 1], [10, 9], [0, 7], [8, 2]]


        true_dual_graph = {0: [4, 5, 7], 1: [6, 8, 9], 2: [3, 7, 8],
                           3: [2, 4, 7], 4: [3, 0, 5], 5: [0, 4, 11],
                           6: [1, 11, 10], 7: [2, 3, 0], 8: [1, 2, 9],
                           9: [1, 8, 10], 10: [6, 9, 11], 11: [5, 6, 10],
                           }


        ladder = [[12, 5, 6, 9, 10, 12, 11, 1, 2, 3, 4, 11],
                  [1, 2, 7, 8, 9, 10, 5, 6, 8, 7, 3, 4]]
        curvepair = CurvePair(ladder, compute=False)
        test_dual_graph = curvepair.graph.dual_graph

        valid = self.check_valid(true_dual_graph, test_dual_graph)
        self.assertTrue(valid)

    def test_2_dual_graph(self):
        true_dual_graph = {0: [5, 4, 3], 1: [2, 3, 4], 2: [1, 3, 2],
                           3: [0, 1, 2], 4: [0, 1, 5], 5: [0, 4, 5],
                           }

        ladder = [[1, 4, 3, 4, 1, 6], [6, 5, 2, 3, 2, 5]]
        curvepair = CurvePair(ladder, compute=False)
        test_dual_graph = curvepair.graph.dual_graph

        valid = self.check_valid(true_dual_graph, test_dual_graph)
        self.assertTrue(valid)

    def test_3_dual_graph(self):

        true_dual_graph = {0: [8, 11], 1: [12, 14, 9], 2: [13, 5, 10],
                           3: [6, 11], 4: [7, 12], 5: [2, 13, 8],
                           6: [3, 9, 15], 7: [16, 4], 8: [5, 0],
                           9: [1, 6, 15], 10: [14, 16, 2], 11: [3, 0],
                           12: [1, 14, 4], 13: [15, 2, 5], 14: [16, 10, 1, 12],
                           15: [13, 6, 9], 16: [10, 14, 7],
                           }

        ladder =[[6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 1, 2, 3, 4, 17, 5],
                 [1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4]]
        curvepair = CurvePair(ladder, compute=False)
        test_dual_graph = curvepair.graph.dual_graph

        valid = self.check_valid(true_dual_graph, test_dual_graph)
        self.assertTrue(valid)

    def test_hempel_dual_graph(self):
        true_dual_graph = {0: [6, 15, 11, 5, 21, 9], 1: [7, 10], 2: [11, 8],
                           3: [12, 9], 4: [13, 10], 5: [0, 6, 15, 11, 21, 14],
                           6: [0, 5, 15, 11, 21, 22], 7: [1, 23], 8: [24, 2],
                           9: [3, 0], 10: [1, 4], 11: [0, 6, 15, 5, 21, 2],
                           12: [16, 3], 13: [17, 4], 14: [18, 5],
                           15: [0, 6, 11, 5, 21, 19], 16: [12, 20],
                           17: [13, 21], 18: [14, 22], 19: [15, 23],
                           20: [24, 16], 21: [0, 6, 15, 11, 5, 17], 22: [18, 6],
                           23: [7, 19], 24: [8, 20]}
        ladder = [[1, 7, 18, 24, 5, 16, 12, 8, 19, 25, 6, 17, 23, 4, 15, 11, 22, 3, 14, 10, 21, 2, 13, 9, 20],
                  [25, 6, 17, 23, 4, 15, 11, 7, 18, 24, 5, 16, 22, 3, 14, 10, 21, 2, 13, 9, 20, 1, 12, 8, 19]]
        curvepair = CurvePair(ladder, compute=False)
        test_dual_graph = curvepair.graph.dual_graph

        valid = self.check_valid(true_dual_graph, test_dual_graph)
        self.assertTrue(valid)

    def test_dual_graph_repeat_is_2(self):
        true_dual_graph = {0+0j: [5+0j, 5+1j, 4+0j, 4+1j, 3+0j, 0+1j],
                           0+1j: [5+0j, 5+1j, 4+0j, 4+1j, 3+1j, 0+0j],
                           1+0j: [2+0j, 2+1j, 3+0j, 3+1j, 4+0j, 1+1j],
                           1+1j: [2+0j, 2+1j, 3+0j, 3+1j, 4+1j, 1+0j],
                           2+0j: [1+0j, 1+1j, 3+0j, 3+1j, 2+0j, 2+1j],
                           2+1j: [1+0j, 1+1j, 3+0j, 3+1j, 2+0j, 2+1j],
                           3+0j: [0+0j, 1+0j, 1+1j, 2+0j, 2+1j, 3+1j],
                           3+1j: [0+1j, 1+0j, 1+1j, 2+0j, 2+1j, 3+0j],
                           4+0j: [0+0j, 0+1j, 1+0j, 5+0j, 5+1j, 4+1j],
                           4+1j: [0+0j, 0+1j, 1+1j, 5+0j, 5+1j, 4+0j],
                           5+0j: [0+0j, 0+1j, 4+0j, 4+1j, 5+0j, 5+1j],
                           5+1j: [0+0j, 0+1j, 4+0j, 4+1j, 5+0j, 5+1j]
                       }
        ladder = [[1, 4, 3, 4, 1, 6], [6, 5, 2, 3, 2, 5]]
        curvepair = CurvePair(ladder, compute=False)
        test_dual_graph, _ = curvepair.graph.create_dual_graph(
            curvepair.concise_boundaries, repeat=2)
        valid = self.check_valid(true_dual_graph, test_dual_graph)
        self.assertTrue(valid)
    # TODO at least two more curves by hand

if __name__ == '__main__':
    unittest.main()