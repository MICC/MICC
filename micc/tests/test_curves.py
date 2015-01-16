__author__ = 'Matt'

import unittest
from micc.curves import RigidGraph
from sys import stderr


class CurveTests(unittest.TestCase):

    def test_rigid_graph1(self):
        cycle = '4+2-3-5-1+'
        ladder = [[1, 2, 3, 2, 4], [5, 3, 4, 1, 5]]
        graphical_rep_ladder = RigidGraph(ladder=ladder)
        graphical_rep_cycle = RigidGraph(cycle=cycle)
        for k, v in graphical_rep_ladder.graph.iteritems():
            stderr.write(str(k)+': '+str(v)+'\n')
        graphical_rep_ladder.determine_boundaries()
        self.assertEqual(graphical_rep_ladder.graph, graphical_rep_cycle.graph)

if __name__ == '__main__':
    unittest.main()