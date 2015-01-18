__author__ = 'Matt'

import unittest
from micc.curves import RigidGraph
from sys import stderr
from micc.curves import  CurvePair


class CurveTests(unittest.TestCase):

    def test_rigid_graph1(self):
        cycle = '4+2-3-5-1+'
        ladder = [[1, 2, 3, 2, 4], [5, 3, 4, 1, 5]]
        curvepair = CurvePair(ladder)
        for boundary in curvepair.concise_boundaries.itervalues():
            stderr.write(str(boundary)+'\n')
        stderr.write(str(curvepair.genus)+'\n')
        #graphical_rep_ladder = RigidGraph(ladder=ladder)
        #graphical_rep_cycle = RigidGraph(cycle=cycle)
        #for k, v in graphical_rep_ladder.graph.iteritems():
        #    stderr.write(str(k)+': '+str(v)+'\n')
        #for boundary in graphical_rep_ladder.determine_boundaries():
        #    stderr.write(str(boundary)+'\n')
        self.assertTrue(True)
        #self.assertEqual(graphical_rep_ladder.graph, graphical_rep_cycle.graph)
    '''
    def test_curvepair(self):
        # Birman Distance 3
        ladder = [[1, 4, 3, 4, 1, 6], [6, 5, 2, 3, 2, 5]]
        curvepair = CurvePair(ladder, compute=False)
        for k, v in curvepair.rigid_graph.graph.iteritems():
            stderr.write(str(k)+': '+str(v)+'\n')
        #for boundary in curvepair.verbose_boundaries:
        #    stderr.write(str(boundary)+'\n')
        for boundary in curvepair.concise_boundaries.values():
            stderr.write(str(boundary)+'\n')
        for k, v in curvepair.graph.dual_graph.iteritems():
            stderr.write(str(k)+': '+str(v)+'\n')
        self.assertTrue(True)
    def test_curvepair_2(self):
        # d = 4, g = 2, i = 12
        ladder = [[12, 5, 6, 9, 10, 12, 11, 1, 2, 3, 4, 11],
                  [1, 2, 7, 8, 9, 10, 5, 6, 8, 7, 3, 4]]
        curvepair = CurvePair(ladder, compute=False)
        for k, v in curvepair.rigid_graph.graph.iteritems():
            stderr.write(str(k)+': '+str(v)+'\n')
        #for boundary in curvepair.verbose_boundaries:
        #    stderr.write(str(boundary)+'\n')
        for boundary in curvepair.concise_boundaries.values():
            stderr.write(str(boundary)+'\n')
        for k, v in curvepair.graph.dual_graph.iteritems():
            stderr.write(str(k)+': '+str(v)+'\n')
        self.assertTrue(True)
    '''
if __name__ == '__main__':
    unittest.main()