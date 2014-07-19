import numpy as np
from curves import fix_matrix_signs, boundary_count, genus, ladder_convert, vector_solution, edges, Three
from graph import Graph

class CurvePair:
    '''
    ladder = None
    beta = None
    top = []
    bottom = []
    n = 0
    matrix = None
    boundaries = None
    genus = None
    edges = None

    solution = None
    distance = 0
    loops = []
    '''
    def __init__(self, top_beta, bottom_beta, dist=1, graph=1, conjectured_dist=3):

        is_ladder = lambda top, bottom: not (0 in top or 0 in bottom)

        if is_ladder(top_beta,bottom_beta):
            self.ladder = [top_beta, bottom_beta]
        else:
            self.ladder = None
        


        if is_ladder(top_beta, bottom_beta):
            self.beta = ladder_convert(top_beta, bottom_beta)
            self.top = self.beta[0]
            self.bottom = self.beta[1]
        else:
            self.top = top_beta
            self.bottom = bottom_beta
            self.beta = [self.top, self.bottom]

        self.n = len(self.top)

        self.matrix = np.zeros((2,self.n,4))
        self.matrix[0,:,0] = [self.n-1] + range(self.n-1)
        self.matrix[0,:,1] = self.top
        self.matrix[0,:,2] = range(1,self.n) +[0]
        self.matrix[0,:,3] = self.bottom

        self.matrix = fix_matrix_signs(self.matrix)

        self.boundaries = boundary_count(self.matrix)
        self.genus = genus(self.matrix)
        self.edges = edges(self.matrix)

        self.solution = vector_solution(self.edges[0])


        if graph is 1:
            graph = Graph(self.edges, rep_num=conjectured_dist-2)
            graph.compute_loops(self.n, self.genus)
            self.loops = graph.gammas
        else:
            self.loops = []

        if dist is 1:
            #from sys import stderr
            #stderr.write(str(self.loops)+'\n')
            self.distance, self.loop_matrices = self.compute_distance(self.matrix, self.loops)
        else:
            self.distance = None

    def __repr__(self):
        return self.ladder[0]+'\n'+self.ladder[1]+'\n'

    def compute_distance(self, M, all_paths):
        '''
        :param M: the matrix
        :type M:
        :param all_paths:
        :type all_paths:
        :returns: dist: the distance if three/four, or 'Higher' if dist is > 4.

        Computes the distance between the two curves embedded in the matrix.
        If this distance is three, tries to use simple paths to extend the distance
        in a different direction. If this fails, simply returns three;
        else it prints a curve that is distance four from alpha.

        '''

        dist_is_three, lib = Three(M, all_paths)
        #stderr.write(str(lib)+'\n')
        #print len(lib.keys())
        #print self.solution, len(self.matrix[0])
        dist = 3 if dist_is_three  else 'at least 4!'
        if dist == 3:
            return dist, lib
        else:
            geodesic_distances = []
            from sys import stderr
            for k, matrix in lib.iteritems():
                #stderr.write(str(matrix))
                if np.array_equal(matrix, self.matrix):
                    continue
                elif self.solution == CurvePair(matrix[0, :, 1], matrix[0, :, 3],0,0).solution \
                        and len(self.matrix[0]) == len(matrix[0]):
                    continue
                cc = CurvePair(matrix[0, :, 1], matrix[0, :, 3])
                stderr.write(str(k)+": "+str(cc.distance)+'\n')
                geodesic_distances.append(cc.distance)
                #print 'computed curve',k,'!'
            #print '\n'
            return min(set(geodesic_distances)) + 1, lib
        #return dist, lib
