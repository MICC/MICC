import numpy as np
from curves import fixMatrixSigns, boundaryCount, genus, ladderConvert, vectorSolution, edges, Three
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
    def __init__(self,topBeta,bottomBeta, dist=1, graph=1,conjectured_dist=3):

        is_ladder = lambda top, bottom: not (0 in top or 0 in bottom)

        if is_ladder(topBeta,bottomBeta):
            self.ladder = [topBeta, bottomBeta]
        else:
            self.ladder = None
        


        if is_ladder(topBeta, bottomBeta):
            self.beta = ladderConvert(topBeta, bottomBeta)
            self.top = self.beta[0]
            self.bottom = self.beta[1]
        else:
            self.top = topBeta
            self.bottom = bottomBeta
            self.beta = [self.top, self.bottom]

        self.n = len(self.top)

        self.matrix = np.zeros((2,self.n,4))
        self.matrix[0,:,0] = [self.n-1] + range(self.n-1)
        self.matrix[0,:,1] = self.top
        self.matrix[0,:,2] = range(1,self.n) +[0]
        self.matrix[0,:,3] = self.bottom

        self.matrix = fixMatrixSigns(self.matrix)

        self.boundaries = boundaryCount(self.matrix)
        self.genus = genus(self.matrix)
        self.edges = edges(self.matrix)

        self.solution = vectorSolution(self.edges[0])


        if graph is 1:
            self.loops = Graph(self, self.edges, rep_num=conjectured_dist-1).gammas
        else:
            self.loops = []

        if dist is 1:
            #from sys import stderr
            #stderr.write(str(self.loops)+'\n')
            self.distance, self.loopMatrices = self.compute_distance(self.matrix, self.loops)
        else:
            self.distance = None

    def __repr__(self):
        return self.ladder[0]+'\n'+self.ladder[1]+'\n'

    def compute_distance(self, M, allPaths):
        '''
        :param M: the matrix
        :type M:
        :param allPaths:
        :type allPaths:
        :returns: dist: the distance if three/four, or 'Higher' if dist is > 4.

        Computes the distance between the two curves embedded in the matrix.
        If this distance is three, tries to use simple paths to extend the distance
        in a different direction. If this fails, simply returns three;
        else it prints a curve that is distance four from alpha.

        '''

        distIsThree, Lib = Three(M, allPaths)
        #stderr.write(str(Lib)+'\n')

        dist = 3 if distIsThree  else 'at least 4!'
        return dist, Lib
        '''
        from sys import stderr
        if dist == 3:
            return dist, Lib
        else:
            geodesic_distances = []
            for k,matrix in Lib.iteritems():
                #stderr.write(str(matrix))
                cc_distance = CurvePair(matrix[0, :, 1], matrix[0, :, 3])
                stderr.write(str(k)+": "+str(cc_distance.distance)+'\n')
                geodesic_distances.append(cc_distance.distance)
            return min(set(geodesic_distances)) + 1, Lib
        '''