# Paul Glenn
# curves.py
# Give curve pairs class structure in preparation for public access
import numpy as np
from itertools import product
from copy import deepcopy


def fixMatrixSigns(M):
    '''
    :param M: The matrix with incorrect signs.
    :type M: numpy.array(dtype=float64) of shape (2,n,4)
    :returns: The matrix of shape (2,n,4) with appropriate signs.

    Given a matrix corresponding to a curve pair,
    fixes the signs so that the matrix can be traversed by function bdycount.
    Starts from the simples point, then assigns based on matching values and
    ENS: No qualitative results about the curve pair will be changed by this,
    since orientation is just a formality.

    '''

    row, col = [-1,1]
    sgn = 1
    #for ii in range(len(M[0,:])):
    while 0 in M[1,:,1:4:2] and row < M.shape[1]-1:
        row += 1
        sgn = -sgn
        # Starting position and sign are arbitrary.
        for i in range(M.shape[1]):
            gval = int(M[0,row,col])
            new = np.array( np.where(M[0,gval,:]==row) )
            #print  np.where(new%2==col%2)
            ind = int( new[ np.where(new%2==col%2)] )
            sgn = -sgn
            M[1,gval,ind] = sgn
            ind += 2
            ind = ind % M.shape[2]
            sgn = -sgn
            M[1,gval,ind] = sgn
            row = gval
            col = ind
        M[1,:,0] = -1
        M[1,:,2] = 1
    return M


def concatenateMatrix(M1, M2):
    '''
    :param M1: First matrix to combine
    :type M1: numpy.array(dtype=float64) of shape (2,n,4)
    :param M2: Second matrix to combine
    :type M2: numpy.array(dtype=float64) of shape (2,m,4)
    :returns: final matrix

    Note that n does not have to equal m in general.
    Combines two curve pairs into one.
    REQ: curve pairs are, of course, encoded as matrices.
    INV: new matrix signs will be correct
    INV: final matrix will correspond to a single curve.

    '''
    topM = np.copy(M1)
    botM = np.copy(M2)
    n = M1.shape[1]
    m = M2.shape[1]
    topM[0,0,0] = n + m - 1
    topM[0,n-1,2] = n
    botM[0] += n * np.ones(M2[0].shape)
    botM[0,m-1,2] = 0
    botM[0,0,0] = n-1
    newM = np.vstack((topM[0], botM[0])) #  Multi-curve
    newM = np.vstack((newM, np.vstack((topM[1], botM[1])) )).reshape((2,n+m,4))

    left = zip(np.where(topM[0] == n-1)[0], np.where(topM[0] == n-1)[1])
    i0, j0 = 0,0
    for pair in left:
        i, j = pair
        if j%2 == 1: i0,j0 = i,j
    newM[0,i0,j0] = n

    tempInd = np.where(newM[0,n-1,:] == i0)[0]
    tempInd = int(tempInd[tempInd%2==1][0])
    tempVal = newM[0, n-1, tempInd]
    switchVal = newM[0, n, tempInd]
    newM[0, n-1, tempInd] = switchVal
    newM[0,n,tempInd] = tempVal
    returnInd = newM[0,switchVal,:] == n
    returnInd[0:3:2] = False
    newM[0,switchVal,returnInd==True] = n-1

    return fixMatrixSigns(newM)

## Path finding methods -- used to find edge paths in the complement of beta-curve.
## Most are helpers for findAllPaths below.
def visited(curr, face , path):
    '''
    :param curr:
    :type curr:
    :param face:
    :type face:
    :param path:
    :type path:


    '''

    myface = list(face)
    myface.remove(curr)
    v = 0
    for edge in myface:
        if edge in path: v = 1
    return v

def isUnique(path, AllPaths):
    '''
    :param path:
    :type path:
    :param AllPaths:
    :type AllPaths:
    :returns: boolean

    '''
    return not ( path in AllPaths)

def shift(path):
    '''
    Reorders path with a minimum intersection as the base point.
    This is useful for determining path uniqueness.

    :param path: list of intersections representing the current path
    :type path: list
    :returns: list with intersections shifted in order such that the lowest intersection is in the first position.


    '''
    temp = path.index(min(path))
    return path[temp:] + path[:temp]

def invert(path):
    '''
    :param path: some path
    :type path: list
    :returns: the inverted path

    '''
    return shift(path[::-1])

def pathFinishedSingle(edge, face, path):
    '''
    :param edge:
    :type edge:
    :param face:
    :type face:
    :param path:
    :type path:
    :returns:

    '''
    C  = len(path)>2 and (edge == path[-1])
    faceL = list(face)
    faceL.remove(edge)
    C = C and not set(faceL).isdisjoint(set(path))
    #meeting = lambda e1, e2, face: e1 in face and e2 in face
    return C

def pathFinishedDouble(edge, face, path):
    '''
    :param edge:
    :type edge:
    :param face:
    :type face:
    :param path:
    :type path:
    :returns:

    '''

    C  = len(path)>2 and (edge == path[-1])
    faceL = list(face)
    faceL.remove(edge)
    C = C and not set(faceL).isdisjoint(set(path))
    if path.count(edge) != 2 : C = 0
    #meeting = lambda e1, e2, face: e1 in face and e2 in face
    return C

def findNewPaths(currentPath, myface, faces, AllPaths, pathFunction):
    '''
    :param currentPath:
    :type currentPath:
    :param myface:
    :type myface:
    :param faces:
    :type faces:
    :param AllPaths:
    :type AllPaths:
    :param pathFunction:
    :type pathFunction:
    :returns:

    '''



    start = currentPath[0]
    nextEdge = None
    subPath = []
    facesWithoutCurrentFace = list(faces)
    facesWithoutCurrentFace.remove(myface)
    for face in faces: #Check all faces...
        if start in face and face is not myface:
            #if we find the start edge in another face...
            for otherface in facesWithoutCurrentFace:
                #go through its edges...
                if start in otherface: # needed?
                    faceWithoutStart = list(otherface)
                    faceWithoutStart.remove(start)

                    for nonStartingEdge in faceWithoutStart:
                        # try all edges in that path
                        nextEdge = nonStartingEdge
                        if  (not visited( start, otherface, currentPath ) ) and\
                            (nextEdge not in myface):
                            subPath = [nextEdge]
                            subPath.extend(currentPath)

                            # Recursive call to take all possible directions
                            findNewPaths(subPath, otherface, faces, AllPaths, pathFunction)

                        elif pathFunction(nextEdge, otherface, currentPath):
                            newFoundPath = shift(currentPath)
                            invertedPath = invert(currentPath)
                            unique = lambda path: isUnique(path, AllPaths)
                            if unique(newFoundPath) and unique(invertedPath):
                                AllPaths.append(newFoundPath)
                    faceWithoutStart.append(start)
        facesWithoutCurrentFace.append(face)

def removeDuplicates(faces, AllPaths):
    '''
    :param faces:
    :type faces:
    :param AllPaths:
    :type AllPaths:
    :returns:

    '''
    paths = list(AllPaths)
    for path in paths:
        for f in faces:
            counter = 0
            for e in f:
                if e in path: counter += 1
            if counter == 3: AllPaths.remove(path)
    return AllPaths

def findAllPaths(faces):
    '''
    :param faces:
    :type faces:
    :returns:

    '''

    AllPaths = []
    forward = lambda path, face, pathFunction: findNewPaths(path, face, faces, AllPaths, pathFunction)
    for face in faces:
        for edge in face:
            forward([edge], face, pathFinishedSingle)
            #forward([edge], face, pathFinishedDouble)
    AllPaths = removeDuplicates(faces,AllPaths)
    return AllPaths

## Now that we have the paths, they need to be
## re- indexed so that matrices can be built from them.
def buildMatrices(edgePaths, AllPaths):
    '''
    :param edgePaths: list of face boundary orientations
    :type edgePaths: list
    :param AllPaths: list
    :type AllPaths: list

    Take the paths in the skeleton of the complement of the transverse curve
    And create matrices.
    '''
    #print 'edgePaths:',edgePaths
    #print 'AllPaths:',AllPaths[0].loops
    MasterList = []

    # Allow paths to be referenced by face
    #print edgePaths
    for itr in range(len(edgePaths)):
        edgePaths[itr] = dict(edgePaths[itr])

    orderedPaths , mappedPaths = [],[]

    # Rescale path 0-len(path) for matrix
    for Path in AllPaths:
        orderedPath = list(np.sort(Path))
        orderedPaths.append(orderedPath)
        mappedPaths.append(dict(zip(orderedPath,range(len(Path)))))

    #Create Matrices using details from edge paths.
    for Path,mappedPath in zip(AllPaths,mappedPaths):
        #Value Matrix
        pathSize = len(Path) ; shape =(pathSize,4)
        last = Path[len(Path)-1]
        M = np.zeros(shape);
        M2 = np.zeros(shape)
        M[:,0] = np.array([pathSize-1]+range(pathSize-1))
        M[:,2] = np.array(range(1,pathSize)+[0])
        pastEdges = dict(); futureEdges =  dict()
        old_vertex = last

        itr = 1
        for vertex in Path:
            flag = False
            for path in edgePaths:
                keys = path.keys()
                if vertex in keys and old_vertex in keys:
                    pastEdges[vertex] = path[vertex]
                    flag = True
                if vertex in keys and Path[itr%pathSize] in keys:
                    if flag:
                        futureEdges[vertex] = (path[vertex]+ 2) %4
                    else:
                        futureEdges[vertex] = path[vertex]
                flag =False
            old_vertex = vertex
            itr += 1
        old_vertex = last;

        itr = 1
        for vertex in Path:
            curr_vertex = vertex
            next_vertex = Path[itr%pathSize]
            M[mappedPath[vertex],pastEdges[vertex]] = mappedPath[old_vertex]
            M[mappedPath[vertex],futureEdges[vertex]] = mappedPath[next_vertex]
            old_vertex = curr_vertex
            itr += 1

        # Sign matrix: Stand-alone function
        M = fixMatrixSigns(np.array([M, M2], dtype=int))
        MasterList.append(M)
    # End while

    return MasterList

def faceParse(alphaEdges):
    '''
    :param alphaEdges: set of faces with alpha edges.
    :type alphaEdges:
    :returns: (Bridges, Isalnds, lengthCheck) Bridges : 4-sided regions; Islands : n > 4 - gons; lengthCheck : the number of alpha edges included.

    Separate set of all faces into bridges (4-sided regions) and
    islands (higher-sided regions) for distance calculator.

    '''
    Bridges, Islands = [],[]
    lengthCheck = []
    for pair in alphaEdges:
        if pair[0] == 4: Bridges.append(pair)
        else: Islands.append(pair)
        lengthCheck.extend(list(pair[1]))
    return Bridges, Islands, len(lengthCheck)

######					For Distance Extension 					#######
def connected(P1, P2):
    '''
    :param P1:
    :type P1:
    :param P2:
    :type P2:
    :returns:

    '''

    S1  = set(P1)
    S2 = set(P2)
    if S1.isdisjoint(S2): return 0
    elif S1.issubset(S2) and S1.issuperset(S2): return 0
    return 1

def shareEdge(path1, path2):
    '''
    :param path1:
    :type path1:
    :param path2:
    :type path2:
    :returns:

    '''
    share = 0
    if not set(path1).isdisjoint(path2): share = 1
    if share == 0: return 0, -1
    else:
        intersectionSet = set(path1) & set(path2)
        numshared = len(intersectionSet)
        if numshared != 1: return 0, -1 #Then they share too much!
        else:
            sharedItem = intersectionSet.pop()
    return 1,path2.index(sharedItem)

def findCombinedPaths( allPaths, MLibrary):
    '''
    :param allPaths:
    :type allPaths:
    :param MLibrary:
    :type MLibrary:
    :returns:
    '''
    ListofConnected = []
    pathLibrary = dict(zip(range(len(allPaths)),allPaths))
    index1 = 0
    for path1 in allPaths:
        index2 = 0
        for path2 in allPaths:
            if shareEdge(path1, path2)[0]:
                ListofConnected.append((MLibrary[index1],
                            MLibrary[index2]))
            index2 += 1
        index1 += 1

    return ListofConnected

def facesShareTwoEdges(faces):
    '''
    :param faces:
    :type faces:
    :returns:

    '''

    distanceThreeFlag = 0

    for face in faces:
        facesWithoutFace = list(faces)
        facesWithoutFace.remove(face)
        for other in facesWithoutFace:
            if len(set(face) & set(other)) >1: distanceThreeFlag = 1

    return distanceThreeFlag

def edges(M):
    '''
    :param M: the matrix representing a pair of curves
    :type M: numpy.array(dtype=float64) of shape (2,n, 4)
    :returns: (allFaces, edges) allFaces: tuple of faces including size and set of alpha-edges which bound them; edges: same as allFaces, exceincluding orientation of boundary edges.
    '''
    #print 'called edges'
    # The em list is needed to hold the tuples of (faceLength, faceEdges)
    allFaces = [ ]
    # INV: Number of faces found.
    faces = 0 ;

    numrows, numcols = M.shape[1:3] #num_rows, num_cols

    old_Vertices = [ ] ##list of previous paths
    bigonFlag = 0
    # Bigons are unwanted structures.
    Paths = [ ]  # alpha - edge paths.
    facesTemp = dict()
    for i,j in product(range(numrows),range(numcols)):
        #Set of edges associated with face
        tr = 1
        face = set()
        if faces==numrows: break #upper bound on possible no. faces
        # Start position in matrix: returning to this means a face has been
        # enclosed
        io =i
        jo=j

        found = 0 #exit condition
        # Number of edges for face. Keeps track of vector solution
        edges=0
        pathTemp = [];

        # Begin traversal
        while not found:

            gval = int(M[0,i,j]);
            #current value at index gives next vertex/row
            #value check
            arr1 = np.where(M[0,gval,:] == i%numrows)
            arr2 = np.where(M[1,gval,:] != M[1,i,j]) #sign check

            i_next = gval
            alpha = (M[0,i,0]+1)%numrows
            i= i_next

            new = np.intersect1d(arr1[0],arr2[0],assume_unique=True)
            #ENS: val and sign correct
            ind = np.where(new%2==j%2) #ENS: beta->beta, alpha->alpha
            j_next = (int(new[ind])+1)%numcols #Always move clockwise
            j_old = j
            j = j_next
            if (i,j) in old_Vertices:
                break
            old_Vertices.append((i,j))
            edges += 1

            alpha_new =( M[0,i,0]+1) % numrows;
            shift = (alpha_new - alpha)

            if shift==1 and j%2 == 1:
                face.add(alpha_new)
                pathTemp.append((alpha_new,(j)%numcols))
            elif shift==-1 and j%2==1:
                face.add(alpha)
                pathTemp.append((alpha,(j)%numcols))
            elif shift == numrows-1 and alpha ==0  and j%2==1: #
                face.add(alpha)
                pathTemp.append((alpha,(j)%numcols))
            elif shift == 1-numrows and alpha_new==0 and j%2==1: #
                face.add(alpha_new)
                pathTemp.append((alpha_new,(j)%numcols))

            if (i,j)==(io,jo):
                facesTemp[edges] = facesTemp.get(edges,0) +1
                if edges==2:
                    bigonFlag = 1
                if not bigonFlag:
                    Paths.append(pathTemp)
                    faces += 1
                    found = 1
                    allFaces.append((edges,face))
    return allFaces, Paths

def boundaryCount(M):
    '''
    :param M: the matrix representing a pair of curves
    :type M: numpy.array(dtype=float64) of shape (2,n, 4)
    :returns: (faces, bigon) faces: number of faces; bigon: 1 iff a bigon is found

    'Simply' count the number of faces bounded by two filling curves on a surface.
    Curves must be encoded as matrix according to vertices of intersection and
    associated orientation.


    '''

    faces = 0
    numrows, numcols = M.shape[1:3] #num_rows, num_cols
    oldEdges = [ ] ##list of previous edge paths
    bigonFlag =0

    for i,j in product(range(numrows),range(numcols)):
        # upper bound on possible no. faces is number of vertices
        if faces==numrows: break

        io =i
        jo=j
        #First matrix element; will go to vertex M[0,i,j]

        found = 0 #Exit condition
        pathLength=0 # Keep track of path length

        while not found:
            gval = int(M[0,i,j]);
            #current value at index gives next vertex/row

            arr1 = np.where(M[0,gval,:] == i%numrows) #value check
            arr2 = np.where(M[1,gval,:] == -M[1,i,j]); #sign should flip +/-
            i = gval # Go to next vertex/row
            new = np.intersect1d(arr1[0],arr2[0],assume_unique=True)
            #ENS: val and sign correct

            ind = np.where(new%2==j%2)
            #ENS: beta->beta, alpha->alpha

            j = (int(new[ind])+1)%numcols
            #Always move clockwise - to next edge

            if (i,j) in oldEdges: break
            oldEdges.append((i,j))
            # To save work and not go on old paths.
            # Also so we don't count faces twice...
            pathLength += 1
            # The path length is the number of edges traversed in the current face.

            if (i,j)==(io,jo):
                if pathLength==2: bigonFlag = 1;
                # Two edges to a face --> bigon
                faces += 1
                found = 1 #INV: found = 1 -> has found a bdy curve
    return faces, bigonFlag

def vectorSolution(edges):

    solution = dict()

    for face in edges:
        if face[0] not in solution.keys():
            solution[face[0]] = 1
        else:
            solution[face[0]] += 1

    return solution

def genus(M, euler=0, boundaries = 0):
    '''
    :param M: the matrix representing a pair of curves
    :type M: numpy.array(dtype=float64) of shape (2,n, 4)
    :param euler: 0 if euler characteristic not needed, else 1
    :type euler: int

    :returns: (g,X): g: genus; X : euler characteristic

    Compute the genus of the span of a curve pair, i.e. the minimal genus surface
    on which they fill.

    '''
    V = M.shape[1] # vertices
    P, bigon = boundaryCount(M) # Polygons in the complement of curve pair
    #if bigon is 1: P -= 1 # pull away one bigon
    #Euler characteristic (since edges = 2*vertices) is P - V
    # originally X = V-E+P
    X = P-V+boundaries

    # genus = 1 - 0.5*euler_characteristic
    Genus =  (2-X)/2
    returnVal = dict([(0,Genus),(1,(Genus,X))])
    # For return purposes only.

    if bigon: Genus -= 1 # Bigons steal genus; this gives it back.

    return returnVal[euler]


def testCollection(matrixList, originalGenus):
    '''
    :param matrixList:
    :type matrixList: list of numpy.array(dtype=float64)
    :param originalGenus:
    :type originalGenus:
    :returns: list of booleans and a dictionary containing the matrices without bigons.

    Test a collection of matrices to see if they fill a given genus.
    Note: Matrices with bigons are still counted in the calculation, but are not returned.
    '''
    genusCollection = []
    index = 0
    matrixLibrary = dict()
    for M in matrixList:

        bigon = boundaryCount(M)[1]
        Genus = genus(M)
        matrixLibrary[index] = M
        index += 1
        if bigon:
            Genus  = 0 # If it has a bigon, it should automatically fail

        genusCollection.append(Genus)
    # Test if all fill and thus distance g.t. 3
    test = [Genus == originalGenus for Genus in genusCollection]
    return test, matrixLibrary

def fourgonTest(F4, Fn):
    '''
    :param F4:
    :type F4:
    :param Fn:
    :type Fn:
    :returns: boolean: True if Fn is a 4-gon, False otherwise

    '''

    for islandFace in Fn:
        for bridgeFace in F4:
            if len(islandFace[1] & bridgeFace[1]) >= 2: return True
    return False

def Three(M, allPaths, ext=0):
    '''
    :param M:
    :type M:
    :param allPaths:
    :type allPaths:
    :param ext:
    :type exr:
    :returns: 1 if the curve pair is distance three and 0 otherwise.

    '''

    three, matrixLibrary = 0, dict()


    F0, bigon = boundaryCount(M)
    originalGenus = genus(M)
    #Calculate face alpha edges and alpha edge paths
    Faces, edgePaths = edges(M)
    Bridges, Islands, lengthCheck = faceParse(Faces)

    # Bridges are faces bounded by four edges.
    # Islands are bounded bt more than four, say six or eight...
    bridgeFaces = [list(face[1]) for face in Bridges]
    islandFaces = [list(face[1]) for face in Islands]

    ############# Quick and dirty checks for distance three #############
    if F0 == 1 :
        three = 1
    #	print '''The complement contains a polygon which shares an edge with
    #			itself, and so is distance 3. '''


    # If any face is larger than the number of vertices, it will definitely
    # Share an edge with itself. Therefore distance three.
    faceSizeTest = [face[0]  > M.shape[1] for face in Faces]
    if True in faceSizeTest:
        three = 1
    #	print '''The complement contains a polygon which shares an edge with
    #			itself, and so is distance 3. '''


    if  fourgonTest(Bridges, Islands):
        three = 1


    if lengthCheck != 2*M.shape[1]:
        three = 1
    #	print '''The complement contains a polygon which shares an edge with
    #			itself, and so is distance 3. '''
    # See function faceParse
    # If this is true, then some face is sharing an edge with itself.
    # It can't be a four-gon, since that would mean we have a multi-curve.
    # So it must be an island, which means that there is a curve
    # in the complement which is distance two. Thus the original pair is
    # Distance three.

    if facesShareTwoEdges(islandFaces):
        three = 1
    #	print ''' Found two faces that share multiple edges: A curve that
    #			  intersects the non-reference curve only two times has been
    #			  found. This curve cannot fill and so the pair is distance 3.
    #		  '''

    # Means there is a path of length two.
    # Since a curve pair with two intersections cannot fill on any genus >1 ,
    # The two curves will be distance three.

    #Find linking edge paths in "mesh"
    faces  = list(bridgeFaces)
    faces += islandFaces

    #Build paths into curves and intersect with alpha
    matrixList = buildMatrices(edgePaths, allPaths)

    genusTest, matrixLibrary = testCollection(matrixList,originalGenus)
    if three != 1:
        three = 1 if False in genusTest else 0

    # Output
    returnVals = [three, matrixLibrary]
    return returnVals

def ladderConvert(ladderTop, ladderBottom):
    #print ladderTop, ladderBottom
    n = len(ladderTop)
    newTop = [' ']*n
    newBottom = list(newTop)

    for j in range(1, n+1):

        if j in ladderTop and j in ladderBottom:
            newTop[ladderTop.index(j)]  = ladderBottom.index(j)
            newBottom[ladderBottom.index(j)] = ladderTop.index(j)

        elif j in ladderTop:
            ladderTopTemp = list(ladderTop)
            indices = [ladderTop.index(j)]; ladderTopTemp[indices[0]] = None#ladderTopTemp.remove(j)
            indices.append(ladderTopTemp.index(j))
            newTop[indices[0]] = indices[1]
            newTop[indices[1]] = indices[0]

        elif j in ladderBottom:
            ladderBottomTemp = list(ladderBottom)
            indices = [ladderBottom.index(j)]; ladderBottomTemp[indices[0]] = None #ladderBottomTemp.remove(j)
            indices.append(ladderBottomTemp.index(j))
            newBottom[indices[0]] = indices[1]
            newBottom[indices[1]] = indices[0]

    return newTop, newBottom


def ladder_is_multicurve(top, bottom):

    n = len(top)

    j0 = top[0]
    counter = 1

    j = bottom[0]
    bottom[0] = None
    oldIndex = 0
    while j != j0:

        old_j = j
        if j in top:
            nextIndex = top.index(j)
            j = bottom[nextIndex]

            if None in top:
                top[top.index(None)] =  old_j
            elif None in bottom:
                bottom[bottom.index(None)] =  old_j

            bottom[nextIndex] = None

        elif j in bottom:
            nextIndex = bottom.index(j)
            j = top[nextIndex]

            if None in top:
                top[top.index(None)] =  old_j
            elif None in bottom:
                bottom[bottom.index(None)] =  old_j

            top[nextIndex] = None

        counter += 1

    if None in top:
        top[top.index(None)] =  j
    elif None in bottom:
        bottom[bottom.index(None)] =  j


    return 1 if counter != n else 0

def matrix_is_multicurve(beta):

    top = beta[0]
    bottom = beta[1]
    index = 0
    j = top[index]
    start = 100
    counter = 0

    while j != start:
        counter += 1
        start = top[0]

        if top[j] == index :
            next_index = j
            j = bottom[j]
            index = next_index

        elif bottom[j] == index:
            next_index = j
            j = top[j]
            index = next_index

        if top[j] == bottom[j]: return True
        #print 'j', j,'next_index: ',  index

    return False if counter == len(top) else True


def test_permutations(original_ladder):
    distance4 = []
    distance3 = []
    from curvepair import CurvePair
    ladder = deepcopy(original_ladder)
    if not original_ladder is None:
        for i in range(len(ladder[0])):
            if not ladder_is_multicurve(*ladder):
                perm = CurvePair(*ladder)
            else:
                perm = None
            if not perm is None :
                if perm.distance is 4:
                    distance4.append(deepcopy(perm))
                else:
                    distance3.append(deepcopy(perm))
            else: pass
            first_vertex = ladder[0].pop(0)
            ladder[0].append(first_vertex)

        if len(distance4) == 0:
            print ' Found no distance four permutations of the ladder. '

        if distance3:
            print 'Distance 3 single curves: '
            for curve in distance3:
                print 'top   : ', curve.ladder[0]
                print 'bottom: ', curve.ladder[1]
        if distance4:
            print 'Distance 4+ single curves: '
            for curve in distance4:
                print 'top   : ', curve.ladder[0]
                print 'bottom: ', curve.ladder[1]

        return distance4
    else:
        print "You didn't give me a ladder! "
        return []


def test_perms(original_ladder):
    distance4 = []
    distance3 = []
    ladder_to_perm = deepcopy(original_ladder)
    from curvepair import CurvePair
    if not original_ladder is None:
        for i in range(len(ladder_to_perm[0])):
            if not ladder_is_multicurve(*ladder_to_perm):
                perm = CurvePair(*deepcopy(ladder_to_perm))
            else:
                perm = None
            if not perm is None :
                if not perm.distance is 3:
                    distance4.append(perm)
                else:
                    distance3.append(perm)
            else: pass

            first_vertex = ladder_to_perm[0].pop(0)
            ladder_to_perm[0].append(first_vertex)
    return distance3, distance4
