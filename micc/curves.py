# Paul Glenn
# curves.py
# Give curve pairs class structure in preparation for public access
import numpy as np
from itertools import product
from copy import deepcopy


def fix_matrix_signs(M):
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

    row, col = [-1, 1]
    sgn = 1
    #for ii in range(len(M[0,:])):
    while 0 in M[1, :, 1:4:2] and row < M.shape[1]-1:
        row += 1
        sgn = -sgn
        # Starting position and sign are arbitrary.
        for i in range(M.shape[1]):
            g_val = int(M[0, row, col])
            new = np.array(np.where(M[0, g_val, :] == row))
            #print  np.where(new%2==col%2)
            ind = int(new[np.where(new % 2 == col % 2)])
            sgn = -sgn
            M[1, g_val, ind] = sgn
            ind += 2
            ind = ind % M.shape[2]
            sgn = -sgn
            M[1, g_val, ind] = sgn
            row = g_val
            col = ind
        M[1, :, 0] = -1
        M[1, :, 2] = 1
    return M



def concatenate_matrix(M1, M2):
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
    top_M = np.copy(M1)
    bot_m = np.copy(M2)
    n = M1.shape[1]
    m = M2.shape[1]
    top_M[0, 0, 0] = n + m - 1
    top_M[0, n-1, 2] = n
    bot_m[0] += n * np.ones(M2[0].shape)
    bot_m[0, m-1, 2] = 0
    bot_m[0, 0, 0] = n-1
    new_M = np.vstack((top_M[0], bot_m[0]))  # Multi-curve
    new_M = np.vstack((new_M, np.vstack((top_M[1], bot_m[1])))).reshape((2, n+m, 4))

    left = zip(np.where(top_M[0] == n-1)[0], np.where(top_M[0] == n-1)[1])
    i0, j0 = 0,0
    for pair in left:
        i, j = pair
        if j % 2 == 1:
            i0, j0 = i, j
    new_M[0, i0, j0] = n

    temp_ind = np.where(new_M[0, n-1, :] == i0)[0]
    temp_ind = int(temp_ind[temp_ind % 2 == 1][0])
    temp_val = new_M[0, n-1, temp_ind]
    switch_val = new_M[0, n, temp_ind]
    new_M[0, n-1, temp_ind] = switch_val
    new_M[0, n, temp_ind] = temp_val
    return_ind = new_M[0, switch_val, :] == n
    return_ind[0:3:2] = False
    new_M[0, switch_val, return_ind == True] = n-1

    return fix_matrix_signs(new_M)

## Path finding methods -- used to find edge paths in the complement of beta-curve.
## Most are helpers for findAllPaths below.



def visited(curr, face, path):
    """

    :param curr:
    :param face:
    :param path:
    :return:
    """
    my_face = list(face)
    my_face.remove(curr)
    v = 0
    for edge in my_face:
        if edge in path:
            v = 1
    return v


def is_unique(path, all_paths):
    '''
    :param path:
    :type path:
    :param AllPaths:
    :type AllPaths:
    :returns: boolean

    '''
    return not (path in all_paths)


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


def path_finished_single(edge, face, path):
    '''
    :param edge:
    :type edge:
    :param face:
    :type face:
    :param path:
    :type path:
    :returns:

    '''
    C = len(path) > 2 and (edge == path[-1])
    faceL = list(face)
    faceL.remove(edge)
    C = C and not set(faceL).isdisjoint(set(path))
    #meeting = lambda e1, e2, face: e1 in face and e2 in face
    return C


def path_finished_double(edge, face, path):
    '''
    :param edge:
    :type edge:
    :param face:
    :type face:
    :param path:
    :type path:
    :returns:

    '''

    C = len(path) > 2 and (edge == path[-1])
    faceL = list(face)
    faceL.remove(edge)
    C = C and not set(faceL).isdisjoint(set(path))
    if path.count(edge) != 2 :
        C = 0
    #meeting = lambda e1, e2, face: e1 in face and e2 in face
    return C



def find_new_paths(current_path, my_face, faces, all_paths, path_function):
    '''
    :param current_path:
    :type current_path:
    :param my_face:
    :type my_face:
    :param faces:
    :type faces:
    :param all_paths:
    :typeall_pathss:
    :param path_function:
    :typepath_functionn:
    :returns:

    '''

    start = current_path[0]
    next_edge = None
    sub_path = []
    faces_without_current_face = list(faces)
    faces_without_current_face.remove(my_face)
    for face in faces: # Check all faces...
        if start in face and face is not my_face:
            #if we find the start edge in another face...
            for other_face in faces_without_current_face:
                #go through its edges...
                if start in other_face: # needed?
                    face_without_start = list(other_face)
                    face_without_start.remove(start)

                    for non_starting_edge in face_without_start:
                        # try all edges in that path
                        next_edge = non_starting_edge
                        if  (not visited( start, other_face, current_path ) ) and\
                            (next_edge not in my_face):
                            sub_path = [next_edge]
                            sub_path.extend(current_path)

                            # Recursive call to take all possible directions
                            find_new_paths(sub_path, other_face, faces, all_paths, path_function)

                        elif path_function(next_edge, other_face, current_path):
                            new_found_path = shift(current_path)
                            inverted_path = invert(current_path)
                            unique = lambda path: is_unique(path, all_paths)
                            if unique(new_found_path) and unique(inverted_path):
                                all_paths.append(new_found_path)
                    face_without_start.append(start)
        faces_without_current_face.append(face)


def remove_duplicates(faces, all_paths):
    '''
    :param faces:
    :type faces:
    :param all_paths:
    :type all_paths:
    :returns:

    '''
    paths = list(all_paths)
    for path in paths:
        for f in faces:
            counter = 0
            for e in f:
                if e in path:
                    counter += 1
            if counter == 3:
                all_paths.remove(path)
    return all_paths


def find_all_paths(faces):
    '''
    :param faces:
    :type faces:
    :returns:

    '''

    all_paths = []
    forward = lambda path, face, path_function: find_new_paths(path, face, faces, all_paths, path_function)
    for face in faces:
        for edge in face:
            forward([edge], face, path_finished_single)
            #forward([edge], face, pathFinishedDouble)
    all_paths = remove_duplicates(faces, all_paths)
    return all_paths

## Now that we have the paths, they need to be
## re- indexed so that matrices can be built from them.

def build_matrices(edge_paths, all_paths):
    '''
    :param edge_paths: list of face boundary orientations
    :type edge_paths: list
    :param all_paths: list
    :type all_paths: list

    Take the paths in the skeleton of the complement of the transverse curve
    And create matrices.
    '''
    #print 'edgePaths:',edgePaths
    #print 'AllPaths:',AllPaths[0].loops
    master_list = []

    # Allow paths to be referenced by face
    #print edgePaths

    for itr in range(len(edge_paths)):
        edge_paths[itr] = dict(edge_paths[itr])
    #print 'all_paths:',all_paths
    #print 'edge_paths:',edge_paths

    ordered_paths , mapped_paths = [],[]

    # Rescale path 0-len(path) for matrix
    for path in all_paths:
        ordered_path = list(np.sort(path))
        ordered_paths.append(ordered_path)
        mapped_paths.append(dict(zip(ordered_path,range(len(path)))))
    #print mapped_paths,'\n'
    #Create Matrices using details from edge paths.
    for Path, mapped_path in zip(all_paths, mapped_paths):
        #Value Matrix
        path_size = len(Path)
        shape = (path_size, 4)
        last = Path[len(Path)-1]
        M = np.zeros(shape)
        M2 = np.zeros(shape)
        M[:, 0] = np.array([path_size-1]+range(path_size-1))
        M[:, 2] = np.array(range(1,path_size)+[0])
        past_edges = dict()
        future_edges = dict()
        old_vertex = last

        itr = 1
        for vertex in Path:
            flag = False
            for path in edge_paths:
                keys = set(path.keys())
                if vertex in keys and old_vertex in keys:
                    past_edges[vertex] = path[vertex]
                    flag = True
                if vertex in keys and Path[itr % path_size] in keys:
                    if flag:
                        future_edges[vertex] = (path[vertex] + 2) % 4
                    else:
                        future_edges[vertex] = path[vertex]
                flag = False
            old_vertex = vertex
            itr += 1
        old_vertex = last

        itr = 1
        for vertex in Path:
            curr_vertex = vertex
            next_vertex = Path[itr % path_size]
            M[mapped_path[vertex], past_edges[vertex]] = mapped_path[old_vertex]
            M[mapped_path[vertex], future_edges[vertex]] = mapped_path[next_vertex]
            old_vertex = curr_vertex
            itr += 1

        # Sign matrix: Stand-alone function
        M = fix_matrix_signs(np.array([M, M2], dtype=int))
        master_list.append(M)
    # End while

    return master_list



def face_parse(alpha_edges):
    '''
    :param alpha_edges: set of faces with alpha edges.
    :typealpha_edgess:
    :returns: (Bridges, Isalnds, lengthCheck) Bridges : 4-sided regions; Islands : n > 4 - gons; lengthCheck : the number of alpha edges included.

    Separate set of all faces into bridges (4-sided regions) and
    islands (higher-sided regions) for distance calculator.

    '''
    bridges, islands = [], []
    length_check = []
    for pair in alpha_edges:
        if pair[0] == 4:
            bridges.append(pair)
        else:
            islands.append(pair)
        length_check.extend(list(pair[1]))
    return bridges, islands, len(length_check)

######					For Distance Extension 					#######


def connected(P1, P2):
    '''
    :param P1:
    :type P1:
    :param P2:
    :type P2:
    :returns:

    '''

    S1 = set(P1)
    S2 = set(P2)
    if S1.isdisjoint(S2) or (S1.issubset(S2) and S1.issuperset(S2)):
        return 0
    else:
        return 1


def share_edge(path1, path2):
    '''
    :param path1:
    :type path1:
    :param path2:
    :type path2:
    :returns:

    '''
    if not set(path1).isdisjoint(path2):
        return 0, -1
    else:
        intersection_set = set(path1) & set(path2)
        numshared = len(intersection_set)
        if numshared != 1:
            return 0, -1 #Then they share too much!
        else:
            shared_item = intersection_set.pop()
    return 1, path2.index(shared_item)


def find_combined_paths(all_paths, M_library):
    '''
    :param all_paths:
    :typeall_pathss:
    :param M_library:
    :typeM_libraryy:
    :returns:
    '''
    list_of_connected = []
    path_library = dict(zip(range(len(all_paths)), all_paths))
    index1 = 0
    for path1 in all_paths:
        index2 = 0
        for path2 in all_paths:
            if share_edge(path1, path2)[0]:
                list_of_connected.append((M_library[index1],
                            M_library[index2]))
            index2 += 1
        index1 += 1

    return list_of_connected


def faces_share_two_edges(faces):
    '''
    :param faces:
    :type faces:
    :returns:

    '''

    distance_three_flag = 0

    for face in faces:
        faces_without_face = list(faces)
        faces_without_face.remove(face)
        for other in faces_without_face:
            if len(set(face) & set(other)) >1: distance_three_flag = 1

    return distance_three_flag



def edges(M):
    '''
    :param M: the matrix representing a pair of curves
    :type M: numpy.array(dtype=float64) of shape (2,n, 4)
    :returns: (allFaces, edges) allFaces: tuple of faces including size and set of alpha-edges which bound them; edges: same as allFaces, exceincluding orientation of boundary edges.
    '''
    #print 'called edges'
    # The em list is needed to hold the tuples of (faceLength, faceEdges)
    all_faces = []
    # INV: Number of faces found.
    faces = 0

    num_rows, num_cols = M.shape[1:3] #num_rows, num_cols

    old_vertices = [] ##list of previous paths
    bigonFlag = 0
    # Bigons are unwanted structures.
    Paths = []  # alpha - edge paths.
    facesTemp = dict()
    for i,j in product(range(num_rows),range(num_cols)):
        #Set of edges associated with face
        tr = 1
        face = set()
        if faces == num_rows:
            break  # upper bound on possible no. faces
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
            arr1 = M[0,gval,:] == i%num_rows
            arr2 = M[1,gval,:] != M[1,i,j] #sign check
            i_next = gval
            alpha = (M[0,i,0]+1)%num_rows
            i= i_next
            new = np.where(arr1 & arr2)[0]
            #new = np.intersect1d(arr1[0],arr2[0],assume_unique=True)
            #ENS: val and sign correct
            ind = np.where(new%2==j%2) #ENS: beta->beta, alpha->alpha
            j_next = (int(new[ind])+1)%num_cols #Always move clockwise
            j_old = j
            j = j_next
            if (i,j) in old_vertices:
                break
            old_vertices.append((i,j))
            edges += 1

            alpha_new =( M[0,i,0]+1) % num_rows;
            shift = (alpha_new - alpha)

            if (shift==1 and j%2 == 1) or (shift == 1-num_rows and alpha_new==0 and j%2==1):
                face.add(alpha_new)
                pathTemp.append((alpha_new,(j)%num_cols))
            elif (shift==-1 and j%2==1 ) or (shift == num_rows-1 and alpha ==0  and j%2==1):
                face.add(alpha)
                pathTemp.append((alpha,(j)%num_cols))

            if (i,j)==(io,jo):
                facesTemp[edges] = facesTemp.get(edges,0) +1
                if edges==2:
                    bigonFlag = 1
                if not bigonFlag:
                    Paths.append(pathTemp)
                    faces += 1
                    found = 1
                    all_faces.append((edges,face))
    return all_faces, Paths

def boundary_count(M):
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

def vector_solution(edges):

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
    P, bigon = boundary_count(M)  # Polygons in the complement of curve pair
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

def test_collection(matrix_list, original_genus):
    '''
    :param matrix_list:
    :typematrix_listt: list of numpy.array(dtype=float64)
    :param original_genus:
    :typeoriginal_genuss:
    :returns: list of booleans and a dictionary containing the matrices without bigons.

    Test a collection of matrices to see if they fill a given genus.
    Note: Matrices with bigons are still counted in the calculation, but are not returned.
    '''
    genusCollection = []
    index = 0
    matrixLibrary = dict()
    for M in matrix_list:

        bigon = boundary_count(M)[1]
        Genus = genus(M)
        matrixLibrary[index] = M
        index += 1
        if bigon:
            Genus  = 0 # If it has a bigon, it should automatically fail

        genusCollection.append(Genus)
    # Test if all fill and thus distance g.t. 3
    test = [Genus == original_genus for Genus in genusCollection]
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

def Three(M, allPaths, ext=0, originalGenus = False, boundaries = False, edges = False):
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

    if not originalGenus:
        originalGenus = genus(M)
    if not boundaries:
        F0, bigon = boundary_count(M)
    else:
        F0, bigon = boundaries
    if not edges:
        #Calculate face alpha edges and alpha edge paths
        Faces, edgePaths = edges(M)
    else:
        Faces, edgePaths = edges

    Bridges, Islands, lengthCheck = face_parse(Faces)

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

    if faces_share_two_edges(islandFaces):
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
    matrixList = build_matrices(edgePaths, allPaths)

    genusTest, matrixLibrary = test_collection(matrixList, originalGenus)
    if three != 1:
        three = 1 if False in genusTest else 0

    # Output
    returnVals = [three, matrixLibrary]
    return returnVals

def ladder_convert(ladder_top, ladder_bottom):
    #print ladderTop, ladderBottom
    n = len(ladder_top)
    newTop = [' ']*n
    newBottom = list(newTop)

    for j in range(1, n+1):

        if j in ladder_top and j in ladder_bottom:
            newTop[ladder_top.index(j)]  = ladder_bottom.index(j)
            newBottom[ladder_bottom.index(j)] = ladder_top.index(j)

        elif j in ladder_top:
            ladderTopTemp = list(ladder_top)
            indices = [ladder_top.index(j)]; ladderTopTemp[indices[0]] = None#ladderTopTemp.remove(j)
            indices.append(ladderTopTemp.index(j))
            newTop[indices[0]] = indices[1]
            newTop[indices[1]] = indices[0]

        elif j in ladder_bottom:
            ladderBottomTemp = list(ladder_bottom)
            indices = [ladder_bottom.index(j)]; ladderBottomTemp[indices[0]] = None #ladderBottomTemp.remove(j)
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
    #from curvepair import CurvePair
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
    #from curvepair import CurvePair
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






#import numpy as np
#from curves import fix_matrix_signs, boundary_count, genus, ladder_convert, vector_solution, edges, Three
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
    def __init__(self, top_beta, bottom_beta, dist=1, conjectured_dist=3,recursive=False):

        is_ladder = lambda top, bottom: not (0 in top or 0 in bottom)

        if is_ladder(top_beta, bottom_beta):
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

        self.loops = []

        if dist is 1:
            graph = Graph(self.edges, rep_num=conjectured_dist-2)
            graph.compute_loops(self.n, self.genus)
            self.loops = graph.gammas
            #from sys import stderr
            #stderr.write(str(self.loops)+'\n')
            self.distance, self.loop_matrices = self.compute_distance(self.matrix, self.loops, recursive=recursive)
        else:
            self.distance = None

    def __repr__(self):
        return self.ladder[0]+'\n'+self.ladder[1]+'\n'

    def compute_distance(self, M, all_paths,recursive=True):
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

        dist_is_three, lib = Three(M, all_paths, originalGenus=self.genus, boundaries=self.boundaries, edges=self.edges)
        dist = 3 if dist_is_three  else 'at least 4!'
        if dist == 3:
            return dist, lib
        else:
            if recursive:
                geodesic_distances = []
                for k, matrix in lib.iteritems():
                    #stderr.write(str(matrix))
                    if np.array_equal(matrix, self.matrix):
                        continue
                    elif self.solution == CurvePair(matrix[0, :, 1], matrix[0, :, 3],0).solution \
                            and len(self.matrix[0]) == len(matrix[0]):
                        continue
                    cc = CurvePair(matrix[0, :, 1], matrix[0, :, 3])
                    #stderr.write(str(k)+": "+str(cc.distance)+'\n')
                    geodesic_distances.append(cc.distance)
                    #print 'computed curve',k,'!'
                #print '\n'
                return min(set(geodesic_distances)) + 1, lib
            else:
                return dist, lib

