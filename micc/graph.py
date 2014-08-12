from copy import deepcopy
import curves as c
from sys import stderr


def shift(path):
    '''
    init

    '''
    temp = path.index(min(path))
    return path[temp:] + path[:temp]


def invert(path):
    '''
    init

    '''
    return shift(path[::-1])


def contains(small, big):
    print type(small), type(big)
    for i in xrange(len(big)-len(small)+1):
        for j in xrange(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return False


class Graph:

    def add_node(self, node):
        self.nodes[node] = []

    def __init__(self, edges, rep_num=2):
        self.edges = edges
        self.rep_num = rep_num
        self.nodes = {}
        self.counter = 0
        self.loops = []
        self.gammas = []
        self.nodes_to_faces = {}
        self.rep_num = rep_num

    def compute_loops(self, n, genus):
        edges = self.edges[0]
        fourgons = [i[1] for i in edges if i[0] == 4]
        non_fourgons = [i[1] for i in edges if i[0] != 4]
        keys = self.nodes_to_faces.keys()
        for i,face in enumerate(non_fourgons):
            for node in face:
                if not node in keys:
                    self.nodes_to_faces[node] = [i]
                else:
                    self.nodes_to_faces[node].append(i)
                keys = self.nodes_to_faces.keys()
        for face in fourgons:
            for node in face:
                if not node in keys:
                    self.nodes_to_faces[node] = [None]
                else:
                    self.nodes_to_faces[node].append(None)

                keys = self.nodes_to_faces.keys()
        self.nodes_to_faces = {int(k): tuple(v) for k,v in self.nodes_to_faces.iteritems()}
        nodes = range(n)

        for i in nodes:
            self.add_node(i)
        self.find_all_edges(fourgons, non_fourgons, nodes, self.rep_num)
        graph_copy = deepcopy(self.nodes)
        #graph_copy = {i : set(j) for i,j in self.nodes.iteritems()}
        #for k,v in graph_copy.iteritems():
        #    stderr.write(str(k)+": "+str(v)+'\n')
        #raw_input()
        for start_node in nodes:
            #self.loops.extend(self.iter_loop_dfs(graph_copy, start_node, start_node))
            for adj_node in graph_copy[start_node]:
                self.loops += self.loop_dfs(start_node,adj_node,graph_copy,[start_node], self.nodes_to_faces)
        '''
        #Johnson circuit locating algorithm
        from johnson import Johnson
        johnny = Johnson(graph_copy)
        johnny.find_all_circuits()
        self.loops = johnny.circuits
        '''
        #print len(self.loops)
        from itertools import chain

        self.loops = [j for j in set([tuple(i) for i in self.loops])]
        edges = self.edges[1]
        for path in list(self.loops):
            temp_len = len(path)
            #in_loops = path in set(self.loops)
            removed = False
            if temp_len < 3:
                if not removed:
                    self.loops.remove(path)
                    removed = True

            elif invert(path) in self.loops:
                if not removed:
                    self.loops.remove(path)
                    removed = True

            # Trial: remove all duplicates
            else:
                temp_path = list(path)
                #temp_len = len(temp_path)
                temp_path = shift(temp_path)
                for face in non_fourgons:
                    for triple in [temp_path[i:i+3] \
                            for i in xrange(temp_len-2)]:
                        if set(triple) <= face:
                            if not removed:
                                self.loops.remove(path)
                                removed = True
                                break
                    if removed:
                        break


                    temp_path = invert(temp_path)

                    for triple in [temp_path[i:i+3] \
                            for i in xrange(temp_len-2)]:
                        if set(triple) <= face:
                            if not removed:
                                self.loops.remove(path)
                                removed = True
                                break

                    for i in xrange(temp_len):
                        temp_path = temp_path[1:] + temp_path[:1]
                        for triple in (temp_path[i:i+3] for i in xrange(temp_len-2)):
                            boolA = set(triple) <= face

                            #boolA = contains(triple,face)
                            #if set(triple) <= set(face) and path in self.loops:
                            if boolA :
                                if not removed:
                                    self.loops.remove(path)
                                    removed = True
                                    break
                        if removed:
                            break

        for loop in list(self.loops):
            path = list(loop)
            path_matrix = c.build_matrices(edges, [path])
            ladder = [list(path_matrix[0][0,:,1]),list(path_matrix[0][0,:,3])]

            gamma = c.CurvePair(ladder[0],ladder[1],0, 0)
            if gamma.genus <= genus:
                self.gammas.append(loop)

    @staticmethod
    def get_value(pos_to_insert, ladder, path):
        return int(path[int(ladder[int(pos_to_insert)])])+1

    def find_all_edges(self, fourgons, non_fourgons, alpha_edge_nodes, rep_num):
        '''
        Determines all edges between boundary componenets and adds adjacencies between
        appropriate edges.

        '''

        #find all direct connections between non-fourgon regions
        regions = fourgons + non_fourgons
        for alpha_edge in alpha_edge_nodes:
            for region in regions:
                if alpha_edge in region:
                    region.remove(alpha_edge)
                    for other_edge in region:
                        self.add_adjacency(alpha_edge, int(other_edge), rep_num)
                    region.add(alpha_edge)

    def add_adjacency(self,node, adjacent_node, rep_num):
        '''
        Adds adjacencies to a graph represented by an adjacency list.
        This is useful when we would like to replicate adjacencies
        between nodes.

        :param self:
        :type self: Graph
        :param node: node to add adjacency
        :param type: int
        :param adjacent_node: the adjacent node
        :type adjacent_node: int
        :param rep_num: set to 1
        :type rep_num: int

        '''
        for i in range(rep_num):
            adjacency_list = self.nodes[node]
            if Graph.count(adjacent_node, adjacency_list) < rep_num:
                adjacency_list.append(adjacent_node)

            adjacency_list = self.nodes[adjacent_node]
            if Graph.count(node, adjacency_list) < rep_num:
                adjacency_list.append(node)

    @staticmethod
    def count(adj_node, adj_list):
        '''
        Determines the number of adjacencies between two nodes in a graph, given the
        adjacency list of one of node.

        :param self:
        :type self: Graph
        :param adj_node: Adjacent node
        :type adj_node: int
        :param adj_list: Adjacency list of the node in question
        :type adj_list: list<int>
        :returns: number of edges from the adjacent node to the original node


        count = 0
        for i in adj_list:
            if i == adj_node:
                count += 1
        return count
        '''
        return adj_list.count(adj_node)

    def iter_loop_dfs(self, graph, start, goal):
        loops = []
        stack = [(start, [start])]
        while stack:
            vertex, path = stack.pop()
            in_path = set(path)
            for next in set(graph[vertex]):
                if next in in_path:
                    if len(path) >= 3:
                        if self.faces_share_edges(self.nodes_to_faces, path):
                            continue
                    if next == goal:
                        loops.append(list(path))
                    else:
                        continue
                else:
                    stack.append((next, list(path + [next])))
        return loops

    def faces_share_edges(self,nodes_to_faces, path):
        path_head_3 = path[-3:]
        previous_three_faces = [nodes_to_faces[edge] for edge in path_head_3]
        previous_three_faces = [set(i) for i in previous_three_faces]
        intersection_all = set.intersection(*previous_three_faces)
        return len(intersection_all) == 2


    def loop_dfs(self, current_node, start_node, graph, current_path, nodes_to_faces):
        '''
        Recursively finds all closed cycles in a given graph that begin and end at start_node.
        As one would guess, it employs a standard depth-first search algorithm on the graph,
        appending current_path to all_loops when it returns to start_node.

        In the overall distance computation, this function is computationally dominating with
        exponential complexity, so take care with its use.

        :param self:
        :type self: Graph
        :param current_node: the current alpha edge in the recursion
        :type current_node: int
        :param start_node: the source node of the current recursive search
        :type start_node: int
        :param graph: graph of the overall graph mid-recursion
        :type graph: dict<int,list<int> >
        :param current_path: list of nodes in the current path
        :type current_path: list<int>
        :param all_loops: list of all current paths
        :type all_loops: list< list<int> >
        :returns: set of all closeds cycles in the graph starting and ending at start_node

        '''

        #stderr.write(str(current_path)+'\n')
        #stderr.write(str([nodes_to_faces[i] for i in current_path])+'\n')
        if len(current_path) >= 3:
            path_head_3 = current_path[-3:]
            previous_three_faces = [set(nodes_to_faces[edge]) for edge in path_head_3]
            intersection_all = set.intersection(*previous_three_faces)
            if len(intersection_all) == 2:
                return []

        if current_node == start_node:
            #stderr.write("Found one! \n")
            #all_loops.append(shift(list(current_path)))
            return [shift(list(current_path))]

        else:
            loops = []
            for adjacent_node in set(graph[current_node]):
                if Graph.count(adjacent_node, current_path) < self.rep_num:
                    current_path.append(adjacent_node)
                    graph[current_node].remove(adjacent_node)
                    graph[adjacent_node].remove(current_node)
                    loops += list(self.loop_dfs(adjacent_node, start_node, graph, current_path, nodes_to_faces))
                    graph[current_node].append(adjacent_node)
                    graph[adjacent_node].append(current_node)
                    current_path.pop()
            return loops
'''
for i in range(len(path)):
    index = path[i]
    beta[0].insert(index+1,None)
    beta[1].insert(index+1,None)
    beta[0].append('____')
    beta[1].append('____')
    for j,tup in enumerate(zip(beta[0],beta[1])):

        t,b = tup
        if type(t) == str or type(b) == str: continue
        if t > index:
            beta[0][j] +=1
        if b > index:
            beta[1][j] +=1
    beta[0].remove('____')
    beta[1].remove('____')


    for j in range(len(path)):
        if path[j] > index:
            path[j] += 1
path.sort()
for i in range(len(path)):
    index = path[i]
    beta[0][index+1]=int(self.getValue(i,ladder[0],path))
    beta[1][index+1] =int(self.getValue(i,ladder[1],path))
'''
'''
count = {}
count[None] = 0
for i,j in zip(beta[0],beta[1]):
    if i in count.keys():
        count[i] += 1
    else:
        count[i] = 1
    if j in count.keys():
        count[j] += 1
    else:
        count[j] = 1
'''
