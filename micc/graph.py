from collections import defaultdict, OrderedDict
from copy import copy, deepcopy
from itertools import izip, chain, combinations, permutations, product
from math import floor
from functools import partial
import random
from micc.utils import shift, complex_cmp
import multiprocessing as mp
import numpy as np
from sys import stderr
import networkx as nx

def SuperVertex(object):
    """
    SuperVertex aims to generalize the notion of a normal vertex to allow for
    paths through vertices multiple times, while altering minimal code for the
    standard case of passing through vertices once.

    id - the integer identification that is used in the graph for this vertex
    rep - the number of times we aim to cross a given vertex (at most once,
    at most twice, three times, etc.)
    """
    def __init__(self, id, rep):
        pass


class Graph(object):

    def __init__(self, boundaries, n, repeat=1):
        self.n = n
        self.repeat = repeat
        self.boundaries = boundaries
        self._d3_dual_graph = None#self.create_dual_graph(self.boundaries)
        self.dual_graph, self.faces_containing_arcs = \
            self.create_dual_graph(self.boundaries, repeat=repeat)
        self.cycles = set()
        self.dual_graph_copy = deepcopy(self.dual_graph)
        #self._d3_cycles = self.find_d3_cycles()
        # stderr.write('dualgraph\n')
        # for k,v in self.dual_graph.iteritems():
        #     stderr.write(str(k)+': '+str(v)+'\n')
        # stderr.write('\n')

    def create_dual_graph(self, boundaries, repeat=1):
        """
        This function constructs the pseudo-dual graph of our reference curve
        arcs by inspecting the connectivity information of the complementary
        polygons.

        The repeat parameter allows for cycles that pass through arcs more than
        once, by interpreting the cycles as follows:

            |                   |
        ------i_1-i_2- ... -i_k---(i+1)_1-(i+1)_2- ... -(i+1)_k-
            |                   |

        Observe that for the ith arc and the jth/kth repeatition of the ith arc,
        i_k \not \in self.dual_graph[i_j] and i_j \not \in self.dual_graph[i_k].
        This removes homotopically trivial extensions to cycles.

        :param boundaries: The return value of boundary_reduction()
        :param repeat: number of parallel copies are allowed in a cycle
        :return: An adjacency list ({int: list}) that:
            - Vertices are reference curve arcs
            - Edge connect vertices if the arcs are connected by a polygonal
            region.
        """
        def inverted_arc_index(boundaries):
            # form an inverted index for reference arcs; this allows the ability
            # to determine which polygons the ith arc bounds (at most two)

            faces_containing_arcs = {arc: [] for arc in xrange(self.n)}
            for face_id, arcs in boundaries.iteritems():
                for arc in arcs:
                    faces_containing_arcs[arc].append(face_id)

            return faces_containing_arcs

        faces_containing_arcs = inverted_arc_index(boundaries)
        self.fourgons = [arcs for arcs in boundaries.itervalues()
                         if len(arcs) == 2]  # 2 reference arcs + 2 transverse
                                             # arcs = 4 sided polygon
        self.non_fourgons = [arcs for arcs in boundaries.itervalues()
                             if len(arcs) != 2]  # similar logic here

        # Create a map of arcs to which non_fourgons contain that arc
        # this is more specific than faces_containing arcs,
        # though a bit redundant
        self.non_fourgon_map = {arc: [] for arc in xrange(self.n)}
        for face_id, face in enumerate(self.non_fourgons):
            for arc in self.non_fourgon_map:
                if arc in face:
                    self.non_fourgon_map[arc].append(face_id)

        dual_graph = {arc: [] for arc in xrange(self.n)}

        regions = self.fourgons + self.non_fourgons

        for region in regions:
            altered_region = list(region)

            for arc in region:
                altered_region.remove(arc)

                for other_arc in altered_region:

                    if other_arc not in dual_graph[arc]:
                        dual_graph[arc].append(other_arc)

                    if arc not in dual_graph[other_arc]:
                        dual_graph[other_arc].append(arc)

                altered_region.append(arc)  # COULD BE SO BAD BREAKS LIST ORDER

        # From the normal dual graph of identified arcs, we incorporate the
        # repetition of arc usage by utilizing complex numbers. for the ith arc
        # and the kth repetition of the arc, we denote this as vertex i+k*j,
        # where j is the Python imaginary unit. The imaginary value denotes
        # which arc multiplicity we're interested in, and the real value
        # denotes the arc of interest.

        # Change types to complex
        complex_dual_graph = {k+0j: [val+0j for val in v] for k, v in
                      dual_graph.iteritems()}
        self._d3_dual_graph = deepcopy(complex_dual_graph)
        if repeat >= 1:
            for arc, adj_list in dual_graph.iteritems():
                if len(adj_list) == 2:
                # This means we the arc is in a 4-gon. We don't want to have
                # total adjacency, since that's making the graph more complex,
                # so we only add the parallel arcs.
                    for i in xrange(1, repeat):
                        complex_dual_graph[arc+i*1j] = [v+i*1j for v in adj_list]
                        complex_dual_graph[arc+0j] = [v+0j for v in adj_list]
                    continue

                # add repetitions based on the number required.
                for i in xrange(1, repeat):
                    complex_dual_graph[arc+i*1j] = [v+0j for v in adj_list]
                    complex_dual_graph[arc+i*1j].append(arc+0j)
                    complex_dual_graph[arc+0j].append(arc+i*1j)
                    for adj_arc in list(adj_list):
                        complex_dual_graph[arc+0j].append(adj_arc+i*1j)
                        complex_dual_graph[arc+i*1j].append(adj_arc+i*1j)
            # We must iterate over the graph again to remove spurious edges
            # at the boundary between 4-gons and >4-gons
            for arc, adj_list in complex_dual_graph.iteritems():
                # Yank out the >4-gons
                if len(adj_list) > 2:
                    for adj_arc in adj_list:
                        region_with_arc = [region for region in self.fourgons
                                           if arc.real in region]
                        # If the part is part of another 4-gon...
                        if any(adj_arc.real in region
                               for region in region_with_arc) and\
                               adj_arc.real != arc.real:
                            # ... remove the extra edges
                            arc_index = arc.imag
                            for i in xrange(repeat):
                                if i == arc_index:
                                    continue
                                to_remove = adj_arc.real+i*1j
                                if to_remove in complex_dual_graph[arc]:
                                    complex_dual_graph[arc].remove(to_remove)

        return complex_dual_graph, faces_containing_arcs

    def path_is_valid(self, current_path):
        """
        Aims to implicitly filter during dfs to decrease output size. Observe
        that more complex filters are applied further along in the function.
        We'd rather do less work to show the path is invalid rather than more,
        so filters are applied in order of increasing complexity.
        :param current_path:
        :return: Boolean indicating a whether the given path is valid
        """
        length = len(current_path)
        if length < 3:
            # The path is too short
            return False

        # Passes through arcs twice... Sketchy for later.
        if len(set(current_path)) != len(current_path):
            return False

        # The idea here is take a moving window of width three along the path
        # and see if it's contained entirely in a polygon.
        for i in xrange(int((length-2)/2)):
            for face in self.non_fourgons:
                if current_path[i] in face and \
                        current_path[i+1] in face and\
                        current_path[i+2] in face:
                    return False

        # This is all kinds of unclear when looking at. There is an edge case
        # pertaining to the beginning and end of a path existing inside of a
        # polygon. The previous filter will not catch this, so we cycle the path
        # a reasonable amount and recheck moving window filter.
        path_copy = [v.real for v in current_path]
        for j in xrange(length):
            path_copy = path_copy[1:] + path_copy[:1]  # wtf
            for i in xrange(int((length-2)/2)):
                for face in (self.non_fourgons[face_id]
                             for face_id in self.non_fourgon_map[path_copy[i]]):
                    #if path_copy[i] in face and \
                    #        path_copy[i+1] in face and \
                    #        path_copy[i+2] in face:
                    if set(path_copy[i:i+3]) <= set(face):
                        return False
        return True

    def faces_share_edges(self, current_path):
        '''
        last_three_vertices = current_path[-3:]
        previous_three_faces = [set(self.faces_containing_arcs[vertex])
                                for vertex in last_three_vertices]
        intersection_all = set.intersection(*previous_three_faces)
        return len(intersection_all) == 2
        '''
        # If it's shorter than 3, it can't have travelled through a face yet
        if len(current_path) >= 3:
            # Count the number of times each face appears by incrementing values
            # of face_id's
            containing_faces = defaultdict(lambda: 0)
            for face in (self.faces_containing_arcs[v.real] for v in
                         current_path[-3:]):
                for f in face:
                    containing_faces[f] += 1
            # If there's any face_id has a value of three, that means that there
            #  is one face that all three arcs bound. This is a trivial path
            # so we discard it.
            return 3 in containing_faces.values()

        return False

    def trim_dual_graph(self, dual_graph, current_path):
        last_vertex, second_last_vertex = current_path[-2:]

        last_face = set(self.faces_containing_arcs[last_vertex.real])
        second_last_face = set(self.faces_containing_arcs[second_last_vertex.real])
        shared_face = last_face.intersection(second_last_face).pop()

        # this is the face who's edges we can remove based on the previous edge
        shared_face_arcs = self.boundaries[shared_face]
        ordered_index = lambda vertex: \
            int(self.repeat*shared_face_arcs.index(vertex.real)+vertex.imag)

        stderr.write(str(last_vertex)+'\t'+str(second_last_vertex)+'\t'+
                     str(shared_face)+'\t'+str(shared_face_arcs)+'\t'+
                     str(ordered_index(last_vertex))+'\t'+str(ordered_index(second_last_vertex))+'\n')

        return dual_graph

    def untrim_dual_graph(self, dual_graph, current_path):
        last_vertex, second_last_vertex = current_path[-2:]

        last_face = set(self.faces_containing_arcs[last_vertex.real])
        second_last_face = set(self.faces_containing_arcs[second_last_vertex.real])
        shared_face = last_face.intersection(second_last_face).pop()

        return dual_graph

    def cycle_dfs(self, current_node,  start_node, current_path):
        """
        Naive depth first search applied to the pseudo-dual graph of the
        reference curve. This sucker is terribly inefficient. More to come.
        :param current_node:
        :param start_node:
        :param graph:
        :param current_path:
        :return:
        """
        #stderr.write(str(current_path)+'\n')
        if self.faces_share_edges(current_path):
            return []
        if current_node == start_node:
            if self.path_is_valid(current_path):
                return [tuple(shift(list(current_path)))]
            else:
                return []
        loops = []
        if not current_path:
            current_path.append(current_node)
            self._d3_dual_graph_copy[start_node].remove(current_node)
            self._d3_dual_graph_copy[current_node].remove(start_node)
            loops += list(self.cycle_dfs(current_node, start_node,
                                         current_path))
            self._d3_dual_graph_copy[start_node].append(current_node)
            self._d3_dual_graph_copy[current_node].append(start_node)
            current_path.pop()
        else:
            #if len(current_path) > 1:
            #    self.trim_dual_graph(self.dual_graph_copy, current_path)

            for adjacent_node in set(self._d3_dual_graph_copy[current_node]):
                current_path.append(adjacent_node)
                self._d3_dual_graph_copy[current_node].remove(adjacent_node)
                self._d3_dual_graph_copy[adjacent_node].remove(current_node)
                loops += list(self.cycle_dfs(adjacent_node, start_node,
                                             current_path))
                self._d3_dual_graph_copy[current_node].append(adjacent_node)
                self._d3_dual_graph_copy[adjacent_node].append(current_node)
                current_path.pop()

            #if len(current_path) > 1:
            #    self.untrim_dual_graph(self.dual_graph_copy, current_path)
        return loops

    def find_d3_cycles(self):
        def remove_vertex_from_graph(vertex, graph):
            # The this removes vertices that have been previously visited in
            # the total traversal. Finding all paths starting at a vertex
            # ultimately finds all paths including that vertex, up to cyclic
            # ordering. Removing them reduces the complexity of the search.
            del graph[vertex]
            for adjacency_list in graph.itervalues():
                if vertex in adjacency_list:
                    adjacency_list.remove(vertex)
            return graph
        d3_cycles = set()
        for vertex in self._d3_dual_graph:

            if vertex in self._d3_dual_graph_copy:
                for adjacent_vertex in self._d3_dual_graph_copy[vertex]:
                    some_cycles = self.cycle_dfs(adjacent_vertex, vertex,[])
                    d3_cycles = d3_cycles | set(some_cycles)

            self._d3_dual_graph_copy = \
                remove_vertex_from_graph(vertex, self._d3_dual_graph_copy)

        d3_cycles = set(d3_cycles)
        return d3_cycles

    def find_cycles(self, repeat=1):
        """
        Finds all possible cycles in the pseudo-dual graph and removes
        remove duplicates to the fullest extend possible.
        :return: a set of filtered candidate cycles
        """
        '''
        def remove_vertex_from_graph(vertex, graph):
            # The this removes vertices that have been previously visited in
            # the total traversal. Finding all paths starting at a vertex
            # ultimately finds all paths including that vertex, up to cyclic
            # ordering. Removing them reduces the complexity of the search.
            del graph[vertex]
            for adjacency_list in graph.itervalues():
                if vertex in adjacency_list:
                    adjacency_list.remove(vertex)
            return graph

        if not self.cycles:
            for vertex in self.dual_graph:

                if vertex in self.dual_graph_copy:
                    for adjacent_vertex in self.dual_graph_copy[vertex]:
                        some_cycles = self.cycle_dfs(adjacent_vertex, vertex,[])
                        self.cycles = self.cycles | set(some_cycles)

                self.dual_graph_copy = \
                    remove_vertex_from_graph(vertex, self.dual_graph_copy)

        self.cycles = set(self.cycles)
        '''
        self.cycles = self.cycle_basis_linear_combination(self.dual_graph, repeat=repeat)
        return self.cycles

    def cycle_basis_linear_combination(self, graph, repeat=1):


        def cycle_vertices_to_edges(cycle):
            n = len(cycle)

            # make a list of edges
            edges_of_cycle = set([(cycle[i % n], cycle[(i+1) % n])
                                  for i in xrange(n)])

            #add in the reverse edges
            edges_of_cycle |= set([edge[::-1]for edge in edges_of_cycle])
            return edges_of_cycle

        def produce_spanning_set(graph, rep, boundaries):
            '''
            The intention here is to produce a set of cycles that spans the same
            space as the space spanned by the cycle basis. There will be more
            elements of this spanning set, but it's much more structured and
            easier to come up with a generic algorithm on this set.

            This spanning set will contain:
                1. the non-trivial basis cycles in the dual graph to compute
                    distance 3
                2. parallel copies of the cycles in 1.
                3. all possible triangles of each 2n-sided polygonal regions
            :return: Structured spanning set of cycles
            '''


            def generate_trivial_cycles(rep, boundaries):
                trivial_cycles = []
                for boundary in boundaries:
                    repeated_boundary = []
                    for i in range(rep):
                        repeated_boundary += [v + i*1j for v in boundary]
                    trivial_cycles += list(combinations(repeated_boundary, 3))

                trivial_cycles_edges = []
                for cycle in trivial_cycles:
                    edges_of_cycle = cycle_vertices_to_edges(cycle)
                    trivial_cycles_edges.append(edges_of_cycle)

                return trivial_cycles_edges

            def generate_nontrivial_cycles(rep, d3_graph):
                # determine the homotopically nontrivial cycles on the original
                # graph (passing through arcs at most once) (Note: these are
                # already in complex form (v_1+0j, v_2+0j, ... v_k+0j)
                stderr.write(str(d3_graph)+'\n')
                nx_graph = nx.Graph(d3_graph)
                cycle_basis = nx.cycle_basis(nx_graph)

                #pull out the ones of length 3 or more
                cycles = [c for c in cycle_basis if len(c) >= 3]

                # remove the cycles that are strictly contained in a single
                # polygon
                d3_non_trivial_cycles = []
                for cycle in cycles:
                    regions_of_cycle = set.intersection(*[set(self.non_fourgon_map[arc])
                                                   for arc in cycle])
                    if len(regions_of_cycle) == 0:
                        d3_non_trivial_cycles.append(cycle)

                non_trivial_cycles = []
                for cycle in d3_non_trivial_cycles:
                    for i in range(rep):
                        parallel_cycle = [arc+i*1j for arc in cycle]

                        # this produces (v_1+i*j, v_2+i*j, ... v_k+i*j), which
                        # is our parallel copy of the original cycle
                        non_trivial_cycles.append(parallel_cycle)

                return non_trivial_cycles

            return generate_nontrivial_cycles(rep, graph)

        # from itertools cookbook
        def powerset(iterable):
            # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
            s = list(iterable)
            return chain.from_iterable(combinations(s, r)
                                       for r in range(len(s)+1))
        #non_trivial_cycles = produce_spanning_set(self._d3_dual_graph, repeat,
        #                         self.boundaries.values())
        cycle_basis = nx.Graph(self.dual_graph)

        cycles = set()
        # produce all possible additions of non-trivial basis elements
        p = powerset(non_trivial_cycles)
        pool = mp.Pool(processes=mp.cpu_count())

        for i, linear_combination in enumerate(p):
            stderr.write('lin_comb_num: '+str(i)+'\n')

            # We need to pull out the edges, both forward and backward, in order
            # to properly perform the symmetric difference of paths. The edges
            # define an ordering that can be recovered through post processing.
            cycles_to_add = []

            # There are three possible operations for each face shared between
            # two cycles. Therefore, for each linear combination of non-trivial
            # cycles, we again need all possible surgeries on these cycles.

            for cycle in linear_combination:
                edges_of_cycle = cycle_vertices_to_edges(cycle)
                cycles_to_add.append(set(edges_of_cycle))

            # Now that we have the cycles we're interested in, we need to
            # determine the cycles in the complete subgraphs to surger these
            # larger cycles together. Some of these are in non_trivial_cycles,
            # but it's possible that the cycles needed at this step are not
            # in that list. Since the set of all possible triangles in all
            # complete subgraphs spans the same space as the original arbitrary
            # basis, we can hand-pick the ones we need for this operation.

            # Even more so, we can determine exactly the three cycles we need to
            # produce the three possible surgeries on two cycles. The can be
            # achieved through iterated cartesian products of interest edges.
            # We elaborate below.

            # Note that because we need to perform 3 different surgeries per
            # face shared between each pair of cycles, and that we're initially
            # using cycle indices instead of the cycles themselves, we will
            # keep track of which cycles are needed to combine each pair of
            # cycles in a dictionary, and pull them out later to symmetric
            # difference them all at once.
            surgery_cycles = {}

            # The first order of business is to determine in which face
            # these cycles intersect, pairwise (addition is a binary operation).
            for cycle_id1, cycle_id2 in\
                    combinations(range(len(cycles_to_add)), 2):

                cycle1_list = linear_combination[cycle_id1]
                cycle2_list = linear_combination[cycle_id2]

                cycle1_set = cycles_to_add[cycle_id1]
                cycle2_set = cycles_to_add[cycle_id2]

                # Identify which non_fourgon each arc passes through, by id.
                cycle1_face_ids_by_arc = [set(self.non_fourgon_map[arc.real])
                                          for arc in cycle1_list]
                cycle2_face_ids_by_arc = [set(self.non_fourgon_map[arc.real])
                                          for arc in cycle2_list]

                cycle1_face_ids = set.union(*cycle1_face_ids_by_arc)
                cycle2_face_ids = set.union(*cycle2_face_ids_by_arc)

                # Determine which ids are shared among the two cycles
                shared_face_ids = cycle1_face_ids & cycle2_face_ids

                def find_cycle_edge_in_face(cycle, face):
                    for edge in cycle:
                        if edge[0].real in face and edge[1].real in face:
                            return edge
                    return None

                # it's possible we have more than one face shared between
                # cycles. We have to include the cycles for each
                for face_id in shared_face_ids:
                    #stderr.write('number of cycles: '+str(count)+'\n')
                    current_face_surgery_cycles = []
                    face = set(self.non_fourgons[face_id])
                    # Pull out the edges that are actually in the face
                    # of interest
                    edge1_in_face = find_cycle_edge_in_face(cycle1_set, face)
                    edge2_in_face = find_cycle_edge_in_face(cycle2_set, face)
                    if not (edge1_in_face and edge2_in_face):
                        continue

                    # One should note here that the cartesian product of the
                    # edges (as collections of vertices) produce one of three
                    # surgery cycles.
                    to_product = set(product(edge1_in_face, edge2_in_face))
                    surger_cycle1 = set([e[::-1] for e in to_product]) | to_product
                    current_face_surgery_cycles.append(surger_cycle1)

                    # Then, of the edges in that surgery cycle, pulling out the
                    # pairs of edges whose element-wise union is the four
                    # original vertices (hence len(...) == 4), the cartesian
                    # product of these edge pairs produce the other two surgery
                    # cycles we need.
                    for edge_pair in combinations(to_product, 2):
                        e1 = set(edge_pair[0])
                        e2 = set(edge_pair[1])

                        if len(set.union(e1, e2)) == 4:
                            alt_surger_cycle = set(product(*edge_pair))
                            alt_surger_cycle |= set([e[::-1] for e in alt_surger_cycle])

                            current_face_surgery_cycles.append(alt_surger_cycle)

                    current_face_surgery_cycles = [c for c in current_face_surgery_cycles
                                                   if c & (cycle1_set | cycle2_set)]
                    # We keep track of what cycles are needed to surger the two
                    # cycles in a given face.
                    surgery_cycles[(cycle_id1, cycle_id2, face_id)] = \
                        current_face_surgery_cycles
            '''
            for curve_surgery_cycles in product(*surgery_cycles.values()):
                cycles_to_add_w_surgeries = \
                    cycles_to_add + list(curve_surgery_cycles)

                resulting_cycle = set()
                for cycle in cycles_to_add_w_surgeries:
                    resulting_cycle ^= cycle

                if not resulting_cycle:
                    continue

                cycle_graph = defaultdict(set)
                for edge in resulting_cycle:
                    cycle_graph[edge[0]].add(edge[1])
                    cycle_graph[edge[1]].add(edge[0])

                if any(len(v) > 2 for v in cycle_graph.itervalues()):
                    continue

                path = list(resulting_cycle.pop())
                current_vertex = path[-1]
                start_vertex = path[0]
                previous_vertex = start_vertex

                while start_vertex != current_vertex:

                    next_vertex = [v for v in cycle_graph[current_vertex]
                                   if v != previous_vertex]
                    previous_vertex = current_vertex
                    current_vertex = next_vertex[0]
                    path.append(next_vertex[0])
                path = path[:-1]

                cycles.add(tuple(shift(path)))
            '''
            #stderr.write('before product\n')
            # for k, curve_surgery_cycles in enumerate(product(*surgery_cycles.values())):
            #     #if k % 100000 == 0: stderr.write(str(k)+'\n')
            #     cycles.add(cycle_addition(curve_surgery_cycles,
            #                               cycles_to_add))#, self.n, self.repeat))
            '''
            sublist_to_map = []
            for k, curve_surgery_cycles in enumerate(product(*surgery_cycles.values())):
                sublist_to_map.append(curve_surgery_cycles)
                if k % 200000 == 0:
                    stderr.write(str(k)+'\n')
                    some_cycles = pool.map(partial(cycle_addition,
                                                   cycles_to_add=cycles_to_add),
                                           sublist_to_map)
                    cycles |= set(some_cycles)
                    sublist_to_map = []
            some_cycles = pool.map(partial(cycle_addition,
                                           cycles_to_add=cycles_to_add),
                                   sublist_to_map)
            '''
            '''
            some_cycles = pool.imap(partial(cycle_addition,
                                           cycles_to_add=cycles_to_add),
                                   product(*surgery_cycles.values()),
                                   chunksize=200000)
            cycles |= set(some_cycles)
            '''
            for curve_surgery_cycles in product(*surgery_cycles.values()):
                resulting_cycle = cycle_addition(curve_surgery_cycles, cycles_to_add)
                if resulting_cycle:
                    stderr.write('resulting cycle: '+str(resulting_cycle)+'\n')
                    stderr.write('non trivial cycles: '+str(cycles_to_add)+'\n')
                    stderr.write('surgery cycle: '+str(curve_surgery_cycles)+'\n\n')
                cycles.add(resulting_cycle)
            #if i >=30: break
        cycles.discard(tuple())
        #pool.close()
        return cycles


def cycle_addition(curve_surgery_cycles, cycles_to_add):#, n, rep):
    cycles_to_add_w_surgeries = \
        cycles_to_add + list(curve_surgery_cycles)

    resulting_cycle = set()
    for cycle in cycles_to_add_w_surgeries:
        resulting_cycle ^= cycle

    if not resulting_cycle:
        return ()

    cycle_graph = defaultdict(set)
    for edge in resulting_cycle:
        cycle_graph[edge[0]].add(edge[1])

    if any(len(v) > 2 for v in cycle_graph.itervalues()):
        return ()
    path = list(resulting_cycle.pop())
    current_vertex = path[-1]
    start_vertex = path[0]
    previous_vertex = start_vertex

    while start_vertex != current_vertex:
        next_vertex = filter(lambda v: v != previous_vertex,
                             cycle_graph[current_vertex])
        previous_vertex = current_vertex
        current_vertex = next_vertex[0]
        path.append(current_vertex)
    path = path[:-1]

    return tuple(shift(path))
