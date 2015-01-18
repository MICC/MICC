import random
import numpy as np
from micc.utils import cycle_to_ladder, ladder_to_cycle, relabel, shift
from copy import copy
from sys import stderr


class RigidGraph(object):
    """
    This class is the generic data structure describing a graphical
    representation of a given curve pair. The constructor is written to populate
    member variables using different process for different inputs, which
    allows for a reasonable level of backwards compatibility and modularity.
    """

    def __init__(self, *args, **kwargs):
        """
        The main purpose of this constructor is to filter out the representation
        that was sent into MICC and call the appropriate function once that has
        been determined. Please be reasonably cautious; only kwargs['ladder'] OR
        kwargs['cycle'] should be true at a given time (only use one rep.
        at a time!). Also, class variables are initialized prior to the argument
        parsing.
        """
        self.graph = {}
        self.n = 0
        self.w = set()  # the vertical curve
        self.v = set()  # the horizontal reference curve
        ladder = kwargs.get('ladder', False)
        cycle = kwargs.get('cycle', False)

        if ladder:
            self.ladder_init(ladder)
        elif cycle:
            self.cycle_init(cycle)

        self.boundaries = self.determine_boundaries()

    def ladder_init(self, ladder):
        ladder_top, ladder_bottom = ladder

        if len(ladder_bottom) != len(ladder_top):
            # raise a custom exception
            pass

        self.n = len(ladder_top)

        # defining some useful modulo arithmetic functions
        mod_n = lambda i: i % self.n
        mod_2n = lambda i: i % (2 * self.n)

        # Initialize the graph with each intersection, denoted here as
        # super_vertex, with sub-vertices that encode the rigidity of the
        # edges. We take advantage of the current iteration and assign the left
        # and right vertex values
        for super_vertex in xrange(self.n):
            self.graph[super_vertex] = {'T': None, 'R': mod_n(super_vertex + 1),
                                        'B': None, 'L': mod_n(super_vertex - 1)}

            # This may seem arbitrary, but we're taking advantage of the fact
            # this ladder_init() performs curve traversals. It's silly to
            # rewrite all of this again, so it's placed here.
            self.v.add((mod_n(super_vertex - 1), super_vertex))

        # Now we populate the 'T' and 'B' portions of the graph. This requires
        # iterating over the identifications of the ladder and determining the
        # endpoints.
        full_ladder = list(ladder_top + ladder_bottom)
        pos = 0

        # quick function to determine 'T' vs 'B'
        # The top of the ladder is the first n elements, while the bottom is
        # second n elements.
        def top_or_bottom(pos):
            if mod_n(pos) == mod_2n(pos):
                return 'T'
            else:
                return 'B'

        while pos < 2 * self.n:
            # Ladders must have two copies of each identification by definition,
            # so we iterate over the entire ladder and kill off identifications
            # as we see them, thus allowing .index(...) to always return the
            # proper endpoint.
            identification = full_ladder[pos]
            if not identification:
                # then we've already seen this identification; skip it
                pos += 1
                continue
            full_ladder[pos] = None
            endpoint = full_ladder.index(identification)
            full_ladder[endpoint] = None

            start_intersection = mod_n(pos)
            end_intersection = mod_n(endpoint)
            self.graph[start_intersection][top_or_bottom(pos)] = mod_n(endpoint)
            self.graph[end_intersection][top_or_bottom(endpoint)] = mod_n(pos)

            # Again, we're just building up the transverse curve as we traverse
            # it here.
            self.w.add((mod_n(endpoint), mod_n(pos)))

            pos += 1

    def cycle_init(self, cycle):
        ladder = relabel(cycle_to_ladder(cycle))
        self.ladder_init(ladder)

    def determine_boundaries(self):
        """
        Returns the polygonal regions in the complement of the union of the two
        curves in the surface. Not only does it produce the number of
        components and the size of each one, but it also returns the sequence of
        alternating curve arcs that ultimately bound each region.
        """
        # We need a set of unused edges to "seed" our boundary search. Edges are
        # written as 1B, 1R, 1T, 1L, 2B, 2R, etc. This doubles as our finishing
        # criteria.

        get_key = lambda vert, dir: str(vert) + dir
        split_key = lambda key: (int(key[:-1]), key[-1])
        unused_edges = {}

        for v in xrange(self.n):
            for direction in ['B', 'R', 'T', 'L']:
                unused_edges[get_key(v, direction)] = 0
        boundaries = []

        def get_boundary(start_edge, unused_edges):
            boundary = []

            # We need to explicitly define the traversal of the boundary of the
            # polygonal region we're considering. The following to dictionary
            # defines a clockwise traversal and easily determines the next
            # edge to use. Since we can't cross an edge, the "next" edge in the
            # boundary path is the "next" element of the cyclic sequence
            # (B, R, T, L). This is read as:
            # B => R => T => L => B... etc.
            clockwise_traversal = {'B': 'R', 'R': 'T', 'T': 'L', 'L': 'B'}

            def locate_identification(source_edge):
                source_vertex, source_direction = split_key(source_edge)

                # get the next edge
                rotated = clockwise_traversal[source_direction]
                # get the end point of that edge...
                target_vertex = self.graph[source_vertex][rotated]
                target_dir = None
                if rotated in ('B', 'T'):
                    # there are only two options for each case, here we
                    # consider top/bottom. We're pulling T/B depending on which
                    # value corresponds to source_vertex
                    for k, val in self.graph[target_vertex].iteritems():
                        if k in ('B','T') and val == source_vertex:
                            target_dir = k
                else:  # it's in ('R', 'L')
                    for k, val in self.graph[target_vertex].iteritems():
                        if k in ('R', 'L') and val == source_vertex:
                            target_dir = k
                return [get_key(source_vertex, rotated),
                        get_key(target_vertex, target_dir)]

            current_edge = locate_identification(start_edge)

            previous_edge = start_edge

            boundary += [start_edge]
            unused_edges[start_edge] = 1

            while current_edge != start_edge:
                boundary += locate_identification(previous_edge)
                current_edge = boundary[-1]
                previous_edge = current_edge
                unused_edges[previous_edge] = 1

                unused_edges = {vert: count for vert, count in
                                unused_edges.iteritems() if count < 1}
            return boundary[:-1], unused_edges

        while unused_edges:
            boundary, unused_edges = get_boundary(
                random.choice(unused_edges.keys()), unused_edges)
            boundaries.append(boundary)

        return boundaries


class CurvePair(object):
    """
    This class encapsulates all of the information required to compute and study
    distance in the curve complex. It functions as an interface between the user
    and the MICC internal API. This means that a user wishing to purely compute
    distance should do by instantiating and manipulating this object, not by
    calling MICC API functions.

    Attributes:
        ladder: a list of lists:
        cycle: a string of alternating integers and +/- signs that characterize
        the curve pair.
        compute: boolean indicating whether or not to compute distance
    """

    def __init__(self, input_rep, compute=False, *args, **kwargs):
        self.rigid_graph = None

        if '+' in input_rep or '-' in input_rep:
            self.cycle = input_rep
            self.ladder = relabel(cycle_to_ladder(input_rep))
        else:
            self.ladder = input_rep
            self.cycle = ladder_to_cycle(*input_rep)

        self.rigid_graph = RigidGraph(ladder=self.ladder)
        self.n = self.rigid_graph.n
        self.verbose_boundaries = self.rigid_graph.boundaries
        self.concise_boundaries = \
            self.boundary_reduction(self.verbose_boundaries)
        self.graph = None
        if compute:
            self.graph = Graph(self.concise_boundaries, self.n)
            pass

    def boundary_reduction(self, verbose_boundaries):
        """
        Further parse down the polygonal boundaries to only hold arcs of the
        reference curve. This allows for producing the curves in the complement
        of the transverse curve.
        :param verbose_boundaries: The boundaries produced by
        RigidGraph.determine_boundaries(), in the form:
        ['4T', '4L', '3R', '3T', '1T', '1L', '0R', '0T', '3B', '3R', '4L', '4B',
         '0B', '0R', '1L', '1B', '2T', '2L', '1R', '1T', '3T', '3L', '2R', '2T',
         '1B', '1R', '2L', '2B']]
        :return: polygonal boundaries that strictly contain only arcs of the
        reference curve (L/R arcs), not the transverse curve (T/B). Edges are
        indexed by the vertex id of the LEFT endpoint; for example the above
        verbose output should become:
        [3, 0, 3, 0, 1, 2, 1]
        """
        concise_boundaries = {}

        for index, boundary in enumerate(verbose_boundaries):
            stderr.write(str(boundary)+'\n')
            concise_boundary = [int(edge[:-1]) for edge in boundary
                                if edge[-1] == 'R']  # if unclear, look at
                                                     # example above
            # Indexing polygons gives a unique identification from P_i to
            # the proper ith region. It's arbitrary, but useful later
            concise_boundaries[index] = concise_boundary

        return concise_boundaries


class Graph(object):

    def __init__(self, boundaries, n):
        self.n = n
        self.boundaries = boundaries
        self.dual_graph, self.faces_containing_arcs = \
            self.create_dual_graph(self.boundaries)
        self.find_cycles()

    def create_dual_graph(self, boundaries):
        """
        This function constructs the pseudo-dual graph of our reference curve
        arcs by inspecting the connectivity information of the complementary
        polygons.
        :param boundaries: The return value of boundary_reduction()
        :return: An adjacency list ({int: list}) that:
            - Vertices are reference curve arcs
            - Edge connect vertices if the arcs are connected by a polygonal
            region.
        """
        #boundary_sets = [set(b) for b in boundaries]
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

        # one vertex per reference arc
        dual_graph = {v: [] for v in xrange(self.n)}
        regions = self.fourgons + self.non_fourgons
        for region in regions:
            altered_region = list(region)

            for arc in region:
                altered_region.remove(arc)

                for other_edge in altered_region:
                    if other_edge not in dual_graph[arc]:
                        dual_graph[arc].append(other_edge)

                    if arc not in dual_graph[other_edge]:
                        dual_graph[other_edge].append(arc)

                altered_region.append(arc)  # COULD BE SO BAD

        return dual_graph, faces_containing_arcs

    def path_is_valid(self, current_path):
        """
        Aims to implicitly filter during dfs to decrease output size. Observe
        that more complex filters are applied further down the function.
        We'd rather do less work to show the path is invalid rather than more,
        so filters are applied in order of increasing complexity.
        :param current_path:
        :return: Boolean indicating a whether the given path is valid
        """
        length = len(current_path)
        if length < 3:
            # The path is too short
            return False

        # The idea here is take a moving window of width three along the path
        # and see if it's contained entirely in a polygon.
        arc_triplets = (current_path[i:i+3] for i in xrange(length-2))
        for triplet in arc_triplets:
            for face in self.non_fourgons:
                if set(triplet) <= set(face):
                    return False

        # This is all kinds of unclear when looking at. There is an edge case
        # pertaining to the beginning and end of a path existing inside of a
        # polygon. The previous filter will not catch this, so we cycle the path
        # a reasonable amount and recheck moving window filter.
        path_copy = list(current_path)
        for i in xrange(int(length/4)):
            path_copy = path_copy[1:] + path_copy[:1]  # wtf
            arc_triplets = (path_copy[i:i+3] for i in xrange(length-2))
            for triplet in arc_triplets:
                for face in self.non_fourgons:
                    if set(triplet) <= set(face):
                        return False

        return True

    def cycle_dfs(self, current_node,  start_node,  graph, current_path):
        """
        Naive depth first search applied to the pseudo-dual graph of the
        reference curve. This sucker is terribly inefficient. More to come.
        :param current_node:
        :param start_node:
        :param graph:
        :param current_path:
        :return:
        """
        if len(current_path) >= 3:
            last_three_vertices = current_path[-3:]
            previous_three_faces = [set(self.faces_containing_arcs[vertex])
                                    for vertex in last_three_vertices]
            intersection_all = set.intersection(*previous_three_faces)
            if len(intersection_all) == 2:
                return []

        if current_node == start_node:
            if self.path_is_valid(current_path):
                return [tuple(shift(list(current_path)))]
            else:
                return []

        else:
            loops = []
            for adjacent_node in set(graph[current_node]):
                current_path.append(adjacent_node)
                graph[current_node].remove(adjacent_node)
                graph[adjacent_node].remove(current_node)
                loops += list(self.cycle_dfs(adjacent_node, start_node,
                                             graph, current_path))
                graph[current_node].append(adjacent_node)
                graph[adjacent_node].append(current_node)
                current_path.pop()
            return loops

    def find_cycles(self):
        """
        Finds all possible cycles in the pseudo-dual graph and removes
        remove duplicates to the fullest extend possible.
        """
        cycles = []
        cycle_set = set()
        for vertex in self.dual_graph:
            for adjacent_vertex in self.dual_graph[vertex]:
                cycles = self.cycle_dfs(vertex, adjacent_vertex,
                                        self.dual_graph, [])
                cycle_set = cycle_set | set(cycles)
        for cycle in cycle_set:
            stderr.write(str(cycle)+'\n')
            if len(cycle) < 3:
                continue
        stderr.write(str(len(cycle_set))+'\n')

