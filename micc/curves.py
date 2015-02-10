from itertools import izip
import random
from micc.graph import Graph
from micc.utils import cycle_to_ladder, ladder_to_cycle, relabel, shift, invert, \
    complex_cmp
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
                        if k in ('B', 'T') and val == source_vertex:
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

    def __init__(self, input_rep, compute=False, arc_path=None,
                 recursive=False, *args, **kwargs):
        self.rigid_graph = None
        if arc_path:
            self.arc_path = arc_path
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

        self.genus = self.genus(self.concise_boundaries)
        self.graph = Graph(self.concise_boundaries, self.n)

        if compute:
            self.distance, self.complementary_curves, self.graph = \
                self.compute_distance(recursive=recursive)


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
            concise_boundary = [int(edge[:-1]) for edge in boundary
                                if edge[-1] == 'R']  # if unclear, look at
                                                     # example above
            # Indexing polygons gives a unique identification from P_i to
            # the proper ith region. It's arbitrary, but useful later
            concise_boundaries[index] = concise_boundary

        return concise_boundaries

    def genus(self, boundaries):
        """
        Compute lowest genus surface that the pair of curves fill upon.
        :param boundaries:
        :return:
        """
        number_of_bigons = sum([1 for bigon in boundaries.itervalues()
                                if len(bigon) == 1])

        v = self.n  # by definition
        e = 2 * self.n  # this should be clear from looking at RigidGraph.graph
        f = len(boundaries)
        euler = v - e + f
        genus = 1 - euler/2
        if number_of_bigons:
            genus -= 1
        return genus

    def compute_distance(self, recursive=False):
        """
        Returns distance of the CurvePair object.
        :return: An integer distance (3, 4, etc.)
        """
        distance = 0

        def regions_share_two_edges(regions):
            for region_index in regions:
                for other_index in regions:
                    if region_index == other_index:
                        continue
                    if len(set(regions[region_index]) &
                           set(regions[other_index])) > 1:
                        # This means they share more than one side
                        return True
            return False

        def regions_share_itself(regions):
            for region in regions.itervalues():
                n = len(region)
                if n == 1:
                    # This means there's a bigon
                    return True
                if len(region) != len(set(region)):
                    # One of the edges appears twice
                    return True
            return False

        if regions_share_itself(self.concise_boundaries):
            return 3
        elif regions_share_two_edges(self.concise_boundaries):
            return 3

        else:
            def is_valid(cycle):
                n = len(cycle)
                for i in xrange(n+1):
                    arc = cycle[i % n]
                    next_arc = cycle[(i+1) % n]
                    if next_arc not in self.graph.dual_graph[arc]:
                        return False
                return True

            for rep in xrange(1, 2+1):  # max of distance 4

                # Create a graph of appropriate replication number
                self.graph = Graph(self.concise_boundaries, self.n, repeat=rep)
                cycles = self.graph.find_cycles()

                # Quick removal of inverted paths:
                cycles_no_inversion = set()
                for cycle in cycles:
                    if invert(cycle) in cycles_no_inversion:
                        continue
                    cycles_no_inversion.add(cycle)
                cycles = cycles_no_inversion

                for cycle in cycles:
                    stderr.write(str([int(v.real) for v in cycle])+'\n')
                stderr.write(str(len(cycles))+'\n')

                complement_curves = []
                for cycle in cycles:
                    if not is_valid(cycle):
                        continue
                    curvepair_in_comp = self.curvepair_from_arc_path(cycle,
                                                            compute=recursive)

                    if curvepair_in_comp.genus <= self.genus:
                        complement_curves.append(curvepair_in_comp)
                        # This creates the short-list of curves in the
                        # complement of the transverse curve intersection the
                        # reference curve.

                if recursive:
                    distances = set([curvepair.distance for curvepair in
                                     complement_curves])

                    if len(distances) == 1:
                        d = distances.pop()
                        distance = 'at least '+str(d+1)
                    else:
                        d = min(distances)
                        distance = d+1

                else:
                    # we can only test for distance 3 and 'at least 4'
                    genuses = set([curvepair.genus for curvepair in
                                   complement_curves])

                    if len(genuses) == 1:
                        distance = 'at least '+str(4)
                    else:
                        min_genus = min(genuses)
                        if min_genus < self.genus:
                            distance = 3
        return distance, complement_curves, self.graph

    def curvepair_from_arc_path(self, arc_path, compute=False):
        """
        Reconstruct a CurvePair object from a sequence of reference arcs
        defining a path in the complement of the transverse curve. This forms
        a sort of hierarchy that is useful for higher distance; the member
        variables of the target CurvePair object are being used to spawn a new
        CurvePair in the complement. This allows for a straightforward tree of
        parent-child CurvePair objects, where the children recursively determine
        the distances of their parents.
        :param arc_path: A sequence of reference arcs from the current CurvePair
        :param compute: pass along the distance computation parameter
        :return: A child CurvePair whose intersections are determined by the
                polygonal connections of the current CurvePair
        """
        parent_arcs = sorted(arc_path, key=complex_cmp)
        child_arcs = range(len(parent_arcs))  # arcs are labeled 0 - n-1
        # For easier access, we map the resulting child arcs to
        # the old parent arcs to keep track of what goes where
        map_parent_to_child = {p_arc: c_arc for c_arc, p_arc in
                               izip(child_arcs, parent_arcs)}
        to_child = lambda a: map_parent_to_child[a]

        # Recall that the reference arcs are labeled by the left endpoints.
        # These are determined by the edges in verbose_boundaries that contain
        # R's:
        # For example:
        #                            |
        #                            T
        #                            |
        #                            ^
        #                            |
        #                          --+--<- R --
        #                            |
        # The R labels in verbose boundaries correspond to the arcs of arc_path.
        # The T/B that is attached to a specific R will determine how the arc
        # connects to the next arc in the path.

        # fill up the ladder to allow for indexing later
        ladder = [[None] * len(arc_path), [None] * len(arc_path)]

        def find_region_with_arcs(current_arc, next_arc):
            # These is only going to be one region that has current_arc+R
            # and next_arc+R; if more than one had each, they would share two
            # edges and get caught earlier on.
            current_str = str(current_arc)+'R'
            next_str = str(next_arc)+'R'

            for region in [r for r in self.verbose_boundaries]:
                if current_str in region and next_str in region:
                    return region

        # Cleaner way to iterate over sequential arc pairs cyclically

        def find_direction(arc, region):
            # Again looking for the R arcs
            arc_str = str(arc)+'R'
            arc_location = region.index(arc_str)  # unique edges so this works
            if arc_location == 0:
                surrounding_arcs = [region[-1]]+region[:2]
            else:
                surrounding_arcs = region[arc_location-1:arc_location+2]
            # Filter out arc_str and the other spurious arc to extract the
            # true arc direction.
            direction = [edge[-1] for edge in surrounding_arcs
                         if int(edge[:-1]) == arc and
                         edge != arc_str].pop()
            return direction

        for i, (arc, next_arc) in enumerate(izip(arc_path, list(arc_path[1:]) +
                                            [arc_path[0]])):
            arc_id = int(arc.real)
            next_arc_id = int(next_arc.real)

            region = find_region_with_arcs(arc_id, next_arc_id)

            # Figure out where the edges are actually leading
            arc_direction = find_direction(arc_id, region)
            next_arc_direction = find_direction(next_arc_id, region)
            # Fill the ladder based on the directions
            # TODO make this less ugly and stupid

            child_ind = to_child(arc)
            next_child_ind = to_child(next_arc)

            if 'T' == arc_direction and 'T' == next_arc_direction:
                ladder[0][child_ind] = i
                ladder[0][next_child_ind] = i

            elif 'T' == arc_direction and 'B' == next_arc_direction:
                ladder[0][child_ind] = i
                ladder[1][next_child_ind] = i

            elif 'B' == arc_direction and 'T' == next_arc_direction:
                ladder[1][child_ind] = i
                ladder[0][next_child_ind] = i

            elif 'B' == arc_direction and 'B' == next_arc_direction:
                ladder[1][child_ind] = i
                ladder[1][next_child_ind] = i

        ladder[0] = [j+1 for j in ladder[0]]
        ladder[1] = [j+1 for j in ladder[1]]
        return CurvePair(ladder, arc_path=arc_path, compute=compute)