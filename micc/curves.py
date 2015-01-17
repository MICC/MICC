import random
import numpy as np
from micc.utils import cycle_to_ladder, relabel
from copy import copy
from sys import stderr


class RigidGraph(object):
    """
    This class is the generic data structure describing a graphical
    representation of a given curve pair. The constructor is written to populate
    member variables using different algorithms for differing inputs, which
    allows for a reasonable level of backwards compatibility.
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
        split_key = lambda key: (int(key[0]), key[1])
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

    def __init__(self, compute=False, *args, **kwargs):
        if compute:
            pass
        pass
