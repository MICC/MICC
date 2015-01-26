from collections import defaultdict
from itertools import izip
from micc.utils import shift
from sys import stderr


class Graph(object):

    def __init__(self, boundaries, n):
        self.n = n
        self.boundaries = boundaries
        self.dual_graph, self.faces_containing_arcs = \
            self.create_dual_graph(self.boundaries)
        self.cycles = set()

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
        self.non_fourgon_map = {arc: [] for arc in xrange(self.n)}
        for face_id, face in enumerate(self.non_fourgons):
            for arc in self.non_fourgon_map:
                if arc in face:
                    self.non_fourgon_map[arc].append(face_id)

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
        for i in xrange(length-2):
            for face in self.non_fourgons:
                if current_path[i] in face and \
                        current_path[i+1] in face and\
                        current_path[i+2] in face:
                    return False

        # This is all kinds of unclear when looking at. There is an edge case
        # pertaining to the beginning and end of a path existing inside of a
        # polygon. The previous filter will not catch this, so we cycle the path
        # a reasonable amount and recheck moving window filter.
        path_copy = list(current_path)
        for j in xrange(length):
            path_copy = path_copy[1:] + path_copy[:1]  # wtf
            for i in xrange(length-2):
                for face in (self.non_fourgons[face_id]
                             for face_id in self.non_fourgon_map[path_copy[i]]):
                    if path_copy[i] in face and \
                            path_copy[i+1] in face and \
                            path_copy[i+2] in face:
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
            for face in (self.faces_containing_arcs[v]
                         for v in current_path[-3:]):
                for f in face:
                    containing_faces[f] += 1
            # If there's any face_id has a vlue of three, that means that there
            #  is one face that all three arcs bound. This is a trivial path
            # so we discard it.
            return 3 in containing_faces.values()

        return False

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
        if self.faces_share_edges(current_path):
            return []
        if current_node == start_node:
            if self.path_is_valid(current_path):
                return [tuple(shift(list(current_path)))]
            else:
                return []

        else:
            loops = []
            for adjacent_node in set(self.dual_graph[current_node]):
                current_path.append(adjacent_node)
                self.dual_graph[current_node].remove(adjacent_node)
                self.dual_graph[adjacent_node].remove(current_node)
                loops += list(self.cycle_dfs(adjacent_node, start_node,
                                             current_path))
                self.dual_graph[current_node].append(adjacent_node)
                self.dual_graph[adjacent_node].append(current_node)
                current_path.pop()
            return loops

    def find_cycles(self):
        """
        Finds all possible cycles in the pseudo-dual graph and removes
        remove duplicates to the fullest extend possible.
        :return: a set of filtered candidate cycles
        """
        if not self.cycles:
            verts = set([item for sublist in self.non_fourgons
                         for item in sublist])
            for vertex in verts: #self.dual_graph:
                for adjacent_vertex in self.dual_graph[vertex]:
                    some_cycles = self.cycle_dfs(vertex, adjacent_vertex, [])
                    self.cycles = self.cycles | set(some_cycles)
        self.cycles = set(self.cycles)

        return self.cycles



