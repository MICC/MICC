from copy import deepcopy
from sys import maxint
index = 0

class Johnson:

    def __init__(self, graph):
        self.n = len(graph.keys())
        self.k = 0
        self.graph = graph
        self.circuits = []
        self.blocked = {}
        self.blocked_nodes = {}

    def circuit(self, subgraph, v, s, stack):
        if len(subgraph.keys()) is 0:
            return False

        f = False
        stack.append(v)
        self.blocked[v] = True

        for w in self.graph[v]:
            if w is s:
                #stack.append(s)
                self.circuits.append(deepcopy(stack))
                #stack.pop()
                f = True
            else:
                if not self.blocked[w]:
                    if self.circuit(subgraph, w, s, stack):
                        f = True

        if f:
            self.unblock(v)
        else:
            for w in self.graph[v]:
                if v not in self.blocked_nodes[w]:
                    self.blocked_nodes[w].append(v)

        stack.pop()
        return f

    def unblock(self, u):
        self.blocked[u] = False

        while len(self.blocked_nodes[u]) > 0:
            w = self.blocked_nodes[u].pop(0)
            if self.blocked[w]:
                self.unblock(w)

    @staticmethod
    def least_vertex(graph):
        return min(graph.keys())

    @staticmethod
    def subgraph_from_vertex(i, graph):
        result = {j : [] for j in graph.keys()}
        for end_point in graph.keys():
            if end_point >= i:
                for start_point in graph[end_point]:
                    if start_point >= i:
                        result[start_point].append(end_point)
                        result[end_point].append(start_point)

        return result

    @staticmethod
    def least_scc(graph):
        def get_sccs(graph):
            sccs = []
            index = 0
            stack = []
            index_map = {}
            low_link_map = {}

            def strongly_connected(v):
                global index
                index_map[v] = index
                low_link_map[v] = index
                index += 1
                stack.append(v)
                for w in graph[v]:
                    if w not in index_map:
                        strongly_connected(w)
                        low_link_map[v] = min([low_link_map[v], low_link_map[w]])
                    else:
                        if w in stack:
                            low_link_map[v] = min([low_link_map[v], index_map[w]])

                result = []
                if low_link_map[v] == index_map[v]:
                    scc_list = []
                    while True:
                        w = stack.pop()
                        scc_list.append(w)
                        if w == v:
                            break
                    if len(scc_list) > 1:
                        result.append(scc_list)
                return result

            for v in graph.keys():
                if v not in index_map:
                    sccs += strongly_connected(v)
            return sccs

        def add_edges(min_scc, graph):  #remove this and make a generic funciton to call
            result = {i: [] for i in min_scc}
            for i in min_scc:
                for edge in [(i, j) for j in graph[i]]:
                    to = edge[1]
                    if to in min_scc:
                        result[i].append(to)
                        result[to].append(i)
            return result

        sccs = get_sccs(graph)
        minint = maxint
        min_scc = []
        for scc in sccs:
            if len(scc) == 1:
                continue
            for i in scc:
                if i < minint:
                    min_scc = scc
                    minint = i
        return add_edges(min_scc, graph)

    def find_all_circuits(self):
        #self.blocked = {i : False for i in range(self.n)}
        #self.blocked_nodes = {i : [] for i in range(self.n)}
        stack = []
        s = 0  #possibly zero in this setting
        while s < self.n:
            subgraph = Johnson.subgraph_from_vertex(s, self.graph)
            least_scc = Johnson.least_scc(subgraph)
            if len(least_scc.keys()) > 0:
                s = Johnson.least_vertex(least_scc)
                for i in least_scc.keys():
                    self.blocked[i] = False
                    self.blocked_nodes[i] = []
                self.circuit(least_scc, s, s, stack)
                s += 1
            else:
                s = self.n
if __name__ == '__main__':
    j = Johnson({
        1: [2, 6],
        2: [3, 1],
        3: [4, 2, 6],
        4: [5, 3],
        5: [6, 4],
        6: [5, 1, 4]
    })

    j.find_all_circuits()
    print j.circuits