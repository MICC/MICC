import cplex as c
#from cplex._internal._parameter_classes import CPX_PARAM_SOLNPOOLINTENSITY,CPX_PARAM_SOLNPOOLAGAP,CPX_PARAM_POPULATELIM
from itertools import izip


def weights_to_ladder(weights_list, top_order, bottom_order):
    '''
    Genus 2
    
    w1   w2   w6   w3   w5   w6'
    -|----|----|----|----|----|-
    w1   w4   w5   w3   w4'  w2
    #top_order =    [1, 2, 6, 3, 5, -6]
    #bottom_order = [1, 4, 5, 3, -4, 2]
    '''
    weights_list = [int(i) for i in weights_list]
    num_intersections = sum(weights_list)
    w_vals = {w : weight for w, weight in izip(range(1, len(weights_list)+1), weights_list)}
    edges = range(1,num_intersections+1)
    w_edges = {}
    for i in range(1, len(weights_list)+1):
        w_edges[i] = []

        for j in range(w_vals[i]):
            w_edges[i].append(edges[0])
            edges.pop(0)

    top    = []
    bottom = []

    for t,b in izip(top_order,bottom_order):
        if t > 0:
            top.extend(w_edges[abs(t)])
        elif t < 0:
            top.extend(w_edges[abs(t)][::-1])

        if b > 0:
            bottom.extend(w_edges[abs(b)])
        elif b < 0:
            bottom.extend(w_edges[abs(b)][::-1])
    return [top, bottom]


def count(adj_node, adj_list):
    return adj_list.count(adj_node)


def shift(path):
    temp = path.index(min(path))
    return path[temp:] + path[:temp]


def ilp_loop_dfs( current_node,  start_node,  graph,  current_path):
    if current_node == start_node:
        return [shift(list(current_path))]
    else:
        loops = []
        for adjacent_node in set(graph[current_node]):
            if count(adjacent_node, current_path) < 1:
                current_path.append(adjacent_node)
                graph[current_node].remove(adjacent_node)
                graph[adjacent_node].remove(current_node)
                loops += list(ilp_loop_dfs(adjacent_node, start_node, graph, current_path))
                graph[current_node].append(adjacent_node)
                graph[adjacent_node].append(current_node)
                current_path.pop()
        return loops

'''
graph = { 0 : [2,1], 1 : [0,3,6], 2 : [0,3], 3: [1,2,5], 4:[5,7], 5: [3,4,6],6: [1,5,7], 7: [6,4]}
cycles = []
for v in graph.keys():
    for adj_v in graph[v]:
        cycles += ilp_loop_dfs(v,adj_v, graph, [v])

for cycle in set([tuple(c) for c in cycles]):
    print cycle
'''
if __name__ == '__main__':
	c = c.Cplex('genus3.lp')
	c.parameters.mip.limits.populate.set(2000000000)
	#c.parameters.mip.limits.populate.set(2000000)
	##c.parameters.mip.pool.absgap.set(0.0)
	#c.parameters.mip.pool.intensity.set(4)
	#c.CPX_PARAM_SOLNPOOLINTENSITY = 4
	#c.CPX_PARAM_SOLNPOOLAGAP = 0.0
	#c.CPX_PARAM_POPULATELIM = 2000000000
	c.solve()
	c.populate_solution_pool()
	for i in range(c.solution.pool.get_num()):
		test = c.solution.pool.get_values(i)
		print test
'''
print test
l = weights_to_ladder(test)
print l[0]
print l[1]
'''
#print'number', c.solution.pool.get_num()
#print c.solution.pool.get_num_replaced()


#test = c.solution.pool.get_values(0)
#ladder_type_1(test)


