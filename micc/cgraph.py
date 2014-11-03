from ctypes import cdll, byref, py_object,c_float, c_int, Structure
import subprocess
from json import dumps,loads

def shift(path):
    '''
    init

    '''
    temp = path.index(min(path))
    return path[temp:] + path[:temp]

def loop_dfs( current_node,  start_node,  graph,  current_path,  nodes_to_faces):
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
            if current_path.count(adjacent_node) < 1:
                current_path.append(adjacent_node)
                graph[current_node].remove(adjacent_node)
                graph[adjacent_node].remove(current_node)
                loops += list(loop_dfs(adjacent_node, start_node, graph, current_path, nodes_to_faces))
                graph[current_node].append(adjacent_node)
                graph[adjacent_node].append(current_node)
                current_path.pop()
        return loops

#lib.loop_dfs(0,0,py_object({1:'hello'}),py_object([]),py_object({}))
to_cpp = [0, 8, {0: [11, 8, 15], 1: [9, 12], 2: [10, 13, 15], 3: [11, 6, 14],
                              4: [12, 7], 5: [13, 8], 6: [3, 9, 14], 7: [4, 16, 10], 8: [0, 5],
                              9: [1, 6], 10: [2, 7, 16], 11: [0, 3, 15], 12: [1, 4],
                              13: [2, 5, 15], 14: [3, 6, 16], 15: [0, 2, 11, 13], 16: [7, 10, 14]}, [0],
                               {0: (0, None), 1: (None, None), 2: (1, None), 3: (2, None),
                                    4: (None, None), 5: (None, None), 6: (2, None), 7: (3, None),
                                    8: (None, None), 9: (None, None), 10: (3, None), 11: (0, None),
                                    12: (None, None), 13: (1, None), 14: (2, None), 15: (0, 1),16: (3, None)}]

'''
l = loop_dfs(0, 8, {0: [11, 8, 15], 1: [9, 12], 2: [10, 13, 15], 3: [11, 6, 14],
                      4: [12, 7], 5: [13, 8], 6: [3, 9, 14], 7: [4, 16, 10], 8: [0, 5],
                      9: [1, 6], 10: [2, 7, 16], 11: [0, 3, 15], 12: [1, 4],
                      13: [2, 5, 15], 14: [3, 6, 16], 15: [0, 2, 11, 13], 16: [7, 10, 14]},[0],
          {0: (0, None), 1: (None, None), 2: (1, None), 3: (2, None),
           4: (None, None), 5: (None, None), 6: (2, None), 7: (3, None),
           8: (None, None), 9: (None, None), 10: (3, None), 11: (0, None),
           12: (None, None), 13: (1, None), 14: (2, None), 15: (0, 1),16: (3, None)})
'''


def cdfs(current_node, start_node, graph, current_path, nodes_to_face):
    to_cpp = [current_node, start_node, current_path, graph, nodes_to_face]
    import os
    j = dumps(to_cpp,separators=(',' , ':'))
    f = open('intermed.json','w')
    f.write(j)
    f.close()
    #os.system("./util-dfs intermed.json")
    #s = subprocess.check_output("./util-dfs "+dumps(to_cpp,separators=(',' , ':')), shell=True)
    s = subprocess.check_output("micc/util-dfs intermed.json", shell=True)
    '''
    from sys import stderr
    stderr.write("FUCK THIS HORSESHIT")
    stderr.write(s)
    '''
    return loads(s)

#cdfs(to_cpp[0], to_cpp[1], to_cpp[2], to_cpp[3], to_cpp[4] )

#s = subprocess.check_output("./util-dfs "+dumps(to_cpp,separators=(',' , ':')), shell=True)

#g = Graph()
#g.bar()

