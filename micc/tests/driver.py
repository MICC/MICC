from micc.curves import CurvePair, edges, build_matrices, fix_matrix_signs,Three
from micc.graph import Graph
from line_profiler import LineProfiler


def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner


def get_number():
    for x in xrange(5000000):
        yield x

@do_profile(follow=[get_number])
def expensive_function():
    for x in get_number():
        i = x ^ x ^ x
    return 'some result!'

#result = expensive_function()


def get_dist(top,bot):
    return CurvePair(top,bot).distance

#@do_profile(follow=[get_dist,Graph.compute_loops, Graph.iter_loop_dfs,Graph.faces_share_edges,Graph.loop_dfs, edges, build_matrices,fix_matrix_signs])
@do_profile(follow=[get_dist,CurvePair.compute_distance, Three, Graph.compute_loops, Graph.loop_dfs])
def profile_distance(top,bot):
    return get_dist(top,bot)


#print profile_distance([6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 1, 2, 3, 4, 17, 5],
#                [1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4])
#print profile_distance([23, 22, 1, 2, 3, 4, 5, 22, 23, 24, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24],
#                       [1, 2, 3, 4, 19, 20, 21, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 20, 19, 5])

#print profile_distance([1,11,3,27,8,15,7,24,29,10,2,12,4,21,19,17,24,14,6,23,28,9,16,25,13,5,20,18,16],[29,10,2,26,7,14,6,23,28,9,1,11,3,22,20,18,25,13,5,22,27,8,15,26,12,4,21,19,17])
print profile_distance([1,2,3,4,12,10,14,15,9,13,11,12,4,5,6,7,21,22,23,24,16,17,18,19],[24,1,2,3,13,9,15,16,8,14,10,11,5,6,7,8,20,21,22,23,17,18,19,20])
