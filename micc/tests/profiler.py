from micc.curves import CurvePair
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


def get_dist(ladder):
    return CurvePair(ladder, compute=True).distance

#@do_profile(follow=[get_dist,Graph.faces_share_edges, Graph.cycle_dfs, CurvePair.compute_distance,Graph.find_cycles, Graph.path_is_valid])
@do_profile(follow=[get_dist,Graph.faces_share_edges,Graph.cycle_basis_linear_combination,Graph.find_cycles, Graph.path_is_valid])
def profile_distance(ladder):
    return get_dist(ladder)

#ladder = [[1, 7, 18, 24, 5, 16, 12, 8, 19, 25, 6, 17, 23, 4, 15, 11, 22, 3, 14, 10, 21, 2, 13, 9, 20],
#          [25, 6, 17, 23, 4, 15, 11, 7, 18, 24, 5, 16, 22, 3, 14, 10, 21, 2, 13, 9, 20, 1, 12, 8, 19]]


ladder = [[6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 1, 2, 3, 4, 17, 5],
          [1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4]]
#ladder = [[23, 22, 1, 2, 3, 4, 5, 22, 23, 24, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24],
#          [1, 2, 3, 4, 19, 20, 21, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 20, 19, 5]]
profile_distance(ladder)
