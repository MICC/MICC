import multiprocessing as mp
from graph import shift
from time import sleep

def count(adj_node, adj_list):
    return adj_list.count(adj_node)

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
            if count(adjacent_node, current_path) < 1:
                current_path.append(adjacent_node)
                graph[current_node].remove(adjacent_node)
                graph[adjacent_node].remove(current_node)
                loops += list(loop_dfs(adjacent_node, start_node, graph, current_path, nodes_to_faces))
                graph[current_node].append(adjacent_node)
                graph[adjacent_node].append(current_node)
                current_path.pop()
        return loops


def dfs_partial( current_node,  start_node,  graph,  current_path,  nodes_to_faces):

    if len(current_path) >= 3:
        path_head_3 = current_path[-3:]
        previous_three_faces = [set(nodes_to_faces[edge]) for edge in path_head_3]
        intersection_all = set.intersection(*previous_three_faces)
        if len(intersection_all) == 2:
            return []

    if current_node == start_node:
        return [shift(list(current_path))]

    if len(current_path) >= graph.keys()/4:
        return  [('work', current_node,  start_node,  graph,  current_path,  nodes_to_faces)]

    else:
        loops = []
        for adjacent_node in set(graph[current_node]):
            if count(adjacent_node, current_path) < 1:
                current_path.append(adjacent_node)
                graph[current_node].remove(adjacent_node)
                graph[adjacent_node].remove(current_node)
                loops += list(loop_dfs(adjacent_node, start_node, graph, current_path, nodes_to_faces))
                graph[current_node].append(adjacent_node)
                graph[adjacent_node].append(current_node)
                current_path.pop()
        return loops


class PGraph:

    class Master:
        num_threads = 0
        workers = []

        def __init__(self):
            self.num_threads = mp.cpu_count()
            self.work_queue = mp.Queue()
            self.results_queue = mp.Queue()
            self.workers = [PGraph.Worker(self.work_queue, self.results_queue) for i in range(self.num_threads+2)]

        def execute(self):
            #starts all workers, then begins searching for work for them to complete
            p = mp.Process(target=self.find_work())
            p.start()
            for worker in self.workers:
                worker.run()
            p.join()

        def find_work(self):
            pass

    class Worker:
        threads = []

        def __init__(self, work_queue, results_queue):
            self.results_queue = results_queue
            self.work_queue = work_queue

        def run(self):
            while not self.work_queue.empty():
                proc = mp.Process(target=loop_dfs, args=self.work_queue.get(0))
                proc.start()
                self.threads.append(proc)
            for thread in self.threads:
                thread.join()

    def __init__(self):
        pass
