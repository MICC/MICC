from ctypes import cdll, byref, py_object,c_float, c_int, Structure
lib = cdll.LoadLibrary('./libdfs.so')


class Graph(object):

    def __init__(self):
        self.obj = lib.Graph_new()

    def bar(self):
        lib.Graph_bar(self.obj)

lib.loop_dfs(0,0,py_object({1:'hello'}),py_object([]),py_object({}))
#g = Graph()
#g.bar()

