from ctypes import cdll, c_float, c_int, Structure
lib = cdll.LoadLibrary('./libdfs.so')


class CGraphWrapper(object):

    class CNodesToFaces(Structure):
        def __init__(self, py_ntf):
            self._fields_ = []
            string_py_ntf = {str(k) : v for k, v in py_ntf.iteritems()}
            for k, v in string_py_ntf:
                self._fields.append( (k, c_int * len(v)) )
            super(Structure, self).__init__(**string_py_ntf)

    class CGraph():
        def __init__(self, py_graph):
            self._fields_ = []
            string_py_graph = {str(k) : v for k, v in py_graph.iteritems()}
            for k, v in string_py_graph:
                self._fields.append( (k, c_int * len(v)) )
            super(Structure, self).__init__(**string_py_graph)

    def __init__(self, nodes_to_faces, graph):
        self.obj = lib.Graph_new()
        self.nodes_to_faces = self.CNodesToFaces(nodes_to_faces)
        self.graph = self.CGraph(graph)

    def bar(self):
        #not thread-safe currently...
        lib.Graph_bar(self.obj, self.nodes_to_faces, self.graph)
