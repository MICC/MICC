#include <iostream>
#include "Python.h"
//current_node,  start_node,  graph,  current_path,  nodes_to_face

class Graph{
	public:
		void bar(){
			std::cout << "fuck yeah" << std::endl;
		}
};

extern "C"{
    PyObject* loop_dfs(PyIntObject* current_node, PyIntObject* start_node,
                                PyObject* graph, PyObject* current_path,
                                PyObject* nodes_to_face);

	Graph* Graph_new(){ return new Graph(); }
	void Graph_bar(Graph* graph){ graph->bar(); }
}

PyObject* loop_dfs(PyIntObject* current_node, PyIntObject* start_node, PyObject* graph, PyObject* current_path, PyObject* nodes_to_face){

    Py_ssize_t current_path_length = PyList_Size(current_path);
    if(current_path_length >= 3){

        PyObject* path_head_3 = PyList_GetSlice(current_path, current_path_length - 3, current_path_length);
        //PyObject* list_of_faces = PyList_New(3);
        PyObject* intersection_all = PyDict_New();
        //verify****
        /*
            Should make a dictionary that increments values based on the number of appearances of an edge
            If any key has a value of 2, we're done
        */
        for(Py_ssize_t i = 0; i < 3; i ++){

            PyObject* face = PyList_GetItem(path_head_3, i);
            PyObject* face_set = PyDict_GetItem(nodes_to_face,face);

            for(Py_ssize_t j = 0; j < PyList_Size(face_set); j ++){
                PyObject* edge = PyList_GetItem(face_set,j);

                if(PyDict_Contains(intersection_all, edge)){
                    PyObject* value = PyDict_GetItem(intersection_all, edge);
                    PyDict_SetItem(intersection_all, edge, PyInt_FromLong(PyInt_AsLong(value) + 1));
                } else {
                    PyDict_SetItem(intersection_all, edge, PyInt_FromLong(1));
                }
            }
        }
        Py_ssize_t len = 0;
        return PyList_New(len);
        //std::cout << 'wooo' << std::endl;
    }

    if()
/*
    #stderr.write(str(current_path)+'\n')
    #stderr.write(str([nodes_to_faces[i] for i in current_path])+'\n')
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
            if Graph.count(adjacent_node, current_path) < 1:
                current_path.append(adjacent_node)
                graph[current_node].remove(adjacent_node)
                graph[adjacent_node].remove(current_node)
                loops += list(loop_dfs(adjacent_node, start_node, graph, current_path, nodes_to_faces))
                graph[current_node].append(adjacent_node)
                graph[adjacent_node].append(current_node)
                current_path.pop()
        return loops

*/


    return 0;
}








