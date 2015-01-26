#include <iostream>
#include <map>
#include <vector>
#include "Python.h"
#include "limits.h"
//current_node,  start_node,  graph,  current_path,  nodes_to_face

class Graph{
	public:
		void bar(){
			std::cout << "fuck yeah" << std::endl;
		}
};

extern "C"{
    PyObject* loop_dfs(PyObject* current_node, PyObject* start_node,
                                PyObject* pygraph, PyObject* current_path,
                                PyObject* py_nodes_to_face);

	Graph* Graph_new(){ return new Graph(); }
	void Graph_bar(Graph* graph){ graph->bar(); }
}

std::map<int, std::vector<int> > pydict_to_map(PyObject* dict){

    std::map<int, std::vector<int> > ret;
    std::vector<int> temp_value;
    int map_key;
    Py_ssize_t index = 0;
    PyObject *pykey, *pyvalue;
    long temp;
    Py_XINCREF(Py_None);

    while(PyDict_Next(dict, &index, &pykey, &pyvalue)){
        std::vector<int> map_value;
        map_key = (int) PyInt_AsLong(pykey);

        for(Py_ssize_t i = 0; i < PyList_Size(pyvalue); i++){

            PyObject* list_element = PyList_GetItem(pyvalue, i);
            std::string s = PyString_AsString(PyObject_Repr(list_element));

            if(s.compare("None") == 0){
                temp = INT_MAX;
            } else{
                temp = PyInt_AsLong(list_element);
            }
            map_value.push_back((int) temp);
        }

        ret.insert(std::pair<int, std::vector<int> >(map_key, map_value));
    }
    Py_DECREF(Py_None);
    return ret;
}
std::vector< std::vector<int> > cdfs(int current_node, int start_node, std::map<int, std::vector<int> > graph, std::vector<int> current_path, std::map<int, std::vector<int> > nodes_to_faces){
    std::vector< std::vector<int> > loops;
    int path_size = current_path.size();
    if(path_size >= 3){
        std::vector<int> path_head_3;
        std::vector< std::vector<int> > sets;
        for(int k = 0; k < 3; k++)
            sets.push_back(nodes_to_faces[current_path.at(path_size - 3 + k)]);


    }
/*
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

    return loops;
}

PyObject* loop_dfs(PyObject* pycurrent_node, PyObject* pystart_node, PyObject* pygraph, PyObject* pycurrent_path, PyObject* pynodes_to_faces){
    std::map<int, std::vector<int> > graph = pydict_to_map(pygraph);
    std::map<int, std::vector<int> > nodes_to_faces = pydict_to_map(pynodes_to_faces);
    int current_node = (int) PyInt_AsLong(pycurrent_node);
    int start_node = (int) PyInt_AsLong(pystart_node);
    std::vector<int> current_path;
    for(Py_ssize_t i = 0; i < PyList_Size(pycurrent_path); i++){
        current_path.push_back((int) PyInt_AsLong(PyList_GetItem(pycurrent_path,i)));
    }


   std::vector< std::vector<int> > loops = cdfs(current_node, start_node, graph, current_path, nodes_to_faces);

    for(int i = 0; i < current_path.size(); i++)
        std::cout << current_path.at(i) << std::endl;

/*
    for (std::map<int, std::vector<int> >::iterator it=graph.begin(); it!=graph.end(); it++){

        std::cout << it->first << " => ";
        for(int i = 0; i < it->second.size(); i++){
            std::cout << it->second.at(i) << ", ";
        }
        std::cout << '\n';//<< it->second << '\n';
    }
*/



}



PyObject* loop_dfs_py(PyObject* current_node, PyObject* start_node, PyObject* graph, PyObject* current_path, PyObject* nodes_to_faces){
    //Py_Initialize();
    Py_XINCREF(current_node);
    Py_XINCREF(graph);

    std::cout << "start of dfs call: " << PyInt_AsLong(current_node) << "   " << PyInt_AsLong(start_node) << std::endl;
    Py_ssize_t current_path_length = PyList_Size(current_path);
    std::cout << current_path_length << std::endl;

    Py_ssize_t len = 0;
    if(current_path_length >= 3){

        PyObject* path_head_3 = PyList_GetSlice(current_path, current_path_length - 3, current_path_length);
        PyObject* intersection_all = PyDict_New();
        //verify****
        /*
            Should make a dictionary that increments values based on the number of appearances of an edge
            If any key has a value of 2, we're done
        */
        for(Py_ssize_t i = 0; i < 3; i ++){

            PyObject* face = PyList_GetItem(path_head_3, i);
            PyObject* face_set = PyDict_GetItem(nodes_to_faces,face);

            for(Py_ssize_t j = 0; j < PyTuple_Size(face_set); j ++){
                PyObject* edge = PyList_GetItem(face_set,j);

                if(PyDict_Contains(intersection_all, edge)){
                    PyObject* value = PyDict_GetItem(intersection_all, edge);
                    PyDict_SetItem(intersection_all, edge, PyInt_FromLong(PyInt_AsLong(value) + 1));
                } else {
                    PyDict_SetItem(intersection_all, edge, PyInt_FromLong(1));
                }
            }
        }
        len = 0;
        return PyList_New(len);
    }

    if(PyInt_AsLong(current_node) == PyInt_AsLong(start_node)){
        len = 1;
        PyObject* ret = PyList_New(len);
        PyList_Append(ret, current_path);
        return ret;
    } else {
        len = 0;
        PyObject* loops = PyList_New(len);

        PyObject* adjacent_nodes = PyDict_GetItem(graph, current_node);
        Py_XINCREF(adjacent_nodes);
        Py_XINCREF(loops);

        Py_ssize_t num_adj_nodes = PyList_Size(adjacent_nodes);

       PyObject* adjacent_node;
        for(Py_ssize_t i = 0; i < num_adj_nodes; i++){
            adjacent_node = PyList_GetItem(adjacent_nodes, i);
            std::cout << PyString_AsString(PyObject_Repr(adjacent_node)) << std::endl;
            Py_XINCREF(adjacent_node);

            int count = 0;

            for(Py_ssize_t j = 0; j < PyList_Size(current_path); j++){
                PyObject* node = PyList_GetItem(current_path, j);
                Py_XINCREF(node);
                std::cout << "shit " << i << "  " << j << std::endl;
                if(PyInt_AsLong(node) == PyInt_AsLong(adjacent_node)){
                    count++;
                }
                Py_DECREF(node);
            }
            if(count < 1){ //this conditional is the generalization for higher distance (!!!)
                //remove adjacent_node from the adj_list of current node
                PyObject* temp = PyDict_GetItem(graph, current_node);
                PyObject* reset = PyList_New(PyList_Size(temp)-1);

                Py_XINCREF(temp);
                Py_XINCREF(reset);

                for(Py_ssize_t j = 0; j < PyList_Size(temp); j++){
                    PyObject* list_element = PyList_GetItem(temp, j);
                    Py_XINCREF(list_element);
                    if(PyInt_AsLong(list_element) != PyInt_AsLong(adjacent_node)){
                        PyList_Append(reset, list_element);
                    }
                    Py_DECREF(list_element);
                }
                PyDict_SetItem(graph, current_node, reset);

                //test removal:
                std::cout << "works " << std::endl;
                PyObject* t= PyDict_GetItem(graph, current_node);
                Py_XINCREF(t);
                std::cout << "works " << std::endl;
                t = PySet_New(t);
                std::cout << "works " << std::endl;
                Py_XINCREF(t);
                std::cout << "removal success: " << PySet_Contains(t,adjacent_node) << std::endl;

                std::cout << "works " << std::endl;

            }
            Py_DECREF(adjacent_node);
        }
         /*
                PyList_Append(current_path, adjacent_node);


                //remove current_node from the adj_list of adjacent_node
                temp = PyDict_GetItem(graph, adjacent_node);
                reset = PyList_New(PyList_Size(temp)-1);
                for(Py_ssize_t j = 0; j < PyList_Size(temp); j++){
                    PyObject* list_element = PyList_GetItem(temp, j);
                    if(PyInt_AsLong(list_element) != PyInt_AsLong(current_node)){
                        PyList_Append(reset, list_element);
                    }
                }
                PyDict_SetItem(graph, adjacent_node, reset);

                PyObject* recursive = loop_dfs(adjacent_node, start_node, graph, current_path, nodes_to_faces);


                Py_XINCREF(recursive);
                PyList_Append(loops, recursive);
                Py_DECREF(recursive);


                temp = PyDict_GetItem(graph, adjacent_node);
                PyList_Append(temp, current_node);
                PyDict_SetItem(graph, adjacent_node, temp);

                temp = PyDict_GetItem(graph, current_node);
                PyList_Append(temp, adjacent_node);
                PyDict_SetItem(graph, current_node, temp);
                current_path = PyList_GetSlice(current_path, 0, current_path_length - 1);
                Py_DECREF(temp); Py_DECREF(reset);


            }
            Py_DECREF(adjacent_node);
        }
        return loops;*/
    }
}










