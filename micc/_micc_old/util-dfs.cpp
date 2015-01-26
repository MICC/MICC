#include <iostream>
#include <map>
#include "json.h"
#include <vector>
#include <string>
#include <fstream>
#include "limits.h"
#include <set>

using std::map;
using std::vector;

void read_fileargs(std::string file, int &current_node, int &start_node,
                    vector<int> &current_path, map< int, vector<int> > &graph,
                    map<int, vector<int> > &nodes_to_faces);
void dfs(int &current_node, int &start_node, vector<int> &current_path,
            map< int, vector<int> > &graph, map<int, vector<int> > &nodes_to_faces , vector< vector<int> > &loops);
void shift(vector<int> &path);
int count(vector<int> &path, int val);
void write_output(std::string file, vector< vector<int> > &loops);

void print_loops(vector< vector<int> > &loops){
    for(int i = 0; i < loops.size(); i++){
        std::cout << "[";
        for(int j = 0; j < loops.size(); j++){
            std::cout << j << ", ";
        }
        std::cout << "]" << std::endl;

    }
}

void write_output(std::string file, vector< vector<int> > &loops){
    //Make a writer and empty JSON array
    Json::FastWriter w;
    Json::Value loops_to_write(Json::arrayValue);
    //dump each loop into a JSON array and append it to the above one
    for(int i = 0; i < loops.size(); i++){
        Json::Value json_loop(Json::arrayValue);
        vector<int> current_loop = loops[i];
        for(int j = 0; j < current_loop.size(); j++){
            json_loop.append(current_loop[j]);
        }
        loops_to_write.append(json_loop);
    }
    //write JSON to output file
    std::string jsonstr = w.write(loops_to_write);
    std::cout << jsonstr;
    //std::ofstream myfile (file);
    //myfile << jsonstr;
    //myfile.close();

}
int main(int argc, char* argv[]){

    map<int, vector<int> > graph, nodes_to_faces;
    int current_node = 0, start_node = 0;
    vector< vector<int> > loops;
    //Parse the input file and dump the contents into the above variables
    vector<int> __;
    read_fileargs(argv[1], current_node, start_node, __, graph, nodes_to_faces);

    //do the depth first search
    for(map<int, vector<int> >::iterator it =graph.begin(); it != graph.end(); it++){
        int start_node = it->first;
        vector<int> adjs = it->second;
        for(int i = 0; i < adjs.size();i++){
            current_node = adjs[i];
            vector<int> current_path;
            current_path.push_back(current_node);
            //std::cout << current_node << "  " << start_node << std::endl;
            dfs(current_node, start_node, current_path, graph, nodes_to_faces, loops);
        }
    }
    //write to output file
    write_output("out.json", loops);
    return 0;

}

//count the number of instances of a value in a path
int count(vector<int> &path, int val){
    int count = 0;
    for(int i = 0; i < path.size(); i++){
        if(path[i] == val){
            count++;
        }
    }
    return count;
}



void shift(vector<int> &path){
    //find the minimum vertex in the path and where it occurs
    int min = INT_MAX;
    int min_at;
    for(int i = 0; i < path.size(); i++){
        int val = path[i];
        if(val < min){
            min = val;
            min_at = i;
        }
    }
    //reshape the vector with the minimum value in the first position
    vector<int> ret;
    for(int i = 0; i < path.size(); i++){
        int index = (min_at + i ) % path.size();
        ret.push_back(path[index]);
    }
    path = ret;

}

//remove val_to_remove from the adjacency list associated with key
void remove_from_graph(map<int, vector<int> > &graph, int key, int val_to_remove ){
    vector<int> temp;
    vector<int> adj_list = graph[key];
    for( int i = 0; i < adj_list.size(); i++){
        if(adj_list[i] != val_to_remove){
            temp.push_back(adj_list[i]);
        }
    }
    graph[key]= temp;
}


void dfs(int &current_node, int &start_node, vector<int> &current_path,
            map< int, vector<int> > &graph, map<int, vector<int> > &nodes_to_faces , vector< vector<int> > &loops){
    //check if the current_path is longer than 3 vertices long
    if(current_path.size() >= 3){
        //determine whether or not the three most recent vertices are in the same face
        //make a map of all elements of nodes_to_faces of the last three vertices of the current_path
        map<int,int> intersection;
        for(int i = 0; i < 3; i++){
            vector<int> temp = nodes_to_faces[current_path[current_path.size()-3+i]];
            for(int j = 0; j < temp.size(); j++){
                int value = temp[j];
                //if the vertex is in the map, increment the value
                if(intersection.count(value)){
                    intersection[value] = intersection[value] + 1;
                } else {
                    //otherwise, pop a 1 in there
                    intersection.insert(std::pair<int,int>(value, 1));
                }
            }
        }
        int intersection_size = 0;
        for(map<int, int>::iterator it =intersection.begin(); it != intersection.end(); it++){
            if(it->second == 3){
                intersection_size++;
            }
            if(intersection_size == 2){
                return;
            }
        }
    }
    if(current_node == start_node){
        //if the path is complete, add it to the list of loops
        vector<int> path;
        for(int i = 0; i <current_path.size(); i++){
            path.push_back(current_path[i]);
        }
        shift(path);
        loops.push_back(path);
        return;
    } else {
        //remove duplicates from adjacency list and iterate over the set
        vector<int> current_adjacencies = graph[current_node];
        std::set<int> adj_list(current_adjacencies.begin(), current_adjacencies.end());
        vector<int> temp;
        for(std::set<int>::iterator it = adj_list.begin(); it != adj_list.end(); it++){
            int adjacent_node = *it;
            if(count(current_path,adjacent_node) < 1){
                //add the adjacent node to the current_path
                current_path.push_back(adjacent_node);
                //remove adjcent_node from the adj_list of current node and vis-versa
                remove_from_graph(graph, current_node, adjacent_node);
                remove_from_graph(graph, adjacent_node, current_node);

                //recursive call
                dfs(adjacent_node,start_node, current_path,graph, nodes_to_faces,loops);

                //re-add the adjacencies removed before the recursion
                temp = graph[current_node];
                temp.push_back(adjacent_node);
                graph[current_node] = temp;

                temp = graph[adjacent_node];
                temp.push_back(current_node);
                graph[adjacent_node] = temp;
                //remove the adjacent node to the current_path
                current_path.pop_back();
            }
        }
        return;

    }
}

/*


        for(map<int, vector<int> >::iterator it =graph.begin(); it != graph.end(); it++){
            std::cout << it->first << " ";
            for(int i = 0; i < it->second.size();i++){
                std::cout << it->second[i] << ", ";
            }
            std::cout << std::endl;
        }

*/

void read_fileargs(std::string file, int &current_node, int &start_node,
                    vector<int> &current_path, map< int, vector<int> > &graph,
                    map<int, vector<int> > &nodes_to_faces){

    std::string line, jsonfile;
    //read JSON from an input file
    std::ifstream myfile (file);
    if (myfile.is_open()){
        while ( getline (myfile,line) ){
            jsonfile = jsonfile +line;
        }
        myfile.close();
    }


    Json::Value root;
    Json::Reader reader;
    //parse the JSON with JSONCPP
    //successful_parse == 1 if parse properly
    bool successful_parse = reader.parse(jsonfile, root);
    if(successful_parse){
        if(root[0].isInt()){ //check that the current_node is an int
            current_node = root[0].asInt();
        }
        if(root[1].isInt()){ //check that start_node is an int
            start_node = root[1].asInt();
        }

        if(root[2].isArray() && root[2].size() == 1){ //check current_path is a list and that it's length one
            current_path.push_back(root[2][0].asInt());
        }
        if(root[3].isObject()){
            // get the number and names of nodes in the graph
            vector<std::string> members = root[3].getMemberNames();
            //iterate over the number of nodes
            for(int i = 0; i < members.size(); i++){
                //reinitialize the adjacency list each time
                vector<int> temp_adjacencies;

                //get the JSON array of the adj_list
                Json::Value json_adj_list = root[3][members[i]];
                int size = json_adj_list.size();
                int node = std::atoi(members[i].c_str());

                //iterate over the JSON array and shove it in the temporary vector
                for(int k = 0; k < size; k++)
                    temp_adjacencies.push_back(json_adj_list[k].asInt());

                //take temporary vector and the node of the graph and insert it into the map
                graph.insert(std::pair<int, vector<int> >(node, temp_adjacencies));


            }
        }
        if(root[4].isObject()){
            // get the number and names of nodes in nodes_to_faces
            vector<std::string> members = root[4].getMemberNames();
            //iterate over the number of nodes
            for(int i = 0; i < members.size(); i++){
                //reinitialize the adjacency list each time
                vector<int> temp_adjacencies;

                //get the JSON array of the adj_list
                Json::Value json_adj_list = root[4][members[i]];
                int size = json_adj_list.size();
                int node = std::atoi(members[i].c_str());

                //iterate over the JSON array and shove it in the temporary vector
                for(int k = 0; k < size; k++){
                    std::string s = json_adj_list[k].asString();
                    int value;
                    //if our adjacency is non-null, use it, otherwise, replace nulls with MAX_INT
                    if(s != ""){
                        value = std::atoi(s.c_str());
                    } else {
                        value = INT_MAX;
                    }
                    //dump it into our temporary vector
                    temp_adjacencies.push_back(value);
                }
                //take temporary vector and the node of the nodes_to_faces and insert it into the map
                nodes_to_faces.insert(std::pair<int, vector<int> >(node, temp_adjacencies));


            }

        }



    }
}

