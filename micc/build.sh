g++ -c -fPIC dfs.cpp -o dfs.o
g++ -shared -Wl,-install_name,libfoo.so -o libdfs.so  dfs.o

