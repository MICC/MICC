g++ -c -fPIC dfs.cpp -o dfs.o -I/Users/Matt/anaconda/include/python2.7 -I/Users/Matt/anaconda/include/python2.7 -fno-strict-aliasing -I/Users/Matt/anaconda/include -arch x86_64 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes
g++ -shared -lpython -Wl,-install_name,libfoo.so -o libdfs.so  dfs.o

