# compile

g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Dbscan.cpp" -o obj/Release/Dbscan.o
gcc -Wall -fexceptions -O3 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/fast_copy.c" -o obj/Release/fast_copy.o
gcc -Wall -fexceptions -O3 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/fht.c" -o obj/Release/fht.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Header.cpp" -o obj/Release/Header.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/InputParser.cpp" -o obj/Release/InputParser.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/main.cpp" -o obj/Release/main.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Test.cpp" -o obj/Release/Test.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Utilities.cpp" -o obj/Release/Utilities.o


g++  -o bin/Release/DbscanCEOs obj/Release/Dbscan.o obj/Release/fast_copy.o obj/Release/fht.o obj/Release/Header.o obj/Release/InputParser.o obj/Release/main.o obj/Release/Test.o obj/Release/Utilities.o  -O3 -s -m64 -lgomp -pthread 



