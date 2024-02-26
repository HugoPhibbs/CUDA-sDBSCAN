# compile

g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Dbscan.cpp" -o obj/Release/Dbscan.o
gcc -Wall -fexceptions -O3 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/fast_copy.c" -o obj/Release/fast_copy.o
gcc -Wall -fexceptions -O3 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/fht.c" -o obj/Release/fht.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Header.cpp" -o obj/Release/Header.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/InputParser.cpp" -o obj/Release/InputParser.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/main.cpp" -o obj/Release/main.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Optics.cpp" -o obj/Release/Optics.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Test.cpp" -o obj/Release/Test.o
g++ -Wall -fexceptions -O3 -std=c++17 -m64 -fopenmp -march=native -I/home/npha145/Library/boost_1_78_0 -I/home/npha145/Library/eigen-3.4.0 -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/DbscanCEOs/Utilities.cpp" -o obj/Release/Utilities.o
g++  -o bin/Release/DbscanCEOs obj/Release/Dbscan.o obj/Release/fast_copy.o obj/Release/fht.o obj/Release/Header.o obj/Release/InputParser.o obj/Release/main.o obj/Release/Optics.o obj/Release/Test.o obj/Release/Utilities.o  -O3 -s -m64 -lgomp -pthread 


# Run sOPTICS

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X" --alg sOptics --eps 2000 --minPts 50 --numEmbed 1024 --numProj 1024 --topKProj 10 --topMProj 50 --dist L2 --sigma 4000 --output y_optics_sigma_4000 --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_cosine" --alg sOptics --eps 0.25 --minPts 50 --numProj 1024 --topKProj 10 --topMProj 50 --dist Cosine  --output y_optics --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_prob" --alg sOptics --eps 0.25 --minPts 50 --numEmbed 1024 --numProj 1024 --topKProj 10 --topMProj 50 --output y_optics --dist Chi2 --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_prob" --alg sOptics --eps 0.25 --minPts 50 --numEmbed 1024 --numProj 1024 --topKProj 10 --topMProj 50 --output y_optics --dist JS --numThreads 64

# Run sDBSCAN

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X" --alg sDbscan --eps 1350 --minPts 50 --numEmbed 1024 --numProj 1024 --topKProj 10 --topMProj 50 --dist L2 --clusterNoise 0 --sigma 2500 --output y_dbscan --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_cosine" --alg sDbscan --eps 0.16 --minPts 50 --numProj 1024 --topKProj 10 --topMProj 50 --dist Cosine --clusterNoise 0 --output y_dbscan --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_prob" --alg sDbscan --eps 0.17 --minPts 50 --numEmbed 1024 --numProj 1024 --topKProj 10 --topMProj 50 --dist Chi2 --clusterNoise 0 --output y_dbscan --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_prob" --alg sDbscan --eps 0.14 --minPts 50 --numEmbed 1024 --numProj 1024 --topKProj 10 --topMProj 50 --dist JS --clusterNoise 0 --output y_dbscan --numThreads 64

# Run sDBSCAN-1NN

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_cosine" --alg sDbscan_1NN --eps 0.16 --minPts 50 --numProj 1024 --topKProj 10 --topMProj 50 --dist Cosine --output y_dbscan --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_cosine" --alg sDbscan --eps 0.16 --minPts 50 --numProj 1024 --topKProj 10 --topMProj 50 --dist Cosine --clusterNoise 4 --output y_dbscan --numThreads 64

