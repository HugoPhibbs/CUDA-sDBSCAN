This is a C++ implementation of scalable DBSCAN and OPTICS, called sDBSCAN and sOPTICS, in high dimensional space.
sDBSCAN and sOPTICS use a significantly large number of random projection vectors to utilize the neighborhood preserving property of a few random vectors.
This lead to a multi-thread friendly implementation with significant speedup compared to scikit-learn.


sDBSCAN and sOPTICS use FFHT (Fast Fast Hadamard Transform) (https://github.com/FALCONN-LIB/FFHT) that provides a heavily optimized C99 implementation of the Fast Hadamard Transform.
It uses AVX to speed up the computation. This lib is part of FalconnLib.

sDBSCAN also needs EigenLib (https://eigen.tuxfamily.org) with vectorization to fast compute distance
and boost (https://www.boost.org/) with binary histogram

# Building with CMake
- From the root directory run:
```bash
cmake . -B ./cmake-build-debug
```

# Run sOptics, sDbscan, sDbscan-1NN

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_cosine" --alg sOptics --eps 0.25 --minPts 50 --numProj 1024 --topKProj 10 --topMProj 50 --dist Cosine  --output y_optics --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_cosine" --alg sDbscan --eps 0.16 --minPts 50 --numProj 1024 --topKProj 10 --topMProj 50 --dist Cosine --clusterNoise 0 --output y_dbscan --numThreads 64

./DbscanCEOs --numPts 8100000 --numDim 784 --X "/home/npha145/Dataset/Clustering/mnist8m_X_cosine" --alg sDbscan_1NN --eps 0.16 --minPts 50 --numProj 1024 --topKProj 10 --topMProj 50 --dist Cosine --output y_dbscan --numThreads 64

See the Compile.sh for compiling and running scripts.


# Configuration
Please make sure that you have the following libraries installed:
- Eigen3
- Google Test
- Array Fire

