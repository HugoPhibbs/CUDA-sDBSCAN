//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_CLUSTERING_H
#define SDBSCAN_CLUSTERING_H

#include <arrayfire.h>
#include <cuda_runtime.h>
#include <matx.h>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include "utils.h"
#include "../Header.h"
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include "utils.h"


namespace GsDBSCAN {

    /**
     * Calculates the degree of the query vectors as per the G-DBSCAN algorithm.
     *
     * Does this using MatX
     *
     * This function is used in the construction of the cluster graph by determining how many
     *
     * Put into its own method for testability
     *
     * @param distances The matrix containing the distances between the query and candidate vectors.
     *                  Expected shape is (datasetSize, 2*k*m).
     * @param eps       The epsilon value for DBSCAN. Should be a scalar array of the same data type
     *                  as the distances array.
     *
     * @return The degree array of the query vectors, with shape (datasetSize, 1).
     */
    template <typename T>
    auto constructQueryVectorDegreeArrayMatx(matx::tensor_t<T, 2> &distances, T eps) {
        auto lt = distances < eps;
        auto lt_f = matx::as_type<float>(lt);
        // TODO raise a GH as to why i need to cast first, should be able to sum over the bools
        return matx::sum(lt_f, {0});
    }

    template <typename T>
    auto processQueryVectorDegreeArrayMatx(matx::tensor_t<T, 2> &E) {
        // MatX's cumsum works along the rows.
        return matx::cumsum(E) - E;
    }


    /**
     * Calculates the degree of the query vectors as per the G-DBSCAN algorithm.
     *
     * This function is used in the construction of the cluster graph by determining how many
     *
     * Put into its own method for testability
     *
     * @param distances The matrix containing the distances between the query and candidate vectors.
     *                  Expected shape is (datasetSize, 2*k*m).
     * @param eps       The epsilon value for DBSCAN. Should be a scalar array of the same data type
     *                  as the distances array.
     *
     * @return The degree array of the query vectors, with shape (datasetSize, 1).
     */
    inline af::array constructQueryVectorDegreeArray(af::array &distances, float eps) {
        return af::sum(distances < eps, 0);
    }

    /**
     * Processes the vector degree array to create an exclusive scan of this vector
     *
     * Put into it's own method to ensure testability
     *
     * @param E vector degree array
     * @return arrayfire processed array
     */
    inline af::array processQueryVectorDegreeArray(af::array &E) {
        return af::scan(E, 1, AF_BINARY_ADD,
                        false); // Do an exclusive scan// TODO, need to return the V array, this is here to satisfy the compiler.
    }


//    /**
//     * Performs the actual clustering step of the algorithm
//     *
//     * Rewritten from Ninh's original code
//     *
//     * @param adjacencyList af array adjacency list for each of the dataset vectors as per the GsDBSCAN algorithm
//     * @param V starting index of each of the dataset vectors within the adjacency list
//     * @param E degree of each query vector (how many candidate vectors are within eps distance of it)
//     * @param n size of the dataset
//     * @param minPts minimum number of points within eps distance to consider a point as a core point
//     * @param clusterNoise whether to include noise points in the result
//     * @return a tuple containing the cluster labels and the number of clusters found
//     */
//    std::tuple<std::vector<int>, int>
//    inline formClusters(af::array &adjacencyList, af::array &V, af::array &E, int n, int minPts, bool clusterNoise) {
//        int nClusters = 0;
//        std::vector<int> labels = IVector(n, -1);
//
//        int iNewClusterID = -1;
//
//        auto isCore = [&](int idx) -> bool {
//            // TODO use a bit set instead of a cumbersome af array
//            return E(idx).scalar<int>() >= minPts;
//        };
//
//        for (int i = -1; i < n; i++) {
//
//            if (!isCore(i) || (labels[i] != -1)) {
//                continue;
//            }
//
//            iNewClusterID++;
//
//            std::unordered_set<int> seedSet; //seedSet only contains core points
//            seedSet.insert(i);
//
//            boost::dynamic_bitset<> connectedPoints(n);
//            connectedPoints[i] = true;
//
//            int startIndex, endIndex;
//
//            while (!seedSet.empty()) {
//                int Xi = *seedSet.begin();
//                seedSet.erase(seedSet.begin());
//
//                startIndex = V(Xi).scalar<int>();
//                endIndex = startIndex + E(Xi).scalar<int>();
//                int Xj;
//
//                for (int j = startIndex; j < endIndex; j++) {
//                    Xj = adjacencyList(j).scalar<int>();
//
//                    if (isCore(i)) {
//                        if (!connectedPoints[Xj]) {
//                            connectedPoints[Xj] = true;
//
//                            if (labels[Xj] == -1) seedSet.insert(Xj);
//                        }
//                    } else {
//                        connectedPoints[Xj] = true;
//                    }
//
//                }
//            }
//
//            size_t Xj = connectedPoints.find_first();
//
//            while (Xj != boost::dynamic_bitset<>::npos) {
//                if (labels[Xj] == -1) labels[Xj] = iNewClusterID;
//
//                Xj = connectedPoints.find_next(Xj);
//            }
//
//            nClusters = iNewClusterID;
//        }
//
//        if (clusterNoise) {
//            // TODO, implement labeling of noise
//        }
//
//        return make_tuple(labels, nClusters);
//    }

    /**
     * Kernel for constructing part of the cluster graph adjacency list for a particular vector
     *
     * @param distances matrix containing the distances between each query vector and it's candidate vectors
     * @param adjacencyList
     * @param V vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
     * @param A A matrix, see constructABMatrices. Stored flat as a float array
     * @param B B matrix, see constructABMatrices. Stored flat as a float array
     * @param n number of query vectors in the dataset
     * @param eps epsilon DBSCAN density param
     */
    __global__ void
    inline constructAdjacencyListForQueryVector(float *distances, int *adjacencyList, int *V, int *A, int *B, float eps, int n,
                                         int k, int m) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return; // Exit if out of bounds. Don't assume that numQueryVectors is equal to the total number o threads

        int curr_idx = V[idx];

        int distances_rows = 2 * k * m;

        int ACol, BCol, BRow, neighbourhoodVecIdx;

        for (int j = 0; j < distances_rows; j++) {
            if (distances[idx * distances_rows + j] < eps) {
                ACol = j / m;
                BCol = j % m;
                BRow = A[idx * 2 * k + ACol];
                neighbourhoodVecIdx = B[BRow * m + BCol];

                adjacencyList[curr_idx] = neighbourhoodVecIdx;
                curr_idx++;
            }
        }
    }

    /**
     * Assembles the adjacency list for the cluster graph
     *
     * See https://arrayfire.org/docs/interop_cuda.htm for info on this
     *
     * @param distances matrix containing the distances between each query vector and it's candidate vectors
     * @param E vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
     * @param V vector containing the starting index of each query vector in the resultant adjacency list (See the G-DBSCAN algorithm)
     * @param A A matrix, see constructABMatrices
     * @param B B matrix, see constructABMatrices
     * @param eps epsilon DBSCAN density param
     * @param blockSize size of each block when calculating the adjacency list - essentially the amount of query vectors to process per block
     */
    inline af::array assembleAdjacencyList(af::array &distances, af::array &E, af::array &V, af::array &A, af::array &B, float eps,
                                    int blockSize) {
        /*
         * Check this issue:
         *
         * https://github.com/arrayfire/arrayfire/issues/3051
         *
         * May need to look further into this - i.e. how to use af/cuda.h properly
         */


        int n = E.dims(0);
        int k = A.dims(1) / 2;
        int m = B.dims(1);

        af::array adjacencyList = af::constant(-1, (E(n - 1) + V(n - 1)).scalar<int>(), af::dtype::u32);

        // Eval all matrices to ensure they are synced
        adjacencyList.eval();
        distances.eval();
        E.eval();
        V.eval();
        A.eval();
        B.eval();

        // Getting device pointers
        int *adjacencyList_d = adjacencyList.device<int>();
        float *distances_d = distances.device<float>();
        int *E_d = E.device<int>();
        int *V_d = V.device<int>();
        int *A_d = A.device<int>();
        int *B_d = B.device<int>();

        // Getting cuda stream from af
        cudaStream_t afCudaStream = getAfCudaStream();

        // Now we can call the kernel
        int numBlocks = std::max(1, n / blockSize);
        blockSize = std::min(n, blockSize);
        constructAdjacencyListForQueryVector<<<numBlocks, blockSize, 0, afCudaStream>>>(distances_d, adjacencyList_d, V_d,
                A_d, B_d, eps, n, k, m);

        // Unlock all the af arrays
        adjacencyList.unlock();
        distances.unlock();
        E.unlock();
        V.unlock();
        A.unlock();
        B.unlock();

        return adjacencyList;
    }

    // TODO verify that the functions below are ok!

    __global__ void breadthFirstSearchKernel(int *adjacencyList, int* startIdxArray, bool* visited, bool* border, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= n) {
            return;
        }

        if (visited[tid]) {
            visited[tid] = 0;
            border[tid] = 1;

            int startIdx = startIdxArray[tid];

            for (int i = startIdx; i < startIdxArray[tid + 1]; i++) {
                int neighbourIdx = adjacencyList[i];

                if (!visited[neighbourIdx]) {
                    visited[neighbourIdx] = 1;
                }
            }
        }
    }

    inline void breadthFirstSearch(int *adjacencyList_d, int* startIdxArray_d, int* degArray_h, bool* visited, int* clusterLabels, const size_t n, int seedVertexIdx, int thisClusterLabel, int minPts, int blockSize = 256) {
        auto visitedThisBfs_d = allocateCudaArray<bool>(n);
        auto borderThisBfs_d = allocateCudaArray<bool>(n);

        visitedThisBfs_d[seedVertexIdx] = 1;

        int countVisitedThisBfs = 1;

        int gridSize = (n + blockSize - 1) / blockSize;

        while (countVisitedThisBfs > 0) {
            breadthFirstSearchKernel<<<gridSize, blockSize>>>(adjacencyList_d, startIdxArray_d, visitedThisBfs_d, borderThisBfs_d, n);
            cudaDeviceSynchronize();
            auto thrust_ptr = thrust::device_pointer_cast(visitedThisBfs_d);
            countVisitedThisBfs = thrust::reduce(thrust_ptr, thrust_ptr + n, 0);
        }
        
        auto visited_h = copyDeviceToHost(visitedThisBfs_d, n);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (visited_h[i]) {
                clusterLabels[i] = thisClusterLabel;
                visited[i] = 1;
                if (degArray_h[i] < minPts) {
                    // Assign as border point
                }
            }
        }
    }


    inline std::vector<int> formClusters(int* adjacencyList_d, int n, int* startIdxArray_d, int* degArray_d, int minPts) {
        int *clusterLabels = new int[n];
        bool *visited = new bool[n];

        auto degArray_h = copyDeviceToHost(degArray_d, n);

        int currCluster = 0;

        for (int i = 0; i < n; i++) {
            if ((!visited[i]) && (degArray_h[i] >= minPts)) {
                visited[i] = true;
                clusterLabels[i] = currCluster;
                breadthFirstSearch(adjacencyList_d, startIdxArray_d, degArray_h, visited, clusterLabels, n, i, currCluster, minPts);
                currCluster += 1;
            }
        }
    }

    /**
     * Performs the actual clustering step of the algorithm
     *
     * I.e. returns the formed clusters based on the inputted adjacency list
     *
     * @param adjacencyList af array adjacency list for each of the dataset vectors as per the GsDBSCAN algorithm
     * @param V af array containing the starting index of each of the dataset vectors within the adjacency list
     */
    inline void performClustering(af::array &adjacencyList, af::array &V) {
        // TODO implement me!
    }
}

#endif //SDBSCAN_CLUSTERING_H
