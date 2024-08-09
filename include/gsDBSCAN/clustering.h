//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_CLUSTERING_H
#define SDBSCAN_CLUSTERING_H

#include <arrayfire.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include "utils.h"
#include "../Header.h"

namespace GsDBSCAN {

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
    af::array constructQueryVectorDegreeArray(af::array &distances, float eps) {
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
    af::array processQueryVectorDegreeArray(af::array &E) {
        return af::scan(E, 1, AF_BINARY_ADD,
                        false); // Do an exclusive scan// TODO, need to return the V array, this is here to satisfy the compiler.
    }


    /**
     * Performs the actual clustering step of the algorithm
     *
     * Rewritten from Ninh's original code
     *
     * @param adjacencyList af array adjacency list for each of the dataset vectors as per the GsDBSCAN algorithm
     * @param V starting index of each of the dataset vectors within the adjacency list
     * @param E degree of each query vector (how many candidate vectors are within eps distance of it)
     * @param n size of the dataset
     * @param minPts minimum number of points within eps distance to consider a point as a core point
     * @param clusterNoise whether to include noise points in the result
     * @return a tuple containing the cluster labels and the number of clusters found
     */
    std::tuple<std::vector<int>, int>
    formClusters(af::array &adjacencyList, af::array &V, af::array &E, int n, int minPts, bool clusterNoise) {
        int nClusters = 0;
        std::vector<int> labels = IVector(n, -1);

        int iNewClusterID = -1;

        auto isCore = [&](int idx) -> bool {
            // TODO use a bit set instead of a cumbersome af array
            return E(idx).scalar<int>() >= minPts;
        };

        for (int i = -1; i < n; i++) {

            if (!isCore(i) || (labels[i] != -1)) {
                continue;
            }

            iNewClusterID++;

            std::unordered_set<int> seedSet; //seedSet only contains core points
            seedSet.insert(i);

            boost::dynamic_bitset<> connectedPoints(n);
            connectedPoints[i] = true;

            int startIndex, endIndex;

            while (!seedSet.empty()) {
                int Xi = *seedSet.begin();
                seedSet.erase(seedSet.begin());

                startIndex = V(Xi).scalar<int>();
                endIndex = startIndex + E(Xi).scalar<int>();
                int Xj;

                for (int j = startIndex; j < endIndex; j++) {
                    Xj = adjacencyList(j).scalar<int>();

                    if (isCore(i)) {
                        if (!connectedPoints[Xj]) {
                            connectedPoints[Xj] = true;

                            if (labels[Xj] == -1) seedSet.insert(Xj);
                        }
                    } else {
                        connectedPoints[Xj] = true;
                    }

                }
            }

            size_t Xj = connectedPoints.find_first();

            while (Xj != boost::dynamic_bitset<>::npos) {
                if (labels[Xj] == -1) labels[Xj] = iNewClusterID;

                Xj = connectedPoints.find_next(Xj);
            }

            nClusters = iNewClusterID;
        }

        if (clusterNoise) {
            // TODO, implement labeling of noise
        }

        return make_tuple(labels, nClusters);
    }

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
    constructAdjacencyListForQueryVector(float *distances, int *adjacencyList, int *V, int *A, int *B, float eps, int n,
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
    af::array assembleAdjacencyList(af::array &distances, af::array &E, af::array &V, af::array &A, af::array &B, float eps,
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

    /**
     * Performs the actual clustering step of the algorithm
     *
     * I.e. returns the formed clusters based on the inputted adjacency list
     *
     * @param adjacencyList af array adjacency list for each of the dataset vectors as per the GsDBSCAN algorithm
     * @param V af array containing the starting index of each of the dataset vectors within the adjacency list
     */
    void performClustering(af::array &adjacencyList, af::array &V) {
        // TODO implement me!
    }
}

#endif //SDBSCAN_CLUSTERING_H
