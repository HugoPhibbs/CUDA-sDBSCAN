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
//#include <execution>
#include "utils.h"
#include "../Header.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include "utils.h"


namespace GsDBSCAN {
    namespace clustering {

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
         * @return Pointer to the degree array. Since this is intended to be how this is used for later steps
         */
        template<typename T>
        inline int *constructQueryVectorDegreeArrayMatx(matx::tensor_t<T, 2> &distances, T eps) {
            std::cout << eps << std::endl;
            print(distances);

            auto lt = distances < eps;
            auto lt_int = matx::as_type<int>(lt);

            print(lt_int);
            auto res = matx::make_tensor<int>({1, distances.Shape()[1]}, matx::MATX_MANAGED_MEMORY);
            (res = matx::sum(lt_int, {0})).run();

            print(res);
            return res.Data();
        }

        template<typename T>
        inline T *processQueryVectorDegreeArrayMatx(matx::tensor_t<T, 2> &E) {
            // MatX's cumsum works along the rows.
            auto res = matx::make_tensor<T, 2>(); // TODO use thrust here!
            (res = matx::cumsum(E) - E).run();
            return res.Data();
        }

        inline int *processQueryVectorDegreeArrayThrust(int *degArray_d, int n) {
            int *startIdxArray_d = utils::allocateCudaArray<int>(n);
            thrust::device_ptr<int> startIdxArray_thrust(startIdxArray_d);
            thrust::device_ptr<int> degArray_thrust(degArray_d);
            thrust::exclusive_scan(degArray_thrust, degArray_thrust + n, startIdxArray_thrust); // Somehow this still runs anyhow?
            return startIdxArray_d;
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
         * @param A A matrix, see constructABMatricesAF. Stored flat as a float array
         * @param B B matrix, see constructABMatricesAF. Stored flat as a float array
         * @param n number of query vectors in the dataset
         * @param eps epsilon DBSCAN density param
         */
        __global__ void
        inline
        constructAdjacencyListForQueryVector(float *distances, int *adjacencyList, int *V, int *A, int *B, float eps,
                                             int n,
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
         * @param A A matrix, see constructABMatricesAF
         * @param B B matrix, see constructABMatricesAF
         * @param eps epsilon DBSCAN density param
         * @param blockSize size of each block when calculating the adjacency list - essentially the amount of query vectors to process per block
         */
        inline af::array
        constructAdjacencyListAF(af::array &distances, af::array &E, af::array &V, af::array &A, af::array &B,
                                 float eps,
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
            cudaStream_t afCudaStream = utils::getAfCudaStream();

            // Now we can call the kernel
            int numBlocks = std::max(1, n / blockSize);
            blockSize = std::min(n, blockSize);
            constructAdjacencyListForQueryVector<<<numBlocks, blockSize, 0, afCudaStream>>>(distances_d,
                                                                                            adjacencyList_d, V_d,
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

        inline std::tuple<int *, int>
        constructAdjacencyList(float *distances_d, int *degArray, int *startIdxArray, int *A_d, int *B_d, int n, int k,
                               int m, float eps, int blockSize = 256) {
            int adjacencyList_size = degArray[n - 1] + startIdxArray[n - 1];

            int *adjacencyList_d = utils::allocateCudaArray<int>(adjacencyList_size);


            int numBlocks = std::max(1, n / blockSize);
            blockSize = std::min(n, blockSize);
            constructAdjacencyListForQueryVector<<<numBlocks, blockSize, 0>>>(distances_d,
                                                                              adjacencyList_d, startIdxArray,
                                                                              A_d, B_d, eps, n, k, m);

            return std::tie(adjacencyList_d, adjacencyList_size);
        }

        // TODO verify that the functions below are ok!

        __global__ void
        inline breadthFirstSearchKernel(int *adjacencyList_d, int *startIdxArray_d, int *visited_d, int *border_d, int n) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid >= n) {
                return; // We allocate more blocks than we need, so we need to check if we are out of bounds
            }

            if (border_d[tid]) {
                border_d[tid] = 0;
                visited_d[tid] = 1;

                int startIdx = startIdxArray_d[tid];

                for (int i = startIdx; i < startIdxArray_d[tid + 1]; i++) {
                    int neighbourIdx = adjacencyList_d[i];

                    if (!visited_d[neighbourIdx]) {
                        border_d[neighbourIdx] = 1;
                    }
                }
            }
        }

        inline void breadthFirstSearch(int *adjacencyList_d, int *startIdxArray_d, int *degArray_h, int *visited,
                                       int *clusterLabels, int *typeLabels, size_t n, int seedVertexIdx,
                                       int thisClusterLabel,
                                       int minPts, int blockSize = 256) {
            // NB: Fa is Border from GsDBSCAN paper, Xa is Visited,
            int* visitedThisBfs_d = utils::allocateCudaArray<int>(n, true); // Managed memory allows to set values from the CPU, and still be used in the GPU
            int* borderThisBfs_d = utils::allocateCudaArray<int>(n, true);

            borderThisBfs_d[seedVertexIdx] = 1;

            int countBordersThisBfs = 1;

            size_t gridSize = (n + blockSize - 1) / blockSize;

            while (countBordersThisBfs > 0) {
                // TODO, infinite loop here, need to fix this
                breadthFirstSearchKernel<<<gridSize, blockSize>>>(adjacencyList_d, startIdxArray_d, visitedThisBfs_d,
                                                                  borderThisBfs_d, n);
                cudaDeviceSynchronize();
                thrust::device_ptr<int> borderThisBfs_thrust(borderThisBfs_d);
                countBordersThisBfs = thrust::reduce(borderThisBfs_thrust, borderThisBfs_thrust + n, 0);
            }

//            auto visited_h = utils::copyDeviceToHost(visitedThisBfs_d, n);

            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                if (visitedThisBfs_d[i]) {
                    clusterLabels[i] = thisClusterLabel;
                    visited[i] = 1;
                    if (degArray_h[i] >= minPts) {
                        typeLabels[i] = 1; // Core pt
                    } else if (degArray_h[i] < minPts) {
                        typeLabels[i] = 0; // Border pt
                    }
                }
            }
        }


        inline std::tuple<int *, int *>
        formClusters(int *adjacencyList_d, int *degArray_d, int *startIdxArray_d, int n, int minPts) {
            int *clusterLabels = new int[n];
            int *typeLabels = new int[n];
//            std::fill(std::execution::par, typeLabels, typeLabels + n, -1); // TODO change to parallel, perhaps could use managed memory for the arrays?
            std::fill(clusterLabels, clusterLabels + n, -1); // Less than 100us for n=70000, so practically negligible in grander scheme
            std::fill(typeLabels, typeLabels + n, -1);
            int *visited = new int[n];

            auto degArray_h = utils::copyDeviceToHost(degArray_d, n);

            int currCluster = 0;

            for (int i = 0; i < n; i++) {
                if ((!visited[i]) && (degArray_h[i] >= minPts)) {
                    visited[i] = 1;
                    clusterLabels[i] = currCluster;
                    breadthFirstSearch(adjacencyList_d, startIdxArray_d, degArray_h, visited, clusterLabels, typeLabels, n, i,
                                       currCluster, minPts);
                    currCluster += 1;
                }
            }
            return std::tie(clusterLabels, typeLabels);
        }
    }
}

#endif //SDBSCAN_CLUSTERING_H
