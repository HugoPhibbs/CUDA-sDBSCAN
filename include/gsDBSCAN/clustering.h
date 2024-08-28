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
#include "algo_utils.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include "algo_utils.h"


namespace GsDBSCAN::clustering {

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
     * @param memorySpace The memory space to allocate the result tensor (and therefore the result) in.
     * @param distanceMetric The distance metric to use. Can be "L1", "L2" or "COSINE". "COSINE" refers to cosine similarity
     *
     * @return Pointer to the degree array. Since this is intended to be how this is used for later steps
     */
    template<typename T>
    inline matx::tensor_t<int, 1> constructQueryVectorDegreeArrayMatx(matx::tensor_t<T, 2> &distances, const T eps,
                                                                      matx::matxMemorySpace_t memorySpace = matx::MATX_MANAGED_MEMORY, const std::string &distanceMetric = "L2") {
        /**
         * Yes, I know the below isn't very clean, but MatX is a bit of a pain when it comes to types.
         *
         * Hence why I'm repeating code across two forloops
         */
        int n = distances.Shape()[0];
        auto res = matx::make_tensor<int>({n}, memorySpace);

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            auto closePoints = distances < eps;
            auto closePoints_int = matx::as_type<int>(closePoints);
            (res = matx::sum(closePoints_int, {1})).run();
            return res;
        } else if (distanceMetric == "COSINE") {
            auto closePoints = distances > eps;
            auto closePoints_int = matx::as_type<int>(closePoints);
            (res = matx::sum(closePoints_int, {1})).run();
            return res;
        } else {
            throw std::runtime_error("Invalid distance metric: " + distanceMetric);
        }

 // Somehow if i return .Data() it casts the pointer to an unregistered host pointer, so I'm returning the tensor itself
    }

    inline int *processQueryVectorDegreeArrayThrust(int *degArray_d, int n) {
        int *startIdxArray_d = algo_utils::allocateCudaArray<int>(n);
        thrust::device_ptr<int> startIdxArray_thrust(startIdxArray_d);
        thrust::device_ptr<int> degArray_thrust(degArray_d);
        thrust::exclusive_scan(degArray_thrust, degArray_thrust + n,
                               startIdxArray_thrust); // Somehow this still runs anyhow?
        return startIdxArray_d;
    }

    template <typename T>
    inline T valueAtIdxDeviceToHost(const T* deviceArray, const int idx) {
        T value;
        cudaMemcpy(&value, deviceArray + idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
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

    /**
     * Kernel for constructing part of the cluster graph adjacency list for a particular vector
     *
     * @param distances matrix containing the distances between each query vector and it's candidate vectors
     * @param adjacencyList
     * @param startIdxArray vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
     * @param A A matrix, see constructABMatricesAF. Stored flat as a float array
     * @param B B matrix, see constructABMatricesAF. Stored flat as a float array
     * @param n number of query vectors in the dataset
     * @param eps epsilon DBSCAN density param
     */
    __global__ void
    inline
    constructAdjacencyListForQueryVector(const float *distances, int *adjacencyList, const int *startIdxArray, const int *A, const int *B, const float eps,
                                         const int n,
                                         const int k, const int m, bool(*pointInCluster)(const float, const float)) {
        // We assume one thread per query vector

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return; // Exit if out of bounds. Don't assume that numQueryVectors is equal to the total number o threads

        int curr_idx = startIdxArray[idx];

        int distances_rows = 2 * k * m;

        int ACol, BCol, BRow, neighbourhoodVecIdx;

        for (int j = 0; j < distances_rows; j++) {

            if (pointInCluster(distances[idx * distances_rows + j], eps)) {
                ACol = j / m;
                BCol = j % m;
                BRow = A[idx * 2 * k + ACol];
                neighbourhoodVecIdx = B[BRow * m + BCol];

                adjacencyList[curr_idx] = neighbourhoodVecIdx;
                curr_idx++;
            }
        }
    }

    inline __device__ bool pointInClusterL1L2(const float distance, const float eps) {
        return distance < eps;
    }

    inline __global__ void setPointInClusterL1L2(bool(**pointInCluster)(const float, const float)) {
        *pointInCluster = pointInClusterL1L2;
    }


    inline __device__ bool pointInClusterCosine(const float distance, const float eps) {
        return distance > eps;
    }


    inline __global__ void setPointInClusterCosine(bool(**pointInCluster)(const float, const float)) {
        *pointInCluster = pointInClusterCosine;
    }

    inline auto getPointInCluster_h(const std::string &distanceMetric) {
        /*
         * Basically the problem is that I can't reference a device function from the host,
         *
         * SO I need to do a slight of hand to get around this.
         *
         * I don't fully understand it, but I used the below link:
         *
         * https://stackoverflow.com/questions/26738079/cuda-kernel-with-function-pointer-and-variadic-templates
         */
        unsigned long long *pointInCluster_d, *pointInCluster_h;
        cudaMalloc(&pointInCluster_d, sizeof(unsigned long long));

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            setPointInClusterL1L2<<<1, 1>>>((bool(**)(const float, const float)) pointInCluster_d);
        } else if (distanceMetric == "COSINE") {
            setPointInClusterCosine<<<1, 1>>>((bool(**)(const float, const float)) pointInCluster_d);
        }

        cudaMemcpy(&pointInCluster_h, pointInCluster_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        return (bool(*)(const float, const float)) pointInCluster_h;
    }

    inline std::tuple<int *, int>
    constructAdjacencyList(const float *distances_d, const int *degArray_d, const int *startIdxArray_d, int *A_d, int *B_d, const int n, const int k,
                           const int m, const float eps, int blockSize = 256, const std::string &distanceMetric= "L2") {
        // Assume the arrays aren't stored in managed memory
        int lastDegree = valueAtIdxDeviceToHost(degArray_d, n - 1);
        int lastStartIdx = valueAtIdxDeviceToHost(startIdxArray_d, n - 1);

        int adjacencyList_size = lastDegree + lastStartIdx; // This will cause a segfault if deg and/or start idx arrays not on the host

        int *adjacencyList_d = algo_utils::allocateCudaArray<int>(adjacencyList_size);


        int gridSize = (n + blockSize - 1) / blockSize;
        blockSize = std::min(n, blockSize);

        auto pointInCluster_h = getPointInCluster_h(distanceMetric);

        constructAdjacencyListForQueryVector<<<gridSize, blockSize>>>(distances_d,
                                                                      adjacencyList_d, startIdxArray_d,
                                                                      A_d, B_d, eps, n, k, m,
                                                                      pointInCluster_h
                                                                      );
        cudaDeviceSynchronize();
        return std::tie(adjacencyList_d, adjacencyList_size);
    }

    // TODO verify that the functions below are ok!

    __global__ void
    inline
    breadthFirstSearchKernel(int *adjacencyList_d, int *startIdxArray_d, int *visited_d, int *border_d, const int n) {
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
//
//    __global__ void inline setVisitedKernel(int* visitedThisBfs_d)

    inline void
    breadthFirstSearch(int *adjacencyList_d, int *degArray_h, int *startIdxArray_d, int *visited, int *clusterLabels,
                       int *typeLabels, const size_t n, const int seedVertexIdx, const int thisClusterLabel, const int minPts, const int blockSize) {
        // NB: Fa is Border from GsDBSCAN paper, Xa is Visited,
        int *borderThisBfs_d = algo_utils::allocateCudaArray<int>(n, true, true);

        int *visitedThisBfs_d = algo_utils::allocateCudaArray<int>(n,
                                                                   true, true); // Managed memory allows to set values from the CPU, and still be used in the GPU

//        cudaMemset(borderThisBfs_d, 0, n * sizeof(int));
//        cudaMemset(visitedThisBfs_d, 0, n * sizeof(int));
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

        #pragma omp parallel for // TODO the below could be a kernel right?, simply allocate the cluster labels to managed memory
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




    inline std::tuple<int *, int *, int>
    formClusters(int *adjacencyList_d, int *degArray_d, int *startIdxArray_d, const int n, const int minPts,
                 const int blockSize = 256) {
        int *clusterLabels = new int[n];
        int *typeLabels = new int[n];
//            std::fill(std::execution::par, typeLabels, typeLabels + n, -1); // TODO change to parallel, perhaps could use managed memory for the arrays?
        std::fill(clusterLabels, clusterLabels + n,
                  -1); // Less than 300us for n=70000, so practically negligible in grander scheme
        std::fill(typeLabels, typeLabels + n, -1);
        int *visited = new int[n];

        std::fill(visited, visited + n, 0);

        auto degArray_h = algo_utils::copyDeviceToHost(degArray_d, n); // TODO may be faster to keep this in managed memory - not sure

        int currCluster = 0;

        for (int i = 0; i < n; i++) {
            if ((!visited[i]) && (degArray_h[i] >= minPts)) {
                visited[i] = 1;
                clusterLabels[i] = currCluster;
                breadthFirstSearch(adjacencyList_d, degArray_h, startIdxArray_d, visited, clusterLabels, typeLabels,
                                   n, i, currCluster, minPts, blockSize);
                currCluster += 1;
            }
        }

        return std::make_tuple(clusterLabels, typeLabels, currCluster + 1);
    }
}

#endif //SDBSCAN_CLUSTERING_H
