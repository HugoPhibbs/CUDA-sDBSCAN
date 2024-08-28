//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_CLUSTERING_H
#define SDBSCAN_CLUSTERING_H

#include <arrayfire.h>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <cuda_runtime.h>
#include <matx.h>
#include <vector>
#include <omp.h>
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
#include "../json.hpp"

namespace au = GsDBSCAN::algo_utils;

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
                                                                      matx::matxMemorySpace_t memorySpace = matx::MATX_MANAGED_MEMORY,
                                                                      const std::string &distanceMetric = "L2") {
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
    constructAdjacencyListForQueryVector(const float *distances, int *adjacencyList, const int *startIdxArray,
                                         const int *A, const int *B, const float eps,
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
            setPointInClusterL1L2<<<1, 1>>>((bool (**)(const float, const float)) pointInCluster_d);
        } else if (distanceMetric == "COSINE") {
            setPointInClusterCosine<<<1, 1>>>((bool (**)(const float, const float)) pointInCluster_d);
        }

        cudaMemcpy(&pointInCluster_h, pointInCluster_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        return (bool (*)(const float, const float)) pointInCluster_h;
    }

    inline std::tuple<int *, int>
    constructAdjacencyList(const float *distances_d, const int *degArray_d, const int *startIdxArray_d, int *A_d,
                           int *B_d, const int n, const int k,
                           const int m, const float eps, int blockSize = 256,
                           const std::string &distanceMetric = "L2") {
        // Assume the arrays aren't stored in managed memory
        int lastDegree = algo_utils::valueAtIdxDeviceToHost(degArray_d, n - 1);
        int lastStartIdx = algo_utils::valueAtIdxDeviceToHost(startIdxArray_d, n - 1);

        int adjacencyList_size =
                lastDegree + lastStartIdx; // This will cause a segfault if deg and/or start idx arrays not on the host

        int *adjacencyList_d = algo_utils::allocateCudaArray<int>(adjacencyList_size);


        int gridSize = (n + blockSize - 1) / blockSize;
        blockSize = std::min(n, blockSize);

        auto pointInCluster_h = getPointInCluster_h(distanceMetric);

        constructAdjacencyListForQueryVector<<<gridSize, blockSize>>>(distances_d,
                                                                      adjacencyList_d,
                                                                      startIdxArray_d,
                                                                      A_d, B_d, eps, n, k, m,
                                                                      pointInCluster_h
        );
        cudaDeviceSynchronize();
        return std::tie(adjacencyList_d, adjacencyList_size);
    }


    inline std::tuple<std::vector<std::vector<int>>, boost::dynamic_bitset<>> processAdjacencyListCpu(int *adjacencyList_d, int *degArray_d, int *startIdxArray_d, int n, int adjacencyList_size,
                            int minPts) {
        auto neighbourhoodMatrix = std::vector<std::vector<int>>(n, std::vector<int>());
        auto corePoints = boost::dynamic_bitset<>(n);

        auto adjacencyList_h = algo_utils::copyDeviceToHost(adjacencyList_d, adjacencyList_size);
        auto startIdxArray_h = algo_utils::copyDeviceToHost(startIdxArray_d, n);
        auto degArray_h = algo_utils::copyDeviceToHost(degArray_d, n);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = startIdxArray_h[i]; j < startIdxArray_h[i] + degArray_h[i]; j++) {
                int candidateIdx = adjacencyList_h[j];
                #pragma omp critical
                    {
                        neighbourhoodMatrix[i].push_back(candidateIdx);
                        neighbourhoodMatrix[candidateIdx].push_back(i);
                    }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            std::unordered_set<int> neighbourhoodSet(neighbourhoodMatrix[i].begin(), neighbourhoodMatrix[i].end());
            neighbourhoodMatrix[i].clear();
            if ((int) neighbourhoodSet.size() >= minPts) { // TODO why did Ninh's code have minPts - 1?
                corePoints[i] = true;
                neighbourhoodMatrix[i].insert(neighbourhoodMatrix[i].end(), neighbourhoodSet.begin(),
                                              neighbourhoodSet.end()); // TODO is this line wrong?, why use .end() of neighbourhoodMatrix
            }
        }

        delete[] adjacencyList_h;
        delete[] startIdxArray_h;
        delete[] degArray_h;

        return std::tie(neighbourhoodMatrix, corePoints);
    }

    inline std::tuple<int *, int>
    formClustersCPU(std::vector<std::vector<int>> &neighbourhoodMatrix, boost::dynamic_bitset<> &corePoints, int n) {
        int* clusterLabels = new int[n];
        std::fill(clusterLabels, clusterLabels + n, -1);
        auto numClusters = 0;

        int currClusterID = 0;

        for (int i = 0; i < n; i++) {
            if ((!corePoints[i]) || (clusterLabels[i] != -1)) {
                continue; // Skip if not a core point or already assigned to a cluster
            }

            // TODO somehow the corePoints bitset is being ignored here for points that are non-core?

            std::unordered_set<int> seedSet;
            seedSet.insert(i);

            boost::dynamic_bitset<> connectedPoints(n);
            connectedPoints[i] = true;

            while (seedSet.size() > 0) {
                int currSeed = *seedSet.begin();
                seedSet.erase(seedSet.begin());

                auto thisNeighbourhood = neighbourhoodMatrix[currSeed];

                for (auto const &neighbourIdx: thisNeighbourhood) {
                    if (corePoints[neighbourIdx]) {
                        if (!connectedPoints[neighbourIdx]) {
                            connectedPoints[neighbourIdx] = true;

                            if (clusterLabels[neighbourIdx] == -1) {
                                seedSet.insert(neighbourIdx);
                            }
                        }
                    } else {
                        connectedPoints[neighbourIdx] = true;
                    }
                }
            }

            size_t neighbourIdx = connectedPoints.find_first();
            while (neighbourIdx != boost::dynamic_bitset<>::npos) {
                if (clusterLabels[neighbourIdx] == -1) {
                    clusterLabels[neighbourIdx] = currClusterID;
                }

                neighbourIdx = connectedPoints.find_next(neighbourIdx);
            }

            currClusterID++;
            numClusters++;
        }

        return std::make_tuple(clusterLabels, numClusters);
    }

    __global__ void
    inline
    breadthFirstSearchKernel(int *adjacencyList_d, int *startIdxArray_d, bool *visited_d, bool *border_d, bool *visited,
                             const size_t n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= n) {
            return; // We allocate more blocks than we need, so we need to check if we are out of bounds
        }

        if (border_d[tid]) {
            border_d[tid] = false;
            visited_d[tid] = true;

            if (!visited[tid]) {
                int startIdx = startIdxArray_d[tid];

                for (int i = startIdx; i < startIdxArray_d[tid + 1]; i++) {
                    int neighbourIdx = adjacencyList_d[i];

                    if (!visited_d[neighbourIdx]) {
                        border_d[neighbourIdx] = true;
                    }
                }
            }
        }
    }

    inline void
    breadthFirstSearch(int *adjacencyList_d, int *degArray_h, int *degArray_d, int *startIdxArray_d, bool *visited,
                       int *clusterLabels,
                       int *typeLabels, const size_t n, const int seedVertexIdx, const int thisClusterLabel,
                       const int minPts, const int blockSize) {
        // NB: Fa is Border from GsDBSCAN paper, Xa is Visited,
        // Managed memory allows to set values from the CPU, and still be used in the GPU
        bool *borderThisBfs_d = algo_utils::allocateCudaArray<bool>(n, true, true, false);

        bool *visitedThisBfs_d = algo_utils::allocateCudaArray<bool>(n, true, true, false);

//        cudaMemset(borderThisBfs_d, 0, n * sizeof(int));
//        cudaMemset(visitedThisBfs_d, 0, n * sizeof(int));
        borderThisBfs_d[seedVertexIdx] = true;

        int countBordersThisBfs = 1;

        size_t gridSize = (n + blockSize - 1) / blockSize;

        while (countBordersThisBfs > 0) {
            // TODO, infinite loop here, need to fix this
            breadthFirstSearchKernel<<<gridSize, blockSize>>>(adjacencyList_d, startIdxArray_d, visitedThisBfs_d,
                                                              borderThisBfs_d, visited, n);
            cudaDeviceSynchronize();
            thrust::device_ptr<bool> borderThisBfs_thrust(borderThisBfs_d);
            countBordersThisBfs = thrust::reduce(borderThisBfs_thrust, borderThisBfs_thrust + n, 0,
                                                 thrust::plus<int>());
        }

        #pragma omp parallel for // TODO the below could be a kernel right?, simply allocate the cluster labels to managed memory
        for (int i = 0; i < n; i++) {
            if (visitedThisBfs_d[i]) {
                clusterLabels[i] = thisClusterLabel;
                visited[i] = true;
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
        bool *visited = algo_utils::allocateCudaArray(n, true, true, false);

        auto degArray_h = algo_utils::copyDeviceToHost(degArray_d,
                                                       n); // TODO may be faster to keep this in managed memory - not sure

        int currCluster = 0;

        for (int i = 0; i < n; i++) {
            if ((!visited[i]) && (degArray_h[i] >= minPts)) {
                clusterLabels[i] = currCluster;
                breadthFirstSearch(adjacencyList_d, degArray_h, degArray_d, startIdxArray_d, visited, clusterLabels,
                                   typeLabels,
                                   n, i, currCluster, minPts, blockSize);
                currCluster += 1;
                visited[i] = true;
            }
        }

        return std::make_tuple(clusterLabels, typeLabels, currCluster);
    }

    inline std::tuple<int *, int>
    performClustering(matx::tensor_t<float, 2> &distances, matx::tensor_t<int, 2> &A_t, matx::tensor_t<int, 2> &B_t,
                      const float eps, const int minPts, const int clusterBlockSize,
                      const std::string &distanceMetric, bool timeIt, nlohmann::ordered_json &times,
                      bool clusterOnCpu = false) {

        int n = distances.Shape()[0];
        int k = A_t.Shape()[1] / 2;
        int m = B_t.Shape()[1];

        auto startClustering = au::timeNow();

        auto startDegArray = au::timeNow();

        auto degArray_t = clustering::constructQueryVectorDegreeArrayMatx(distances, eps, matx::MATX_DEVICE_MEMORY,
                                                                          distanceMetric);
        auto degArray_d = degArray_t.Data(); // Can't embed this in the above function call, bc pointer gets downgraded to a host one

        if (timeIt) times["degArray"] = au::duration(startDegArray, au::timeNow());

        auto startStartIdxArray = au::timeNow();

        int *startIdxArray_d = clustering::processQueryVectorDegreeArrayThrust(degArray_d, n);

        if (timeIt) times["startIdxArray"] = au::duration(startStartIdxArray, au::timeNow());

        auto startAdjacencyList = au::timeNow();

        auto [adjacencyList_d, adjacencyList_size] = clustering::constructAdjacencyList(distances.Data(), degArray_d,
                                                                                        startIdxArray_d, A_t.Data(),
                                                                                        B_t.Data(), n, k, m, eps,
                                                                                        clusterBlockSize,
                                                                                        distanceMetric);

        if (timeIt) times["adjacencyList"] = au::duration(startAdjacencyList, au::timeNow());

        std::tuple<int *, int> result;

        if (clusterOnCpu) {
            auto startProcessAdjacencyList = au::timeNow();

            auto [neighbourhoodMatrix, corePoints] = processAdjacencyListCpu(adjacencyList_d, degArray_d,
                                                                             startIdxArray_d, n,
                                                                             adjacencyList_size, minPts);

            if (timeIt) times["processAdjacencyList"] = au::duration(startProcessAdjacencyList, au::timeNow());

            auto startFormClusters = au::timeNow();

            result = formClustersCPU(neighbourhoodMatrix, corePoints, n);

            if (timeIt) times["formClusters"] = au::duration(startFormClusters, au::timeNow());
        } else {
            auto startFormClusters = au::timeNow();

            auto tup = clustering::formClusters(adjacencyList_d, degArray_d,
                                                startIdxArray_d, n,
                                                minPts, clusterBlockSize);


            if (timeIt) times["formClusters"] = au::duration(startFormClusters, au::timeNow());

            result = std::make_tuple(std::get<0>(tup), std::get<2>(tup));
        }

        cudaFree(adjacencyList_d);
        cudaFree(degArray_d);
        cudaFree(startIdxArray_d);

        if (timeIt) times["clusteringOverall"] = au::duration(startClustering, au::timeNow());

        return result;
    }
}

#endif //SDBSCAN_CLUSTERING_H
