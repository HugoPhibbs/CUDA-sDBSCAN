//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_CLUSTERING_H
#define SDBSCAN_CLUSTERING_H

#include <unordered_set>
#include <vector>
#include <tuple>
#include <unordered_set>
#include "cuda_runtime.h"
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
#include "GsDBSCAN_Params.h"
#include "../pch.h"
#include <mutex>

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
    inline int *constructQueryVectorDegreeArrayMatx(matx::tensor_t<T, 2> &distances, const T eps,
                                                    const std::string &distanceMetric,
                                                    matx::matxMemorySpace_t memorySpace = matx::MATX_MANAGED_MEMORY
    ) {
        /**
         * Yes, I know the below isn't very clean, but MatX is a bit of a pain when it comes to types.
         *
         * Hence why I'm repeating code across two forloops
         */
        int n = distances.Shape()[0];
        auto degArray = au::allocateCudaArray<int>(n);
        auto res = matx::make_tensor<int>(degArray, {n}, false);

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            auto closePoints = distances < eps;
            auto closePoints_int = matx::as_type<int>(closePoints);
            (res = matx::sum(closePoints_int, {1})).run();
        } else if (distanceMetric == "COSINE") {
            auto closePoints = distances > eps;
            auto closePoints_int = matx::as_type<int>(closePoints);
            (res = matx::sum(closePoints_int, {1})).run();
        } else {
            throw std::runtime_error("Invalid distance metric: " + distanceMetric);
        }
        return degArray;
    }

    inline int *constructStartIdxArray(int *degArray_d, int n, int initialStartIdx = 0) {
        int *startIdxArray_d = algo_utils::allocateCudaArray<int>(n);
        thrust::device_ptr<int> startIdxArray_thrust(startIdxArray_d);
        thrust::device_ptr<int> degArray_thrust(degArray_d);
        thrust::exclusive_scan(degArray_thrust, degArray_thrust + n,
                               startIdxArray_thrust, initialStartIdx); // Somehow this still runs anyhow?

        return startIdxArray_d;
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
                                         const int k, const int m, bool(*pointInCluster)(const float, const float),
                                         int AStartIdx) {
        // We assume one thread per query vector

        // TODO Make sure that the startIdx thing works
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
                BRow = A[(AStartIdx + idx) * 2 * k + ACol];
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
                           const int m, const float eps, int blockSize,
                           const std::string &distanceMetric, int AStartNIdx = 0) {
        // Assume the arrays aren't stored in managed memory
        int lastDegree = algo_utils::valueAtIdxDeviceToHost(degArray_d, n - 1);
        int lastStartIdx = algo_utils::valueAtIdxDeviceToHost(startIdxArray_d, n - 1);

        int adjacencyList_size =
                lastDegree + lastStartIdx;

        int *adjacencyList_d = algo_utils::allocateCudaArray<int>(adjacencyList_size);


        int gridSize = (n + blockSize - 1) / blockSize;
        blockSize = std::min(n, blockSize);

        auto pointInCluster_h = getPointInCluster_h(distanceMetric);

        constructAdjacencyListForQueryVector<<<gridSize, blockSize>>>(distances_d,
                                                                      adjacencyList_d,
                                                                      startIdxArray_d,
                                                                      A_d, B_d, eps, n, k, m,
                                                                      pointInCluster_h, AStartNIdx
        );
        cudaDeviceSynchronize();
        return std::tie(adjacencyList_d, adjacencyList_size);
    }


    inline std::tuple<std::vector<std::vector<int>>, boost::dynamic_bitset<>>
    processAdjacencyListCpu(int *adjacencyList_d, int *degArray_d, int *startIdxArray_d,
                            GsDBSCAN::GsDBSCAN_Params &params, int adjacencyList_size,
                            nlohmann::ordered_json *times = nullptr) {
        if (params.verbose) std::cout << "Processing the adj list(CPU)" << std::endl;

        auto neighbourhoodMatrix = std::vector<std::vector<int>>(params.n, std::vector<int>());
        auto corePoints = boost::dynamic_bitset<>(params.n);

        auto timeCopyClusteringArraysStart = au::timeNow();

        auto adjacencyList_h = algo_utils::copyDeviceToHost(adjacencyList_d, adjacencyList_size);
        auto startIdxArray_h = algo_utils::copyDeviceToHost(startIdxArray_d, params.n);
        auto degArray_h = algo_utils::copyDeviceToHost(degArray_d, params.n);

        auto timeCopyClusteringArrays = au::duration(timeCopyClusteringArraysStart, au::timeNow());

        if (times != nullptr && params.timeIt) {
            (*times)["copyClusteringArrays"] = timeCopyClusteringArrays;
        }

        std::vector<std::mutex> rowLocks(params.n);

        std::function<void(int i)> processPoint;

        if (params.ignoreAdjListSymmetry) {
            if (params.verbose) std::cout << "Not ensuring adj list symmetry" << std::endl;
            processPoint = [&](int i) {
                neighbourhoodMatrix[i] = std::vector<int>(adjacencyList_h + startIdxArray_h[i],
                                                          (adjacencyList_h + startIdxArray_h[i]) + degArray_h[i] / sizeof(int)
                );
            };
        } else {
            if (params.verbose) std::cout << "Ensuring adj list symmetry" << std::endl;
            processPoint = [&](int i) {
                for (int j = startIdxArray_h[i]; j < startIdxArray_h[i] + degArray_h[i]; j++) {
                    int candidateIdx = adjacencyList_h[j];
                    {
                        std::lock_guard<std::mutex> lock_i(rowLocks[i]);
                        neighbourhoodMatrix[i].push_back(candidateIdx);
                    }
                    {
                        std::lock_guard<std::mutex> lock_j(rowLocks[candidateIdx]);
                        neighbourhoodMatrix[candidateIdx].push_back(i);
                    }
                }
            };
        }

        #pragma omp parallel for
        for (int i = 0; i < params.n; i++) {
            processPoint(i);
        }

        #pragma omp parallel for
        for (int i = 0; i < params.n; i++) {
            std::sort(neighbourhoodMatrix[i].begin(), neighbourhoodMatrix[i].end());
            auto last_iter = std::unique(neighbourhoodMatrix[i].begin(), neighbourhoodMatrix[i].end());
            neighbourhoodMatrix[i].erase(last_iter, neighbourhoodMatrix[i].end());
            if ((int) neighbourhoodMatrix[i].size() >= params.minPts - 1) {
                corePoints[i] = true;
            }
        }

        delete[] adjacencyList_h;
        delete[] startIdxArray_h;
        delete[] degArray_h;

        return std::tie(neighbourhoodMatrix, corePoints);
    }

    inline std::tuple<int *, int>
    formClustersCPU(std::vector<std::vector<int>> &neighbourhoodMatrix, boost::dynamic_bitset<> &corePoints, int n) {
        int *clusterLabels = new int[n];
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

        cudaFree(borderThisBfs_d);
        cudaFree(visitedThisBfs_d);
    }

    inline std::tuple<int *, int *, int>
    formClusters(int *adjacencyList_d, int *degArray_d, int *startIdxArray_d, const int n, const int minPts,
                 const int blockSize) {
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

        delete[] degArray_h;
        delete[] visited;

        return std::tie(clusterLabels, typeLabels, currCluster);
    }

    inline std::tuple<int *, int, int *, int *>
    createClusteringArrays(matx::tensor_t<float, 2> &distances, matx::tensor_t<int, 2> &A_t,
                           matx::tensor_t<int, 2> &B_t, float eps, int clusterBlockSize,
                           const std::string &distanceMetric, nlohmann::ordered_json &times, bool timeIt,
                           int startIdx = 0) {

        int thisN = distances.Shape()[0]; // thisN as distances can be processed in batches - don't use A.shape(0)
        int k = A_t.Shape()[1] / 2;
        int m = B_t.Shape()[1];

        // Deg array
        auto degArrayStart = au::timeNow();

        auto degArray_d = clustering::constructQueryVectorDegreeArrayMatx(distances, eps, distanceMetric,
                                                                          matx::MATX_DEVICE_MEMORY);

        auto degArrayDuration = au::durationSinceStart(degArrayStart);

        // Start Idx array
        auto startIdxArrayStart = au::timeNow();

        auto startIdxArray_d = clustering::constructStartIdxArray(degArray_d, thisN);

        auto startIdxArrayDuration = au::durationSinceStart(startIdxArrayStart);

        // Adj list

        auto adjListStart = au::timeNow();

        auto [adjacencyList_d, adjacencyListSize] = clustering::constructAdjacencyList(
                distances.Data(), degArray_d,
                startIdxArray_d, A_t.Data(),
                B_t.Data(), thisN, k, m, eps,
                clusterBlockSize, distanceMetric, startIdx);

        auto adjListDuration = au::durationSinceStart(adjListStart);

        // Set times, allows for batching by accommodating for existing times
        if (timeIt) {
            times.contains("degArray") ? times["degArray"] = static_cast<int>(times["degArray"]) + degArrayDuration
                                       : times["degArray"] = degArrayDuration;
            times.contains("startIdxArray") ? times["startIdxArray"] =
                                                      static_cast<int>(times["startIdxArray"]) + startIdxArrayDuration
                                            : times["startIdxArray"] = startIdxArrayDuration;
            times.contains("adjList") ? times["adjList"] = static_cast<int>(times["adjList"]) + adjListDuration
                                      : times["adjList"] = adjListDuration;
        }
        return std::make_tuple(adjacencyList_d, adjacencyListSize, degArray_d, startIdxArray_d);
    }

    inline std::tuple<int *, int>
    performClustering(matx::tensor_t<float, 2> &distances, matx::tensor_t<int, 2> &A_t, matx::tensor_t<int, 2> &B_t,
                      GsDBSCAN::GsDBSCAN_Params &params, nlohmann::ordered_json &times) {

        auto startClustering = au::timeNow();

        auto [adjacencyList_d, adjacencyListSize, degArray_d, startIdxArray_d] = createClusteringArrays(distances, A_t,
                                                                                                        B_t, params.eps,
                                                                                                        params.clusterBlockSize,
                                                                                                        params.distanceMetric,
                                                                                                        times,
                                                                                                        params.timeIt);

        std::tuple<int *, int> result;

        if (params.clusterOnCpu) {
            auto startProcessAdjacencyList = au::timeNow();

            auto [neighbourhoodMatrix, corePoints] = processAdjacencyListCpu(adjacencyList_d, degArray_d,
                                                                             startIdxArray_d, params,
                                                                             adjacencyListSize, &times);

            if (params.timeIt) times["processAdjacencyList"] = au::duration(startProcessAdjacencyList, au::timeNow());

            auto startFormClusters = au::timeNow();

            result = formClustersCPU(neighbourhoodMatrix, corePoints, params.n);

            if (params.timeIt) times["formClusters"] = au::duration(startFormClusters, au::timeNow());
        } else {
            auto startFormClusters = au::timeNow();

            auto tup = clustering::formClusters(adjacencyList_d, degArray_d,
                                                startIdxArray_d, params.n,
                                                params.minPts, params.clusterBlockSize);


            if (params.timeIt) times["formClusters"] = au::duration(startFormClusters, au::timeNow());

            result = std::make_tuple(std::get<0>(tup), std::get<2>(tup));
        }

        cudaFree(adjacencyList_d);
        cudaFree(degArray_d);
        cudaFree(startIdxArray_d);

        if (params.timeIt) times["clusteringOverall"] = au::duration(startClustering, au::timeNow());

        return result;
    }
}

#endif //SDBSCAN_CLUSTERING_H
