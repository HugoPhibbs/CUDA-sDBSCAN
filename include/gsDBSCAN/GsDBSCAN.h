//
// Created by hphi344 on 10/05/24.
//

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#include <chrono>
#include <tuple>

#include "../pch.h"
#include "projections.h"
#include "distances.h"
#include "algo_utils.h"
#include "clustering.h"

using json = nlohmann::json;

template<typename T>
using thrustDVec = thrust::device_vector<T>;

namespace au = GsDBSCAN::algo_utils;

namespace GsDBSCAN {

    inline float adjustEps(const std::string &distanceMetric, float eps) {
        if (distanceMetric == "COSINE") {
            return 1 - eps; // We use cosine similarity, thus we need to convert the eps to a cosine distance.
        }
        return eps;
    }

    inline std::tuple<thrustDVec<int>, thrustDVec<int>, thrustDVec<int>>
    batchCreateClusteringVecs(torch::Tensor X, torch::Tensor A, torch::Tensor B, int miniBatchSize, float eps, nlohmann::ordered_json &times,
                              const std::string &distanceMetric = "L2", float alpha = 1.2,
                              int distancesBatchSize = -1, int clusterBlockSize = 256) {
        int n = X.size(0);
        int k = A.size(1) / 2;
        int m = B.size(1);

        thrustDVec<int> adjacencyListVec(1);
        thrustDVec<int> degVec(n);
        thrustDVec<int> startIdxVec(n);

        auto A_matx = au::torchTensorToMatX<int>(A);
        auto B_matx = au::torchTensorToMatX<int>(B);

        int currAdjacencyListSize = 0;

        int totalTimeDistances = 0;
        int totalTimeDegArray = 0;
        int totalTimeAdjList = 0;
        int totalTimeStartIdxArray = 0;
        int totalTimeCopyMerge = 0;

        int startIdxArrayInitialValue = 0;

        for (int i = 0; i < n; i += miniBatchSize) {
            int endIdx = std::min(i + miniBatchSize, n);

            auto distanceBatchStart = au::timeNow();

            auto distancesBatch = distances::findDistancesTorch(X, A, B, alpha, distancesBatchSize, distanceMetric, i,
                                                                endIdx);

            cudaDeviceSynchronize();

            totalTimeDistances += au::duration(distanceBatchStart, au::timeNow());

            auto thisN = distancesBatch.size(0);

            auto distancesBatchMatx = au::torchTensorToMatX<float>(distancesBatch);

            auto degArrayBatchStart = au::timeNow();

            auto degArrayBatch_d = clustering::constructQueryVectorDegreeArrayMatx(distancesBatchMatx, eps,
                                                                                   matx::MATX_DEVICE_MEMORY,
                                                                                   distanceMetric);

            cudaDeviceSynchronize();

            totalTimeDegArray += au::duration(degArrayBatchStart, au::timeNow());

            auto startIdxArrayBatchStart = au::timeNow();

            auto startIdxArrayBatch_d = clustering::constructStartIdxArray(degArrayBatch_d, thisN,
                                                                           startIdxArrayInitialValue);

            cudaDeviceSynchronize();

            totalTimeStartIdxArray += au::duration(startIdxArrayBatchStart, au::timeNow());

            auto adjacencyListBatchStart = au::timeNow();

            auto [adjacencyListBatch_d, adjacencyListBatchSize] = clustering::constructAdjacencyList(
                    distancesBatchMatx.Data(), degArrayBatch_d,
                    startIdxArrayBatch_d, A_matx.Data(),
                    B_matx.Data(), thisN, k, m, eps,
                    clusterBlockSize, distanceMetric, i);

            cudaDeviceSynchronize();

            totalTimeAdjList += au::duration(adjacencyListBatchStart, au::timeNow());

            auto copyMergeStart = au::timeNow();

            // Set the last element in degArrayBatch_d to startIdxArrayInitialValue
            startIdxArrayInitialValue = au::valueAtIdxDeviceToHost(degArrayBatch_d, thisN - 1) + au::valueAtIdxDeviceToHost(startIdxArrayBatch_d, thisN - 1);

            // Copy Results
            thrust::copy(degArrayBatch_d, degArrayBatch_d + thisN, degVec.begin() + i);
            thrust::copy(startIdxArrayBatch_d, startIdxArrayBatch_d + thisN, startIdxVec.begin() + i);

            adjacencyListVec.resize(currAdjacencyListSize + adjacencyListBatchSize);
            thrust::copy(adjacencyListBatch_d, adjacencyListBatch_d + adjacencyListBatchSize,
                         adjacencyListVec.begin() + currAdjacencyListSize);
            currAdjacencyListSize += adjacencyListBatchSize;

            cudaDeviceSynchronize();

            totalTimeCopyMerge += au::duration(copyMergeStart, au::timeNow());
        }

        cudaDeviceSynchronize();

        times["totalTimeDistances"] = totalTimeDistances;
        times["totalTimeDegArray"] = totalTimeDegArray;
        times["totalTimeStartIdxArray"] = totalTimeStartIdxArray;
        times["totalTimeAdjList"] = totalTimeAdjList;
        times["totalTimeCopyMerge"] = totalTimeCopyMerge;

        return std::make_tuple(adjacencyListVec, degVec, startIdxVec);
    }

    inline std::tuple<int *, int>
    performClusteringBatch(torch::Tensor X, torch::Tensor A, torch::Tensor B, int miniBatchSize, float eps, int minPts,
                           nlohmann::ordered_json &times,
                           const std::string &distanceMetric = "L2", float alpha = 1.2,
                           int distancesBatchSize = -1, int clusterBlockSize = 256, bool timeIt = false) {
        int n = X.size(0);

        auto [adjacencyListVec, degVec, startIdxVec] = batchCreateClusteringVecs(X, A, B, miniBatchSize, eps, times,
                                                                                 distanceMetric, alpha,
                                                                                 distancesBatchSize, clusterBlockSize);

        int *adjacencyList_d = thrust::raw_pointer_cast(adjacencyListVec.data());
        int *degArray_d = thrust::raw_pointer_cast(degVec.data());
        int *startIdxArray_d = thrust::raw_pointer_cast(startIdxVec.data());

        auto adjacencyListSize = adjacencyListVec.size();

        auto processAdjacencyListStart = au::timeNow();

        auto [neighbourhoodMatrix, corePoints] = clustering::processAdjacencyListCpu(adjacencyList_d, degArray_d,
                                                                                     startIdxArray_d, n,
                                                                                     adjacencyListSize, minPts, &times);

        if (timeIt)
            times["processAdjacencyList"] = au::duration(processAdjacencyListStart, au::timeNow());

        auto startFormClusters = au::timeNow();

        auto result = clustering::formClustersCPU(neighbourhoodMatrix, corePoints, n);

        if (timeIt)
            times["formClusters"] = au::duration(startFormClusters, au::timeNow());

        return result;
    }

    /**
    * Performs the gs dbscan algorithm
    *
    * @param X array storing the dataset. Should be in COL major order. Stored on the GPU
    * @param n number of entries in the X dataset
    * @param d dimension of the X dataset
    * @param D int for number of random vectors to generate
    * @param minPts min number of points as per the DBSCAN algorithm
    * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
    * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
    * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
    * @param alpha float to tune the batch size when calculating distances
    * @param distanceMetric string for the distance metric to use. Options are "L1", "L2" or "COSINE"
    * @param distancesBatchSize int for the batch size to use when calculating distances, set to -1 to calculate automatically
    * @param clusterBlockSize int for the block size to use when clustering
    * @param timeIt bool to indicate whether to time the algorithm or not
    * @param clusterOnCpu bool to indicate whether to cluster on the CPU or not
    * @return a tuple containing:
    *  An integer array of size n containing the cluster labels for each point in the X dataset
    *  An integer array of size n containing the type labels for each point in the X dataset - e.g. Noise, Core, Border // TODO decide on how this will work?
    *  A nlohmann json object containing the timing information
    */
    inline std::tuple<int *, int, nlohmann::ordered_json>
    performGsDbscan(float *X, int n, int d, int D, int minPts, int k, int m, float eps, float alpha = 1.2,
                    int distancesBatchSize = -1, const std::string &distanceMetric = "L2", int clusterBlockSize = 256,
                    bool timeIt = false, bool clusterOnCpu = false, bool needToNormalize = true,
                    int fourierEmbedDim = 1024, float sigmaEmbed = 1) {

        eps = adjustEps(distanceMetric, eps);

        nlohmann::ordered_json times;

        au::Time startOverAll = au::timeNow();

        // Normalise and perform projections

        au::Time startCopyingToDevice = au::timeNow();

        auto X_d_col_major = au::copyHostToDevice(X, n * d);
        auto X_d_row_major = au::colMajorToRowMajorMat(X_d_col_major, n, d);

        if (timeIt)
            times["copyingAndConvertData"] = au::duration(startCopyingToDevice, au::timeNow());

        au::Time startProjections = au::timeNow();

        auto X_torch = au::torchTensorFromDeviceArray<float, torch::kFloat32>(X_d_row_major, n, d);

        auto [projections_torch, X_torch_norm] = projections::normaliseAndProjectTorch(X_torch, D, needToNormalize,
                                                                                       distanceMetric, fourierEmbedDim,
                                                                                       sigmaEmbed);

        if (timeIt)
            times["projectionsAndNormalize"] = au::duration(startProjections, au::timeNow());

        // AB matrices

        auto startABMatrices = au::timeNow();

        //        auto [A_torch, B_torch] = projections::constructABMatricesTorch(projections_torch, k, m, distanceMetric);

        auto [A_torch, B_torch] = projections::constructABMatricesBatch(X_torch, D, k, m, needToNormalize,
                                                                        distanceMetric, fourierEmbedDim, sigmaEmbed,
                                                                        10000, 128);

        if (timeIt)
            times["constructABMatrices"] = au::duration(startABMatrices, au::timeNow());

        //        // Distances
        //
        //        auto startDistances = au::timeNow();
        //
        //        auto distances_torch = distances::findDistancesTorch(X_torch, A_torch, B_torch, alpha, distancesBatchSize,
        //                                                             distanceMetric);
        //
        //        cudaDeviceSynchronize();
        //
        //        if (timeIt) times["distances"] = au::duration(startDistances, au::timeNow());

        //        auto timeMatXToTorch = au::timeNow();
        //
        //        auto distances_matx = au::torchTensorToMatX<float>(distances_torch);
        //        auto A_matx = au::torchTensorToMatX<int>(A_torch);
        //        auto B_matx = au::torchTensorToMatX<int>(B_torch);
        //
        //        if (timeIt) times["matXToTorch"] = au::duration(timeMatXToTorch, au::timeNow());
        //
        //        auto [clusterLabels, numClusters] = clustering::performClustering(distances_matx, A_matx, B_matx, eps, minPts,
        //                                                                          clusterBlockSize, distanceMetric, timeIt,
        //                                                                          times, clusterOnCpu);

        auto [clusterLabels, numClusters] = performClusteringBatch(X_torch, A_torch, B_torch, 10000, eps, minPts, times,
                                                                   distanceMetric, alpha, distancesBatchSize,
                                                                   clusterBlockSize, timeIt);

        if (timeIt)
            times["overall"] = au::duration(startOverAll, au::timeNow());

        return std::tie(clusterLabels, numClusters, times);
    }

};

#endif // DBSCANCEOS_GSDBSCAN_H
