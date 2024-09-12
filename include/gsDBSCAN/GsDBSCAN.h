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
#include "GsDBSCAN_Params.h"

using json = nlohmann::json;

template<typename T>
using thrustDVec = thrust::device_vector<T>;

namespace au = GsDBSCAN::algo_utils;

namespace GsDBSCAN {

    inline std::tuple<thrustDVec<int>, thrustDVec<int>, thrustDVec<int>>
    batchCreateClusteringVecs(torch::Tensor X, torch::Tensor A, torch::Tensor B, nlohmann::ordered_json &times, GsDBSCAN_Params params)  {
        thrustDVec<int> adjacencyListVec(0);
        thrustDVec<int> degVec(params.n);
        thrustDVec<int> startIdxVec(params.n);

        auto A_matx = au::torchTensorToMatX<int>(A);
        auto B_matx = au::torchTensorToMatX<int>(B);

        int currAdjacencyListSize = 0;

        int totalTimeDistances = 0;
        int totalTimeDegArray = 0;
        int totalTimeAdjList = 0;
        int totalTimeStartIdxArray = 0;
        int totalTimeCopyMerge = 0;

        int startIdxArrayInitialValue = 0;

        for (int i = 0; i < params.n; i += params.miniBatchSize) {
            int endIdx = std::min(i + params.miniBatchSize, params.n);

            /*
             * Get the batch distances
             */

            auto distanceBatchStart = au::timeNow();

            auto distancesBatch = distances::findDistancesTorch(X, A, B, params.alpha, params.distancesBatchSize, params.distanceMetric, i,
                                                                endIdx);

            cudaDeviceSynchronize();

            totalTimeDistances += au::duration(distanceBatchStart, au::timeNow());

            auto thisN = distancesBatch.size(0);

            auto distancesBatchMatx = au::torchTensorToMatX<float>(distancesBatch);

            /*
             * Get the clustering arrays
             */

            auto [adjacencyListBatch_d,
                  adjacencyListBatchSize,
                  degArrayBatch_d,
                  startIdxArrayBatch_d
              ] = clustering::createClusteringArrays(distancesBatchMatx, A_matx, B_matx,
                                                     params.eps, params.clusterBlockSize, params.distanceMetric,
                                                     times, i);

            auto copyMergeStart = au::timeNow();

            /*
             * Copy Results
             */

            // For degArray simply copy
            thrust::copy(degArrayBatch_d, degArrayBatch_d + thisN, degVec.begin() + i);

            // For startIdx, need to account for the current start idx
            thrust::device_ptr<int> startIdxArray_thrustPtr(startIdxArrayBatch_d);
            thrust::transform(startIdxArray_thrustPtr, startIdxArray_thrustPtr+thisN, startIdxArray_thrustPtr,
                              [startIdxArrayInitialValue] __device__ (float x) { return x + startIdxArrayInitialValue;}
                              );
            thrust::copy(startIdxArrayBatch_d, startIdxArrayBatch_d + thisN, startIdxVec.begin() + i);

            // For adj list, need to resize the vec, and add the results to the end
            adjacencyListVec.resize(currAdjacencyListSize + adjacencyListBatchSize);
            thrust::copy(adjacencyListBatch_d, adjacencyListBatch_d + adjacencyListBatchSize,
                         adjacencyListVec.begin() + currAdjacencyListSize);
            currAdjacencyListSize += adjacencyListBatchSize;

            cudaDeviceSynchronize();

            // Set the last element in degArrayBatch_d to startIdxArrayInitialValue
            startIdxArrayInitialValue = currAdjacencyListSize;

            totalTimeCopyMerge += au::duration(copyMergeStart, au::timeNow());

            // Free memory
            cudaFree(degArrayBatch_d);
            cudaFree(startIdxArrayBatch_d);
            cudaFree(adjacencyListBatch_d); // TODO do i also need to remove the distances array?
        }

        cudaDeviceSynchronize();

        times["totalTimeDistances"] = totalTimeDistances;
        times["totalTimeCopyMerge"] = totalTimeCopyMerge;

        return std::make_tuple(adjacencyListVec, degVec, startIdxVec);
    }

    inline std::tuple<int *, int>
    performClusteringBatch(torch::Tensor X, torch::Tensor A, torch::Tensor B, nlohmann::ordered_json &times, GsDBSCAN_Params params) {

        auto [adjacencyListVec, degVec, startIdxVec] = batchCreateClusteringVecs(X, A, B, times, params);

        int *adjacencyList_d = thrust::raw_pointer_cast(adjacencyListVec.data());
        int *degArray_d = thrust::raw_pointer_cast(degVec.data());
        int *startIdxArray_d = thrust::raw_pointer_cast(startIdxVec.data());

        auto adjacencyListSize = adjacencyListVec.size();

        auto processAdjacencyListStart = au::timeNow();

        auto [neighbourhoodMatrix, corePoints] = clustering::processAdjacencyListCpu(adjacencyList_d, degArray_d,
                                                                                     startIdxArray_d, params.n,
                                                                                     adjacencyListSize, params.minPts, &times);

        if (params.timeIt)
            times["processAdjacencyList"] = au::duration(processAdjacencyListStart, au::timeNow());

        auto startFormClusters = au::timeNow();

        auto result = clustering::formClustersCPU(neighbourhoodMatrix, corePoints, params.n);

        if (params.timeIt)
            times["formClusters"] = au::duration(startFormClusters, au::timeNow());

        return result;
    }

    /**
    * Performs the gs dbscan algorithm
    *
    * @param X a float array of size n * d containing the data points
    * @param params a GsDBSCAN_Params object containing the parameters for the algorithm
    * @return a tuple containing:
    *  An integer array of size n containing the cluster labels for each point in the X dataset
    *  An integer array of size n containing the type labels for each point in the X dataset - e.g. Noise, Core, Border // TODO decide on how this will work?
    *  A nlohmann json object containing the timing information
    */
    inline std::tuple<int *, int, nlohmann::ordered_json>
    performGsDbscan(float *X, GsDBSCAN_Params params) {

        nlohmann::ordered_json times;

        au::Time startOverAll = au::timeNow();

        // Normalise and perform projections

        au::Time startCopyingToDevice = au::timeNow();

        auto X_d_col_major = au::copyHostToDevice(X, params.n * params.d);
        auto X_d_row_major = au::colMajorToRowMajorMat(X_d_col_major, params.n, params.d);

        cudaDeviceSynchronize();

        cudaFree(X_d_col_major);

        if (params.timeIt)
            times["copyingAndConvertData"] = au::duration(startCopyingToDevice, au::timeNow());

        au::Time startProjections = au::timeNow();

        auto X_torch = au::torchTensorFromDeviceArray<float, torch::kFloat32>(X_d_row_major, params.n, params.d);

        if (params.timeIt)
            times["projectionsAndNormalize"] = au::duration(startProjections, au::timeNow());

        // AB matrices

        auto startABMatrices = au::timeNow();

        params.ABatchSize = 10000;
        params.BBatchSize = 128;

        if (params.needToNormalise) {
            X_torch = projections::normaliseDatasetTorch(X_torch);
        }

        auto [A_torch, B_torch] = projections::constructABMatricesBatch(X_torch, params);

        if (params.timeIt)
            times["constructABMatrices"] = au::duration(startABMatrices, au::timeNow());

        cudaDeviceSynchronize();

        // Calculate distances and cluster at the same time

        auto [clusterLabels, numClusters] = performClusteringBatch(X_torch, A_torch, B_torch, times, params);

        if (params.timeIt)
            times["overall"] = au::duration(startOverAll, au::timeNow());

        cudaDeviceSynchronize();

        cudaFree(X_d_row_major);

        return std::tie(clusterLabels, numClusters, times);
    }

};

#endif // DBSCANCEOS_GSDBSCAN_H
