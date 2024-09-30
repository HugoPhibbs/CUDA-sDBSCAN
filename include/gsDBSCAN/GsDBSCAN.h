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
    batchCreateClusteringVecs(torch::Tensor X, torch::Tensor A, torch::Tensor B, nlohmann::ordered_json &times, GsDBSCAN_Params &params)  {
        thrustDVec<int> adjacencyListVec(0);
        thrustDVec<int> degVec(params.n);
        thrustDVec<int> startIdxVec(params.n);

        auto A_matx = au::torchTensorToMatX<int>(A);
        auto B_matx = au::torchTensorToMatX<int>(B);

        int currAdjacencyListSize = 0;

        int totalTimeDistances = 0;
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

//            auto distancesBatch = distances::findDistancesTorchWithScripts(X, A, B, params.alpha, params.distancesBatchSize, params.distanceMetric, i,
//                                                        endIdx);

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
                                                     times, params.timeIt, i);

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

            if (params.verbose) au::printCUDAMemoryUsage();
            if (params.verbose) std::cout << "Curr adjacency list size: " << currAdjacencyListSize << std::endl;
            if (params.verbose) std::cout << "Batch adj list size: " << adjacencyListBatchSize << std::endl;
        }

        cudaDeviceSynchronize();

        if (params.timeIt) {
            times["totalTimeDistances"] = totalTimeDistances;
            times["totalTimeCopyMerge"] = totalTimeCopyMerge;
        }
        return std::make_tuple(adjacencyListVec, degVec, startIdxVec);
    }

    inline std::tuple<int *, int>
    performClusteringBatch(torch::Tensor X, torch::Tensor A, torch::Tensor B, nlohmann::ordered_json &times, GsDBSCAN_Params &params) {

        if (params.verbose) std::cout << "Creating clustering vecs (batching)" << std::endl;
        auto [adjacencyListVec, degVec, startIdxVec] = batchCreateClusteringVecs(X, A, B, times, params);

        if (params.verbose) std::cout << "Clustering vecs created" << std::endl;

        int *adjacencyList_d = thrust::raw_pointer_cast(adjacencyListVec.data());
        int *degArray_d = thrust::raw_pointer_cast(degVec.data());
        int *startIdxArray_d = thrust::raw_pointer_cast(startIdxVec.data());

        auto adjacencyListSize = adjacencyListVec.size();

        if (params.verbose) std::cout << "Adjacency List Size: " << adjacencyListSize << std::endl;

        auto processAdjacencyListStart = au::timeNow();

        if (params.verbose) std::cout << "Processing adjacency list" << std::endl;

        auto [neighbourhoodMatrix, corePoints] = clustering::processAdjacencyListCpu(adjacencyList_d, degArray_d,
                                                                                     startIdxArray_d, params,
                                                                                     adjacencyListSize, &times);

        if (params.verbose) std::cout << "Adjacency list processed" << std::endl;

        if (params.timeIt)
            times["processAdjacencyList"] = au::duration(processAdjacencyListStart, au::timeNow());

        auto startFormClusters = au::timeNow();

        if (params.verbose) std::cout << "Forming clusters (CPU)" << std::endl;

        auto result = clustering::formClustersCPU(neighbourhoodMatrix, corePoints, params.n);

        if (params.verbose) std::cout << "Clusters formed" << std::endl;

        if (params.timeIt)
            times["formClusters"] = au::duration(startFormClusters, au::timeNow());

        return result;
    }

    /**
    * Performs the gs dbscan algorithm
    *
    * @param X an array of size n * d containing the data points. For f32 use 'float' for f16, use 'uint_16' (this will be reinterpreted by Torch to a f16).
    * Elements should be in *row* major order
    * @param params a GsDBSCAN_Params object containing the parameters for the algorithm
    * @return a tuple containing:
    *  An integer array of size n containing the cluster labels for each point in the X dataset
    *  An integer array of size n containing the type labels for each point in the X dataset - e.g. Noise, Core, Border // TODO decide on how this will work?
    *  A nlohmann json object containing the timing information
    */
    template <typename XType, typename torch::Dtype TorchType>
    inline std::tuple<int *, int, nlohmann::ordered_json>
    performGsDbscan(XType *X, GsDBSCAN_Params &params) {

        nlohmann::ordered_json times;

        au::Time startOverAll = au::timeNow();

        // Normalise and perform projections

        au::Time startCopyingToDevice = au::timeNow();

        if (params.verbose) std::cout << "Preparing the X tensor" << std::endl;

        torch::TensorOptions XOptions = torch::TensorOptions().dtype(TorchType).device(torch::kCPU);
        auto XTorchCpu = torch::from_blob(X, {params.n, params.d}, XOptions);
        auto XTorchGPU = XTorchCpu.to(torch::kCUDA);

        cudaDeviceSynchronize();

        if (params.timeIt)
            times["copyingAndConvertData"] = au::duration(startCopyingToDevice, au::timeNow());

        // Normalise dataset

        auto startNormalise = au::timeNow();

        if (params.needToNormalise) {
            if (params.verbose) std::cout << "Normalising dataset" << std::endl;
            XTorchGPU = projections::normaliseDataset(XTorchGPU, params);
        }

        if (params.timeIt) times["normalise"] = au::duration(startNormalise, au::timeNow());

        int *clusterLabels = nullptr;
        int numClusters = -1;

        if (params.useBatchClustering) {
            if (params.verbose) std::cout << "Using batch clustering" << std::endl;

            auto startABMatrices = au::timeNow();

            auto [A_torch, B_torch] = projections::constructABMatricesBatch(XTorchGPU, params);

            if (params.timeIt)
                times["constructABMatrices"] = au::duration(startABMatrices, au::timeNow());

            cudaDeviceSynchronize();

            // Calculate distances and cluster at the same time

            if (params.verbose) std::cout << "Performing clustering (batching)" << std::endl;

            std::tie(clusterLabels, numClusters) = performClusteringBatch(XTorchGPU, A_torch, B_torch, times, params);

        } else {
            if (params.verbose) std::cout << "Not using batch clustering" << std::endl;

            au::Time startProjections = au::timeNow();

            if (params.verbose) std::cout << "Performing projections" << std::endl;

            auto projections_torch = projections::projectDataset(XTorchGPU, params.D, params.distanceMetric, params.fourierEmbedDim, params.sigmaEmbed);

            if (params.timeIt) times["projections"] = au::duration(startProjections, au::timeNow());

            // AB matrices

            auto startABMatrices = au::timeNow();

            if (params.verbose) std::cout << "Constructing AB matrices" << std::endl;

            auto [A_torch, B_torch] = projections::constructABMatrices(projections_torch, params.k, params.m,
                                                                       params.distanceMetric);

            if (params.timeIt) times["constructABMatrices"] = au::duration(startABMatrices, au::timeNow());

            // Distances

            auto startDistances = au::timeNow();

            if (params.verbose) std::cout << "Calculating distances" << std::endl;

            auto distances_torch = distances::findDistancesTorch(XTorchGPU, A_torch, B_torch, params.alpha, params.distancesBatchSize, params.distanceMetric);

            cudaDeviceSynchronize();

            if (params.timeIt) times["distances"] = au::duration(startDistances, au::timeNow());

            auto distances_matx = matx::make_tensor<float>(distances_torch.data_ptr<float>(), {params.n, 2*params.k*params.m}, matx::MATX_DEVICE_MEMORY);
            auto A_t = matx::make_tensor<int>(A_torch.data_ptr<int>(), {params.n, 2*params.k}, matx::MATX_DEVICE_MEMORY);
            auto B_t = matx::make_tensor<int>(B_torch.data_ptr<int>(), {2*params.D, params.m}, matx::MATX_DEVICE_MEMORY);

            if (params.verbose) std::cout << "Performing clustering" << std::endl;

            std::tie(clusterLabels, numClusters) = clustering::performClustering(distances_matx, A_t, B_t, params, times);
        }

        if (params.timeIt)
            times["overall"] = au::duration(startOverAll, au::timeNow());

        cudaDeviceSynchronize();

        if (params.verbose) std::cout << "Finished" << std::endl;

        return std::tie(clusterLabels, numClusters, times);
    }

};

#endif // DBSCANCEOS_GSDBSCAN_H
