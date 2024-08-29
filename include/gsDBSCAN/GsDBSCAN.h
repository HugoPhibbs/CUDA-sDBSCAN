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

namespace au = GsDBSCAN::algo_utils;

namespace GsDBSCAN {

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
                    bool timeIt = false, bool clusterOnCpu = false, const std::string &projectionsMethod="AF", bool needToNormalize = true) {

        if (distanceMetric == "COSINE") {
            eps = 1 - eps; // We use cosine similarity, thus we need to convert the eps to a cosine distance.
        }

        nlohmann::ordered_json times;

        au::Time startOverAll = au::timeNow();

        // Normalise and perform projections

        auto X_d_col_major = au::copyHostToDevice(X, n*d);
        auto X_d_row_major = au::colMajorToRowMajorMat(X_d_col_major, n, d);

        au::Time startProjections = au::timeNow();

        auto X_torch = au::torchTensorFromDeviceArray<float, torch::kFloat32>(X_d_row_major, n, d);

        auto [projections_torch, X_torch_norm] = projections::normaliseAndProjectTorch(X_torch, D, needToNormalize);

        if (timeIt) times["projections"] = au::duration(startProjections, au::timeNow());

        // AB matrices

        auto startABMatrices = au::timeNow();

        auto [A_torch, B_torch] = projections::constructABMatricesTorch(projections_torch, k, m, distanceMetric);

        if (timeIt) times["constructABMatrices"] = au::duration(startABMatrices, au::timeNow());

        // Distances

        auto startDistances = au::timeNow();

        auto distances_torch = distances::findDistancesTorch(X_torch, A_torch, B_torch, alpha, distancesBatchSize, distanceMetric);

        cudaDeviceSynchronize();

        if (timeIt) times["distances"] = au::duration(startDistances, au::timeNow());

        auto distances_matx = matx::make_tensor<float>(distances_torch.data_ptr<float>(), {n, 2*k*m}, matx::MATX_DEVICE_MEMORY);
        auto A_t = matx::make_tensor<int>(A_torch.data_ptr<int>(), {n, 2*k}, matx::MATX_DEVICE_MEMORY);
        auto B_t = matx::make_tensor<int>(B_torch.data_ptr<int>(), {2*D, m}, matx::MATX_DEVICE_MEMORY);

        auto [clusterLabels, numClusters] = clustering::performClustering(distances_matx, A_t, B_t, eps, minPts, clusterBlockSize, distanceMetric, timeIt, times, clusterOnCpu);

        if (timeIt) times["overall"] = au::duration(startOverAll, au::timeNow());

        return std::tie(clusterLabels, numClusters, times);
    }


};

#endif //DBSCANCEOS_GSDBSCAN_H
