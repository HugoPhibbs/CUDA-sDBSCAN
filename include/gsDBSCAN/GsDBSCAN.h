//
// Created by hphi344 on 10/05/24.
//

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#include <arrayfire.h>
#include "cuda_runtime.h"
#include <af/cuda.h>
#include <arrayfire.h>
#include <matx.h>
#include <chrono>
#include <tuple>

#include "../json.hpp"
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
                    bool timeIt = false, bool clusterOnCpu = false) {

        if (distanceMetric == "COSINE") {
            eps = 1 - eps; // We use cosine similarity, thus we need to convert the eps to a cosine distance.
        }

        nlohmann::ordered_json times;

        au::Time startOverAll = au::timeNow();

        // Normalise and perform projections

        au::Time startProjections = au::timeNow();

        auto X_af = af::array(n, d, X);
        X_af.eval();

        X_af = projections::normaliseDatasetAF(X_af);
        X_af.eval();

        auto projections = projections::performProjectionsAF(X_af, D);

        projections.eval();

        if (timeIt) times["projections"] = au::duration(startProjections, au::timeNow());


        // Get a tensor for X

        auto X_t = algo_utils::afMatToMatXTensor<float, float>(X_af, matx::MATX_DEVICE_MEMORY);


        // AB matrices

        auto startABMatrices = au::timeNow();

        auto [A_af, B_af] = projections::constructABMatricesAF(projections, k, m, distanceMetric);

        auto A_t = algo_utils::afMatToMatXTensor<int, int>(A_af,
                                                           matx::MATX_DEVICE_MEMORY); // TODO use MANAGED or DEVICE memory?
        auto B_t = algo_utils::afMatToMatXTensor<int, int>(B_af,
                                                           matx::MATX_DEVICE_MEMORY); // TODO use MANAGED or DEVICE memory?

        if (timeIt) times["constructABMatrices"] = au::duration(startABMatrices, au::timeNow());


        // Distances

        auto startDistances = au::timeNow();

        matx::tensor_t<float, 2> distances = distances::findDistancesMatX(X_t, A_t, B_t, alpha, distancesBatchSize,
                                                                          distanceMetric,
                                                                          matx::MATX_DEVICE_MEMORY);
        cudaDeviceSynchronize();

        if (timeIt) times["distances"] = au::duration(startDistances, au::timeNow());


        // Clustering

        auto [clusterLabels, numClusters] = clustering::performClustering(distances, A_t, B_t, eps, minPts, clusterBlockSize, distanceMetric, timeIt, times, clusterOnCpu);

        // Free memory
        A_af.unlock();
        B_af.unlock();

        if (timeIt) times["overall"] = au::duration(startOverAll, au::timeNow());

        return std::make_tuple(clusterLabels, numClusters, times);
    }


};

#endif //DBSCANCEOS_GSDBSCAN_H
