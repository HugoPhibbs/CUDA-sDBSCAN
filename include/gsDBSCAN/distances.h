//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_DISTANCES_H
#define SDBSCAN_DISTANCES_H

#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../pch.h"

#include <cstdio>
#include "../../include/gsDBSCAN/algo_utils.h"

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                           \
        cudaError_t e = cudaGetLastError();                              \
        if (e != cudaSuccess) {                                          \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(e));                               \
            exit(EXIT_FAILURE);                                          \
        } else {                                                         \
            printf("CUDA call successful: %s:%d\n", __FILE__, __LINE__); \
        }                                                                \
    }

enum class DistanceMetric {
    L1,
    L2,
    COSINE
};

namespace GsDBSCAN::distances {
    /**
     * Calculates the batch size for distance calculations
     *
     * @param n size of the X dataset
     * @param d dimension of the X dataset
     * @param k k parameter of the DBSCAN algorithm
     * @param m m parameter of the DBSCAN algorithm
     * @param alpha alpha param to tune the batch size
     * @return int for the calculated batch size
     */
    inline int findDistanceBatchSize(const float alpha, const int n, const int d, const int k, const int m) {
        int batchSize = static_cast<int>((static_cast<long long>(n) * d * 2 * k * m) / (std::pow(1024, 3) * alpha));

        if (batchSize == 0) {
            return n;
        }

        for (int div = batchSize; div > 0; div--) {
            if (n % div == 0) {
                return div;
            }
        }

        return -1; // Should never reach here
    }

    inline torch::Tensor
    findDistancesTorchWithScripts(torch::Tensor &X, torch::Tensor &A, torch::Tensor &B, const float alpha,
                                  int batchSize, const std::string &distanceMetric, int XStartIdx = 0,
                                  int XEndIdx = -1) {
        if (XEndIdx == -1) {
            XEndIdx = X.size(0);
        }

        int k = A.size(1) / 2;
        int m = B.size(1);
        int d = X.size(1);

        int effectiveN = XEndIdx - XStartIdx;
        int actualN = X.size(0);

        batchSize = (batchSize != -1) ? batchSize : findDistanceBatchSize(alpha, actualN, d, k, m);

        torch::Tensor distances = torch::empty({effectiveN, 2 * k * m},
                                               torch::device(torch::kCUDA).dtype(torch::kFloat32));


        std::string moduleName = "";
        if (distanceMetric == "L1") {
            moduleName = "/home/hphi344/Documents/GS-DBSCAN-CPP/torch_scripts/distances_l1.pt";
        } else if (distanceMetric == "L2") {
            moduleName = "/home/hphi344/Documents/GS-DBSCAN-CPP/torch_scripts/distances_l2.pt";
        } else if (distanceMetric == "COSINE") {
            moduleName = "/home/hphi344/Documents/GS-DBSCAN-CPP/torch_scripts/distances_cosine.pt";
        } else {
            throw std::invalid_argument("Unsupported distance metric");
        }

        torch::jit::script::Module processBatchModule;

        try {
            processBatchModule = torch::jit::load(moduleName);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading the module: " << e.what() << std::endl;
            std::exit(-1);
        }

        const int numStreams = 4;
        std::vector<c10::cuda::CUDAStream> streams;
        for (int i = 0; i < numStreams; i++) {
            streams.push_back(c10::cuda::getStreamFromPool(false));
        }

        for (int i = 0; i < effectiveN; i += batchSize) {
            int maxDistancesIdx = std::min(i + batchSize, effectiveN);
            int thisBatchSize = maxDistancesIdx - i;

            int thisXStartIdx = i + XStartIdx;
            int thisXMaxIdx = thisXStartIdx + thisBatchSize;

            std::vector<torch::jit::IValue> inputs;

            inputs.push_back(X);
            inputs.push_back(A);
            inputs.push_back(B);
            inputs.push_back(distances);
            inputs.push_back(k);
            inputs.push_back(m);
            inputs.push_back(d);
            inputs.push_back(thisXStartIdx);
            inputs.push_back(thisXMaxIdx);
            inputs.push_back(thisBatchSize);
            inputs.push_back(i);
            inputs.push_back(maxDistancesIdx);

            c10::cuda::CUDAStreamGuard guard(streams[i % numStreams]);

            processBatchModule.forward(inputs);
        }

        for (int i = 0; i < numStreams; ++i) {
            streams[i].synchronize();
        }

        return distances;
    }

    inline torch::Tensor
    findDistancesTorch(torch::Tensor &X, torch::Tensor &A, torch::Tensor &B, const float alpha,
                       int batchSize, const std::string &distanceMetric, int XStartIdx = 0, int XEndIdx = -1) {


        if (XEndIdx == -1) {
            XEndIdx = X.size(0);
        }

        int k = A.size(1) / 2;
        int m = B.size(1);
        int d = X.size(1);

        int effectiveN = XEndIdx - XStartIdx;
        int actualN = X.size(0);

        batchSize = (batchSize != -1) ? batchSize : findDistanceBatchSize(alpha, actualN, d, k, m);

        torch::Tensor distances = torch::empty({effectiveN, 2 * k * m},
                                               torch::device(torch::kCUDA).dtype(torch::kFloat32));

        std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &)> calculate_distance;

        if (distanceMetric == "L2") {
            calculate_distance = [](const torch::Tensor &Z_batch_adj, const torch::Tensor &X_batch) {
                return torch::norm(Z_batch_adj - X_batch, 2, /*dim=*/2);
            };
        } else if (distanceMetric == "L1") {
            calculate_distance = [](const torch::Tensor &X_subset_adj, const torch::Tensor &X_batch) {
                return torch::norm(X_subset_adj - X_batch, 1, /*dim=*/2);
            };
        } else if (distanceMetric == "COSINE") {
            calculate_distance = [](const torch::Tensor &Z_batch_adj, const torch::Tensor &X_batch) {
                torch::Tensor product = Z_batch_adj * X_batch;
                return torch::sum(product, /*dim=*/2);
            };
        } else {
            throw std::invalid_argument("Unsupported distance metric");
        }

        for (int i = 0; i < effectiveN; i += batchSize) {
            int maxDistancesIdx = std::min(i + batchSize, effectiveN);
            int thisBatchSize = maxDistancesIdx - i;

            int thisXStartIdx = i + XStartIdx;
            int thisXMaxIdx = thisXStartIdx + thisBatchSize;

            // Equivalent to X[B[A[i:max_batch_idx]]] in Python
            torch::Tensor X_subset = X.index_select(0, B.index_select(0, A.slice(0, thisXStartIdx,
                                                                                 thisXMaxIdx).flatten()).flatten());
            torch::Tensor X_subset_adj = X_subset.view({thisBatchSize, 2 * k * m, d});

            torch::Tensor X_batch = X.slice(0, thisXStartIdx, thisXMaxIdx).unsqueeze(1);

            distances.slice(0, i, maxDistancesIdx) = calculate_distance(X_subset_adj, X_batch);
        }

        return distances;
    }
}


#endif //SDBSCAN_DISTANCES_H