//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_DISTANCES_H
#define SDBSCAN_DISTANCES_H

#include <cmath>
#include <chrono>
#include <iostream>
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

    inline torch::Tensor findDistancesTorch(torch::Tensor X, torch::Tensor A, torch::Tensor B, const float alpha = 1.2, int batchSize = -1, const std::string &distanceMetric = "L2",
                              matx::matxMemorySpace_t memorySpace = matx::MATX_DEVICE_MEMORY) {
        int k = A.size(1) / 2;
        int m = B.size(1);
        int n = X.size(0);
        int d = X.size(1);
        int D = B.size(0) / 2;

        batchSize = (batchSize != -1) ? batchSize : findDistanceBatchSize(alpha, n, d, k, m);

        // Prepare the distances tensor
        torch::Tensor distances = torch::empty({n, 2 * k * m}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n; i += batchSize) {
            int max_batch_idx = std::min(i + batchSize, n);

            // Equivalent to X[B[A[i:max_batch_idx]]] in Python
            torch::Tensor Z_batch = X.index_select(0, B.index_select(0, A.slice(0, i, max_batch_idx).flatten()).flatten());
            torch::Tensor Z_batch_adj = Z_batch.view({max_batch_idx - i, 2 * k * m, d});

            if (distanceMetric == "L2") {
                distances.slice(0, i, max_batch_idx) = torch::norm(
                        Z_batch_adj - X.slice(0, i, max_batch_idx).unsqueeze(1),
                        2, /*dim=*/2
                );
            } else if (distanceMetric == "L1") {
                distances.slice(0, i, max_batch_idx) = torch::norm(
                        Z_batch_adj - X.slice(0, i, max_batch_idx).unsqueeze(1),
                        1, /*dim=*/2
                );
            } else if (distanceMetric == "COSINE") {
                torch::Tensor product = Z_batch_adj * X.slice(0, i, max_batch_idx).unsqueeze(1);
                distances.slice(0, i, max_batch_idx) = torch::sum(product, /*dim=*/2);
            }
        }

        auto loop_time = std::chrono::high_resolution_clock::now() - start_time;
        std::chrono::duration<double> loop_time_seconds = loop_time;
        std::cout << "Time for the loop: " << loop_time_seconds.count() << " seconds" << std::endl;

        return distances;
    }
}


#endif //SDBSCAN_DISTANCES_H