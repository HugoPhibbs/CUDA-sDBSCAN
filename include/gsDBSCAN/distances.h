//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_DISTANCES_H
#define SDBSCAN_DISTANCES_H

#include <matx.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include "../pch.h"

#include <cstdio>
#include <arrayfire.h>
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

    /**
     * Finds the distances between each of the query points and their candidate neighbourhood vectors
     *
     * Uses L2 norm. I.e. Euclidean norm
     *
     * @param X matrix containing the X dataset vectors
     * @param A A matrix, see constructABMatricesAF
     * @param B B matrix, see constructABMatricesAF
     * @param alpha float for the alpha parameter to tune the batch size
     */
    inline af::array findDistancesL2AF(af::array &X, af::array &A, af::array &B, float alpha = 1.2) {
        int k = A.dims(1) / 2;
        int m = B.dims(1);

        int n = X.dims(0);
        int d = X.dims(1);
        int D = B.dims(0) / 2;

        int batchSize = findDistanceBatchSize(alpha, n, d, k, m);

        af::array distances(n, 2 * k * m, af::dtype::f32);
        af::array ABatch(batchSize, 2 * k, A.type());
        af::array BBatch(batchSize, m, B.type());
        af::array XBatch(batchSize, 2 * k, m, d, X.type());
        af::array XBatchAdj(batchSize, 2 * k * m, d,
                            X.type()); // This is very large, around 7gb. Possible to do this without explicitly allocating the memory?
        af::array XSubset(batchSize, d, X.type());
        af::array XSubsetReshaped = af::constant(0, XBatchAdj.dims(), XBatchAdj.type());
        af::array YBatch = af::constant(0, XBatchAdj.dims(), XBatchAdj.type());

        for (int i = 0; i < n; i += batchSize) {
            int maxBatchIdx = i + batchSize - 1;

            af::array AIndex = af::seq(i, maxBatchIdx);
            ABatch = A(AIndex, af::span);

            BBatch = B(ABatch, af::span);

            BBatch = af::moddims(BBatch, batchSize, 2 * k, m);

            XBatch = X(BBatch, af::span);

            XBatchAdj = af::moddims(XBatch, batchSize, 2 * k * m, d);

            af::array XIndex = af::seq(i, maxBatchIdx);
            XSubset = X(XIndex, af::span);

            XSubsetReshaped = moddims(XSubset, batchSize, 1, d); // Insert new dim

            YBatch = XBatchAdj - XSubsetReshaped;

            // sqrt(sum(sq(...)))

            af::array distancesIndex = af::seq(i, maxBatchIdx);
//            distances(distancesIndex, af::span) =
            af::sqrt(af::sum(YBatch * YBatch, 1));

            printf("%d\n", i);
        }

        return distances;
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


    template<typename T>
    matx::tensor_t<T, 2> inline
    findDistancesMatX(matx::tensor_t<T, 2> &X_t, matx::tensor_t<int, 2> &A_t, matx::tensor_t<int, 2> &B_t,
                      const float alpha = 1.2, int batchSize = -1, const std::string &distanceMetric = "L2",
                      matx::matxMemorySpace_t memorySpace = matx::MATX_DEVICE_MEMORY) {
        const int k = A_t.Shape()[1] / 2;
        const int m = B_t.Shape()[1];

        const int n = X_t.Shape()[0];
        const int d = X_t.Shape()[1];

        batchSize = (batchSize != -1) ? batchSize : findDistanceBatchSize(alpha, n, d, k, m);

        auto distances_t = matx::make_tensor<T>({n, 2 * k * m}, memorySpace);

        auto processBatches = [&](auto computeOperation) {
            auto AFlat_t = matx::flatten(A_t);

            for (int i = 0; i < n; i += batchSize) {
                int maxBatchIdx = i + batchSize;

                auto XSubset_t_op = matx::slice(X_t, {i, 0}, {maxBatchIdx, matx::matxEnd});
                auto ABatchFlat_t_op = matx::slice(AFlat_t, {i * 2 * k}, {maxBatchIdx * 2 * k});
                auto BBatch_t_op = matx::remap<0>(B_t, ABatchFlat_t_op);
                auto XBatch_t_op = matx::remap<0>(X_t, matx::flatten(BBatch_t_op));
                auto XBatchReshaped_t_op = matx::reshape(XBatch_t_op, {batchSize, 2 * k * m, d});
                auto XSubsetReshaped_t_op = matx::reshape(XSubset_t_op, {batchSize, 1, d});

                computeOperation(XBatchReshaped_t_op, XSubsetReshaped_t_op, distances_t, i, maxBatchIdx);
            }
        };

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            auto normOrder = (distanceMetric == "L1") ? matx::NormOrder::L1 : matx::NormOrder::L2;

            processBatches([&](auto& XBatchReshaped_t_op, auto& XSubsetReshaped_t_op, auto& distances, int i, int maxBatchIdx) {
                auto YBatch_t_op = XBatchReshaped_t_op - matx::repmat(XSubsetReshaped_t_op, {1, 2 * k * m, 1});
                auto YBatch_t_norm_op = matx::vector_norm(YBatch_t_op, {2}, normOrder);
                (matx::slice(distances, {i, 0}, {maxBatchIdx, matx::matxEnd}) = YBatch_t_norm_op).run();
            });
        } else if (distanceMetric == "COSINE") {
            processBatches([&](auto& XBatchReshaped_t_op, auto& XSubsetReshaped_t_op, auto& distances, int i, int maxBatchIdx) {
                auto product_op = XBatchReshaped_t_op * matx::repmat(XSubsetReshaped_t_op, {1, 2 * k * m, 1});
                auto product_sum_op = matx::sum(product_op, {2});
                (matx::slice(distances, {i, 0}, {maxBatchIdx, matx::matxEnd}) = product_sum_op).run();
            });
        } else {
            throw std::runtime_error("Invalid distance metric: " + distanceMetric);
        }

        return distances_t;
    }
}


#endif //SDBSCAN_DISTANCES_H