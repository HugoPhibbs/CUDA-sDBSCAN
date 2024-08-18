//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_DISTANCES_H
#define SDBSCAN_DISTANCES_H

#include <matx.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cstdio>
#include <arrayfire.h>

namespace GsDBSCAN {
    namespace distances {
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
        inline int findDistanceBatchSize(float alpha, int n, int d, int k, int m) {
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
                af::sqrt(af::sum(YBatch*YBatch, 1));

                printf("%d\n", i);
            }

            return distances;
        }



        template<typename T>
        matx::tensor_t<T, 2> inline
        findDistancesMatX(matx::tensor_t<T, 2> &X_t, matx::tensor_t<int32_t, 2> &A_t, matx::tensor_t<int32_t, 2> &B_t,
                          float alpha = 1.2, int batchSize = -1,  std::string distanceMetric="L2",  matx::matxMemorySpace_t memorySpace = matx::MATX_MANAGED_MEMORY) {
            // Handle distance metric


            const int k = A_t.Shape()[1] / 2;
            const int m = B_t.Shape()[1];

            const int n = X_t.Shape()[0];
            const int d = X_t.Shape()[1];

            batchSize = (batchSize != -1) ? batchSize : findDistanceBatchSize(alpha, n, d, k, m);

            auto AFlat_t = matx::flatten(A_t);

            auto distances_t = matx::make_tensor<T>({n, 2 * k * m}, memorySpace);

            for (int i = 0; i < n; i += batchSize) {
                auto start = std::chrono::high_resolution_clock::now(); // TODO remove me!

                int maxBatchIdx = i + batchSize; // Index within X along the ROWS

                auto XSubset_t_op = matx::slice(X_t, {i, 0}, {maxBatchIdx, matx::matxEnd});

                auto ABatchFlat_t_op = matx::slice(AFlat_t, {i * 2 * k}, {maxBatchIdx * 2 * k});

                auto BBatch_t_op = matx::remap<0>(B_t, ABatchFlat_t_op);

                auto XBatch_t_op = matx::remap<0>(X_t, matx::flatten(BBatch_t_op));

                auto XBatchReshaped_t_op = matx::reshape(XBatch_t_op, {batchSize, 2 * k * m, d});

                auto XSubsetReshaped_t_op = matx::reshape(XSubset_t_op, {batchSize, 1, d});

                auto YBatch_t_op = (XBatchReshaped_t_op - matx::repmat(XSubsetReshaped_t_op, {1, 2 * k * m,
                                                                                              1})); // Repmat is a workaround for minusing naively incompatibhle tensor shapes

                if (distanceMetric == "L1") {
                    auto YBatch_t_norm_op = matx::vector_norm(YBatch_t_op, {2}, matx::NormOrder::L1);
                    (matx::slice(distances_t, {i, 0}, {maxBatchIdx, matx::matxEnd}) = YBatch_t_norm_op).run();
                } else if (distanceMetric == "L2") {
                    auto YBatch_t_norm_op = matx::vector_norm(YBatch_t_op, {2}, matx::NormOrder::L2);
                    (matx::slice(distances_t, {i, 0}, {maxBatchIdx, matx::matxEnd}) = YBatch_t_norm_op).run(); // TODO: TBH I don't know the type of YBatch_norm_op, so I'm repeating the call like a noob
                } else if (distanceMetric == "COSINE") {
                    // TODO implement me! - can be a smart way to do this with pre processing of dot products
                }
            }

            return distances_t;
        }

    }
}

#endif //SDBSCAN_DISTANCES_H
