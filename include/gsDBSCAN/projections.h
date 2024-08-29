//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_PROJECTIONS_H
#define SDBSCAN_PROJECTIONS_H

#include <tuple>
#include <cmath>
#include "../pch.h"

#include <arrayfire.h>
#include "algo_utils.h"

namespace GsDBSCAN::projections {
    /**
     * Constructs the A and B matrices as per the GS-DBSCAN algorithm
     *
     * A stores the indices of the closest and furthest random vectors per data point.
     *
     * @param D projections, projections between query vectors and the random vectors, has shape (N, D)
     * @param k k parameter as per the DBSCAN algorithm
     * @param m m parameter as per the DBSCAN algorithm
     * @param distanceMetric distance metric to use, one of "L1", "L2", or "COSINE"
     */
    inline std::tuple<af::array, af::array> constructABMatricesAF(const af::array &projections, int k, int m, const std::string &distanceMetric="L2") {
        // Assume projections has shape (n, D)
        int n = projections.dims(0);
        int D = projections.dims(1);

        af::array A(n, 2 * k, af::dtype::u32);
        af::array B(2 * D, m, af::dtype::u32);

        af::array sortedValsTemp, dataToRandomIdxSorted;

        af::sort(sortedValsTemp, dataToRandomIdxSorted, projections, 1); // Sort across the rows

        af::seq BEvenIdx = af::seq(0, 2 * D - 1, 2); // Down the rows
        af::seq BOddIdx = af::seq(1, 2 * D - 1, 2);

        af::array sortedValsTemp2, randomToDataIdxSorted;
        af::sort(sortedValsTemp2, randomToDataIdxSorted, projections, 0); // Sort down the cols

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            // For L1 and L2 difference, we take a low projection rating to be close
            A(af::span, af::seq(0, k - 1)) = 2 * dataToRandomIdxSorted(af::span, af::seq(0, k - 1));
            A(af::span, af::seq(k, af::end)) = 2 * dataToRandomIdxSorted(af::span, af::seq(D - k, af::end)) + 1;

            B(BEvenIdx, af::span) = af::transpose(randomToDataIdxSorted(af::seq(0, m - 1), af::span));
            B(BOddIdx, af::span) = af::transpose(randomToDataIdxSorted(af::seq(n - m, af::end), af::span));
        } else if (distanceMetric == "COSINE") {
            // For COSINE similarity. We take +1 as close, 0 as orthogonal, and -1 as far
            A(af::span, af::seq(0, k - 1)) = 2 * dataToRandomIdxSorted(af::span, af::seq(D - k, af::end));
            A(af::span, af::seq(k, af::end)) = 2 * dataToRandomIdxSorted(af::span, af::seq(0, k - 1)) + 1;

            B(BEvenIdx, af::span) = af::transpose(randomToDataIdxSorted(af::seq(n - m, af::end), af::span)); // close -> close
            B(BOddIdx, af::span) = af::transpose(randomToDataIdxSorted(af::seq(0, m - 1), af::span)); // far -> far
        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }

        return std::tie(A, B);
    }

    inline std::tuple<torch::Tensor, torch::Tensor> constructABMatricesTorch(const torch::Tensor &projections, int k, int m, const std::string &distanceMetric="L2") {
        // Assume projections has shape (n, D)
        int n = projections.size(0);
        int D = projections.size(1);

        torch::Tensor A = torch::empty({n, 2 * k}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        torch::Tensor B = torch::empty({2 * D, m}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        bool sortDescending;

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            sortDescending = false;
        } else if (distanceMetric == "COSINE") {
            sortDescending = true;
        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }

        auto dataToRandomIdxSorted = projections.argsort(1, sortDescending).toType(torch::kInt32);
        auto randomToDataIdxSorted = projections.argsort(0, sortDescending).toType(torch::kInt32);

        auto BEvenIdx = torch::arange(0, 2 * D, 2);
        auto BOddIdx = BEvenIdx + 1;

        A.slice(1, 0, k) = 2 * dataToRandomIdxSorted.slice(1, 0, k); // Closest
        A.slice(1, k, 2 * k) = 2 * dataToRandomIdxSorted.slice(1, D - k, D) + 1; // Furthest

        B.index_put_({BEvenIdx, torch::indexing::Ellipsis}, randomToDataIdxSorted.slice(0, 0, m).transpose(0, 1)); // Close -> close
        B.index_put_({BOddIdx, torch::indexing::Ellipsis}, randomToDataIdxSorted.slice(0, n - m, n).transpose(0, 1)); // Far -> far

        return std::tie(A, B);
    }

    inline torch::Tensor normaliseDatasetTorch(torch::Tensor &X) {
        auto rowNorms = torch::sqrt(torch::sum(X * X, 1));
        return X / rowNorms.unsqueeze(1);
    }

    inline torch::Tensor performProjectionsTorch(torch::Tensor &X, int D) {
        int d = X.size(1);
        auto Y = torch::randn({d, D}, torch::TensorOptions().device(X.device()));
        return torch::matmul(X, Y);
    }

    inline std::tuple<torch::Tensor, torch::Tensor> normaliseAndProjectTorch(torch::Tensor &X, int D, bool needToNormalize=true) {
        if (needToNormalize) {
            X = normaliseDatasetTorch(X);
        }
        return std::make_tuple(performProjectionsTorch(X, D), std::ref(X));
    }

    inline af::array performProjectionsAF(af::array X, int D) {
        int d = X.dims(1);
        auto Y = af::randn(d, D); // TODO should these be signed?
//        return (1 / std::sqrt(D)) * af::matmul(X, Y);
        return af::matmul(X, Y); // No normalisation (not needed), since the 1 / sqrt(D) term is a constant
    }

    inline af::array normaliseDatasetAF(af::array X) {
        // Normalise along the rows
        auto rowNorms = af::sqrt(af::sum(X * X, 1));
        return X / af::tile(rowNorms, 1, X.dims(1));
    }

    template <typename Op>
    inline auto performProjectionsMatX(Op X, int D) {
        int d = matx::Shape(X)[1];
        auto Y = matx::random<float>({d, D}, matx::NORMAL);
        auto res = matx::make_tensor<float>({matx::Shape(X)[0], D});
        (res = matx::matmul(X, Y)).run();
        return res;
    }

//    template <typename T>
//    inline matx::tensor_t<T, 2> normaliseDatasetMatX(matx::tensor_t<T, 2> X) {
//        auto rowNorms_op = matx::vector_norm(X, {1}, matx::NormOrder::L2);
//        auto rowNorms_op_2 = matx::clone<2>(rowNorms_op, {X.Shape()[0], X.Shape()[1]});
//        auto res = matx::make_tensor<float>({X.Shape()[0], X.Shape()[1]});
//
//        (res = X / rowNorms_op_2).run();
//    }

    template <typename T>
    inline std::tuple<af::array, matx::tensor_t<T, 2>> normaliseAndProject(T* X, int n, int d, int D, const std::string &projectionsMethod="AF", bool needToNormalize=true) {
        // Assume X is a col-major array, can assume it's on the host - arrayfire accounts for this.

        auto X_af = af::array(n, d, X);

        std::cout<< "NN " << needToNormalize << std::endl;

        if (needToNormalize) {
            X_af = normaliseDatasetAF(X_af); // MatX normalise is broken, so we'll do it in AF
        }
        X_af.eval();

        auto X_t = algo_utils::afMatToMatXTensor<float, float>(X_af, matx::MATX_DEVICE_MEMORY);

        af::array projections;

        if (projectionsMethod == "AF") {
            projections = performProjectionsAF(X_af, D);
            projections.eval();
            auto projections_t = algo_utils::afMatToMatXTensor<T, T>(projections, matx::MATX_DEVICE_MEMORY);
        } else if (projectionsMethod == "MATX") {
            auto projections_t = performProjectionsMatX(X_t, D);
            projections = algo_utils::matXTensorToAfMat<T, T>(projections_t);
        } else {
            // TODO can add FHT as a method
            throw std::runtime_error("Unknown projectionsMethod: '" + projectionsMethod + "'");
        }

        return std::tie(projections, X_t);
    }
}


#endif //SDBSCAN_PROJECTIONS_H
