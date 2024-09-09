//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_PROJECTIONS_H
#define SDBSCAN_PROJECTIONS_H

#include <tuple>
#include <cmath>
#include "../pch.h"
#include <optional>

#include "algo_utils.h"

template<typename T>
using opt = std::optional<T>;

namespace GsDBSCAN::projections {

    inline bool getSortDescending(const std::string &distanceMetric) {
        if (distanceMetric == "L1" || distanceMetric == "L2") {
            return false;
        } else if (distanceMetric == "COSINE") {
            return true;
        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }
    }

    inline torch::Tensor
    constructAMatrix(const torch::Tensor &projections, int k, bool sortDescending = false,
                     opt <torch::Tensor> A = std::nullopt, int startIdx = 0) {
        // Assume projections has shape (n, D)
        int n = projections.size(0);
        int D = projections.size(1);

        if (!A.has_value()) {
            A = torch::empty({n, 2 * k}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        }

        auto dataToRandomIdxSorted = projections.argsort(1, sortDescending).toType(torch::kInt32);

        auto AColCloseSlice = torch::indexing::Slice(0, k);
        auto AColFarSlice = torch::indexing::Slice(k, 2 * k);
        auto ARowSlice = torch::indexing::Slice(startIdx, startIdx + n);

        auto closeProjectionsIdx = 2 * dataToRandomIdxSorted.slice(1, 0, k);
        auto farProjectionsIdx = 2 * dataToRandomIdxSorted.slice(1, D - k, D) + 1;
//
//        std::cout << "ARowSlice size: " << ARowSlice.sizes() << std::endl;
//        std::cout << "AColCloseSlice size: " << AColCloseSlice.sizes() << std::endl;
//        std::cout << "closeProjectionsIdx size: " << closeProjectionsIdx.sizes() << std::endl;

        A->index_put_({ARowSlice, AColCloseSlice}, closeProjectionsIdx); // Closest
        A->index_put_({ARowSlice, AColFarSlice}, farProjectionsIdx); // Furthest

        return *A;
    }

    inline torch::Tensor
    constructBMatrix(const torch::Tensor &projections, int m, bool sortDescending = false,
                     opt <torch::Tensor> B = std::nullopt, int startIdx = 0) {
        // Assume projections has shape (n, D)
        int n = projections.size(0);
        int D = projections.size(1);

        if (!B.has_value()) {
            B = torch::empty({2 * D, m}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        }

        auto randomToDataIdxSorted = projections.argsort(0, sortDescending).toType(torch::kInt32);

        auto BRowEvenIdx = torch::arange(2 * startIdx + 0, 2 * (startIdx + D), 2);
        auto BRowOddIdx = BRowEvenIdx + 1;

        auto closeProjectionsIdx = randomToDataIdxSorted.slice(0, 0, m).transpose(0, 1);
        auto farProjectionsIdx = randomToDataIdxSorted.slice(0, n - m, n).transpose(0, 1);

        B->index_put_({BRowEvenIdx, torch::indexing::Ellipsis}, closeProjectionsIdx); // Close -> close
        B->index_put_({BRowOddIdx, torch::indexing::Ellipsis}, farProjectionsIdx); // Far -> far

        return *B;
    }

    inline std::tuple<torch::Tensor, torch::Tensor>
    constructABMatricesTorch(const torch::Tensor &projections, int k, int m, const std::string &distanceMetric = "L2") {
        bool sortDescending = getSortDescending(distanceMetric);

        auto A = constructAMatrix(projections, k, sortDescending);
        auto B = constructBMatrix(projections, m, sortDescending);

        return std::tie(A, B);
    }

    inline torch::Tensor normaliseDatasetTorch(torch::Tensor &X) {
        auto rowNorms = torch::sqrt(torch::sum(X * X, 1));
        return X / rowNorms.unsqueeze(1);
    }

    inline std::tuple<torch::Tensor, torch::Tensor>
    normaliseAndProjectTorch(torch::Tensor &X, int D, bool needToNormalize = true,
                             const std::string &distanceMetric = "L2", int fourierEmbedDim = 1024,
                             float sigmaEmbed = 1) {
        int d = X.size(1);
        torch::Tensor projections;

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            torch::Tensor W;
            float std = 1 / sigmaEmbed;

            if (distanceMetric == "L1") {
                auto uniform = torch::rand({fourierEmbedDim, d}, torch::TensorOptions().device(X.device()));
                W = ((1 / 2) * (std * std)) * torch::tan(M_PI * (uniform - 0.5)); // Cauchy ~ Laplace
            } else { // L2
                W = std * torch::randn({fourierEmbedDim, d}, torch::TensorOptions().device(X.device())); // Gaussian
            }

            auto WX = torch::matmul(W, X.t()); // Shape (fourierEmbedDim, n)
            auto XEmbed = torch::concat({torch::cos(WX), torch::sin(WX)}, 0); // Shape (2 * fourierEmbedDim, n)

            auto Y = torch::randn({2 * fourierEmbedDim, D}, torch::TensorOptions().device(X.device()));
            projections = torch::matmul(XEmbed.t(), Y);

        } else if (distanceMetric == "COSINE") {
            if (needToNormalize) {
                X = normaliseDatasetTorch(X);
            }

            auto Y = torch::randn({d, D}, torch::TensorOptions().device(X.device()));
            projections = torch::matmul(X, Y);

        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }

        return std::make_tuple(std::ref(projections), std::ref(X));
    }


//    inline std::tuple<torch::Tensor, torch::Tensor>
//    constructABMatricesBatch(torch::Tensor X, int D, bool needToNormalize = true,
//                             const std::string &distanceMetric = "L2", int fourierEmbedDim = 1024,
//                             float sigmaEmbed = 1, int ABatchSize = 500000, int BBatchSize = 128) {
//        int n = X.size(0);
//
//        torch::Tensor A = torch::empty({n, 2 * D}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//
//        for (int i = 0; i < n; i += ABatchSize) {
//            auto thisX = X.slice(0, i, std::min(i + ABatchSize, n));
//            auto [thisProjections, _] = normaliseAndProjectTorch(thisX, D, needToNormalize, distanceMetric,
//                                                                 fourierEmbedDim,
//                                                                 sigmaEmbed);
//
//            auto thisX
//
//        }
//
//        torch::Tensor B = torch::empty({2 * D, n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//    }
}

#endif //SDBSCAN_PROJECTIONS_H
