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
#include "GsDBSCAN_Params.h"

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

    inline torch::Tensor
    getRandomVectorsMatrix(int d, int D, const std::string &distanceMetric = "L2", int fourierEmbedDim = 1024,
                           float sigmaEmbed = 1) {
        if (distanceMetric == "L1" || distanceMetric == "L2") {
            return torch::randn({2 * fourierEmbedDim, D}, torch::TensorOptions().device(torch::kCUDA));
        } else if (distanceMetric == "COSINE") {
            return torch::randn({d, D}, torch::TensorOptions().device(torch::kCUDA));
        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }
    }

    inline std::tuple<torch::Tensor, torch::Tensor>
    projectTorch(torch::Tensor &X, int D, const std::string &distanceMetric = "L2", int fourierEmbedDim = 1024,
                 float sigmaEmbed = 1, opt <torch::Tensor> Y = std::nullopt) {
        int d = X.size(1);
        torch::Tensor projections;

        if (!Y.has_value()) {
            Y = getRandomVectorsMatrix(d, D, distanceMetric, fourierEmbedDim, sigmaEmbed);
        }

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

            projections = torch::matmul(XEmbed.t(), Y.value());

        } else if (distanceMetric == "COSINE") {
            projections = torch::matmul(X, Y.value());

        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }

        return std::make_tuple(std::ref(projections), std::ref(X));
    }


    inline std::tuple<torch::Tensor, torch::Tensor>
    constructABMatricesBatch(torch::Tensor &X, GsDBSCAN::GsDBSCAN_Params &params) {

        int n = X.size(0);
        auto Y = getRandomVectorsMatrix(X.size(1), params.D, params.distanceMetric, params.fourierEmbedDim,
                                        params.sigmaEmbed);

        // Construct A matrix
        torch::Tensor A = torch::empty({n, 2 * params.k},
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        for (int i = 0; i < n; i += params.ABatchSize) {
            auto thisX = X.slice(0, i, std::min(i + params.ABatchSize, n));
            auto [thisProjections, _] = projectTorch(thisX, params.D, params.distanceMetric,
                                                     params.fourierEmbedDim,
                                                     params.sigmaEmbed, Y);

            constructAMatrix(thisProjections, params.k, getSortDescending(params.distanceMetric), A, i);
        }

        // Construct B matrix
        torch::Tensor B = torch::empty({2 * params.D, params.m},
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        for (int j = 0; j < params.D; j += params.BBatchSize) {
            auto thisY = Y.slice(1, j, std::min(j + params.BBatchSize, params.D));

            auto [thisProjections, _] = projectTorch(X, params.D, params.distanceMetric,
                                                     params.fourierEmbedDim,
                                                     params.sigmaEmbed, thisY);

            constructBMatrix(thisProjections, params.m, getSortDescending(params.distanceMetric), B, j);
        }

        return std::make_tuple(std::ref(A), std::ref(B));
    }
}

#endif //SDBSCAN_PROJECTIONS_H
