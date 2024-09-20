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
    constructABMatrices(const torch::Tensor &projections, int k, int m, const std::string &distanceMetric = "L2") {
        bool sortDescending = getSortDescending(distanceMetric);

        auto A = constructAMatrix(projections, k, sortDescending);
        auto B = constructBMatrix(projections, m, sortDescending);

        return std::tie(A, B);
    }

    inline torch::Tensor normaliseDataset(torch::Tensor &X, GsDBSCAN_Params params) {
        if (params.useBatchNorm) {
            int normBatchSize = params.normBatchSize;
            for (int i = 0; i < X.size(0); i += normBatchSize) {
                auto thisX = X.slice(0, i, std::min(i + normBatchSize, (int) X.size(0)));
                auto rowNorms = torch::linalg_vector_norm(thisX, 2, 1);
                X.index_put_({torch::indexing::Slice(i, i + thisX.size(0)), torch::indexing::Ellipsis},
                             thisX / rowNorms.unsqueeze(1));
            }
            return X;
        } else {
            auto rowNorms = torch::linalg_vector_norm(X, 2, 1);
            return X / rowNorms.unsqueeze(1);
        }
    }

    inline torch::Tensor
    getRandomVectorsMatrix(int d, int D, const std::string &distanceMetric = "L2", int fourierEmbedDim = 1024,
                           std::optional<torch::Dtype> castToType = std::nullopt) {

        torch::Tensor Y;

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            Y = torch::randn({2 * fourierEmbedDim, D}, torch::TensorOptions().device(torch::kCUDA));
        } else if (distanceMetric == "COSINE") {
            Y = torch::randn({d, D}, torch::TensorOptions().device(torch::kCUDA));
        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }

        if (castToType.has_value()) {
            Y = Y.to(castToType.value());
        }

        return Y;
    }

    inline torch::Tensor
    projectDataset(torch::Tensor &X, int D, const std::string &distanceMetric = "L2", int fourierEmbedDim = 1024,
                   float sigmaEmbed = 1, opt <torch::Tensor> Y = std::nullopt, bool verbose = false) {
        int d = X.size(1);
        torch::Tensor projections;

//        // TODO this will break the code if X is too big (FIXME)
//        auto X_f32 = X.to(torch::kFloat32); // Ensure float32, so X can be used with random gen'd Tensors

        if (!Y.has_value()) {
            Y = getRandomVectorsMatrix(d, D, distanceMetric, fourierEmbedDim, X.scalar_type());
        }

        if (distanceMetric == "L1" || distanceMetric == "L2") {
            torch::Tensor W;
            float std = 1 / sigmaEmbed;

            if (distanceMetric == "L1") {
                auto uniform = torch::rand({fourierEmbedDim, d}, torch::TensorOptions().device(X.device()));
                W = ((1 / 2) * (std * std)) * torch::tan(M_PI * (uniform - 0.5)); // Cauchy
            } else { // L2
                W = std * torch::randn({fourierEmbedDim, d}, torch::TensorOptions().device(X.device())); // Gaussian
            }

            W = W.to(X.scalar_type());

            auto WX = torch::matmul(W, X.t()); // Shape (fourierEmbedDim, n)
            auto XEmbed = torch::concat({torch::cos(WX), torch::sin(WX)}, 0); // Shape (2 * fourierEmbedDim, n)

            projections = torch::matmul(XEmbed.t(), Y.value());

        } else if (distanceMetric == "COSINE") {
            projections = torch::matmul(X, Y.value());

        } else {
            throw std::runtime_error("Unknown distanceMetric: '" + distanceMetric + "'");
        }

        return projections;
    }


    inline std::tuple<torch::Tensor, torch::Tensor>
    constructABMatricesBatch(torch::Tensor &X, GsDBSCAN::GsDBSCAN_Params &params) {

        int n = X.size(0);
        auto Y = getRandomVectorsMatrix(X.size(1), params.D, params.distanceMetric, params.fourierEmbedDim, X.scalar_type());

        bool sortDescending = getSortDescending(params.distanceMetric);

        if (params.verbose) std::cout << "Creating A matrix" << std::endl;
        // Construct A matrix
        torch::Tensor A = torch::empty({n, 2 * params.k},
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        for (int i = 0; i < n; i += params.ABatchSize) {
            auto thisX = X.slice(0, i, std::min(i + params.ABatchSize, n));
            auto thisProjections = projectDataset(thisX, params.D, params.distanceMetric,
                                                  params.fourierEmbedDim,
                                                  params.sigmaEmbed, Y, params.verbose);

            constructAMatrix(thisProjections, params.k, sortDescending, A, i);
        }

        if (params.verbose) std::cout << "A created" << std::endl;

        if (params.verbose) std::cout << "Creating B matrix" << std::endl;

        // Construct B matrix
        torch::Tensor B = torch::empty({2 * params.D, params.m},
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        for (int j = 0; j < params.D; j += params.BBatchSize) {
            auto thisY = Y.slice(1, j, std::min(j + params.BBatchSize, params.D));

            // TODO should this be projecting across the entire dataset - why don't we adjust for the batch size? - giving params.D here, not params.BBatchSize
            auto thisProjections = projectDataset(X, params.D, params.distanceMetric,
                                                  params.fourierEmbedDim,
                                                  params.sigmaEmbed, thisY);

            constructBMatrix(thisProjections, params.m, sortDescending, B, j);
        }

        if (params.verbose) std::cout << "B created" << std::endl;

        return std::make_tuple(std::ref(A), std::ref(B));
    }
}

#endif //SDBSCAN_PROJECTIONS_H
