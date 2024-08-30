//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_PROJECTIONS_H
#define SDBSCAN_PROJECTIONS_H

#include <tuple>
#include <cmath>
#include "../pch.h"

#include "algo_utils.h"
#include "run_utils.h"
#include "enums.h"

namespace GsDBSCAN::projections {

    inline std::tuple<torch::Tensor, torch::Tensor> constructABMatricesTorch(const torch::Tensor &projections, int k, int m, DistanceMetric distanceMetric=DistanceMetric::L2) {
        // Assume projections has shape (n, D)
        int n = projections.size(0);
        int D = projections.size(1);

        torch::Tensor A = torch::empty({n, 2 * k}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        torch::Tensor B = torch::empty({2 * D, m}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        bool sortDescending;

        if (distanceMetric == DistanceMetric::L1 || distanceMetric == DistanceMetric::L2) {
            sortDescending = false;
        } else if (distanceMetric == DistanceMetric::COSINE) {
            sortDescending = true;
        } else {
            throw std::runtime_error("Unknown distanceMetric: " + distanceMetricToString(distanceMetric));
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
}


#endif //SDBSCAN_PROJECTIONS_H
