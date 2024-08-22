//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_PROJECTIONS_H
#define SDBSCAN_PROJECTIONS_H

#include <tuple>

#include <arrayfire.h>
#include "../Header.h"


namespace GsDBSCAN::projections {
    /**
     * Constructs the A and B matrices as per the GS-DBSCAN algorithm
     *
     * A stores the indices of the closest and furthest random vectors per data point.
     *
     * @param D projections, projections between query vectors and the random vectors, has shape (N, D)
     * @param k k parameter as per the DBSCAN algorithm
     * @param m m parameter as per the DBSCAN algorithm
     */
    inline std::tuple<af::array, af::array> constructABMatricesAF(const af::array &projections, int k, int m) {
        // Assume projections has shape (n, D)
        int n = projections.dims(0);
        int D = projections.dims(1);

        af::array A(n, 2 * k, af::dtype::u32);
        af::array B(2 * D, m, af::dtype::u32);

        af::array sortedValsTemp, dataToRandomIdxSorted;

        af::sort(sortedValsTemp, dataToRandomIdxSorted, projections, 1); // Sort across the rows

        A(af::span, af::seq(0, k - 1)) = 2 * dataToRandomIdxSorted(af::span, af::seq(0, k - 1));
        A(af::span, af::seq(k, af::end)) = 2 * dataToRandomIdxSorted(af::span, af::seq(D - k, af::end)) + 1;

        af::seq BEvenIdx = af::seq(0, 2 * D - 1, 2); // Down the rows
        af::seq BOddIdx = af::seq(1, 2 * D - 1, 2);

        af::array sortedValsTemp2, randomToDataIdxSorted;
        af::sort(sortedValsTemp2, randomToDataIdxSorted, projections, 0); // Sort down the cols

        B(BEvenIdx, af::span) = af::transpose(randomToDataIdxSorted(af::seq(0, m - 1), af::span));
        B(BOddIdx, af::span) = af::transpose(randomToDataIdxSorted(af::seq(n - m, af::end), af::span));

        return std::make_tuple(A, B);
    }

    inline af::array performProjectionsAF(af::array X, int D) {
        int d = X.dims(1);
        auto Y = af::randn(d, D);
        return af::matmul(X, Y);
    }

    inline af::array normaliseDatasetAF(af::array X) {
        // Normalise along the rows
        auto rowNorms = af::sqrt(af::sum(X * X, 1));
        return X / af::tile(rowNorms, 1, X.dims(1));
    }

//        template <typename Op>
//        inline auto performProjectionsMatX(Op X, int D) {
//            int d = matx::Shape(X)[1];
//            auto Y = matx::random<float>({d, D}, matx::UNIFORM);
//            auto res = matx::make_tensor<float>({matx::Shape(X)[0], D});
//            (res = matx::matmul(X, Y)).run();
//            return res;
//        }
//
        template <typename T>
        inline matx::tensor_t<T, 2> normaliseDatasetMatX(matx::tensor_t<T, 2> X) {
            auto rowNorms_op = matx::vector_norm(X, {1}, matx::NormOrder::L2);
            auto rowNorms_op_2 = matx::clone(rowNorms_op, {1, matx::matxKeepDim});
            auto res = matx::make_tensor<float>({X.Shape()[0], X.Shape()[1]});

            (res = X / matx::repmat(rowNorms_op_2, {1, X.Shape()[1]})).run();

            return res;
        }
}


#endif //SDBSCAN_PROJECTIONS_H
