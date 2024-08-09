//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_PREPROCESSING_H
#define SDBSCAN_PREPROCESSING_H

#include <tuple>

#include <arrayfire.h>

namespace GsDBSCAN {

    /*
* TODO:
*
* This method relies on using Eigen matrices. So we either need to convert af matrices to eigens, or use af arrays instead
*/

///**
// * Performs the pre-checks for the gs dbscan algorithm
// *
// * @param X ArrayFire af::array matrix for the X data points
// * @param D int for number of random vectors to generate
// * @param minPts min number of points as per the DBSCAN algorithm
// * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
// * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
// * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
// */
//void GsDBSCAN::preChecks(af::array &X, int D, int minPts, int k, int m, float eps) {
//    assert(X.dims(1) > 0);
//    assert(X.dims(1) > 0);
//    assert(D > 0);
//    assert(D >= k);
//    assert(m >= minPts);
//}
//
///**
// * Performs random projections between the X dataset and the random vector
// *
// * @param X af::array matrix for the X data points
// * @param D int for number of random vectors to generate
// * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
// * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
// * @return af::array matrix for the random projections
// */
//af::array GsDBSCAN::randomProjections(af::array &X, boost::dynamic_bitset<> bitHD3, int D, int k, int m, string distanceMetric, float sigma, int seed, int fhtDim, int nRotate) {
//    // TODO implement me!
//    int n = X.dims(0);
//    int d = X.dims(1);
//
//    Matrix<float, -1, -1> matrixR;
//
//    int iFourierEmbed_D = d / 2;
//
//    // TODO I don't really know what these lines do - just copying and pasting from Ninh's work.
//    if (distanceMetric == "L1") {
//        matrixR = cauchyGenerator(iFourierEmbed_D, d, 0, 1.0 / sigma, seed);
//    } else if (distanceMetric == "L2") {
//        matrixR = cauchyGenerator(iFourierEmbed_D, d, 0, 1.0 / sigma, seed);
//    }
//
//    MatrixXf matrixFHT = MatrixXf::Zero(D, n);
//
//    int log2Project = log2(fhtDim);
//    bitHD3Generator(fhtDim * nRotate, bitHD3, seed);
//
//    Matrix<int, -1, -1> matrixTopK = MatrixXi::Zero(2 * k, n);
//
//    /**
//Parallel for each the point Xi: (1) Compute and store dot product, and (2) Extract top-k close/far random vectors
//**/
//#pragma omp parallel for
//    for (int i = 0; i < sDbscan::n_points; ++i)
//    {
//        /**
//        Random embedding
//        TODO: create buildKernelFeatures and random projection as a new function since sDbscan-1NN also use it
//        **/
//
//        // TODO what is the diff between ker_n_features and n_features?
//
//        VectorXf vecX = sDbscan::matrix_X.col(i);
//        VectorXf vecEmbed = VectorXf::Zero(sDbscan::ker_n_features); // sDbscan::ker_n_features >= D
//
//        // NOTE: must ensure ker_n_features = n_features on Cosine
//        if (sDbscan::distance == "Cosine")
//            vecEmbed.segment(0, sDbscan::n_features) = vecX;
//        else if ((sDbscan::distance == "L1") || (sDbscan::distance == "L2"))
//        {
//            VectorXf vecProject = sDbscan::matrix_R * vecX;
//            vecEmbed.segment(0, iFourierEmbed_D) = vecProject.array().cos();
//            vecEmbed.segment(iFourierEmbed_D, iFourierEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
//        }
//        else if (sDbscan::distance == "Chi2")
//            embedChi2(vecX, vecEmbed, sDbscan::ker_n_features, sDbscan::n_features, sDbscan::ker_intervalSampling);
//        else if (sDbscan::distance == "JS")
//            embedJS(vecX, vecEmbed, sDbscan::ker_n_features, sDbscan::n_features, sDbscan::ker_intervalSampling);
//
//        /**
//        Random projection
//        **/
//
//        VectorXf vecRotation = VectorXf::Zero(sDbscan::fhtDim); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
//        vecRotation.segment(0, sDbscan::ker_n_features) = vecEmbed;
//
//        for (int r = 0; r < sDbscan::n_rotate; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < sDbscan::fhtDim; ++d)
//            {
//                vecRotation(d) *= (2 * (int)sDbscan::bitHD3[r * sDbscan::fhtDim + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecRotation.data(), log2Project);
//        }
//
//        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
//        MATRIX_FHT.col(i) = vecRotation.segment(0, sDbscan::n_proj); // only get up to #n_proj
//
//        /**
//        Extract top-k closes and furtherest random vectors
//        **/
//
//        Min_PQ_Pair minCloseTopK;
//        Min_PQ_Pair minFarTopK;
//
//        for (int d = 0; d < sDbscan::n_proj; ++d) {
//            float fValue = vecRotation(d); // take the value up to n_proj
//
//            // (1) Close: Using priority queue to find top-k closest vectors for each point
//            if ((int)minCloseTopK.size() < sDbscan::topK)
//                minCloseTopK.emplace(d, fValue);
//            else
//            {
//                if (fValue > minCloseTopK.top().m_fValue)
//                {
//                    minCloseTopK.pop();
//                    minCloseTopK.emplace(d, fValue);
//                }
//            }
//
//            // (2) Far: Using priority queue to find top-k furthest vectors
//            if ((int)minFarTopK.size() < sDbscan::topK)
//                minFarTopK.emplace(d, -fValue);
//            else
//            {
//                if (-fValue > minFarTopK.top().m_fValue)
//                {
//                    minFarTopK.pop();
//                    minFarTopK.emplace(d, -fValue);
//                }
//            }
//        }
//
//        for (int k = sDbscan::topK - 1; k >= 0; --k)
//        {
//            sDbscan::matrix_topK(k, i) = minCloseTopK.top().m_iIndex;
//            minCloseTopK.pop();
//
//            sDbscan::matrix_topK(k + sDbscan::topK, i) = minFarTopK.top().m_iIndex;
//            minFarTopK.pop();
//        }
//
//    }
//}

    /**
     * Constructs the A and B matrices as per the GS-DBSCAN algorithm
     *
     * A stores the indices of the closest and furthest random vectors per data point.
     *
     * @param D projections, projections between query vectors and the random vectors, has shape (N, D)
     * @param k k parameter as per the DBSCAN algorithm
     * @param m m parameter as per the DBSCAN algorithm
     */
    std::tuple<af::array, af::array> constructABMatrices(const af::array &projections, int k, int m) {
        // Assume projections has shape (n, D)
        int n = projections.dims(0);
        int D = projections.dims(1);

        af::array A(n, 2 * k);
        af::array B(2 * D, m);

        af::array dataToRandomIdxSorted = af::constant(-1, projections.dims(), af::dtype::u16);
        af::array randomToDataIdxSorted = af::constant(-1, projections.dims(), af::dtype::u16);
        af::array sortedValsTemp;

        af::sort(sortedValsTemp, sortedValsTemp, projections, 1);

        A(af::span, af::seq(0, k - 1)) = 2 * dataToRandomIdxSorted(af::span, af::seq(0, k - 1));
        A(af::span, af::seq(k, af::end)) =
                2 * dataToRandomIdxSorted(af::span, af::seq(dataToRandomIdxSorted.dims(0) - k, af::end));

        af::array BEvenIdx = af::seq(0, 2 * D - 1, 2);
        af::array BOddIdx = BEvenIdx + 1;

        B(BEvenIdx, af::span) = randomToDataIdxSorted(af::seq(0, m - 1), af::span);
        B(BOddIdx, af::span) = randomToDataIdxSorted(af::seq(randomToDataIdxSorted.dims(0) - m, af::end), af::span);

        return std::make_tuple(A, B);
    }
}

#endif //SDBSCAN_PREPROCESSING_H
