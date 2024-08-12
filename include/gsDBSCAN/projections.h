//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_PROJECTIONS_H
#define SDBSCAN_PROJECTIONS_H

#include <tuple>

#include <arrayfire.h>
#include "../Header.h"

namespace GsDBSCAN {

    /**
     * Adds an element to a limited size (pair) priority queue
     *
     * Only adds if the element as sufficient priority
     *
     * @param queue priority queue to add to
     * @param index index of the element to be added
     * @param value value of the element to be added
     * @param maxSize max size of the PQ
     */
    inline void addToLimitedPairPQ(Min_PQ_Pair queue, int index, float value, int maxSize) {
        if ((int) queue.size() < maxSize)
            queue.emplace(index, value);
        else {
            if (value > queue.top().m_fValue) {
                queue.pop();
                queue.emplace(index, value);
            }
        }
    }

    /**
     * Generate Cauchy distribution C(x0, gamma)
     *
     * @param p_iNumRows
     * @param p_iNumCols
     * @param x0
     * @param gamma
     * @param random_seed
     * @return a matrix of size numRow x numCol
     */
    inline MatrixXf cauchyGenerator(int p_iNumRows, int p_iNumCols, float x0, float gamma, int random_seed) {
        MatrixXf MATRIX_C = MatrixXf::Zero(p_iNumRows, p_iNumCols);

        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        if (random_seed >= 0)
            seed = random_seed;

        default_random_engine generator(seed);

        cauchy_distribution<float> cauchyDist(x0, gamma); // {x0 /* a */, 𝛾 /* b */}

        // Always iterate col first, then row later due to the col-wise storage
        for (int c = 0; c < p_iNumCols; ++c)
            for (int r = 0; r < p_iNumRows; ++r)
                MATRIX_C(r, c) = cauchyDist(generator);

        return MATRIX_C;
    }


    /**
     * Generate random bit for FHT
     *
     * @param p_iNumBit
     * @param bitHD
     * @param random_seed
     * return bitHD that contains fhtDim * n_rotate (default of n_rotate = 3)
     */
    inline void bitHD3Generator(int p_iNumBit, boost::dynamic_bitset<> & bitHD, int random_seed) {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        if (random_seed >= 0)
            seed = random_seed;

        default_random_engine generator(seed);
        uniform_int_distribution<uint32_t> unifDist(0, 1);

        bitHD = boost::dynamic_bitset<> (p_iNumBit);

        // Loop col first since we use col-wise
        for (int d = 0; d < p_iNumBit; ++d)
        {
            bitHD[d] = unifDist(generator) & 1;
        }

    }

    /**
     * Handles the choice of the distance metrix for doing projections, changes the embedding vector
     *
     * // TODO speak to Ninh about the below
     *
     * @param distanceMetric distance metric to use
     * @param nFeatures the number of features // TODO what is this?
     * @param randomMatrix matrix containing the random vectors. Shoul dhave shape (d, D). Vecs stored along the COLS
     * @param iFourierEmbed_D// TODO what is this?
     * @param vecX vector to project against. Should be a col vector.
     * @param vecEmbed // TODO what is this?
     */
    inline VectorXf handleDistanceMetric(const string &distanceMetric, int nFeatures, const Matrix<float, -1, -1> &randomMatrix,
                              int iFourierEmbed_D, const VectorXf &vecX, int kerNFeatures) {
        // NOTE: must ensure kerNFeatures = nFeatures on Cosine // TODO what to do with this informatjon?

        VectorXf vecEmbed = VectorXf::Zero(kerNFeatures); // sDbscan::ker_n_features >= D

        if (distanceMetric == "Cosine") {
            vecEmbed.segment(0, nFeatures) = vecX;
        } else if ((distanceMetric == "L1") || (distanceMetric == "L2")) {
            VectorXf vecProject = randomMatrix * vecX; // TODO adjust this for change of randomMatrix shape.
            vecEmbed.segment(0, iFourierEmbed_D) = vecProject.array().cos();
            vecEmbed.segment(iFourierEmbed_D,
                             iFourierEmbed_D) = vecProject.array().sin();// start from iEmbbed, copy iEmbed elements
        } else {
            throw runtime_error("Unknown distance metric: " + distanceMetric);
        }
        // TODO add back support for Chi2 and JS, I've removed them to make things a bit cleaner

        return vecEmbed;
    }

    /**
     * Constructs the A matrix for the GsDBSCAN algorithm
     *
     * @param X dataset matrix. Has shape (n, d). Has vectors stored along the ROWS
     * @param randomMatrix matrix containing the random vectors. Has shape (d, D). Vectors are stored along the COLS
     * @param bitHD3 // TODO what is this?
     * @param distanceMetric Distance metric to use when finding distances between random and dataset vecs
     * @param k how many closest/furthest random vecs to choose for each dataset vec
     * @param fhtDim dimension of the FHT // TODO what is the purpose of this?
     * @param nRotate // TODO what is this?
     * @param kerNFeatures // TODO what is this?
     * @param nFeatures // TODO what is this?
     * @param iFourierEmbed_D // TODO what is this?
     * @return a tuple containing:
     *  The constructed A matrix. Has shape (n, 2*k). The ith row contains the entries of the index to access for the B matrix when doing random projections
     *  The FHT matrix to use for the next steps
     */
    inline std::tuple<Matrix<int, -1, -1>, Matrix<float, -1, -1>> constructAMatrixEigen(MatrixXf &X,
                                              Matrix<float, -1, -1, RowMajor> &randomMatrix,
                                              boost::dynamic_bitset<> &bitHD3,
                                              string distanceMetric,
                                              int k, int fhtDim,
                                    int nRotate, int kerNFeatures, int nFeatures, int iFourierEmbed_D) {

        int n = X.rows();
        int D = randomMatrix.cols();

        Matrix<int, -1, -1, RowMajor> A = MatrixXi::Constant(n, 2*k, -1);
        Matrix<float, -1, -1, RowMajor> matrixFHT = MatrixXf::Zero(n, D);

        int log2Project = log2(fhtDim);

#pragma omp parallel for
        for (int dataSetIdx = 0; dataSetIdx < n; ++dataSetIdx) {
            VectorXf vecX = X.row(dataSetIdx); // TODO change me to row

            auto vecEmbed = handleDistanceMetric(distanceMetric, nFeatures, randomMatrix, iFourierEmbed_D, vecX, kerNFeatures);

            VectorXf vecRotation = VectorXf::Zero(fhtDim); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
            vecRotation.segment(0, kerNFeatures) = vecEmbed;

            for (int r = 0; r < nRotate; ++r) {
                for (int d = 0; d < fhtDim; ++d){
                    vecRotation(d) *= (2 * (int)bitHD3[r * fhtDim + d] - 1);
                }
                fht_float(vecRotation.data(), log2Project);
            }

            matrixFHT.col(dataSetIdx) = vecRotation.segment(0, D); // only get up to #n_proj

            Min_PQ_Pair minCloseTopK, minFarTopK;

            for (int projIdx = 0; projIdx < D; ++projIdx) {
                float fValue = vecRotation(projIdx); // take the value up to n_proj

                addToLimitedPairPQ(minCloseTopK, projIdx, fValue, k);
                addToLimitedPairPQ(minFarTopK, projIdx, -fValue, k);
            }

            for (int pqIdx = k - 1; pqIdx >= 0; --pqIdx) {
                A(dataSetIdx, pqIdx) = 2 * minCloseTopK.top().m_iIndex;
                minCloseTopK.pop();

                A(dataSetIdx, k + pqIdx) = 2 * minCloseTopK.top().m_iIndex + 1;
                minFarTopK.pop();
            }
        }

        return std::tie(A, matrixFHT);
    }

    /**
     * Constructs the B matrix for the GsDBSCAN algorithm
     *
     * @param m number of closest/furthest dataset vecs to save for each random vec
     * @param D the number of random vectors
     * @param n the number of dataset vectors
     * @param matrixFHT the FHT projections mat
     * @return the created B matrix. Has shape (2*D, m).
     *  Essentially, the 2*i row contains the m closest dataset vecs to the ith random vec,
     *  and the 2*i+1 row containing the m furthest dataset vecs to the ith random vec
     */
    inline Matrix<int, -1, -1> constructBMatrixEigen(int m, int D, int n, MatrixXf &matrixFHT) {

        Matrix<int, -1, -1, RowMajor> B = MatrixXi::Constant(2 * D, m, -1);

#pragma omp parallel for
        for (int projIdx = 0; projIdx < D; projIdx++) {
            Min_PQ_Pair randomToDataClosePQ;
            Min_PQ_Pair randomToDataFarPQ;

            VectorXf vecProject = matrixFHT.row(projIdx); // it must be row since D x N

            for (int j = 0; j < n; ++j) {
                float fValue = vecProject(j);
                addToLimitedPairPQ(randomToDataClosePQ, j, fValue, m);

                addToLimitedPairPQ(randomToDataFarPQ, j, -fValue, m);
            }

            for (int pqIdx = m - 1; pqIdx >= 0; pqIdx--) {
                B(2*projIdx, pqIdx) = randomToDataClosePQ.top().m_iIndex; // Closest data vecs to random vecs
                randomToDataClosePQ.pop();

                B(2*projIdx + 1, pqIdx) = randomToDataFarPQ.top().m_iIndex; // Furthest data vecs to random vecs
                randomToDataFarPQ.pop();
            }
        }

        return B;
    }


    /**
     * Constructs the A and B matrices for the GsDBSCAN algorithm
     *
     * @param X eigen matrix containing the dataset vectors. Has shape (n, d). Vectors stored along the ROWS
     * @param bitHD3 // TODO what is this?
     * @param D The number of random vectors to generate
     * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
     * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
     * @param distanceMetric the distance metric to use for comparing random to dataset vecs
     * @param sigma // TODO what is this?
     * @param seed random number generation seed
     * @param fhtDim Dimension of the FHT to use for projections
     * @param nRotate // TODO what is this?
     * @param kerNFeatures // TODO what is this?
     * @param nFeatures // TODO what is this?
     * @return af::array matrix for the random projections
     */
    inline std::tuple<MatrixXi, MatrixXi> constructABMatricesEigen(MatrixXf &X, boost::dynamic_bitset<> bitHD3,
                                      int D, int k, int m,
                                      string distanceMetric, float sigma, int seed, int fhtDim,
                                      int nRotate, int kerNFeatures, int nFeatures) {
        int n = X.rows();
        int d = X.cols();

        Matrix<float, -1, -1, RowMajor> randomMatrix; // Ran

        int iFourierEmbed_D = kerNFeatures / 2; // TODO what is this?

        // TODO I don't really know what these lines do - just copying and pasting from Ninh's work.
        if (distanceMetric == "L1") {
            randomMatrix = cauchyGenerator(d, iFourierEmbed_D, 0, 1.0 / sigma, seed);
        } else if (distanceMetric == "L2") {
            randomMatrix = cauchyGenerator(d, iFourierEmbed_D, 0, 1.0 / sigma, seed);
        }

        bitHD3Generator(fhtDim * nRotate, bitHD3, seed);

        auto [A, matrixFHT] = constructAMatrixEigen(X, randomMatrix, bitHD3, distanceMetric, k, fhtDim, nRotate, kerNFeatures, nFeatures, iFourierEmbed_D);

        auto B = constructBMatrixEigen(m, D, n, matrixFHT);

        return std::tie(A, B);
    }


    /**
     * Constructs the A and B matrices as per the GS-DBSCAN algorithm
     *
     * A stores the indices of the closest and furthest random vectors per data point.
     *
     * @param D projections, projections between query vectors and the random vectors, has shape (N, D)
     * @param k k parameter as per the DBSCAN algorithm
     * @param m m parameter as per the DBSCAN algorithm
     */
    inline std::tuple<af::array, af::array> constructABMatrices(const af::array &projections, int k, int m) {
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

#endif //SDBSCAN_PROJECTIONS_H
