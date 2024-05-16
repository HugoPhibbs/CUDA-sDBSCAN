//
// Created by hugop on 10/05/2024.
//

#include <omp.h>
#include "../../include/preprocessing.h"
#include "../../include/Header.h"
#include "../../include/Utilities.h"


/**
Have to store the projection matrix for parallel processing
- Input: MATRIX_EMBED
**/

MatrixXf constructBMatrix(int DSize, int m, int minPts, const MatrixXf &randomProjections);

MatrixXf constructAMatrix(int DSize, int n, int k);

/**
 * Computes random projections between the dataset and the random vectors
 * 
 * @param D_size number of random vectors to use
 * @param n 
 */
void parallelRandomProjections(int DSize, int n, int k, int m, int minPts) {

    MatrixXf randomProjections = constructAMatrix(DSize, n, k);

    MatrixXf B = constructBMatrix(DSize, m, minPts, randomProjections);

    // TODO return top M and K here

    // TODO can refactor the above to construct the A B matrices instead
}

/**
 * Constructs the A matrix as per the GS-DBSCAN algorithm
 *
 * @param DSize Number of random vectors
 * @param m m parameter as per the sDBSCAN algorithm
 * @param minPts minPts param as per the DBSCAN algorithm
 * @return the A matrix (shape (n, 2*k)). The i'th row of A stores the indices of the 2*k closest and furthest random vectors to the i'th point in X. Closest in first k slots and furthest in the last k slots. Indices are stored in the form 2*j+s to allow for easy indexing of B
 *         the randomProjections matrix containing the projections of the dataset vectors to the random vectors
 */
MatrixXf constructAMatrix(int DSize, int n, int k) {
    MatrixXf randomProjections = MatrixXf::Zero(D_size, n); // TODO how to get type hints for MatrixXf?

    MatrixXf A = MatrixXi::Zero(n, 2 * k);

    boost::dynamic_bitset<> bitHD3;
    bitHD3Generator(DSize * PARAM_INTERNAL_NUM_ROTATION, bitHD3);

    int log2Project = log2(DSize);

    // Fast Hadamard transform
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);

#pragma omp parallel for
    for (int n = 0; n < n; ++n) {
        // Get data
        VectorXf vecPoint = VectorXf::Zero(DSize); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = MATRIX_X_EMBED.col(n);

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r) {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < DSize; ++d) {
                vecPoint(d) *= (2 * (int) bitHD3[r * DSize + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Project);
        }

        // We scale with DSize since we apply HD2HD1, each need a scale of sqrt(DSize).
        // The last HD3 is similar to Gaussian matrix.
        vecPoint /= DSize;

        randomProjections.col(n) = vecPoint;

        // Store potential candidate and find top-k closes and furtherest random vector
        Min_PQ_Pair minCloseTopK;
        Min_PQ_Pair minFarTopK;

        for (int d = 0; d < DSize; ++d) {
            float fValue = vecPoint(d);

            // Using priority queue to find top-k closest vectors for each point
            if ((int) minCloseTopK.size() < k)
                minCloseTopK.push(IFPair(d, fValue));
            else {
                if (fValue > minCloseTopK.top().m_fValue) {
                    minCloseTopK.pop();
                    minCloseTopK.push(IFPair(d, fValue));
                }
            }

            // Using priority queue to find top-k furthest vectors
            if ((int) minFarTopK.size() < k)
                minFarTopK.push(IFPair(d, -fValue));
            else {
                if (-fValue > minFarTopK.top().m_fValue) {
                    minFarTopK.pop();
                    minFarTopK.push(IFPair(d, -fValue));
                }
            }
        }

        // Get (sorted by projection value) top-k closest and furthest vector for each point
        for (int k = k - 1; k >= 0; --k) {
            A(n, k) = 2 * minCloseTopK.top().m_iIndex;
            minCloseTopK.pop();

            A(n, 2 * k) = 2 * minFarTopK.top().m_iIndex + 1;
            minFarTopK.pop();
        }
    }

    return randomProjections; // TODO how to return both the random projections and the A matrix?
}

bool addToPairPriorityQueue(Min_PQ_Pair priorityQueue, float pointValue, float pointIndex, int priorityQueueMaxSize) {

}

/**
 * Constructs the B matrix as per the GS-DBSCAN algorithm
 *
 * @param DSize Number of random vectors
 * @param m m parameter as per the sDBSCAN algorithm
 * @param minPts minPts param as per the DBSCAN algorithm
 * @param randomProjections matrix containing the random projections between the dataset and the random vectors
 * @return the B matrix. Each row of form 2*j+s stores the closest/furthest dataset vector to the ith random vector. For s = 0 store closest, for s = 1 store farthest
 */
MatrixXf constructBMatrix(int DSize, int m, int minPts,
                          const MatrixXf &randomProjections) {// After getting all neighborhood points for each random vector, we sort based on value, extract only index
    MatrixXi B = MatrixXi::Zero(2 * DSize, m);

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
#pragma omp parallel for
    for (int d = 0; d < DSize; ++d) {
        // sort(begin(matProject.col(d)), end(matProject.col(d)), [](float lhs, float rhs){return rhs > lhs});

        Min_PQ_Pair minPQ_Close;
        Min_PQ_Pair minPQ_Far;

        VectorXf vecProject = randomProjections.row(d);

        for (int n = 0; n < n; ++n) {
            float fValue = vecProject(n);

            if ((int) minPQ_Close.size() < minPts)
                minPQ_Close.push(IFPair(n, fValue));
            else {
                if (fValue > minPQ_Close.top().m_fValue) {
                    minPQ_Close.pop();
                    minPQ_Close.push(IFPair(n, fValue));
                }
            }

            // Single thread: (1) Far: PQ to find Top-MinPts for each random vector
            if ((int) minPQ_Far.size() < minPts)
                minPQ_Far.push(IFPair(n, -fValue));
            else {
                if (-fValue > minPQ_Far.top().m_fValue) {
                    minPQ_Far.pop();
                    minPQ_Far.push(IFPair(n, -fValue));
                }
            }
        }

        for (int k = minPts - 1; k >= 0; --k) {
            // Close
            B(2 * d, k) = minPQ_Close.top().m_iIndex;
            minPQ_Close.pop();

            // Far
            B(2 * d + 1, m + k) = minPQ_Far.top().m_iIndex;
            minPQ_Far.pop();
        }
    }

    return B;
}
