//
// Created by hphi344 on 10/05/24.
//

#include "../../include/gsDBSCAN.h"
#include <arrayfire.h>
#include <cmath>

// Constructor to initialize the DBSCAN parameters
GsDBSCAN::GsDBSCAN(const af::array &X, const af::array &D, int minPts, int k, int m, float eps, bool skip_pre_checks)
        : X(X), D(D), minPts(minPts), k(k), m(m), eps(eps), skip_pre_checks(skip_pre_checks) {
        n = X.dims(0);
        d = X.dims(1);

}

/**
 * Performs the gs dbscan algorithm
 *
 * @param X ArrayFire af::array matrix for the X data points
 * @param D ArrayFire af::array matrix for the D random vectors
 * @param minPts min number of points as per the DBSCAN algorithm
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 * @param skip_pre_checks boolean flag to skip the pre-checks
 */
void GsDBSCAN::performGsDbscan() {
    if (!skip_pre_checks) {
        preChecks(X, D, minPts, k, m, eps);
    }
    // Something something ... TODO
}

/**
 * Performs the pre-checks for the gs dbscan algorithm
 *
 * @param X ArrayFire af::array matrix for the X data points
 * @param D ArrayFire af::array matrix for the D random vectors
 * @param minPts min number of points as per the DBSCAN algorithm
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 */
void GsDBSCAN::preChecks(af::array X, af::array D, int minPts, int k, int m, float eps) {
    // TODO implement me!
}

/**
 * Performs the pre-processing for the gs dbscan algorithm
 *
 * @param D ArrayFire af::array matrix for the D random vectors
 * @return A, B matrices as per the GS-DBSCAN algorithm. A has size (n, 2*k) and B has size (2*k, m)
 */
void GsDBSCAN::preProcessing(af::array D) {
    // TODO implement me!
}

/**
 * Performs random projections between the X dataset and the random vector
 *
 * @param X af::array matrix for the X data points
 * @param D af::array matrix for the D data points
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 * @return af::array matrix for the random projections
 */
void GsDBSCAN::randomProjections(af::array X, af::array D, int k, int m, float eps) {
    // TODO implement me!
}

/**
 * Constructs the A and B matrices as per the GS-DBSCAN algorithm
 *
 * A stores the indices of the closest and furthest random vectors per data point.
 *
 * @param X matrix containing the X dataset vectors
 * @param D matrix containing the random vectors
 * @param k k parameter as per the DBSCAN algorithm
 * @param m m parameter as per the DBSCAN algorithm
 */
void GsDBSCAN::constructABMatrices(af::array X, af::array D, int k, int m) {
    // TODO implement me!
}

/**
 * Finds the distances between each of the query points and their candidate neighbourhood vectors
 *
 * @param X matrix containing the X dataset vectors
 * @param A A matrix, see constructABMatrices
 * @param B B matrix, see constructABMatrices
 * @param alpha float for the alpha parameter to tune the batch size
 */
af::array GsDBSCAN::findDistances(af::array X, af::array A, af::array B, float alpha) {
    int k = A.dims(1) / 2;
    int m = B.dims(1) / 2;

    int n = X.dims(0);
    int d = X.dims(1);

    int batchSize = GsDBSCAN::findDistanceBatchSize(alpha, n, d, k, m);

    af::array distances(n, 2*k*m, af::dtype::f32);
    af::array ABatch(batchSize, A.dims(1), A.type());
    af::array BBatch(batchSize, B.dims(1), B.type());
    af::array XBatch(batchSize, 2*k, m, d, X.type());
    af::array XBatchAdj(batchSize, 2*k*m, d, X.type());
    af::array XSubset(batchSize, d, X.type());
    af::array XSubsetReshaped = af::constant(0, XBatchAdj.dims(), XBatchAdj.type());
    af::array YBatch = af::constant(0, XBatchAdj.dims(), XBatchAdj.type());

    for (int i = 0; i < n; i += batchSize) {
        int maxBatchIdx = i + batchSize - 1;
        ABatch = A(af::seq(i, maxBatchIdx));
        BBatch = B(A);

        XBatch = X(BBatch); // TODO need to create XBatch before for loop?
        XBatchAdj = af::moddims(XBatch, XBatch.dims(0), XBatch.dims(1) * XBatch.dims(2), XBatch.dims(3));

        XSubset = X(af::seq(i, maxBatchIdx), af::span);

        XSubsetReshaped = moddims(XSubset, XSubset.dims(0), 1, XSubset.dims(1)); // Insert new dim

        YBatch = XBatchAdj - XSubsetReshaped;

        distances(af::seq(i, maxBatchIdx), af::span) = af::sqrt(af::sum(af::pow(YBatch, 2), 2)); // af doesn't have norms across arbitrary axes
    }

    return distances;
}

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
int GsDBSCAN::findDistanceBatchSize(float alpha, int n, int d, int k, int m) {
    int batchSize = static_cast<int>(static_cast<long long>(n) * d * 2 * k * m / (std::pow(1024, 3) * alpha));

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
 * Constructs the cluster graph for the DBSCAN algorithm
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param eps epsilon DBSCAN density param
 * @param k k parameter of the DBSCAN algorithm
 * @param m m parameter of the DBSCAN algorithm
 * @return a vector containing the (flattened) adjacency list of the cluster graph, along with another list V containing the starting index of each query vector in the adjacency list
 */
void GsDBSCAN::constructClusterGraph(af::array distances, float eps, int k, int m) {
    // TODO implement me!
}

/**
 * Assembles the adjacency list for the cluster graph
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param E vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
 * @param V vector containing the starting index of each query vector in the resultant adjacency list (See the G-DBSCAN algorithm)
 * @param A A matrix, see constructABMatrices
 * @param B B matrix, see constructABMatrices
 * @param eps epsilon DBSCAN density param
 * @param blockSize size of each block when calculating the adjacency list - essentially the amount of query vectors to process per block
 */
void GsDBSCAN::assembleAdjacencyList(af::array distances, int E, int V, af::array A, af::array B, float eps, int blockSize) {
    // TODO implement me!
}

/**
 * Kernel for constructing part of the cluster graph adjacency list for a particular vector
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param adjacencyList
 * @param V vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
 * @param E vector containing the starting index of each query vector in the adjacency list
 * @param eps epsilon DBSCAN density param
 */
//__global__ void constructAdjacencyListForQueryVector(float *distances, int *adjacencyList, int V, int E, float eps) {
//    // TODO implement me!
//}
