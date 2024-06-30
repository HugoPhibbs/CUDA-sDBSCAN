//
// Created by hphi344 on 10/05/24.
//

#include "../../include/GsDBSCAN.h"
#include <cmath>
#include <cassert>
#include <tuple>
#define AF_DEFINE_CUDA_TYPES

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
 * @param D int for number of random vectors to generate
 * @param minPts min number of points as per the DBSCAN algorithm
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 * @param skip_pre_checks boolean flag to skip the pre-checks
 */
void GsDBSCAN::performGsDbscan(af::array &X, int D, int minPts, int k, int m, float eps, float alpha) {
    if (!skip_pre_checks) {
        preChecks(X, D, minPts, k, m, eps);
    }
    // Something something ... TODO

    /*
     * Steps:
     *
     * 1. Preprocessing - Perform random projections and create A and B matrices
     * 2. Find distances between query vectors and their candidate vectors
     * 3. Use vectors to create E and V vectors
     * 4. Create the adjacency list
     * 5. Finally, create the cluster graph - can use Ninh's pre-existing clustering method
     *
     *
     */
    af::array projections = GsDBSCAN::randomProjections(X, D, k, m);

    af::array A, B;
    std::tie(A, B) = GsDBSCAN::constructABMatrices(projections, k, m);

    af::array distances = GsDBSCAN::findDistances(X, A, B, alpha);

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);
    af::array V = GsDBSCAN::processQueryVectorDegreeArray(E);

    af::array adjacencyList = GsDBSCAN::assembleAdjacencyList(distances, E, V, A, B, eps, 1024);

    // Create clusters, then return the clusters. Thats all
}

/**
 * Performs the pre-checks for the gs dbscan algorithm
 *
 * @param X ArrayFire af::array matrix for the X data points
 * @param D int for number of random vectors to generate
 * @param minPts min number of points as per the DBSCAN algorithm
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 */
void GsDBSCAN::preChecks(af::array &X, int D, int minPts, int k, int m, float eps) {
    assert(X.dims(1) > 0);
    assert(X.dims(1) > 0);
    assert(D > 0);
    assert(D >= k);
    assert(m >= minPts);
}

/**
 * Performs random projections between the X dataset and the random vector
 *
 * @param X af::array matrix for the X data points
 * @param D int for number of random vectors to generate
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @return af::array matrix for the random projections
 */
af::array GsDBSCAN::randomProjections(af::array &X, int D, int k, int m) {
    // TODO implement me!
    return af::constant(1, 1, 1);
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
std::tuple<af::array, af::array> GsDBSCAN::constructABMatrices(const af::array& projections, int k, int m) {
    // Assume projections has shape (n, D)
    int n = projections.dims(0);
    int D = projections.dims(1);

    af::array A(n, 2*k);
    af::array B(2*D, m);

    af::array dataToRandomIdxSorted = af::constant(-1, projections.dims(), af::dtype::u16);
    af::array randomToDataIdxSorted = af::constant(-1, projections.dims(), af::dtype::u16);
    af::array sortedValsTemp;

    af::sort(sortedValsTemp, sortedValsTemp, projections, 1);

    A(af::span, af::seq(0, k-1)) = 2 * dataToRandomIdxSorted(af::span, af::seq(0, k - 1));
    A(af::span, af::seq(k, af::end)) = 2 * dataToRandomIdxSorted(af::span, af::seq(dataToRandomIdxSorted.dims(0)-k, af::end));

    af::array BEvenIdx = af::seq(0, 2*D-1, 2);
    af::array BOddIdx = BEvenIdx + 1;

    B(BEvenIdx, af::span) = randomToDataIdxSorted(af::seq(0, m-1), af::span);
    B(BOddIdx, af::span) = randomToDataIdxSorted(af::seq( randomToDataIdxSorted.dims(0)-m, af::end), af::span);

    return std::make_tuple(A, B);
}

/**
 * Finds the distances between each of the query points and their candidate neighbourhood vectors
 *
 * @param X matrix containing the X dataset vectors
 * @param A A matrix, see constructABMatrices
 * @param B B matrix, see constructABMatrices
 * @param alpha float for the alpha parameter to tune the batch size
 */
af::array GsDBSCAN::findDistances(af::array &X, af::array &A, af::array &B, float alpha) {
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
 * Calculates the degree of the query vectors as per the G-DBSCAN algorithm.
 *
 * This function is used in the construction of the cluster graph by determining how many
 *
 * Put into its own method for testability
 *
 * @param distances The matrix containing the distances between the query and candidate vectors.
 *                  Expected shape is (datasetSize, 2*k*m).
 * @param eps       The epsilon value for DBSCAN. Should be a scalar array of the same data type
 *                  as the distances array.
 *
 * @return The degree array of the query vectors, with shape (datasetSize, 1).
 */
af::array GsDBSCAN::constructQueryVectorDegreeArray(af::array &distances, float eps) {
    return af::sum( distances < eps, 0);
}

/**
 * Processes the vector degree array to create an exclusive scan of this vector
 *
 * Put into it's own method to ensure testability
 *
 * @param E vector degree array
 * @return arrayfire processed array
 */
af::array GsDBSCAN::processQueryVectorDegreeArray(af::array &E) {
    return af::scan(E, 1, AF_BINARY_ADD, false); // Do an exclusive scan// TODO, need to return the V array, this is here to satisfy the compiler.
}

/**
 * Performs the actual clustering step of the algorithm
 *
 * I.e. returns the formed clusters based on the inputted adjacency list
 *
 * @param adjacencyList af array adjacency list for each of the dataset vectors as per the GsDBSCAN algorithm
 * @param V af array containing the starting index of each of the dataset vectors within the adjacency list
 */
void static performClustering(af::array &adjacencyList, af::array &V) {
    // TODO implement me!
}

/**
 * Kernel for constructing part of the cluster graph adjacency list for a particular vector
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param adjacencyList
 * @param V vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
 * @param A A matrix, see constructABMatrices. Stored flat as a float array
 * @param B B matrix, see constructABMatrices. Stored flat as a float array
 * @param n number of query vectors in the dataset
 * @param eps epsilon DBSCAN density param
 */
__global__ void constructAdjacencyListForQueryVector(float *distances, int *adjacencyList, int *V, int *A, int *B, float eps, int n, int k, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return; // Exit if out of bounds. Don't assume that numQueryVectors is equal to the total number o threads

    int curr_idx = V[idx];

    int distances_rows = 2 * k * m;

    int ACol, BCol, BRow, neighbourhoodVecIdx;

    for (int j = 0; j < distances_rows; j++) {
        if (distances[idx * distances_rows + j] < eps) {
            ACol = j / m;
            BCol = j % m;
            BRow = A[idx * 2 * k + ACol];
            neighbourhoodVecIdx = B[BRow * m + BCol];

            adjacencyList[curr_idx] = neighbourhoodVecIdx;
            curr_idx++;
        }
    }
}


/**
 * Assembles the adjacency list for the cluster graph
 *
 * See https://arrayfire.org/docs/interop_cuda.htm for info on this
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param E vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
 * @param V vector containing the starting index of each query vector in the resultant adjacency list (See the G-DBSCAN algorithm)
 * @param A A matrix, see constructABMatrices
 * @param B B matrix, see constructABMatrices
 * @param eps epsilon DBSCAN density param
 * @param blockSize size of each block when calculating the adjacency list - essentially the amount of query vectors to process per block
 */
af::array GsDBSCAN::assembleAdjacencyList(af::array &distances, af::array &E, af::array &V, af::array &A, af::array &B, float eps, int blockSize) {
    int n = E.dims(0);
    int k = A.dims(1) / 2;
    int m = B.dims(1);

    af::array adjacencyList = af::constant(-1, (E(n-1) + V(n-1)).scalar<int>(), af::dtype::u32);

    // Eval all matrices to ensure they are synced
    adjacencyList.eval();
    distances.eval();
    E.eval();
    V.eval();
    A.eval();
    B.eval();

    // Getting device pointers
    int *adjacencyList_d= adjacencyList.device<int>();
    float *distances_d = distances.device<float>();
    int *E_d = E.device<int>();
    int *V_d = V.device<int>();
    int *A_d = A.device<int>();
    int *B_d = B.device<int>();

    // Getting cuda stream from af
    cudaStream_t afCudaStream = getAfCudaStream();

    // Now we can call the kernel
    int numBlocks = std::max(1, n / blockSize);
    blockSize = std::min(n, blockSize);
    constructAdjacencyListForQueryVector<<<numBlocks, blockSize, 0, afCudaStream>>>(distances_d, adjacencyList_d, V_d, A_d, B_d, eps, n, k, m);

    // Unlock all the af arrays
    adjacencyList.unlock();
    distances.unlock();
    E.unlock();
    V.unlock();
    A.unlock();
    B.unlock();

    return adjacencyList;
}

/**
 * Gets the CUDA stream from ArrayFire
 *
 * For easy testing of other functions
 *
 * See https://arrayfire.org/docs/interop_cuda.htm for info on this
 *
 * @return the CUDA stream
 */
cudaStream_t GsDBSCAN::getAfCudaStream() {
    int afId = af::getDevice();
    int cudaId= afcu::getNativeId(afId);
    return afcu::getStream(cudaId);
}

/*
 * For the above, check this issue:
 *
 * https://github.com/arrayfire/arrayfire/issues/3051
 *
 * May need to look further into this - i.e. how to use af/cuda.h properly
 */

