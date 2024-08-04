//
// Created by hphi344 on 10/05/24.
//


#include "../../include/GsDBSCAN.h"
#include <cmath>
#include <cassert>
#include <tuple>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "../../include/TestUtils.h"

namespace tu = testUtils;

// Constructor to initialize the DBSCAN parameters
GsDBSCAN::GsDBSCAN(const af::array &X, int D, int minPts, int k, int m, float eps, bool skip_pre_checks, float sigma, int seed, string distanceMetric, float batchAlpha, int fhtDim, int nRotate, bool clusterNoise)
        : X(X), D(D), minPts(minPts), k(k), m(m), eps(eps), skip_pre_checks(skip_pre_checks), sigma(sigma), seed(seed), clusterNoise(clusterNoise), distanceMetric(distanceMetric), batchAlpha(batchAlpha), fhtDim(fhtDim) {
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
void GsDBSCAN::performGsDbscan() {
//    if (!skip_pre_checks) {
//        preChecks(X, D, minPts, k, m, eps);
//    }
    // Something something ...

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
//    af::array projections = GsDBSCAN::randomProjections(X, D, k, m, distanceMetric, sigma, seed, fhtDim, nRotate);
    af::array projections = af::constant(D, k, af::dtype::f32);

    af::array A, B;
    std::tie(A, B) = GsDBSCAN::constructABMatrices(projections, k, m);

    af::array distances = GsDBSCAN::findDistances(X, A, B, batchAlpha);

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);
    af::array V = GsDBSCAN::processQueryVectorDegreeArray(E);

    af::array adjacencyList = GsDBSCAN::assembleAdjacencyList(distances, E, V, A, B, eps, 1024);

    // Create clusters, then return the clusters. Thats all
}



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
    int m = B.dims(1);

    int n = X.dims(0);
    int d = X.dims(1);
    int D = B.dims(0) / 2;

    int batchSize = GsDBSCAN::findDistanceBatchSize(alpha, n, d, k, m);

    af::array distances(n, 2 * k * m, af::dtype::f32);
    af::array ABatch(batchSize, 2 * k, A.type());
    af::array BBatch(batchSize, m, B.type());
    af::array XBatch(batchSize, 2 * k, m, d, X.type());
    af::array XBatchAdj(batchSize, 2 * k * m, d,
                        X.type()); // This is very large, around 7gb. Possible to do this without explicitly allocating the memory?
    af::array XSubset(batchSize, d, X.type());
    af::array XSubsetReshaped = af::constant(0, XBatchAdj.dims(), XBatchAdj.type());
    af::array YBatch = af::constant(0, XBatchAdj.dims(), XBatchAdj.type());

    for (int i = 0; i < n; i += batchSize) {
        int maxBatchIdx = i + batchSize - 1;
        ABatch = A(af::seq(i, maxBatchIdx), af::span);

        BBatch = B(ABatch, af::span);

        BBatch = af::moddims(BBatch, BBatch.dims(0) / (2 * k), 2 * k, BBatch.dims(1));

        XBatch = X(BBatch, af::span);

        XBatchAdj = af::moddims(XBatch, batchSize, 2 * k * m, d);

        XSubset = X(af::seq(i, maxBatchIdx), af::span);

        XSubsetReshaped = moddims(XSubset, XSubset.dims(0), 1, XSubset.dims(1)); // Insert new dim

        YBatch = XBatchAdj - XSubsetReshaped;

        // sqrt(sum(sq(...)))

        distances(af::seq(i, maxBatchIdx), af::span) = af::norm(YBatch, AF_NORM_VECTOR_2,
                                                                2); // af doesn't have norms across arbitrary'
    }

    return distances;
}

void printCudaMemoryUsage() {
    size_t free_mem, total_mem;
    cudaError_t error;

    cudaDeviceSynchronize();

    // Get memory information
    error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    // Convert bytes to gigabytes
    double free_mem_gb = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
    double total_mem_gb = static_cast<double>(total_mem) / (1024.0 * 1024.0 * 1024.0);
    double used_mem_gb = total_mem_gb - free_mem_gb;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Memory Usage: " << used_mem_gb << " GB used, "
              << free_mem_gb << " GB free, " << total_mem_gb << " GB total" << std::endl;
}

matx::tensor_t<matx::matxFp16, 2>  GsDBSCAN::findDistancesMatX(matx::tensor_t<matx::matxFp16, 2> &X_t, matx::tensor_t<int, 2> &A_t, matx::tensor_t<int, 2> &B_t, float alpha, int batchSize) {
    const int k = A_t.Shape()[1] / 2;
    const int m = B_t.Shape()[1];

    const int n = X_t.Shape()[0];
    const int d = X_t.Shape()[1];
    int D = B_t.Shape()[0] / 2;

    batchSize = (batchSize != -1) ? batchSize : GsDBSCAN::findDistanceBatchSize(alpha, n, d, k, m);

    auto AFlat_t = matx::flatten(A_t);

    auto distances_t = matx::make_tensor<matx::matxFp16>({n, 2*k*m}, matx::MATX_DEVICE_MEMORY);

    int j = 0;
    std::vector<double> times;

    auto start_all = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i += batchSize) {
        auto start = std::chrono::high_resolution_clock::now();

        int maxBatchIdx = i + batchSize - 1; // Index within X along the ROWS

        auto XSubset_t_op = matx::slice(X_t, {i, 0}, {maxBatchIdx + 1, matx::matxEnd});

        auto ABatchFlat_t_op = matx::slice(AFlat_t, {i * 2 * k}, {(maxBatchIdx + 1) * 2 * k});

        auto BBatch_t_op = matx::remap<0>(B_t, ABatchFlat_t_op);

        auto XBatch_t_op = matx::remap<0>(X_t, matx::flatten(BBatch_t_op));

        auto XBatchReshaped_t_op = matx::reshape(XBatch_t_op, {batchSize, 2*k*m, d});

        auto XSubsetReshaped_t_op = matx::reshape(XSubset_t_op, {batchSize, 1, d});

        auto YBatch_t_op = (XBatchReshaped_t_op - matx::repmat(XSubsetReshaped_t_op, {1, 2*k*m, 1})); // Repmat is a workaround for minusing naively incompatibhle tensor shapes

        auto YBatch_t_norm_op = matx::vector_norm(YBatch_t_op, {2}, matx::NormOrder::L2);

        (matx::slice(distances_t, {i, 0}, {maxBatchIdx + 1, matx::matxEnd}) = YBatch_t_norm_op).run();

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration
        std::chrono::duration<double> duration = end - start;

        // Cast to double and store in array
        times.push_back(duration.count());
    }

    auto start_sync = std::chrono::high_resolution_clock::now();

    cudaDeviceSynchronize();

    // Record end time
    auto end_sync = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration_sync = end_sync - start_sync;

    // Output the duration
    std::cout << "Time taken: " << duration_sync.count() << " seconds" << std::endl;

    for (const auto& element : times) {
        std::cout << element << std::endl;
    }

    // Record end time
    auto end_all = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end_all - start_all;

    // Output the duration
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return distances_t;
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
    int batchSize = static_cast<int>((static_cast<long long>(n) * d * 2 * k * m) / (std::pow(1024, 3) * alpha));

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

/**
 * Performs the actual clustering step of the algorithm
 *
 * Rewritten from Ninh's original code
 *
 * @param adjacencyList af array adjacency list for each of the dataset vectors as per the GsDBSCAN algorithm
 * @param V starting index of each of the dataset vectors within the adjacency list
 * @param E degree of each query vector (how many candidate vectors are within eps distance of it)
 * @param n size of the dataset
 * @param minPts minimum number of points within eps distance to consider a point as a core point
 * @param clusterNoise whether to include noise points in the result
 * @return a tuple containing the cluster labels and the number of clusters found
 */
tuple<vector<int>, int> GsDBSCAN::formClusters(af::array &adjacencyList, af::array &V, af::array &E, int n, int minPts, bool clusterNoise) {
    int nClusters = 0;
    vector<int> labels = IVector(n, -1);

    int iNewClusterID = -1;

    auto isCore = [&](int idx) -> bool {
        // TODO use a bit set instead of a cumbersome af array
        return E(idx).scalar<int>() >= minPts;
    };

    for (int i = -1; i < n; i++) {

        if (!isCore(i) || (labels[i] != -1)) {
            continue;
        }

        iNewClusterID++;

        unordered_set<int> seedSet; //seedSet only contains core points
        seedSet.insert(i);

        boost::dynamic_bitset<> connectedPoints(n);
        connectedPoints[i] = true;

        int startIndex, endIndex;

        while (!seedSet.empty()) {
            int Xi = *seedSet.begin();
            seedSet.erase(seedSet.begin());

            startIndex = V(Xi).scalar<int>();
            endIndex = startIndex + E(Xi).scalar<int>();
            int Xj;

            for (int j = startIndex; j < endIndex; j++) {
                Xj = adjacencyList(j).scalar<int>();

                if (isCore(i)) {
                    if (!connectedPoints[Xj]) {
                        connectedPoints[Xj] = true;

                        if (labels[Xj] == -1) seedSet.insert(Xj);
                    }
                } else {
                    connectedPoints[Xj] = true;
                }

            }
        }

        size_t Xj = connectedPoints.find_first();

        while (Xj != boost::dynamic_bitset<>::npos) {
            if (labels[Xj] == -1) labels[Xj] = iNewClusterID;

            Xj = connectedPoints.find_next(Xj);
        }

        nClusters = iNewClusterID;
    }

    if (clusterNoise) {
        // TODO, implement labeling of noise
    }

    return make_tuple(labels, nClusters);
}

/*
 * For the above, check this issue:
 *
 * https://github.com/arrayfire/arrayfire/issues/3051
 *
 * May need to look further into this - i.e. how to use af/cuda.h properly
 */

