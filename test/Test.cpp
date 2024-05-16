#include "../test/Test.h"
#include "../include/sngDBSCAN.h"
#include "../include/sDBSCAN.h"
#include "../include/uDBSCAN.h"
#include "../include/sngDBSCAN.h"

MatrixXf MATRIX_X_EMBED;

/**
Simulate Random Fourier Feature for L2 by HD3HD2HD1
**/
void FourierEmbed_L2()
{

    int iEmbed_D = round(PARAM_KERNEL_EMBED_D / 2); // This is becase we need cos() and sin()
    int log2Embed_D = log2(iEmbed_D);

    // PARAM_INTERNAL_FWHT_PROJECTION = 2^log{D}
    // We can use less than D random projection, so call FWHT with 2^log{D} and select top-D-up positions (see Falconn++)
    boost::dynamic_bitset<> bitHD3;
    bitHD3Generator(iEmbed_D * PARAM_INTERNAL_NUM_ROTATION, bitHD3);

    MatrixXf MATRIX_P = MatrixXf::Zero(iEmbed_D, PARAM_DATA_N); // col-wise D x n

    // Fast Hadamard transform
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get data

        VectorXf vecPoint = VectorXf::Zero(iEmbed_D); // iEmbed_D > D
        vecPoint.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < iEmbed_D; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3[r * iEmbed_D + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Embed_D);
        }

        // We scale with iEmbed_D since we apply HD2HD1, each need a scale of sqrt(iEmbed_D). The last HD3 is similar to Gaussian matrix.
        // We further scale with PARAM_DATA_D since we use the standard scale N(0, 1/d^2) hence std = 1/d for kernel embedding
        MATRIX_P.col(n) = vecPoint / (iEmbed_D * PARAM_KERNEL_SIGMA); // vecPoint.segment(0, 128); // 512 projections, but you need only 128 projections
    }


    /**
    Test the accuracy of HD3HD2HD1 that preserve dot product
    **/
//    float error = 0.0;
//    int iNumPairs = 10000;
//
//    // We have to scale MATRIX_P as MATRIX_P.col(n) = vecPoint / (iEmbed_D * sqrt(iEmbed_D));
//    MATRIX_P *= (iEmbed_D * sqrt(PARAM_DATA_D));
//    MATRIX_P /= (iEmbed_D * sqrt(iEmbed_D));
//
//    for (int n = 0; n < iNumPairs; ++n)
//    {
//        // Generate a random pair (i, j) to compute L2 distance
//        int Xi = rand() % PARAM_DATA_N;
//        int Xj = rand() % PARAM_DATA_N;
//
//        // True distance
//        float fDot = MATRIX_X.col(Xi).dot(MATRIX_X.col(Xj));
//        float fEst = MATRIX_P.col(Xi).dot(MATRIX_P.col(Xj)); // HD preserves dot product
//        error += abs(fDot - fEst);
//    }
//
//    cout << "Avg error of HD3HD2HD1 is: " << error / iNumPairs << endl;


    MATRIX_X_EMBED = MatrixXf::Zero(PARAM_KERNEL_EMBED_D, PARAM_DATA_N);
    MATRIX_X_EMBED << MATRIX_P.array().cos() / sqrt(iEmbed_D), MATRIX_P.array().sin() / sqrt(iEmbed_D);


    /**
    Test the accuracy of kernel embedding
    Note that we have to keep MATRIX_X for this testing
    If using N(0, 1), then approximate K(x, y) = exp(-dist(x, y)^2 / 2)
    If using N(0, 1/sigma^2) (std = 1/sigma), then approximate K(x, y) = exp(-dist(x, y)^2/ 2 sigma^2)
    **/
//    MatrixXf MATRIX_G;
//    MATRIX_G = gaussGenerator(iEmbed_D, PARAM_DATA_D, 0, 1, MATRIX_G);
//    MatrixXf tempX = MATRIX_G * MATRIX_X; // col-wise D x n
//    tempX /= PARAM_DATA_D; // scale by d
//    MatrixXf MATRIX_PROJECT = MatrixXf::Zero(PARAM_KERNEL_EMBED_D, PARAM_DATA_N);
//    MATRIX_PROJECT << tempX.array().cos() / sqrt(iEmbed_D), tempX.array().sin() / sqrt(iEmbed_D);
//
//    float error1 = 0.0;
//    float error2 = 0.0;
//    int iNumPairs = 1000;
//    for (int n = 0; n < iNumPairs; ++n)
//    {
//        // Generate a random pair (i, j) to compute L2 distance
//        int Xi = rand() % 1000;
//        int Xj = rand() % 1000;
//
//        // True distance
//        VectorXf vecTemp = MATRIX_X.col(Xi) - MATRIX_X.col(Xj);
//        float fSquaredDist = vecTemp.squaredNorm();
//        float fKernelValue = exp(-fSquaredDist / (2 * PARAM_KERNEL_SIGMA * PARAM_KERNEL_SIGMA));  // Since we use std = 1/sigma
//        float fEst1 = MATRIX_X_EMBED.col(Xi).dot(MATRIX_X_EMBED.col(Xj));
//        float fEst2 = MATRIX_PROJECT.col(Xi).dot(MATRIX_PROJECT.col(Xj));
//
//        if (fSquaredDist <= 16.0) // kNN distance tends to be very small compared to the rest
//        {
//            cout << "Dist: " << sqrt(fSquaredDist) << endl;
//            cout << "Kernel value: " << fKernelValue << endl;
//            cout << "Estimate 1: " << fEst1 << endl;
//            cout << "Estimate 2: " << fEst2 << endl;
//        }
//
//        error1 += abs(fKernelValue - fEst1);
//        error2 += abs(fKernelValue - fEst2);
//    }
//
//    cout << "Avg error of embedding 1 is: " << error1 / iNumPairs << endl;
//    cout << "Avg error of embedding 2 is: " << error2 / iNumPairs << endl;

}

void FourierEmbed_Nonmetric()
{

    MatrixXf MATRIX_X_EMBED = MatrixXf::Zero(PARAM_KERNEL_EMBED_D, PARAM_DATA_N); // col-wise D x n

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        /**
        Deterministic Additive Kernel feature
        **/
        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D); // iEmbed_D > D
        if (PARAM_DISTANCE == 3)
            embedChiSquare(MATRIX_X.col(n), MATRIX_X_EMBED.col(n));
        else if (PARAM_DISTANCE == 4)
            embedJS(MATRIX_X.col(n), MATRIX_X_EMBED.col(n));
    }

        /**
    Test the accuracy of HD3HD2HD1 that preserve dot product
    **/
    float error = 0.0;
    int iNumPairs = 10000;

    for (int n = 0; n < iNumPairs; ++n)
    {
        // Generate a random pair (i, j) to compute L2 distance
        int Xi = rand() % PARAM_DATA_N;
        int Xj = rand() % PARAM_DATA_N;

        // True distance
//        VectorXf vecX = MATRIX_X.col(Xi);
//        vecX.array() += EPSILON;
//        VectorXf vecY = MATRIX_X.col(Xj);
//        vecY.array() += EPSILON;
//
//        VectorXf multTemp = vecX.cwiseProduct(vecY);
//        VectorXf sumTemp = vecX + vecY;
//        VectorXf temp = multTemp.cwiseQuotient(sumTemp);
//        temp.array() *= 2;
//
//        float fDot = temp.sum();

        float fDot = 1.0 - computeDist(MATRIX_X.col(Xi), MATRIX_X.col(Xj));

//        cout << "True distance: " << fDot << endl;
        float fEst = MATRIX_X_EMBED.col(Xi).dot(MATRIX_X_EMBED.col(Xj)); // HD preserves dot product
//        cout << "Approx distance: " << fEst << endl;

        error += abs(fDot - fEst);
    }

    cout << "Avg error of kernel embedding is: " << error / iNumPairs << endl;


}

/**
We process each point seperately, and do not store any additional information, e.g. MATRIX_PROJECT
- Input: MATRIX_EMBED
- Used while processing the data point directly and updating the index on-the-fly
**/
void seqRandomProjection()
{
    vector<Min_PQ_Pair> vecPQ_Close = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());
    vector<Min_PQ_Pair> vecPQ_Far = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());

    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    boost::dynamic_bitset<> bitHD3;
    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3);

    int log2Project = log2(PARAM_NUM_PROJECTION);

    // Fast Hadamard transform
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get data
        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = MATRIX_X_EMBED.col(n);

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3[r * PARAM_NUM_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Project);
        }

        // We scale with PARAM_NUM_PROJECTION since we apply HD2HD1, each need a scale of sqrt(PARAM_NUM_PROJECTION).
        // The last HD3 is similar to Gaussian matrix.

        // Since we get top-k, we might not need to rescale for unbiased estimate
        vecPoint /= PARAM_NUM_PROJECTION;

        // Test min and max of G*x
//        cout << "Min is: " << vecPoint.minCoeff() << " and Max is: " << vecPoint.maxCoeff() << endl;

        // Store potential candidate and find top-k closes and furtherest random vector
        Min_PQ_Pair minCloseTopK;
        Min_PQ_Pair minFarTopK;

        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
        {
            float fValue = vecPoint(d);

            /**
            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
            **/

        #pragma omp critical
        {
            // Single thread: (1) Close: PQ to find Top-MinPts for each random vector
            if ((int)vecPQ_Close[d].size() < PARAM_DBSCAN_MINPTS)
                vecPQ_Close[d].push(IFPair(n, fValue));
            else
            {
                if (fValue > vecPQ_Close[d].top().m_fValue)
                {
                    vecPQ_Close[d].pop();
                    vecPQ_Close[d].push(IFPair(n, fValue));
                }
            }

            // Single thread: (1) Far: PQ to find Top-MinPts for each random vector
            if ((int)vecPQ_Far[d].size() < PARAM_DBSCAN_MINPTS)
                vecPQ_Far[d].push(IFPair(n, -fValue));
            else
            {
                if (-fValue > vecPQ_Far[d].top().m_fValue)
                {
                    vecPQ_Far[d].pop();
                    vecPQ_Far[d].push(IFPair(n, -fValue));
                }
            }
        }
            // (2) Close: Using priority queue to find top-k closest vectors for each point
            if ((int)minCloseTopK.size() < PARAM_PROJECTION_TOP_K)
                minCloseTopK.push(IFPair(d, fValue));
            else
            {
                if (fValue > minCloseTopK.top().m_fValue)
                {
                    minCloseTopK.pop();
                    minCloseTopK.push(IFPair(d, fValue));
                }
            }


            // // (2) Far: Using priority queue to find top-k furthest vectors
            if ((int)minFarTopK.size() < PARAM_PROJECTION_TOP_K)
                minFarTopK.push(IFPair(d, -fValue));
            else
            {
                if (-fValue > minFarTopK.top().m_fValue)
                {
                    minFarTopK.pop();
                    minFarTopK.push(IFPair(d, -fValue));
                }
            }

        }

        // Get (sorted by projection value) top-k closest and furthest vector for each point
        for (int k = PARAM_PROJECTION_TOP_K - 1; k >= 0; --k)
        {
            MATRIX_TOP_K(k, n) = minCloseTopK.top().m_iIndex;
            minCloseTopK.pop();

            MATRIX_TOP_K(k + PARAM_PROJECTION_TOP_K, n) = minFarTopK.top().m_iIndex;
            minFarTopK.pop();
        }


    }



    // After getting all neighborhood points for each random vector, we sort based on value, extract only index
    MATRIX_TOP_M = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_M, PARAM_NUM_PROJECTION);

    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
    {
        for (int k = PARAM_DBSCAN_MINPTS - 1; k >= 0; --k)
        {
            // Close
            MATRIX_TOP_M(k, d) = vecPQ_Close[d].top().m_iIndex;
            vecPQ_Close[d].pop();

            // Far
            MATRIX_TOP_M(k + PARAM_PROJECTION_TOP_M, d) = vecPQ_Far[d].top().m_iIndex;
            vecPQ_Far[d].pop();
        }
    }
}

/**
Have to store the projection matrix for parallel processing
- Input: MATRIX_EMBED
**/
void parRandomProjection()
{

    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);

    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    boost::dynamic_bitset<> bitHD3;
    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3);

    int log2Project = log2(PARAM_NUM_PROJECTION);

    // Fast Hadamard transform
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get data
        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = MATRIX_X_EMBED.col(n);

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3[r * PARAM_NUM_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Project);
        }

        // We scale with PARAM_NUM_PROJECTION since we apply HD2HD1, each need a scale of sqrt(PARAM_NUM_PROJECTION).
        // The last HD3 is similar to Gaussian matrix.
        vecPoint /= PARAM_NUM_PROJECTION;

        MATRIX_RP.col(n) = vecPoint;

        // Test min and max of G*x
//        cout << "Min is: " << vecPoint.minCoeff() << " and Max is: " << vecPoint.maxCoeff() << endl;

        // Store potential candidate and find top-k closes and furtherest random vector
        Min_PQ_Pair minCloseTopK;
        Min_PQ_Pair minFarTopK;

        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
        {
            float fValue = vecPoint(d);

            /**
            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
            - Will process it later using projection matrix
            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
            **/

            // (2) Close: Using priority queue to find top-k closest vectors for each point
            if ((int)minCloseTopK.size() < PARAM_PROJECTION_TOP_K)
                minCloseTopK.push(IFPair(d, fValue));
            else
            {
                if (fValue > minCloseTopK.top().m_fValue)
                {
                    minCloseTopK.pop();
                    minCloseTopK.push(IFPair(d, fValue));
                }
            }

            // // (2) Far: Using priority queue to find top-k furthest vectors
            if ((int)minFarTopK.size() < PARAM_PROJECTION_TOP_K)
                minFarTopK.push(IFPair(d, -fValue));
            else
            {
                if (-fValue > minFarTopK.top().m_fValue)
                {
                    minFarTopK.pop();
                    minFarTopK.push(IFPair(d, -fValue));
                }
            }

        }



        // Get (sorted by projection value) top-k closest and furthest vector for each point
        for (int k = PARAM_PROJECTION_TOP_K - 1; k >= 0; --k)
        {
            MATRIX_TOP_K(k, n) = minCloseTopK.top().m_iIndex;
            minCloseTopK.pop();

            MATRIX_TOP_K(k + PARAM_PROJECTION_TOP_K, n) = minFarTopK.top().m_iIndex;
            minFarTopK.pop();
        }
    }

    // After getting all neighborhood points for each random vector, we sort based on value, extract only index
    MATRIX_TOP_M = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_M, PARAM_NUM_PROJECTION);

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
    {
        // sort(begin(matProject.col(d)), end(matProject.col(d)), [](float lhs, float rhs){return rhs > lhs});

        Min_PQ_Pair minPQ_Close;
        Min_PQ_Pair minPQ_Far;

        VectorXf vecProject = MATRIX_RP.row(d);

        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            float fValue = vecProject(n);

            if ((int)minPQ_Close.size() < PARAM_DBSCAN_MINPTS)
                minPQ_Close.push(IFPair(n, fValue));
            else
            {
                if (fValue > minPQ_Close.top().m_fValue)
                {
                    minPQ_Close.pop();
                    minPQ_Close.push(IFPair(n, fValue));
                }
            }

            // Single thread: (1) Far: PQ to find Top-MinPts for each random vector
            if ((int)minPQ_Far.size() < PARAM_DBSCAN_MINPTS)
                minPQ_Far.push(IFPair(n, -fValue));
            else
            {
                if (-fValue > minPQ_Far.top().m_fValue)
                {
                    minPQ_Far.pop();
                    minPQ_Far.push(IFPair(n, -fValue));
                }
            }
        }

        for (int k = PARAM_DBSCAN_MINPTS - 1; k >= 0; --k)
        {
            // Close
            MATRIX_TOP_M(k, d) = minPQ_Close.top().m_iIndex;
            minPQ_Close.pop();

            // Far
            MATRIX_TOP_M(k + PARAM_PROJECTION_TOP_M, d) = minPQ_Far.top().m_iIndex;
            minPQ_Far.pop();
        }

    }
}

/**
Construct core points and its neighborhood given precomputed Optics data structure
**/
void findCorePoints_PrecomputedOptics()
{
    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Using the set to clear duplicates
        unordered_set<int> setNeighbor;
        for (int i = 0; i < (int)vec2D_OPTICS_NeighborDist[n].size(); ++i)
        {
            if (vec2D_OPTICS_NeighborDist[n][i].second <= PARAM_DBSCAN_EPS)
                setNeighbor.insert(vec2D_OPTICS_NeighborDist[n][i].first);
        }


        // Decide core points
        if ((int)setNeighbor.size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[n] = 1;

//            if ( n < 1000 )
//                cout << setNeighbor.size() << endl;

            // Only need neighborhood if it is core point
            vec2D_DBSCAN_Neighbor[n].insert(vec2D_DBSCAN_Neighbor[n].end(), setNeighbor.begin(), setNeighbor.end());

        }

        // We need to keep the neighborhood of noisy points in case we have to cluster the noise by the case 3
        // Which is consider any core points within its neighborhoods
        else if (PARAM_DBSCAN_CLUSTER_NOISE == 3)
        {
            vec2D_DBSCAN_Neighbor[n].insert(vec2D_DBSCAN_Neighbor[n].end(), setNeighbor.begin(), setNeighbor.end());
        }

    }

    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

/**
vec2D_DBSCAN_Neighbor: For each sampled point, store its true neighborhood (within radius eps)
**/
void findCorePoints_maxEps(float p_fMaxEps)
{
    vec2D_OPTICS_NeighborDist = vector< vector< pair<int, float> > > (PARAM_DATA_N, vector< pair<int, float> >());

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        VectorXf vecXn = MATRIX_X.col(n);

        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far

        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N);

        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
        {
            // Closest
            int Ri = vecTopK(k);

            for (int i = 0; i < PARAM_PROJECTION_TOP_M; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_M(i, Ri);

                if (iPointIdx == n)
                    continue;

                if (!approxNeighbor[iPointIdx]) // cannot find
                {
                    approxNeighbor[iPointIdx] = 1;

//                    cout << "The point: " << n << endl;
//                    cout << vecXn << endl;
//
//                    cout << "The point: " << iPointIdx << endl;
//                    cout << MATRIX_X.col(iPointIdx) << endl;

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

//                    if (fDist < 0)
//                    {
//                        cout << n << " " << iPointIdx << ": " << fDist << endl;
//                        cout << 1 - vecXn.dot(MATRIX_X.col(iPointIdx)) << endl;
//                    }

                    if (fDist <= p_fMaxEps)
                    {

                        #pragma omp critical
                        {
//                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDist));
//                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDist));

                        vec2D_OPTICS_NeighborDist[n].push_back(make_pair(iPointIdx, fDist)); // duplicate at most twice
                        vec2D_OPTICS_NeighborDist[iPointIdx].push_back(make_pair(n, fDist)); // so vector is much better than map()
                        }
                    }
                }
            }

            // Far
            Ri = vecTopK(k + PARAM_PROJECTION_TOP_K);

            for (int i = 0; i < PARAM_PROJECTION_TOP_M; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_M(i + PARAM_PROJECTION_TOP_M, Ri);

                if (iPointIdx == n)
                    continue;

                if (!approxNeighbor[iPointIdx]) // cannot find
                {
                    approxNeighbor[iPointIdx] = 1;

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= p_fMaxEps)
                    {
                        // omp_set_dynamic(0);     // Explicitly disable dynamic teams
                        omp_set_num_threads(PARAM_NUM_THREADS);
                        #pragma omp critical
                        {
//                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDist));
//                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDist));

                        vec2D_OPTICS_NeighborDist[n].push_back(make_pair(iPointIdx, fDist));
                        vec2D_OPTICS_NeighborDist[iPointIdx].push_back(make_pair(n, fDist));

                        }
                    }
                }
            }
        }
    }

    cout << "Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

}


/** Simulate sngOptics to get point and its distance in the neighbor
**/
void findCorePoints_sngDbscan_maxEps(float p_fMaxEps)
{
    vec2D_OPTICS_NeighborDist = vector< vector< pair<int, float> > > (PARAM_DATA_N, vector< pair<int, float> >());

    int iNumSamples = ceil(PARAM_SAMPLING_PROB * PARAM_DATA_N);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        VectorXf vecXn = MATRIX_X.col(n);

        // Sampling points to identify core points
        random_device rd;  // a seed source for the random number engine
        mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
        uniform_int_distribution<> distrib(0, PARAM_DATA_N - 1);

        // Compute distance from sampled Xn to all points in X
        for (int s = 0; s < iNumSamples; ++s) {
            int iPointIdx = distrib(gen);
            if (iPointIdx == n)
                continue;

            float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

            if (fDist <= p_fMaxEps) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
            {
#pragma omp critical
                {
                    vec2D_OPTICS_NeighborDist[n].push_back(make_pair(iPointIdx, fDist)); // duplicate at most twice
                    vec2D_OPTICS_NeighborDist[iPointIdx].push_back(make_pair(n, fDist)); // so vector is much better than map()
                }
            }
        }
    }

}

void test_sDbscan(int p_iTimes)
{
    cout << "Testing sDbscan" << endl;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    parDbscanIndex();

    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();

    float fBaseEps = PARAM_DBSCAN_EPS;
    float fMaxEps = fBaseEps + (PARAM_INTERNAL_TEST_UNITS - 1) * PARAM_INTERNAL_TEST_EPS_RANGE;

    findCorePoints_maxEps(fMaxEps);
    cout << "MaxEps Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    for (int i = 0; i < PARAM_INTERNAL_TEST_UNITS; ++i)
    {
        begin = chrono::steady_clock::now();

        // Get the current Eps for testing
        PARAM_DBSCAN_EPS = fBaseEps + i * PARAM_INTERNAL_TEST_EPS_RANGE;

        // Build neighborhood from pre-computed neighborhoods
        findCorePoints_PrecomputedOptics();

        cout << "Find core points time for eps = " << PARAM_DBSCAN_EPS << " is: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        begin = chrono::steady_clock::now();
        int iNumClusters = 0;
        IVector vecLabels;
        formCluster(iNumClusters, vecLabels);

        cout << "Form clusters time for eps = " << PARAM_DBSCAN_EPS << " is: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;


        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            string sFileName = PARAM_OUTPUT_FILE + int2str(p_iTimes) + "_L" + int2str(PARAM_DISTANCE) +
                            "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                            "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                            "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                            "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                            "_TopM_" + int2str(PARAM_PROJECTION_TOP_M) +
                            "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

            outputDbscan(vecLabels, sFileName);
        }
    }
}

/**
Test FWHT for embedding & CEOs
With FindCore_Asym() - we should use much larger m, e.g. m = 2 * minPts to ensure the coverage of neighborhoods
**/
void test_sDbscan_Asym(int i)
{
    cout << "Testing asymmetric sDbscan" << endl;

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();
    parDbscanIndex();

    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    // findCorePoints();
    findCorePoints_Asym();
    cout << "Find core points Asym() time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    IVector vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + int2str(i) + "_L" + int2str(PARAM_DISTANCE) +
                        "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopM_" + int2str(PARAM_PROJECTION_TOP_M) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputDbscan(vecLabels, sFileName);
	}
}

void test_uDbscan(int i)
{
    cout << "Testing uDBSCAN++" << endl;

    chrono::steady_clock::time_point begin, start;
    begin = chrono::steady_clock::now();
    start = chrono::steady_clock::now();
    findCorePoints_uDbscan();
    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    IVector vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "uDbscan time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = PARAM_OUTPUT_FILE + int2str(i) + "_L" + int2str(PARAM_DISTANCE) +
                           "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                           "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                           "_Prob_" + int2str(round(1000 * PARAM_SAMPLING_PROB));

        outputDbscan(vecLabels, sFileName);
    }
}

/**
Compute all-NN once with max eps, then extract NN and form cluster for smaller eps
**/
void test_sngDbscan(int p_iTimes)
{
    cout << "Testing sngDBSCAN" << endl;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    float fBaseEps = PARAM_DBSCAN_EPS;
    float fMaxEps = fBaseEps + (PARAM_INTERNAL_TEST_UNITS - 1) * PARAM_INTERNAL_TEST_EPS_RANGE;

    findCorePoints_sngDbscan_maxEps(fMaxEps);
    cout << "MaxEps Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    for (int i = 0; i < PARAM_INTERNAL_TEST_UNITS; ++i)
    {
        begin = chrono::steady_clock::now();

        // Get the current Eps for testing
        PARAM_DBSCAN_EPS = fBaseEps + i * PARAM_INTERNAL_TEST_EPS_RANGE;

        // Build neighborhood from pre-computed neighborhoods
        findCorePoints_PrecomputedOptics();

        cout << "Find core points time for eps = " << PARAM_DBSCAN_EPS << " is: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        begin = chrono::steady_clock::now();
        int iNumClusters = 0;
        IVector vecLabels;
        formCluster(iNumClusters, vecLabels);

        cout << "Form clusters time for eps = " << PARAM_DBSCAN_EPS << " is: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;


        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            string sFileName = PARAM_OUTPUT_FILE + int2str(p_iTimes) + "_L" + int2str(PARAM_DISTANCE) +
                               "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                               "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                               "_Prob_" + int2str(round(1000 * PARAM_SAMPLING_PROB));

            outputDbscan(vecLabels, sFileName);
        }

    }

}

/**
Call test_sng with p = 1
**/
void test_naiveDbscan()
{
    cout << "Testing exact dbscan" << endl;

    chrono::steady_clock::time_point begin, start;
    begin = chrono::steady_clock::now();
    start = chrono::steady_clock::now();
    findCorePoints_sngDbscan();
    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    IVector vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "Dbscan time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = PARAM_OUTPUT_FILE + "_L" + int2str(PARAM_DISTANCE) +
                           "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                           "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS);

        outputDbscan(vecLabels, sFileName);
    }
}
