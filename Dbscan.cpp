#include "Header.h"
#include "Utilities.h"
#include "Dbscan.h"

//#include "google/dense_hash_set"
//#include "google/dense_hash_map"


/**
Have to store the projection matrix for parallel processing.
Instead of using priority queue, we use sort() to extract top-k and top-MinPts --> SLOWER
**/
//void parDbscanIndex_Sort()
//{
//
//    /** Param for embedding **/
//    int iEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()
//    int log2Embed_D = log2(iEmbed_D);
//
//    boost::dynamic_bitset<> bitHD3_Embed;
//    bitHD3Generator(iEmbed_D * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Embed);
//
//    /** Param for random projection **/
//    MatrixXf matProject = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);
//    int log2Project = log2(PARAM_NUM_PROJECTION);
//
//    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);
//
//    boost::dynamic_bitset<> bitHD3_Proj;
//    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);
//
//    #pragma omp parallel for
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//    {
//        /**
//        L2 Embedding
//        **/
//
//        VectorXf vecPoint = VectorXf::Zero(iEmbed_D); // iEmbed_D > D
//        vecPoint.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);
//
//        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < iEmbed_D; ++d)
//            {
//                vecPoint(d) *= (2 * (int)bitHD3_Embed[r * iEmbed_D + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecPoint.data(), log2Embed_D);
//        }
//
//        // We scale with iEmbed_D since we apply HD2HD1, each need a scale of sqrt(iEmbed_D). The last HD3 is similar to Gaussian matrix.
//        // We further scale with PARAM_DATA_D since we use the standard scale N(0, 1/d^2) hence std = 1/d for kernel embedding
//        vecPoint /= (iEmbed_D * PARAM_DATA_D); // vecPoint.segment(0, 128); // 512 projections, but you need only 128 projections
//
//
//        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D);
//        vecEmbed << vecPoint.array().cos(), vecPoint.array().sin(); // no need to scale since we only use ranking
//
//        /**
//        Random projection
//        **/
//        vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
//        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;
//
//        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//            {
//                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecPoint.data(), log2Project);
//        }
//
//        // Store projection matrix for faster parallel
//        matProject.col(n) = vecPoint;
//
//        // Store potential candidate and find top-k closes and furtherest random vector
//        IVector vecRandomIdx(PARAM_NUM_PROJECTION);
//        iota(vecRandomIdx.begin(), vecRandomIdx.end(), 0); // init index = 0 to NUM_PROJECT - 1
//
//        sort(vecRandomIdx.begin(), vecRandomIdx.end(), [&](int i, int j){return vecPoint(i) > vecPoint(j);} );
//
//        // Get (sorted by projection value) top-k closest and furthest vector for each point
//        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
//        {
//            MATRIX_TOP_K(k, n) = vecRandomIdx[k];
//            MATRIX_TOP_K(k + PARAM_PROJECTION_TOP_K, n) = vecRandomIdx[PARAM_NUM_PROJECTION - k -1];
//        }
//    }
//
//    /**
//    For each random vector, getting top-MinPts as candidate
//    **/
//    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);
//
//    #pragma omp parallel for
//    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//    {
//        VectorXf vecProject = matProject.row(d);
//
//        IVector vecPointIdx(PARAM_DATA_N);
//        iota(vecPointIdx.begin(), vecPointIdx.end(), 0); // init index = 0 to NUM_PROJECT - 1
//        sort(vecPointIdx.begin(), vecPointIdx.end(), [&](int i, int j){return vecProject(i) > vecProject(j);} );
//
//        for (int k = 0; k < PARAM_DBSCAN_MINPTS; ++k)
//        {
//            // Close
//            MATRIX_TOP_MINPTS(k, d) = vecPointIdx[k];
//            // Far
//            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = vecPointIdx[PARAM_DATA_N - k - 1];
//        }
//    }
//}

/**
Have to store the projection matrix for parallel processing.
- Call HD3HD2HD1 for Fourier embedding
- Call HD3HD2HD1 for CEOs
Using priority queue to extract top-k and top-MinPts
**/
//void parDbscanIndex_L2()
//{
//
//    /** Param for embedding **/
//    int iEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()
//    int log2Embed_D = log2(iEmbed_D);
//
//    boost::dynamic_bitset<> bitHD3_Embed;
//    bitHD3Generator(iEmbed_D * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Embed);
//
//    /** Param for random projection **/
//    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);
//    int log2Project = log2(PARAM_NUM_PROJECTION);
//
//    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);
//
//    boost::dynamic_bitset<> bitHD3_Proj;
//    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);
//
//    // Only for testing
////    MATRIX_X_EMBED = MatrixXf::Zero(PARAM_KERNEL_EMBED_D, PARAM_DATA_N);
//
//    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(PARAM_NUM_THREADS);
//
//    #pragma omp parallel for
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//    {
//        /**
//        L2 Embedding
//        **/
//
//        VectorXf vecProject = VectorXf::Zero(iEmbed_D); // iEmbed_D > D
//        vecProject.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);
//
//        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < iEmbed_D; ++d)
//            {
//                vecProject(d) *= (2 * (int)bitHD3_Embed[r * iEmbed_D + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecProject.data(), log2Embed_D);
//        }
//
//        // We scale with iEmbed_D since we apply HD2HD1, each need a scale of sqrt(iEmbed_D). The last HD3 is similar to Gaussian matrix.
//        // We further scale with sigma since we use the standard scale N(0, 1/sigma^2) hence std = 1/sigma for kernel embedding
//        // K(x, y) = exp(-dist^2 / 2*sigma^2)
//        // See: https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
//        vecProject /= (iEmbed_D * PARAM_KERNEL_SIGMA); // vecPoint.segment(0, 128); // One can call O(DlogD) and get top-iEmbed_D if D > iEmbed_D
//
//        // At this stage, vecProject is identical to random projection values with normal vectors.
//        // We would need cos(), sin() and scale by 1/sqrt(iEmbed_D) to have E <f(x), f(q)> = K(x, q)
//
//        // Only for testing OPTICS plot (Euclidean space vs Embedded space)
////        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D);
////        vecEmbed << vecProject.array().cos(), vecProject.array().sin();
////        MATRIX_X_EMBED.col(n) = vecEmbed / sqrt(iEmbed_D);
//
//        /**
//        Random projection
//        **/
//        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
//        vecPoint.segment(0, iEmbed_D) = vecProject.array().cos();
//        vecPoint.segment(iEmbed_D, iEmbed_D) = vecProject.array().sin();
//        // At this stage, vecPoint = {cos(w_1 . x), sin(w_1 . x)    ... cos(w_d . x), sin(w_d . x)}
//        // We do not scale by 1/sqrt(iEmbed_D) since we only consider ordering, not value
//        // If we consider the set Si and Ri, then we must scale by 1/sqrt(iEmbed_D)
//
//        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//            {
//                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecPoint.data(), log2Project);
//        }
//
//        // Since we only consider top-k and top-m, we do not need any further scaling
//        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-m
//        MATRIX_RP.col(n) = vecPoint;
//
//        // Store potential candidate and find top-k closes and furtherest random vector
//        Min_PQ_Pair minCloseTopK;
//        Min_PQ_Pair minFarTopK;
//
//        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//        {
//            float fValue = vecPoint(d);
//
//            /**
//            For each point Xi, get top-K closest random vector and top-K furthest random vector
//            **/
//
//            // 1) Close: Using priority queue to find top-k closest vectors for each point
//            if ((int)minCloseTopK.size() < PARAM_PROJECTION_TOP_K)
//                minCloseTopK.push(IFPair(d, fValue));
//            else
//            {
//                if (fValue > minCloseTopK.top().m_fValue)
//                {
//                    minCloseTopK.pop();
//                    minCloseTopK.push(IFPair(d, fValue));
//                }
//            }
//
//            // 2) Far: Using priority queue to find top-k furthest vectors
//            if ((int)minFarTopK.size() < PARAM_PROJECTION_TOP_K)
//                minFarTopK.push(IFPair(d, -fValue));
//            else
//            {
//                if (-fValue > minFarTopK.top().m_fValue)
//                {
//                    minFarTopK.pop();
//                    minFarTopK.push(IFPair(d, -fValue));
//                }
//            }
//        }
//
//        // Get (sorted by projection value) top-k closest and furthest vector for each point
//        for (int k = PARAM_PROJECTION_TOP_K - 1; k >= 0; --k)
//        {
//            MATRIX_TOP_K(k, n) = minCloseTopK.top().m_iIndex;
//            minCloseTopK.pop();
//
//            MATRIX_TOP_K(k + PARAM_PROJECTION_TOP_K, n) = minFarTopK.top().m_iIndex;
//            minFarTopK.pop();
//        }
//    }
//
//    /**
//    For each random vector, getting 2*top-m as close and far candidates
//    **/
//    MATRIX_TOP_M = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_M, PARAM_NUM_PROJECTION);
//
//    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(PARAM_NUM_THREADS);
//    #pragma omp parallel for
//    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//    {
//        // sort(begin(matProject.col(d)), end(matProject.col(d)), [](float lhs, float rhs){return rhs > lhs});
//
//        Min_PQ_Pair minPQ_Close;
//        Min_PQ_Pair minPQ_Far;
//
//        VectorXf vecProject = MATRIX_RP.row(d); // it must be row since D x N
//
//        for (int n = 0; n < PARAM_DATA_N; ++n)
//        {
//            float fValue = vecProject(n);
//
//            // Close
//            if ((int)minPQ_Close.size() < PARAM_PROJECTION_TOP_M)
//                minPQ_Close.push(IFPair(n, fValue));
//            else
//            {
//                if (fValue > minPQ_Close.top().m_fValue)
//                {
//                    minPQ_Close.pop();
//                    minPQ_Close.push(IFPair(n, fValue));
//                }
//            }
//
//            // Far
//            if ((int)minPQ_Far.size() < PARAM_PROJECTION_TOP_M)
//                minPQ_Far.push(IFPair(n, -fValue));
//            else
//            {
//                if (-fValue > minPQ_Far.top().m_fValue)
//                {
//                    minPQ_Far.pop();
//                    minPQ_Far.push(IFPair(n, -fValue));
//                }
//            }
//        }
//
//        // Extract elements from PQ
//        for (int k = PARAM_PROJECTION_TOP_M - 1; k >= 0; --k)
//        {
//            // Close
//            MATRIX_TOP_M(k, d) = minPQ_Close.top().m_iIndex;
//            minPQ_Close.pop();
//
//            // Far
//            MATRIX_TOP_M(k + PARAM_PROJECTION_TOP_M, d) = minPQ_Far.top().m_iIndex;
//            minPQ_Far.pop();
//        }
//    }
//}

/**
Do not store Projection-Matrix due to the memory limit
Slow on threading since we have to update the same memory i.e. data structure
Using priority queue to extract top-k and top-MinPts
**/
//void seqDbscanIndex_Metric()
//{
//    vector<Min_PQ_Pair> vecPQ_Close = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());
//    vector<Min_PQ_Pair> vecPQ_Far = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());
//
//
//    /** Param for embedding **/
//    int iEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()
//
//    MatrixXf MATRIX_R = MatrixXf::Zero(iEmbed_D, PARAM_DATA_D); // EigenLib does not allow resize MatrixXf when passing by reference !
//
//    // Can be generalized for other kernel embedding
//    if (PARAM_DISTANCE == 1)
//        cauchyGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // K(x, y) = exp(-gamma * L1_dist(X, y))) where gamma = d^2
//    else if (PARAM_DISTANCE == 2)
//        gaussGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // K(x, y) = exp(-gamma * L2_dist^2(X, y))) where gamma = 1/2 sigma^2
//
//    /** Param for random projection **/
//    int log2Project = log2(PARAM_NUM_PROJECTION);
//
//    boost::dynamic_bitset<> bitHD3_Proj;
//    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);
//
//    /** Param for index **/
//    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);
//
//    #pragma omp parallel for
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//    {
//        /**
//        Random projection
//        **/
//        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D and PARAM_DATA_D
////        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;
//
//        if (PARAM_DISTANCE == 0)
//            vecPoint.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);
//        else // L1 and L2: Random Fourier Features
//        {
//            VectorXf vecProject = MATRIX_R * MATRIX_X.col(n);
//            vecPoint.segment(0, iEmbed_D) = vecProject.array().cos();
//            vecPoint.segment(iEmbed_D, iEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
//        }
//
//        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//            {
//                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecPoint.data(), log2Project);
//        }
//
//        // Store potential candidate and find top-k closes and furtherest random vector
//        Min_PQ_Pair minCloseTopK;
//        Min_PQ_Pair minFarTopK;
//
//        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//        {
//            float fValue = vecPoint(d);
//
//            /**
//            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
//            - Will process it later using projection matrix
//            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
//            **/
//
//        #pragma omp critical
//        {
//            // Single thread: (1) Close: PQ to find Top-MinPts for each random vector
//            if ((int)vecPQ_Close[d].size() < PARAM_DBSCAN_MINPTS)
//                vecPQ_Close[d].push(IFPair(n, fValue));
//            else
//            {
//                if (fValue > vecPQ_Close[d].top().m_fValue)
//                {
//                    vecPQ_Close[d].pop();
//                    vecPQ_Close[d].push(IFPair(n, fValue));
//                }
//            }
//
//            // Single thread: (1) Far: PQ to find Top-MinPts for each random vector
//            if ((int)vecPQ_Far[d].size() < PARAM_DBSCAN_MINPTS)
//                vecPQ_Far[d].push(IFPair(n, -fValue));
//            else
//            {
//                if (-fValue > vecPQ_Far[d].top().m_fValue)
//                {
//                    vecPQ_Far[d].pop();
//                    vecPQ_Far[d].push(IFPair(n, -fValue));
//                }
//            }
//        }
//
//            // (2) Close: Using priority queue to find top-k closest vectors for each point
//            if ((int)minCloseTopK.size() < PARAM_PROJECTION_TOP_K)
//                minCloseTopK.push(IFPair(d, fValue));
//            else
//            {
//                if (fValue > minCloseTopK.top().m_fValue)
//                {
//                    minCloseTopK.pop();
//                    minCloseTopK.push(IFPair(d, fValue));
//                }
//            }
//
//            // // (2) Far: Using priority queue to find top-k furthest vectors
//            if ((int)minFarTopK.size() < PARAM_PROJECTION_TOP_K)
//                minFarTopK.push(IFPair(d, -fValue));
//            else
//            {
//                if (-fValue > minFarTopK.top().m_fValue)
//                {
//                    minFarTopK.pop();
//                    minFarTopK.push(IFPair(d, -fValue));
//                }
//            }
//        }
//
//        // Get (sorted by projection value) top-k closest and furthest vector for each point
//        for (int k = PARAM_PROJECTION_TOP_K - 1; k >= 0; --k)
//        {
//            MATRIX_TOP_K(k, n) = minCloseTopK.top().m_iIndex;
//            minCloseTopK.pop();
//
//            MATRIX_TOP_K(k + PARAM_PROJECTION_TOP_K, n) = minFarTopK.top().m_iIndex;
//            minFarTopK.pop();
//        }
//
//    }
//
//    /**
//    For each random vector, getting 2*top-MinPts as close and far candidates
//    **/
//    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);
//
//    #pragma omp parallel for
//    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//    {
//        for (int k = PARAM_DBSCAN_MINPTS - 1; k >= 0; --k)
//        {
//            // Close
//            MATRIX_TOP_MINPTS(k, d) = vecPQ_Close[d].top().m_iIndex;
//            vecPQ_Close[d].pop();
//
//            // Far
//            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = vecPQ_Far[d].top().m_iIndex;
//            vecPQ_Far[d].pop();
//        }
//    }
//}

/**
Have to store the projection matrix for parallel processing.
Using priority queue to extract top-k and top-MinPts
**/
//void parDbscanIndex_Metric()
//{
//
//    /** Param for embedding **/
//    int iEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()
//
//    MatrixXf MATRIX_R = MatrixXf::Zero(iEmbed_D, PARAM_DATA_D); // EigenLib does not allow resize MatrixXf when passing by reference !
//
//    // See: https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
//    if (PARAM_DISTANCE == 1)
//        cauchyGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // K(x, y) = exp(-gamma * L1_dist(X, y))) where gamma = 1/sigma
//    else if (PARAM_DISTANCE == 2)
//        gaussGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // std = 1/sigma, K(x, y) = exp(-gamma * L2_dist^2(X, y))) where gamma = 1/2 sigma^2
//
//    /** Param for random projection **/
//    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);
//
//    int log2Project = log2(PARAM_NUM_PROJECTION);
//    boost::dynamic_bitset<> bitHD3_Proj;
//    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);
//
//    /** Param for index **/
//    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);
//
//    #pragma omp parallel for
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//    {
//        /**
//        Random projection
//        **/
//        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
////        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;
//
//        if (PARAM_DISTANCE == 0)
//            vecPoint.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);
//        else // L1 and L2: Random Fourier Features
//        {
//            VectorXf vecProject = MATRIX_R * MATRIX_X.col(n);
//            vecPoint.segment(0, iEmbed_D) = vecProject.array().cos();
//            vecPoint.segment(iEmbed_D, iEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
//        }
//
//
//        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//            {
//                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecPoint.data(), log2Project);
//        }
//
//        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
//        MATRIX_RP.col(n) = vecPoint;
//
//        // Store potential candidate and find top-k closes and furtherest random vector
//        Min_PQ_Pair minCloseTopK;
//        Min_PQ_Pair minFarTopK;
//
//        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//        {
//            float fValue = vecPoint(d);
//
//            /**
//            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
//            - Will process it later using projection matrix
//            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
//            **/
//
//            // (2) Close: Using priority queue to find top-k closest vectors for each point
//            if ((int)minCloseTopK.size() < PARAM_PROJECTION_TOP_K)
//                minCloseTopK.push(IFPair(d, fValue));
//            else
//            {
//                if (fValue > minCloseTopK.top().m_fValue)
//                {
//                    minCloseTopK.pop();
//                    minCloseTopK.push(IFPair(d, fValue));
//                }
//            }
//
//            // // (2) Far: Using priority queue to find top-k furthest vectors
//            if ((int)minFarTopK.size() < PARAM_PROJECTION_TOP_K)
//                minFarTopK.push(IFPair(d, -fValue));
//            else
//            {
//                if (-fValue > minFarTopK.top().m_fValue)
//                {
//                    minFarTopK.pop();
//                    minFarTopK.push(IFPair(d, -fValue));
//                }
//            }
//        }
//
//        // Get (sorted by projection value) top-k closest and furthest vector for each point
//        for (int k = PARAM_PROJECTION_TOP_K - 1; k >= 0; --k)
//        {
//            MATRIX_TOP_K(k, n) = minCloseTopK.top().m_iIndex;
//            minCloseTopK.pop();
//
//            MATRIX_TOP_K(k + PARAM_PROJECTION_TOP_K, n) = minFarTopK.top().m_iIndex;
//            minFarTopK.pop();
//        }
//
//    }
//
//    /**
//    For each random vector, getting 2*top-MinPts as close and far candidates
//    **/
//    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);
//
//    #pragma omp parallel for
//    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//    {
//        // sort(begin(matProject.col(d)), end(matProject.col(d)), [](float lhs, float rhs){return rhs > lhs});
//
//        Min_PQ_Pair minPQ_Close;
//        Min_PQ_Pair minPQ_Far;
//
//        VectorXf vecProject = MATRIX_RP.row(d); // it must be row since D x N
//
//        for (int n = 0; n < PARAM_DATA_N; ++n)
//        {
//            float fValue = vecProject(n);
//
//            if ((int)minPQ_Close.size() < PARAM_DBSCAN_MINPTS)
//                minPQ_Close.push(IFPair(n, fValue));
//            else
//            {
//                if (fValue > minPQ_Close.top().m_fValue)
//                {
//                    minPQ_Close.pop();
//                    minPQ_Close.push(IFPair(n, fValue));
//                }
//            }
//
//            // Single thread: (1) Far: PQ to find Top-MinPts for each random vector
//            if ((int)minPQ_Far.size() < PARAM_DBSCAN_MINPTS)
//                minPQ_Far.push(IFPair(n, -fValue));
//            else
//            {
//                if (-fValue > minPQ_Far.top().m_fValue)
//                {
//                    minPQ_Far.pop();
//                    minPQ_Far.push(IFPair(n, -fValue));
//                }
//            }
//        }
//
//        for (int k = PARAM_DBSCAN_MINPTS - 1; k >= 0; --k)
//        {
//            // Close
//            MATRIX_TOP_MINPTS(k, d) = minPQ_Close.top().m_iIndex;
//            minPQ_Close.pop();
//
//            // Far
//            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = minPQ_Far.top().m_iIndex;
//            minPQ_Far.pop();
//        }
//    }
//}

/**
Have to store the projection matrix for parallel processing.
Using priority queue to extract top-k and top-MinPts

Only support ChiSquare and Johnson Shannon divergences
**/
//void parDbscanIndex_NonMetric()
//{
//    /** Param for random projection **/
//    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);
//
//    int log2Project = log2(PARAM_NUM_PROJECTION);
//    boost::dynamic_bitset<> bitHD3_Proj;
//    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);
//
//    /** Param for index **/
//    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);
//
//    #pragma omp parallel for
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//    {
//        /**
//        Deterministic Additive Kernel feature
//        **/
//        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D); // iEmbed_D > D
//        if (PARAM_DISTANCE == 3)
//            embedChiSquare(MATRIX_X.col(n), vecEmbed);
//        else if (PARAM_DISTANCE == 4)
//            embedJS(MATRIX_X.col(n), vecEmbed);
//
//        /**
//        Random projection
//        **/
//        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
//        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;
//
//        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
//        {
//            // Component-wise multiplication with a random sign
//            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//            {
//                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
//            }
//
//            // Multiple with Hadamard matrix by calling FWHT transform
//            fht_float(vecPoint.data(), log2Project);
//        }
//
//        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
//        MATRIX_RP.col(n) = vecPoint;
//
//        // Store potential candidate and find top-k closes and furtherest random vector
//        Min_PQ_Pair minCloseTopK;
//        Min_PQ_Pair minFarTopK;
//
//        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//        {
//            float fValue = vecPoint(d);
//
//            /**
//            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
//            - Will process it later using projection matrix
//            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
//            **/
//
//            // (2) Close: Using priority queue to find top-k closest vectors for each point
//            if ((int)minCloseTopK.size() < PARAM_PROJECTION_TOP_K)
//                minCloseTopK.push(IFPair(d, fValue));
//            else
//            {
//                if (fValue > minCloseTopK.top().m_fValue)
//                {
//                    minCloseTopK.pop();
//                    minCloseTopK.push(IFPair(d, fValue));
//                }
//            }
//
//            // // (2) Far: Using priority queue to find top-k furthest vectors
//            if ((int)minFarTopK.size() < PARAM_PROJECTION_TOP_K)
//                minFarTopK.push(IFPair(d, -fValue));
//            else
//            {
//                if (-fValue > minFarTopK.top().m_fValue)
//                {
//                    minFarTopK.pop();
//                    minFarTopK.push(IFPair(d, -fValue));
//                }
//            }
//        }
//
//        // Get (sorted by projection value) top-k closest and furthest vector for each point
//        for (int k = PARAM_PROJECTION_TOP_K - 1; k >= 0; --k)
//        {
//            MATRIX_TOP_K(k, n) = minCloseTopK.top().m_iIndex;
//            minCloseTopK.pop();
//
//            MATRIX_TOP_K(k + PARAM_PROJECTION_TOP_K, n) = minFarTopK.top().m_iIndex;
//            minFarTopK.pop();
//        }
//
//    }
//
//    /**
//    For each random vector, getting 2*top-MinPts as close and far candidates
//    **/
//    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);
//
//    #pragma omp parallel for
//    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
//    {
//        // sort(begin(matProject.col(d)), end(matProject.col(d)), [](float lhs, float rhs){return rhs > lhs});
//
//        Min_PQ_Pair minPQ_Close;
//        Min_PQ_Pair minPQ_Far;
//
//        VectorXf vecProject = MATRIX_RP.row(d); // it must be row since D x N
//
//        for (int n = 0; n < PARAM_DATA_N; ++n)
//        {
//            float fValue = vecProject(n);
//
//            if ((int)minPQ_Close.size() < PARAM_DBSCAN_MINPTS)
//                minPQ_Close.push(IFPair(n, fValue));
//            else
//            {
//                if (fValue > minPQ_Close.top().m_fValue)
//                {
//                    minPQ_Close.pop();
//                    minPQ_Close.push(IFPair(n, fValue));
//                }
//            }
//
//            // Single thread: (1) Far: PQ to find Top-MinPts for each random vector
//            if ((int)minPQ_Far.size() < PARAM_DBSCAN_MINPTS)
//                minPQ_Far.push(IFPair(n, -fValue));
//            else
//            {
//                if (-fValue > minPQ_Far.top().m_fValue)
//                {
//                    minPQ_Far.pop();
//                    minPQ_Far.push(IFPair(n, -fValue));
//                }
//            }
//        }
//
//        for (int k = PARAM_DBSCAN_MINPTS - 1; k >= 0; --k)
//        {
//            // Close
//            MATRIX_TOP_MINPTS(k, d) = minPQ_Close.top().m_iIndex;
//            minPQ_Close.pop();
//
//            // Far
//            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = minPQ_Far.top().m_iIndex;
//            minPQ_Far.pop();
//        }
//    }
//}

/**
Store the projection matrix Rx for parallel processing.
Store binary bits for FHWT transform, especially we have to use them for sDBSCAN-1NN

Using priority queue to extract top-k and top-m

For each point Xi, compute its dot product, extract top-k close/far random vectors
For each random vector Ri, reuse dot product matrix, extract top-k close/far points
**/
void parDbscanIndex()
{
    /** Param for embedding L1 and L2 **/
    int iFourierEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()

    // MatrixXf MATRIX_R; // EigenLib does not allow resize MatrixXf when passing by reference !

    // See: https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
    if (PARAM_DISTANCE == 1)
        MATRIX_R = cauchyGenerator(iFourierEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA); // K(x, y) = exp(-gamma * L1_dist(X, y))) where gamma = 1/sigma
    else if (PARAM_DISTANCE == 2)
        MATRIX_R = gaussGenerator(iFourierEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA); // std = 1/sigma, K(x, y) = exp(-gamma * L2_dist^2(X, y))) where gamma = 1/2 sigma^2

    /** Param for random projection **/
    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);

    int log2Project = log2(PARAM_INTERNAL_FWHT_PROJECTION);
    //boost::dynamic_bitset<> bitHD3_Proj;
    bitHD3Generator(PARAM_INTERNAL_FWHT_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);

    /** Param for index **/
    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);

    /**
    Parallel for each the point Xi: (1) Compute and store dot product, and (2) Extract top-k close/far random vectors
    **/
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        /**
        Random embedding
        **/
        VectorXf vecX = MATRIX_X.col(n);
        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D); // PARAM_KERNEL_EMBED_D >= D

        if (PARAM_DISTANCE == 0)
            vecEmbed.segment(0, PARAM_DATA_D) = vecX;
        else if ((PARAM_DISTANCE == 1) || (PARAM_DISTANCE == 2))
        {
            VectorXf vecProject = MATRIX_R * vecX;
            vecEmbed.segment(0, iFourierEmbed_D) = vecProject.array().cos();
            vecEmbed.segment(iFourierEmbed_D, iFourierEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
        }
        else if (PARAM_DISTANCE == 3)
            embedChiSquare(vecX, vecEmbed);
        else if (PARAM_DISTANCE == 4)
            embedJS(vecX, vecEmbed);

        /**
        Random projection
        **/

        VectorXf vecRotation = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecRotation.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
            {
                vecRotation(d) *= (2 * (int)bitHD3_Proj[r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecRotation.data(), log2Project);
        }

        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
        MATRIX_RP.col(n) = vecRotation.segment(0, PARAM_NUM_PROJECTION); // only get up to #numProj

        /**
        Extract top-k closes and furtherest random vectors
        **/

        Min_PQ_Pair minCloseTopK;
        Min_PQ_Pair minFarTopK;

        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
        {
            float fValue = vecRotation(d); // take the value up to numProj

            /**
            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
            - Will process it later using projection matrix
            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
            **/

            // (1) Close: Using priority queue to find top-k closest vectors for each point
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

            // (2) Far: Using priority queue to find top-k furthest vectors
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


    /**
    For each random vector, extract top-m close/far data points
    **/
    MATRIX_TOP_M = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_M, PARAM_NUM_PROJECTION);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);

    /**
    Parallel for each random vector, getting 2*top-m as close and far candidates
    **/
    #pragma omp parallel for
    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
    {
        // sort(begin(matProject.col(d)), end(matProject.col(d)), [](float lhs, float rhs){return rhs > lhs});

        Min_PQ_Pair minPQ_Close;
        Min_PQ_Pair minPQ_Far;

        VectorXf vecProject = MATRIX_RP.row(d); // it must be row since D x N

        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            float fValue = vecProject(n);

            // Close
            if ((int)minPQ_Close.size() < PARAM_PROJECTION_TOP_M)
                minPQ_Close.push(IFPair(n, fValue));
            else
            {
                if (fValue > minPQ_Close.top().m_fValue)
                {
                    minPQ_Close.pop();
                    minPQ_Close.push(IFPair(n, fValue));
                }
            }

            // Far
            if ((int)minPQ_Far.size() < PARAM_PROJECTION_TOP_M)
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

        for (int k = PARAM_PROJECTION_TOP_M - 1; k >= 0; --k)
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
Faster and more memory-efficient than parDbscanIndex() if using single-thread

Do not store Projection Matrix
Slow on threading since we have to update the same memory containing the data structure
Using priority queue to extract top-k and top-m
**/
void seqDbscanIndex()
{
    vector<Min_PQ_Pair> vecPQ_Close = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());
    vector<Min_PQ_Pair> vecPQ_Far = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());

    /** Param for embedding **/
    int iFourierEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()

    MatrixXf MATRIX_R; // EigenLib does not allow resize MatrixXf when passing by reference !

    // Can be generalized for other kernel embedding
    if (PARAM_DISTANCE == 1)
        MATRIX_R = cauchyGenerator(iFourierEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA); // K(x, y) = exp(-gamma * L1_dist(X, y))) where gamma = d^2
    else if (PARAM_DISTANCE == 2)
        MATRIX_R = gaussGenerator(iFourierEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA); // K(x, y) = exp(-gamma * L2_dist^2(X, y))) where gamma = 1/2 sigma^2

    /** Param for random projection **/
    int log2Project = log2(PARAM_INTERNAL_FWHT_PROJECTION);

    boost::dynamic_bitset<> bitHD3_Proj;
    bitHD3Generator(PARAM_INTERNAL_FWHT_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);

    /** Param for index **/
    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        /**
        Random embedding
        **/
        VectorXf vecX = MATRIX_X.col(n);
        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D); // PARAM_KERNEL_EMBED_D >= D

        if (PARAM_DISTANCE == 0)
            vecEmbed.segment(0, PARAM_DATA_D) = vecX;
        else if ((PARAM_DISTANCE == 1) || (PARAM_DISTANCE == 2))
        {
            VectorXf vecProject = MATRIX_R * vecX;
            vecEmbed.segment(0, iFourierEmbed_D) = vecProject.array().cos();
            vecEmbed.segment(iFourierEmbed_D, iFourierEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
        }
        else if (PARAM_DISTANCE == 3)
            embedChiSquare(vecX, vecEmbed);
        else if (PARAM_DISTANCE == 4)
            embedJS(vecX, vecEmbed);

        /**
        Random projection
        **/
        VectorXf vecRotation = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecRotation.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
            {
                vecRotation(d) *= (2 * (int)bitHD3_Proj[r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecRotation.data(), log2Project);
        }

        // Store potential candidate and find top-k closes and furtherest random vector
        Min_PQ_Pair minCloseTopK;
        Min_PQ_Pair minFarTopK;

        for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
        {
            float fValue = vecRotation(d);

            /**
            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
            - Will process it later using projection matrix
            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
            **/

        // need to update PQ for many random vectors as the same time, so it is not multi-thread-friendly
        #pragma omp critical
        {
            // Single thread: (1) Close: PQ to find Top-m for each random vector
            if ((int)vecPQ_Close[d].size() < PARAM_PROJECTION_TOP_M)
                vecPQ_Close[d].push(IFPair(n, fValue));
            else
            {
                if (fValue > vecPQ_Close[d].top().m_fValue)
                {
                    vecPQ_Close[d].pop();
                    vecPQ_Close[d].push(IFPair(n, fValue));
                }
            }

            // Single thread: (1) Far: PQ to find Top-m for each random vector
            if ((int)vecPQ_Far[d].size() < PARAM_PROJECTION_TOP_M)
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

    /**
    For each random vector, getting 2*top-MinPts as close and far candidates
    **/
    MATRIX_TOP_M = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_M, PARAM_NUM_PROJECTION);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
    {
        for (int k = PARAM_PROJECTION_TOP_M - 1; k >= 0; --k)
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
vec2D_DBSCAN_Neighbor: For each point, store its true neighborhood (within radius eps)
- Note that for each point X, if dist(X, Y) < eps, we insert Y into X's neighborhood - using vector. Note that set() is rather slow as the size of vector is O(km)
- This approach is very fast (multi-thread friendly) but losing that X is also Y's neighborhood.
This affects the symmetric property of the NN graph, which will degrade the clustering time and accuracy.

bit_CORE_POINTS: Store a bit vector presenting for core point
**/
void findCorePoints_Asym()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    //TODO: If single thread, then we can improve if we store (X1, X2) s.t. <X1,X2> >= threshold

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get top-k closese/furthest vectors
        VectorXf vecXn = MATRIX_X.col(n);
        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far

        IVector vecNeighborhood;

//        unordered_set<int> approxNeighbor; MinPts * 2K * 32 bits >> N
        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N); // As we check several random vectors, there might be duplicates

        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
        {
            // Closest
            int Ri = vecTopK(k);

            // Close
            for (int i = 0; i < PARAM_PROJECTION_TOP_M; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_M(i, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
                //if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
                    //approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    // Can speed up if storing global precomputed matrix as a set((small_ID, large_ID) if have enough RAM
                    // It is not thread-friendly as it has to update this matrix
                    if (fDist <= PARAM_DBSCAN_EPS)
                        vecNeighborhood.push_back(iPointIdx);
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

                // Check already compute distance
                //if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
                    //approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    // Can speed up if storing global precomputed matrix as a set((small_ID, large_ID) if have enough RAM
                    // It is not thread-friendly as it has to update this matrix
                    if (fDist <= PARAM_DBSCAN_EPS)
                        vecNeighborhood.push_back(iPointIdx);
                }
            }
        }


        if ((int)vecNeighborhood.size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[n] = 1;

    //            if ( n < 1000 )
    //                cout << vecNeighborhood.size() << endl;

            vec2D_DBSCAN_Neighbor[n] = vecNeighborhood;
        }

    }

    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

/**
vec2D_DBSCAN_Neighbor: For each point, store its true neighborhood (within radius eps)
- We use a vector<set> to store neighborhood - for dist(x, y) < eps, update both x's and y's neighborhoods
- which is not multi-thread friendly
bit_CORE_POINTS: Store a bit vector presenting for core point
**/
void findCorePoints()
{
    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    //    vector<unordered_set<int>> set2D_DBSCAN_Neighbor(PARAM_DATA_N, unordered_set<int>()); // Space and time overheads are significant compared to vector

    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    //TODO: If single thread, then we can improve if we store (X1, X2) s.t. <X1,X2> >= threshold

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get top-k closese/furthest vectors
//        VectorXf vecXn = MATRIX_X_EMBED.col(n);
        VectorXf vecXn = MATRIX_X.col(n);

        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far


        /**
        Choice of data structure: We are using bitSet of size N bits
        If using vector: It will be faster for 2k * m * 32 (4 bytes) << N (bits)
        If using unorder_set or set: It will be faster for 2k * m * 36 * 8 (36 bytes) << N (bits)
        **/
//        IVector vecNeighborhood;
//        unordered_set<int> approxNeighbor;
//        set<int> approxNeighbor;
        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N); // 8M bits = 1MB

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

                // Check already compute distance
//                if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
//                    approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS)
                    {
                        #pragma omp critical
                        {
//                        set2D_DBSCAN_Neighbor[n].insert(iPointIdx);  // set is very slow and take much memory
//                        set2D_DBSCAN_Neighbor[iPointIdx].insert(n);

                        vec2D_DBSCAN_Neighbor[n].push_back(iPointIdx); // allow duplicate, at most double so vector is much faster than set()
                        vec2D_DBSCAN_Neighbor[iPointIdx].push_back(n); // e.g. 1 = {3, 5}, and 3 = {1 6}
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

                // Check already compute distance
//                if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
//                    approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS)
                    {
                        #pragma omp critical
                        {
//                        set2D_DBSCAN_Neighbor[n].insert(iPointIdx);  // set is very slow, and the overhead is large
//                        set2D_DBSCAN_Neighbor[iPointIdx].insert(n);

                        vec2D_DBSCAN_Neighbor[n].push_back(iPointIdx);
                        vec2D_DBSCAN_Neighbor[iPointIdx].push_back(n);
                        }
                    }
                }
            }
        }

//        cout << "Number of used distances for the point of " << n << " is: " << approxNeighbor.size() << endl;
    }

    cout << "Find neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Using the set to clear duplicates
        unordered_set<int> setNeighbor(vec2D_DBSCAN_Neighbor[n].begin(), vec2D_DBSCAN_Neighbor[n].end());
        vec2D_DBSCAN_Neighbor[n].clear();

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
This is a naive implementation of uDBSCAN++.
Note that DBSCAN++ uses KD Tree to speed up 1NN heuristic for cluster propagation.

vec2D_DBSCAN_Neighbor: For each sampled point, store its true neighborhood (within radius eps)
bit_CORE_POINTS: Store a bit vector presenting for core point
**/
void findCorePoints_uDbscan()
{
    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    int iNumSamples = ceil(1.0 * PARAM_DATA_N * PARAM_SAMPLING_PROB);

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    // Sampling points to identify core points
    random_device rd;  // a seed source for the random number engine
    mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> distrib(0, PARAM_DATA_N - 1);

    // Use distrib to transform the random unsigned int
    // generated by gen into an int in [0, iNumSamples]
    IVector vecSamples(iNumSamples);
    for (int n = 0; n < iNumSamples; ++n)
        vecSamples[n] = distrib(gen);


    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int s = 0; s < iNumSamples; ++s)
    {
        // Get top-k closese/furthest vectors
        int iPointIdx = vecSamples[s];
        VectorXf vecXs = MATRIX_X.col(iPointIdx);
        IVector vecNeighbor;

        // Compute distance from sampled Xn to all points in X
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            if (n == iPointIdx)
                continue;

            float fDist = computeDist(vecXs, MATRIX_X.col(n));

            if (fDist <= PARAM_DBSCAN_EPS)
                vecNeighbor.push_back(n);
        }

        if ((int)vecNeighbor.size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[iPointIdx] = 1;

//            if ( n < 1000 )
//                cout << setNeighbor.size() << endl;

            // Only need neighborhood if it is core point
            vec2D_DBSCAN_Neighbor[iPointIdx] = vecNeighbor; //.insert(vec2D_DBSCAN_Neighbor[iPointIdx].end(), vecNeighbor.begin(), vecNeighbor.end());
        }
    }

    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

/**
This is a multi-thread friendly implementation of sngDBSCAN, much faster than the original sngDBSCAN on large data.

vec2D_DBSCAN_Neighbor: For each sampled point, store its true neighborhood (within radius eps)
bit_CORE_POINTS: Store a bit vector presenting for core point
**/
void findCorePoints_sngDbscan()
{
    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    int iNumSamples = ceil(1.0 * PARAM_DATA_N * PARAM_SAMPLING_PROB);

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        VectorXf vecXn = MATRIX_X.col(n);

        if (PARAM_SAMPLING_PROB < 1.0)
        {
            // Sampling points to identify core points
            random_device rd;  // a seed source for the random number engine
            mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
            uniform_int_distribution<> distrib(0, PARAM_DATA_N - 1);

            // Compute distance from sampled Xn to all points in X
            for (int s = 0; s < iNumSamples; ++s)
            {
                int iPointIdx = distrib(gen);
                if (iPointIdx == n)
                    continue;

                float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                if (fDist <= PARAM_DBSCAN_EPS) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
                {
                    #pragma omp critical
                    {
                        vec2D_DBSCAN_Neighbor[n].push_back(iPointIdx);
                        vec2D_DBSCAN_Neighbor[iPointIdx].push_back(n);
                    }
                }
            }
        }
        else // exact Dbscan as sampling-prob = 1
        {
            // Compute distance from sampled Xn to all points in X
            for (int s = 0; s < PARAM_DATA_N; ++s)
            {
                if (s == n)
                    continue;

                float fDist = computeDist(vecXn, MATRIX_X.col(s));

                if (fDist <= PARAM_DBSCAN_EPS) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
                    vec2D_DBSCAN_Neighbor[n].push_back(s);
            }
        }
    }

    omp_set_num_threads(PARAM_NUM_THREADS);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // We do not need to use the set to clear duplicate as the chance of getting duplicate by sampling is very tiny
        if ((int)vec2D_DBSCAN_Neighbor[n].size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[n] = 1;
        }
        else if (PARAM_DBSCAN_CLUSTER_NOISE != 3)
            vec2D_DBSCAN_Neighbor[n].clear();
    }

    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

/**
We connect core points first, then label its neighborhood later

If the symmetric property of the graph is not preserved, then forming clustering will be very slow

Current implementation is not the final solution!
**/
void formCluster_Asym(int & p_iNumClusters, IVector & p_vecLabels)
{
    p_vecLabels = IVector(PARAM_DATA_N, -1); //noise = -1
    p_iNumClusters = 0;

    int iOldClusterID = 0;
    int iNewClusterID = -1; // The cluster ID starts from 0

    // Fast enough so might not need multi-threading
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Skip: (1) border point with assigned labels, (2) non-core points, (3) core-point with assigned labels

        if ( (!bit_CORE_POINTS[n]) || (p_vecLabels[n] != -1) )  //only consider core points and point without labels
            continue;


        /** Note: There is a tradeoff between multi-threading speedup and clustering time
        TODO: Better parallel forming cluster()
        - If call findCorePoints_Asym(), then we have to consider several seedSet and it will connect to the clustered point before
        Therefore, forming cluster takes time
        - If call findCorePoints(), similar points are inserted into both arrays, then clusters tend to connect each other well.
        Therefore, forming clustering is very fast
        One can test with cout << n << endl;

        **/

        // Always start from the core points without any labels

        iNewClusterID++;

        unordered_set<int> seedSet; //seedSet only contains core points
        seedSet.insert(n);

        boost::dynamic_bitset<> connectedPoints(PARAM_DATA_N);
        connectedPoints[n] = 1;

        // Connecting components in a cluster
        bool bConnected = false;

        // unorder_set<int> is slow if there are many core points - google::dense_hash_set might be faster
        // however, clustering is very fast compared to computing core points and its neighborhood - no need to improve at this stage
//        while (seedSet.count() > 0)
        while (seedSet.size() > 0)
        {
            int Xi = *seedSet.begin();
            seedSet.erase(seedSet.begin());

//            int Xi = seedSet.find_first();
//            seedSet[Xi] = 0;

            // Get neighborhood of the core Xi
            IVector Xi_neighborhood = vec2D_DBSCAN_Neighbor[Xi];

            // Find the core points, connect them together, and check if one of them already has assigned labels
            for (auto const& Xj : Xi_neighborhood)
            {
                // Find the core points first
                if (bit_CORE_POINTS[Xj])
                {
//                    if (connectedCore.find(Xj) == connectedCore.end())
                    if (! connectedPoints[Xj])
                    {
                        connectedPoints[Xj] = 1;

                        if (p_vecLabels[Xj] == -1) // only insert into seedSet for non-labeled core; otherwise used the label of this labeled core
                            seedSet.insert(Xj);
//                            seedSet[Xj] = 1;

                        /** This is the key difference with asymmetric graph
                        If NN(x) = {y, z} but NN(y) and NN(z) do not contain x (not symmetric)
                        what if label(y) != label(z) ?

                        Current slow solution: Updating all label(y) and label(z) with label(x)
                        Have to update labels of many points --> Very slow

                        TODO: Lazy update cluster label
                        - If 1 connects to 2, keep info 1-2
                        - If 2 connects to 4, keep info 2-4
                        Then finally, merge all connected components together and update labels
                        **/
                        // Check the core in neighborhood, whose label is already assigned
                        if ( (! bConnected) && (p_vecLabels[Xj] > -1) && (p_vecLabels[Xj] < iNewClusterID))
                        {
                            iOldClusterID = p_vecLabels[Xj];
                            iNewClusterID--; // since we do not need a new cluster ID
                            bConnected = true;
                        }
                    }
                }
                else
                    connectedPoints[Xj] = 1;
            }
        }

        int iClusterID = iNewClusterID;
        if (bConnected)
            iClusterID = iOldClusterID;


//        cout << "Connecting component time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        // Assign the cluster label
//        begin = chrono::steady_clock::now();
//        for (auto const& Xj : connectedCore)
        size_t Xj = connectedPoints.find_first();
        while (Xj != boost::dynamic_bitset<>::npos)
        {
            // Get all neighborhood
            if (p_vecLabels[Xj] == -1)
                p_vecLabels[Xj] = iClusterID; // assign core

            // This code increases the accuracy for the non-core points Xi which has non-core points neighbor
            // It might be suitable for the data set with many noise
//            IVector Xj_neighborhood = vec2D_DBSCAN_Neighbor[Xj];
//            for (auto const& Xk : Xj_neighborhood)
//            {
//                if (p_vecLabels[Xk] == -1)
//                    p_vecLabels[Xk] = iClusterID;
//            }

            Xj = connectedPoints.find_next(Xj);
        }

        // Update the largest cluster ID
        if (iClusterID > p_iNumClusters)
            p_iNumClusters = iClusterID;

//        cout << "Labelling components time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    }

    p_iNumClusters = p_iNumClusters + 2; // increase by 2 since we count -1 as noisy cluster, and cluster ID start from 0

    cout << "Number of clusters: " << p_iNumClusters << endl;
}

/**
We connect core points first, then label its neighborhood later
**/
void formCluster(int & p_iNumClusters, IVector & p_vecLabels)
{
    p_vecLabels = IVector(PARAM_DATA_N, -1); //noise = -1
    p_iNumClusters = 0;

    int iNewClusterID = -1; // The cluster ID starts from 0

    // Fast enough so might not need multi-threading
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Skip: (1) core-point with assigned labels, (2) non-core points

        if ( (!bit_CORE_POINTS[n]) || (p_vecLabels[n] != -1) )  //only consider core points and point without labels
            continue;


        /** Note: There is a tradeoff between multi-threading speedup and clustering time
        TODO: Better parallel forming cluster()

        - If call findCorePoints_Asym(), then we have to consider several seedSet and it will connect to the clustered point before
        Therefore, forming cluster takes time
        - If call findCorePoints(), similar points are inserted into both arrays, then clusters tend to connect each other well.
        Therefore, forming clustering is very fast
        One can test with cout << n << endl;

        **/

        // Always start from the core points without any labels

        iNewClusterID++;

        unordered_set<int> seedSet; //seedSet only contains core points
        seedSet.insert(n);

        /**
        # core points detected for each connected component is small (due to the approximation), so unorder_set() is fine.
        However, if this number is large, bitSet might be a good choice
//        boost::dynamic_bitset<> seedSet(PARAM_DATA_N);
//        seedSet[n] = 1;
        **/

        /**
        connectedPoints tend to have many points (e.g. n/20), then bitSet is better than unordered_set()
//        unordered_set<int> connectedPoints;
//        connectedPoints.insert(n);
        **/

        boost::dynamic_bitset<> connectedPoints(PARAM_DATA_N);
        connectedPoints[n] = 1;

        // unordered_set<int> is slow if there are many core points - google::dense_hash_set or bitSet might be faster
        // however, clustering is very fast compared to computing core points and its neighborhood - no need to improve at this stage
//        while (seedSet.count() > 0)
        while (seedSet.size() > 0)
        {
            int Xi = *seedSet.begin();
            seedSet.erase(seedSet.begin());

//            int Xi = seedSet.find_first();
//            seedSet[Xi] = 0;

            // Get neighborhood of the core Xi
            IVector Xi_neighborhood = vec2D_DBSCAN_Neighbor[Xi];

            // Find the core points, connect them together, and check if one of them already has assigned labels
            for (auto const& Xj : Xi_neighborhood)
            {
                // If core point and not connected, then add into seedSet
                if (bit_CORE_POINTS[Xj])
                {
//                    if (connectedCore.find(Xj) == connectedCore.end())
                    if (! connectedPoints[Xj])
                    {
                        connectedPoints[Xj] = 1;

                        if (p_vecLabels[Xj] == -1) // only insert into seedSet for non-labeled core; otherwise used the label of this labeled core
                            seedSet.insert(Xj);
//                            seedSet[Xj] = 1;
                    }
                }
                else
                    connectedPoints[Xj] = 1;
            }
        }

//        cout << "Connecting component time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        // Assign the cluster label
//        begin = chrono::steady_clock::now();
//        for (auto const& Xj : connectedCore)
        size_t Xj = connectedPoints.find_first();
        while (Xj != boost::dynamic_bitset<>::npos)
        {
            // Get all neighborhood
            if (p_vecLabels[Xj] == -1)
                p_vecLabels[Xj] = iNewClusterID; // assign core

            // This code increases the accuracy for the non-core points Xi which has non-core points neighbor
            // It might be suitable for the data set with many noise
//            IVector Xj_neighborhood = vec2D_DBSCAN_Neighbor[Xj];
//            for (auto const& Xk : Xj_neighborhood)
//            {
//                if (p_vecLabels[Xk] == -1)
//                    p_vecLabels[Xk] = iClusterID;
//            }

            Xj = connectedPoints.find_next(Xj);
        }

        // Update the largest cluster ID
        p_iNumClusters = iNewClusterID;

//        cout << "Labelling components time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    }

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    // Assign cluster label to noisy point -- it will increase NMI if comparing with the class labels
    if (PARAM_DBSCAN_CLUSTER_NOISE > 0)
        clusterNoise(p_vecLabels, PARAM_DBSCAN_CLUSTER_NOISE);

    cout << "Clustering noisy point time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;


    p_iNumClusters = p_iNumClusters + 2; // increase by 2 since we count -1 as noisy cluster, and cluster ID start from 0

    cout << "Number of clusters: " << p_iNumClusters << endl;
}

/**
Cluster noisy points to increase NMI when comparing with class labels
There are several methods:
1) Assign label to random vectors, and use labels of random vectors
2) Sampling 0.01n core points to compute build 1NN classifier
3) sngDBSCAN: Assign core points' labels to all of its neighborhoods
4) Sampling 0.01n core points, use CEOs to approximate 1NN classifier
**/
void clusterNoise(IVector& p_vecLabels, int p_iOption)
{

    // Counting noisy points
    IVector vecNoise;
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        if (p_vecLabels[n] == -1)
            vecNoise.push_back(n);
    }

    cout << "Number of noisy points: " << vecNoise.size() << endl;

    // assign label for random vectors
    switch (p_iOption)
    {
        // Assign labels of core points to random vectors
        case 1:
        {
            MatrixXi matLabels = MatrixXi::Zero(2, PARAM_NUM_PROJECTION);
            matLabels.array() -= 1;

            omp_set_num_threads(PARAM_NUM_THREADS);
            #pragma omp parallel for
            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
            {
                // vecTopM contains top-M closest and top-M furthest points for a random vector
                VectorXi vecTopM = MATRIX_TOP_M.col(d);

                // Find closest core point and get its label
                for (int m = 0; m < PARAM_PROJECTION_TOP_M; ++m)
                {
                    int iPointIdx = vecTopM(m);
                    if (bit_CORE_POINTS[iPointIdx])
                    {
                        matLabels(0, d) = p_vecLabels[iPointIdx];
                        break;
                    }
                }

                // Find furthest core point and get its labels
                for (int m = 0; m < PARAM_PROJECTION_TOP_M; ++m)
                {
                    int iPointIdx = vecTopM(m + PARAM_PROJECTION_TOP_M);
                    if (bit_CORE_POINTS[iPointIdx])
                    {
                        matLabels(1, d) = p_vecLabels[iPointIdx];
                        break;
                    }
                }
            }

            // Assign the labels of closest random vectors
            omp_set_num_threads(PARAM_NUM_THREADS);
            #pragma omp parallel for
            for (auto const& Xi : vecNoise)
            {
                // Get top-k closest/furthest random vectors
                VectorXi vecRandom = MATRIX_TOP_K.col(Xi);

                for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
                {
                    int closeRi = vecRandom(k);
                    if (matLabels(0, closeRi) > -1)
                    {
                        p_vecLabels[Xi] = matLabels(0, closeRi);
                        break;
                    }

                    int farRi = vecRandom(k + PARAM_PROJECTION_TOP_K);
                    if (matLabels(1, farRi) > -1)
                    {
                        p_vecLabels[Xi] = matLabels(1, farRi);
                        break;
                    }
                }
            }

            break;
        }

        // Get randomly 2000 core points, and use 1NN to find the closest core points and use the label of this point
        // This approach is DBSCAN++
        case 2:
        {
            // First we get vector of sampled core points
            IVector vecSampledCores;
            int iNumCore = bit_CORE_POINTS.count();

            // hack default is p = 0.01
            float fProb = min(0.01 * PARAM_DATA_N / iNumCore, 1.0);

            size_t Xi = bit_CORE_POINTS.find_first();
            while (Xi != boost::dynamic_bitset<>::npos)
            {
                if (rand() / (RAND_MAX + 1.) <= fProb)
                    vecSampledCores.push_back(Xi);

                Xi = bit_CORE_POINTS.find_next(Xi);
            }

            cout << "Number of sampled core points: " << vecSampledCores.size() << endl;

//        auto rd = std::random_device {};
//        auto rng = std::default_random_engine { rd() };
//        shuffle(vecCore.begin(), vecCore.end(), rng);

            // hack only get random 2000 found core points
//            int iNumSamples = min((int)vecCore.size(), 4 * PARAM_PROJECTION_TOP_K * PARAM_PROJECTION_TOP_M);
//            IVector vecSampledCores = samplingWOR(vecCore, iNumSamples);

            omp_set_num_threads(PARAM_NUM_THREADS);
            #pragma omp parallel for
            for (auto const& Xi : vecNoise)
            {
                // Find the closest core points by bruteforce then use the found core point's label (used in uDbscan)
                // Only be efficient if number of noisy points or core poinst is small
                // https://github.com/jenniferjang/dbscanpp/blob/master/dbscanpp.pyx
                VectorXf vecXn = MATRIX_X.col(Xi);

                int iCoreIdx = -1;
                float best_so_far = POS_INF;

                for (auto const& Xj : vecSampledCores)
                {
                    float fDist = computeDist(vecXn, MATRIX_X.col(Xj));
                    if (fDist < best_so_far)
                    {
                        best_so_far = fDist;
                        iCoreIdx = Xj;
                    }
                }

                p_vecLabels[Xi] = p_vecLabels[iCoreIdx];

            }

            break;
        }


        // Get the core point from the neighborhood and use its labels
        // This approach is sngDBSCAN
        case 3:
        {

            omp_set_num_threads(PARAM_NUM_THREADS);
            #pragma omp parallel for
            for (auto const& Xi : vecNoise)
            {
                // Find the core points, connect them together, and check if one of them already has assigned labels
                // This solution slightly improve the NMI
                // Found it here: https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan/blob/master/subsampled_neighborhood_graph_dbscan.h
                // cluster remaining

//                cout << "Noise Point Idx: " << Xi << endl;

                for (auto const& Xj : vec2D_DBSCAN_Neighbor[Xi])
                {
//                    cout << "Neighbor Point Idx: " << Xj << endl;
                    if (bit_CORE_POINTS[Xj])
                    {
                        p_vecLabels[Xi] = p_vecLabels[Xj];
                        break;
                    }
                }
            }

            break;
        }

        // Estimate the distance using CEOs and get closest core point to use its labels.
        // This approach is DBSCAN++
        case 4:
        {
            // First we get vector of sampled core points

            IVector vecCore;
            int iNumCore = bit_CORE_POINTS.count();

            /**
            Sampling 0.01 n # core points
            **/
            // Hack: We only consider 1% * n # core points
            float fProb = 1.0;
            if (iNumCore > 0.01 * PARAM_DATA_N)
                fProb = 0.01 * PARAM_DATA_N / iNumCore;

            size_t Xi = bit_CORE_POINTS.find_first();
            while (Xi != boost::dynamic_bitset<>::npos)
            {
                // Store the core point Idx
                if (fProb == 1.0)
                    vecCore.push_back(Xi);
                else if (rand() / (RAND_MAX + 1.) <= fProb)
                    vecCore.push_back(Xi);

                Xi = bit_CORE_POINTS.find_next(Xi);
            }

            /** Compute again their random projections **/
            iNumCore = vecCore.size();
            cout << "Number of sampled core points: " << vecCore.size() << endl;

            MatrixXf matCoreEmbeddings = MatrixXf::Zero(iNumCore, PARAM_NUM_PROJECTION);

            /** Param for embedding L1 and L2 **/
            int iFourierEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()
            int log2Project = log2(PARAM_INTERNAL_FWHT_PROJECTION);

            omp_set_num_threads(PARAM_NUM_THREADS);
            #pragma omp parallel for
            for (int i = 0; i < iNumCore; ++i)
            {
                int Xi = vecCore[i];

                /**
                Random embedding
                **/
                VectorXf vecX = MATRIX_X.col(Xi);
                VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D); // PARAM_KERNEL_EMBED_D >= D

                if (PARAM_DISTANCE == 0)
                    vecEmbed.segment(0, PARAM_DATA_D) = vecX;
                else if ((PARAM_DISTANCE == 1) || (PARAM_DISTANCE == 2))
                {
                    VectorXf vecProject = MATRIX_R * vecX;
                    vecEmbed.segment(0, iFourierEmbed_D) = vecProject.array().cos();
                    vecEmbed.segment(iFourierEmbed_D, iFourierEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
                }
                else if (PARAM_DISTANCE == 3)
                    embedChiSquare(vecX, vecEmbed);
                else if (PARAM_DISTANCE == 4)
                    embedJS(vecX, vecEmbed);

                /**
                Random projection
                **/

                VectorXf vecRotation = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
                vecRotation.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;

                for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
                {
                    // Component-wise multiplication with a random sign
                    for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
                    {
                        vecRotation(d) *= (2 * (int)bitHD3_Proj[r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                    }

                    // Multiple with Hadamard matrix by calling FWHT transform
                    fht_float(vecRotation.data(), log2Project);
                }

                // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
                matCoreEmbeddings.row(i) = vecRotation.segment(0, PARAM_NUM_PROJECTION); // only get up to #numProj
            }

            /** Estimate distance using CEOs and 1NN classifer **/

            omp_set_num_threads(PARAM_NUM_THREADS);
            #pragma omp parallel for
            for (auto const& Xi : vecNoise)
            {
                // Get top-k closest/furthest random vectors
                VectorXi vecRandom = MATRIX_TOP_K.col(Xi);
                VectorXf vecDotEst = VectorXf::Zero(iNumCore);

                for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
                {
                    int closeRi = vecRandom(k);
                    vecDotEst += matCoreEmbeddings.col(closeRi);

                    int farRi = vecRandom(k + PARAM_PROJECTION_TOP_K);
                    vecDotEst -= matCoreEmbeddings.col(farRi);
                }

                int iCoreIdx = -1;
                float best_so_far = NEG_INF;

                for (int i = 0; i < iNumCore; ++i)
                {
                    float fEst = vecDotEst(i);
                    if (fEst > best_so_far)
                    {
                        best_so_far = fEst;
                        iCoreIdx = vecCore[i];
                    }
                }

                p_vecLabels[Xi] = p_vecLabels[iCoreIdx];

            }

            break;
        }
    }



    // Counting noisy points after label assignment
    vecNoise.clear();
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        if (p_vecLabels[n] == -1)
            vecNoise.push_back(n);
    }
    cout << "After labelling, the number of noisy points: " << vecNoise.size() << endl;

}

/** Multi-thread sDbscan **/
void sDbscan()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    parDbscanIndex();
    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
//    findCorePoints_Asym();
    findCorePoints();
    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    IVector vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + + "_L" + int2str(PARAM_DISTANCE) +
                        "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopM_" + int2str(PARAM_PROJECTION_TOP_M) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputDbscan(vecLabels, sFileName);
	}

}

/** Single thread sDbscan **/
void seq_sDbscan()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();
    seqDbscanIndex();
    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    findCorePoints();
    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    IVector vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + + "_L" + int2str(PARAM_DISTANCE) +
                        "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopM_" + int2str(PARAM_PROJECTION_TOP_M) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputDbscan(vecLabels, sFileName);
	}

}

/** Multi-thread sngDbscan **/
void sngDbscan()
{
    cout << "Running sngDBSCAN" << endl;

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

    cout << "sngDbscan time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = PARAM_OUTPUT_FILE + "_L" + int2str(PARAM_DISTANCE) +
                           "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                           "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                           "_Prob_" + int2str(round(1000 * PARAM_SAMPLING_PROB));

        outputDbscan(vecLabels, sFileName);
    }

}



