#include "Header.h"
#include "Utilities.h"
#include "Dbscan.h"



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
Using priority queue to extract top-k and top-MinPts
**/
void parDbscanIndex_L2()
{

    /** Param for embedding **/
    int iEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()
    int log2Embed_D = log2(iEmbed_D);

    boost::dynamic_bitset<> bitHD3_Embed;
    bitHD3Generator(iEmbed_D * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Embed);

    /** Param for random projection **/
    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);
    int log2Project = log2(PARAM_NUM_PROJECTION);

    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    boost::dynamic_bitset<> bitHD3_Proj;
    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);

    // Only for testing
//    MATRIX_X_EMBED = MatrixXf::Zero(PARAM_KERNEL_EMBED_D, PARAM_DATA_N);

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        /**
        L2 Embedding
        **/

        VectorXf vecProject = VectorXf::Zero(iEmbed_D); // iEmbed_D > D
        vecProject.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < iEmbed_D; ++d)
            {
                vecProject(d) *= (2 * (int)bitHD3_Embed[r * iEmbed_D + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecProject.data(), log2Embed_D);
        }

        // We scale with iEmbed_D since we apply HD2HD1, each need a scale of sqrt(iEmbed_D). The last HD3 is similar to Gaussian matrix.
        // We further scale with sigma since we use the standard scale N(0, 1/sigma^2) hence std = 1/sigma for kernel embedding
        // See: https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
        vecProject /= (iEmbed_D * PARAM_KERNEL_SIGMA); // vecPoint.segment(0, 128); // 512 projections, but you need only 128 projections

//        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D);
//        vecEmbed << vecProject.array().cos(), vecProject.array().sin(); // no need to scale since we only use ranking

        // Only for testing OPTICS plot (Euclidean space vs Embedded space)
//        MATRIX_X_EMBED.col(n) = vecEmbed / sqrt(iEmbed_D);

        /**
        Random projection
        **/
        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecPoint.segment(0, iEmbed_D) = vecProject.array().cos();
        vecPoint.segment(iEmbed_D, iEmbed_D) = vecProject.array().sin();

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Project);
        }

        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
        MATRIX_RP.col(n) = vecPoint;

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

    /**
    For each random vector, getting 2*top-MinPts as close and far candidates
    **/
    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);

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
            MATRIX_TOP_MINPTS(k, d) = minPQ_Close.top().m_iIndex;
            minPQ_Close.pop();

            // Far
            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = minPQ_Far.top().m_iIndex;
            minPQ_Far.pop();
        }
    }
}

/**
Do not store Projection-Matrix due to the memory limit
Slow on threading since we have to update the same memory i.e. data structure
Using priority queue to extract top-k and top-MinPts
**/
void seqDbscanIndex_Metric()
{
    vector<Min_PQ_Pair> vecPQ_Close = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());
    vector<Min_PQ_Pair> vecPQ_Far = vector<Min_PQ_Pair> (PARAM_NUM_PROJECTION, Min_PQ_Pair());


    /** Param for embedding **/
    int iEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()

    MatrixXf MATRIX_R = MatrixXf::Zero(iEmbed_D, PARAM_DATA_D); // EigenLib does not allow resize MatrixXf when passing by reference !

    // Can be generalized for other kernel embedding
    if (PARAM_DISTANCE == 1)
        cauchyGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // K(x, y) = exp(-gamma * L1_dist(X, y))) where gamma = d^2
    else
        gaussGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // K(x, y) = exp(-gamma * L2_dist^2(X, y))) where gamma = 1/2 sigma^2

    /** Param for random projection **/
    int log2Project = log2(PARAM_NUM_PROJECTION);

    boost::dynamic_bitset<> bitHD3_Proj;
    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);

    /** Param for index **/
    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        /**
        Random Fourier Transform
        **/

        VectorXf vecProject = VectorXf::Zero(iEmbed_D); // iEmbed_D > D
        vecProject = MATRIX_R * MATRIX_X.col(n);

        /**
        Random projection
        **/
        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecPoint.segment(0, iEmbed_D) = vecProject.array().cos();
        vecPoint.segment(iEmbed_D, iEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements


        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Project);
        }

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

    /**
    For each random vector, getting 2*top-MinPts as close and far candidates
    **/
    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);

    #pragma omp parallel for
    for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
    {
        for (int k = PARAM_DBSCAN_MINPTS - 1; k >= 0; --k)
        {
            // Close
            MATRIX_TOP_MINPTS(k, d) = vecPQ_Close[d].top().m_iIndex;
            vecPQ_Close[d].pop();

            // Far
            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = vecPQ_Far[d].top().m_iIndex;
            vecPQ_Far[d].pop();
        }
    }
}

/**
Have to store the projection matrix for parallel processing.
Using priority queue to extract top-k and top-MinPts
**/
void parDbscanIndex_Metric()
{

    /** Param for embedding **/
    int iEmbed_D = PARAM_KERNEL_EMBED_D / 2; // This is becase we need cos() and sin()

    MatrixXf MATRIX_R = MatrixXf::Zero(iEmbed_D, PARAM_DATA_D); // EigenLib does not allow resize MatrixXf when passing by reference !

    // See: https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
    if (PARAM_DISTANCE == 1)
        cauchyGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // K(x, y) = exp(-gamma * L1_dist(X, y))) where gamma = 1/sigma
    else
        gaussGenerator(iEmbed_D, PARAM_DATA_D, 0, 1.0 / PARAM_KERNEL_SIGMA, MATRIX_R); // std = 1/sigma, K(x, y) = exp(-gamma * L2_dist^2(X, y))) where gamma = 1/2 sigma^2

    /** Param for random projection **/
    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);

    int log2Project = log2(PARAM_NUM_PROJECTION);
    boost::dynamic_bitset<> bitHD3_Proj;
    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);

    /** Param for index **/
    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        /**
        Random Fourier Transform
        **/

        VectorXf vecProject = VectorXf::Zero(iEmbed_D); // iEmbed_D > D
        vecProject = MATRIX_R * MATRIX_X.col(n);

//        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D);
//        vecEmbed << vecPoint.array().cos(), vecPoint.array().sin(); // no need to scale since we only use ranking


        /**
        Random projection
        **/
        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
//        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;
        vecPoint.segment(0, iEmbed_D) = vecProject.array().cos();
        vecPoint.segment(iEmbed_D, iEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements


        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Project);
        }

        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
        MATRIX_RP.col(n) = vecPoint;

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

    /**
    For each random vector, getting 2*top-MinPts as close and far candidates
    **/
    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);

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
            MATRIX_TOP_MINPTS(k, d) = minPQ_Close.top().m_iIndex;
            minPQ_Close.pop();

            // Far
            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = minPQ_Far.top().m_iIndex;
            minPQ_Far.pop();
        }
    }
}

/**
Have to store the projection matrix for parallel processing.
Using priority queue to extract top-k and top-MinPts

Only support ChiSquare and Johnson Shannon divergences
**/
void parDbscanIndex_NonMetric()
{
    /** Param for random projection **/
    MatrixXf MATRIX_RP = MatrixXf::Zero(PARAM_NUM_PROJECTION, PARAM_DATA_N);

    int log2Project = log2(PARAM_NUM_PROJECTION);
    boost::dynamic_bitset<> bitHD3_Proj;
    bitHD3Generator(PARAM_NUM_PROJECTION * PARAM_INTERNAL_NUM_ROTATION, bitHD3_Proj);

    /** Param for index **/
    MATRIX_TOP_K = MatrixXi::Zero(2 * PARAM_PROJECTION_TOP_K, PARAM_DATA_N);

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        /**
        Deterministic Additive Kernel feature
        **/
        VectorXf vecEmbed = VectorXf::Zero(PARAM_KERNEL_EMBED_D); // iEmbed_D > D
        if (PARAM_DISTANCE == 3)
            embedChiSquare(MATRIX_X.col(n), vecEmbed);
        else if (PARAM_DISTANCE == 4)
            embedJS(MATRIX_X.col(n), vecEmbed);

        /**
        Random projection
        **/
        VectorXf vecPoint = VectorXf::Zero(PARAM_NUM_PROJECTION); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
        vecPoint.segment(0, PARAM_KERNEL_EMBED_D) = vecEmbed;

        for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_NUM_PROJECTION; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3_Proj[r * PARAM_NUM_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), log2Project);
        }

        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
        MATRIX_RP.col(n) = vecPoint;

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

    /**
    For each random vector, getting 2*top-MinPts as close and far candidates
    **/
    MATRIX_TOP_MINPTS = MatrixXi::Zero(2 * PARAM_DBSCAN_MINPTS, PARAM_NUM_PROJECTION);

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
            MATRIX_TOP_MINPTS(k, d) = minPQ_Close.top().m_iIndex;
            minPQ_Close.pop();

            // Far
            MATRIX_TOP_MINPTS(k + PARAM_DBSCAN_MINPTS, d) = minPQ_Far.top().m_iIndex;
            minPQ_Far.pop();
        }
    }
}


/**
vec2D_DBSCAN_Neighbor: For each point, store its true neighborhood (within radius eps)
- Note that for each point x, we keep insert into x's neighborhood y - using vector
- This approach is very fast (multi-thread friendly) but losing that x is also y's neighborhood
- Also, this approach might store duplicate if y appears several time on close/far random project
- We might fix with set()
bit_CORE_POINTS: Store a bit vector presenting for core point
**/
void findCorePoints_Asym()
{
//    float fThresDot = exp(-PARAM_DBSCAN_EPS * PARAM_DBSCAN_EPS / (2.0 * PARAM_DATA_D * PARAM_DATA_D);

    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    //TODO: If single thread, then we can improve if we store (X1, X2) s.t. <X1,X2> >= threshold
    //TODO: We can use EPS and MATRIX_X to compute the distance if embedding_d >> original_d

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get top-k closese/furthest vectors
//        VectorXf vecXn = MATRIX_X_EMBED.col(n);
        VectorXf vecXn = MATRIX_X.col(n);

        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far

        IVector vecNeighborhood;

//        unordered_set<int> approxNeighbor; MinPts * 2K * 32 bits >> N
        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N);

        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
        {
            // Closest
            int Ri = vecTopK(k);

            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
                //if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
                    //approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if ( fDist <= PARAM_DBSCAN_EPS) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
                        vecNeighborhood.push_back(iPointIdx);
                }
            }

            // Far
            Ri = vecTopK(k + PARAM_PROJECTION_TOP_K);

            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i + PARAM_DBSCAN_MINPTS, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
                //if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
                    //approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS) // Can speed up if storing global precomputed matrix as a set((small_ID, large_ID)
                        vecNeighborhood.push_back(iPointIdx);
                }
            }
        }

//        cout << "Number of used distances for the point of " << n << " is: " << approxNeighbor.size() << endl;

        // Decide core points
        if ((int)vecNeighborhood.size() >= PARAM_DBSCAN_MINPTS - 1)
            bit_CORE_POINTS[n] = 1;

        // Store to use while forming cluster
        vec2D_DBSCAN_Neighbor[n] = vecNeighborhood;
    }

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
//    float fThresDot = exp(-PARAM_DBSCAN_EPS * PARAM_DBSCAN_EPS / (2.0 * PARAM_DATA_D * PARAM_DATA_D);

    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    vector<unordered_set<int>> set2D_DBSCAN_Neighbor(PARAM_DATA_N, unordered_set<int>());

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    //TODO: If single thread, then we can improve if we store (X1, X2) s.t. <X1,X2> >= threshold
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get top-k closese/furthest vectors
//        VectorXf vecXn = MATRIX_X_EMBED.col(n);
        VectorXf vecXn = MATRIX_X.col(n);

        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far

//        IVector vecNeighborhood;

//        unordered_set<int> approxNeighbor; // it might be faster with binary vector for small data set if MinPts * 2K * 32 bits >> N
//        set<int> approxNeighbor;
        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N);

        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
        {
            // Closest
            int Ri = vecTopK(k);

            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
//                if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
//                    approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
                    {
                        #pragma omp critical
                        {
                        set2D_DBSCAN_Neighbor[n].insert(iPointIdx);
                        set2D_DBSCAN_Neighbor[iPointIdx].insert(n);

//                        vec2D_DBSCAN_Neighbor[n].push_back(iPointIdx);
//                        vec2D_DBSCAN_Neighbor[iPointIdx].push_back(n);
                        }
                    }

                }
            }

            // Far
            Ri = vecTopK(k + PARAM_PROJECTION_TOP_K);

            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i + PARAM_DBSCAN_MINPTS, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
//                if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
//                    approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS) // Can speed up if storing global precomputed matrix as a set((small_ID, large_ID)
                    {
                        #pragma omp critical
                        {
                        set2D_DBSCAN_Neighbor[n].insert(iPointIdx);
                        set2D_DBSCAN_Neighbor[iPointIdx].insert(n);

//                        vec2D_DBSCAN_Neighbor[n].push_back(iPointIdx);
//                        vec2D_DBSCAN_Neighbor[iPointIdx].push_back(n);
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
        int iNeighborSize = set2D_DBSCAN_Neighbor[n].size();
        vec2D_DBSCAN_Neighbor[n].insert(vec2D_DBSCAN_Neighbor[n].end(), set2D_DBSCAN_Neighbor[n].begin(), set2D_DBSCAN_Neighbor[n].end());

        // Decide core points
        if (iNeighborSize >= PARAM_DBSCAN_MINPTS - 1)
//        if (vec2D_DBSCAN_Neighbor[n].size() >= PARAM_DBSCAN_MINPTS - 1)
            bit_CORE_POINTS[n] = 1;


    }

    cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

/**
We connect core points first, then label its neighborhood later
**/
void formCluster(int & p_iNumClusters, IVector & p_vecLabels)
{
    p_vecLabels = IVector(PARAM_DATA_N, -1); //noise = -1
    p_iNumClusters = 0;

    int iOldClusterID = 0;
    int iNewClusterID = -1; // The cluster ID starts from 0

    // Fast enough so might not need multi-threading
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {

        if ((! bit_CORE_POINTS[n]) || (p_vecLabels[n] != -1))
            continue;

        // Always start from the core points

        iNewClusterID++;

        unordered_set<int> seedSet; //seedSet only contains core points
        seedSet.insert(n);

        // Note: For small data set (e.g. < 10K), bitset might be faster
        // However, for large data set (1M), iterative the bitset is O(n) --> much slower, especially if OPTICS with large radius
//        boost::dynamic_bitset<> seedSet(PARAM_DATA_N);
//        seedSet[n] = 1;

//        unordered_set<int> connectedPoints; // can be replaced by a histogram to increase the searching process
//        connectedPoints.insert(n);
        // We should use bitset since a cluster tends to contain many points (e.g. n / 10) so iterative bitset is fine
        // Otherwise, unorder_set might be faster
        boost::dynamic_bitset<> connectedPoints(PARAM_DATA_N);
        connectedPoints[n] = 1;

        // Connecting components in a cluster
        bool bConnected = false;

//        while (seedSet.count() > 0)
        while (seedSet.size() > 0) // unorder_set<int> is slow if there are many core points
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

                        seedSet.insert(Xj);
//                        seedSet[Xj] = 1;

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


void fastDbscan()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();
//    parDbscanIndex_L2();

    if ((PARAM_DISTANCE == 1) || (PARAM_DISTANCE == 2))
        parDbscanIndex_Metric();
    else if ((PARAM_DISTANCE == 3) || (PARAM_DISTANCE == 4))
        parDbscanIndex_NonMetric();
    else
        cerr << "Error: The distance is not supported !" << endl;

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
                        "_Eps_" + int2str(round(10 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputDbscan(vecLabels, sFileName);
	}

}

void memoryDbscan()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();
    seqDbscanIndex_Metric();
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
                        "_Eps_" + int2str(round(10 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputDbscan(vecLabels, sFileName);
	}

}


void findCoreDist_Asym()
{
//    float fThresDot = exp(-PARAM_DBSCAN_EPS * PARAM_DBSCAN_EPS / (2.0 * PARAM_DATA_D * PARAM_DATA_D);
    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    vec2D_DBSCAN_NeighborDist = vector<FVector> (PARAM_DATA_N, FVector());

    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);
    vec_CORE_DIST = FVector(PARAM_DATA_N);

    //TODO: If single thread, then we can improve if we store (X1, X2) s.t. <X1,X2> >= threshold
    //TODO: We can use EPS and MATRIX_X to compute the distance if embedding_d >> original_d

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get top-k closese/furthest vectors
//        VectorXf vecXn = MATRIX_X_EMBED.col(n);
        VectorXf vecXn = MATRIX_X.col(n);

        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far

        vector<IFPair> vecNeighborhood;

        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N);

        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
        {
            // Closest
            int Ri = vecTopK(k);
            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
                if (!approxNeighbor[iPointIdx]) // cannot find
                {
//                    approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS)
                        vecNeighborhood.push_back(IFPair(iPointIdx, fDist));
                }
            }

            // Far
            Ri = vecTopK(k + PARAM_PROJECTION_TOP_K);
            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i + PARAM_DBSCAN_MINPTS, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
                if (!approxNeighbor[iPointIdx]) // cannot find
                {
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS)
                        vecNeighborhood.push_back(IFPair(iPointIdx, fDist));
                }
            }
        }

//        cout << "Number of used distances for the point of " << n << " is: " << approxNeighbor.size() << endl;

        // Store information
        for (int i = 0; i < (int)vecNeighborhood.size(); ++i)
        {
            vec2D_DBSCAN_Neighbor[n].push_back(vecNeighborhood[i].m_iIndex);
            vec2D_DBSCAN_NeighborDist[n].push_back(vecNeighborhood[i].m_fValue);
        }

        // Decide core points and its core distance
        if ((int)vecNeighborhood.size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[n] = 1;

            // Only sort if this is the core
            nth_element(vecNeighborhood.begin(), vecNeighborhood.begin() + PARAM_DBSCAN_MINPTS - 2, vecNeighborhood.end()); // default is X1 < X2 < ...

            // Store core dist
            vec_CORE_DIST[n] = vecNeighborhood[PARAM_DBSCAN_MINPTS - 2].m_fValue; // scikit learn includes the point itself, and index start from 0

            // test
//            cout << "Core dist: " << VECTOR_CORE_DIST(n) << endl;
//            sort(vecNeighborhood.begin(), vecNeighborhood.end());
//
//            cout << "Begin testing " << endl;
//            for (int i = 0; i < vecNeighborhood.size(); ++i)
//            {
//                cout << vecNeighborhood[i].m_fValue << endl;
//            }
//
//            cout << "End testing " << endl;
        }

    }

    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

void findCoreDist()
{
//    float fThresDot = exp(-PARAM_DBSCAN_EPS * PARAM_DBSCAN_EPS / (2.0 * PARAM_DATA_D * PARAM_DATA_D);
    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
    vec2D_DBSCAN_NeighborDist = vector<FVector> (PARAM_DATA_N, FVector());

    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);
    vec_CORE_DIST = FVector(PARAM_DATA_N);

    vector<unordered_map<int, float>> map2D_DBSCAN_Neighbor(PARAM_DATA_N, unordered_map<int, float>());

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get top-k closese/furthest vectors
//        VectorXf vecXn = MATRIX_X_EMBED.col(n);
        VectorXf vecXn = MATRIX_X.col(n);

        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far

        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N);

        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
        {
            // Closest
            int Ri = vecTopK(k);

            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i, Ri);

                if (iPointIdx == n)
                    continue;

                if (!approxNeighbor[iPointIdx]) // cannot find
                {
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS)
                    {
                        // vecNeighborhood.push_back(IFPair(iPointIdx, fDist));
                        #pragma omp critical
                        {
                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDist));
                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDist));
                        }
                    }
                }
            }

            // Far
            Ri = vecTopK(k + PARAM_PROJECTION_TOP_K);

            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = MATRIX_TOP_MINPTS(i + PARAM_DBSCAN_MINPTS, Ri);

                if (iPointIdx == n)
                    continue;

                if (!approxNeighbor[iPointIdx]) // cannot find
                {
                    approxNeighbor[iPointIdx] = 1;

//                    float fDist = 0.0;
//                    if (PARAM_DISTANCE == 1)
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).cwiseAbs().sum(); // or (vecXn - MATRIX_X.col(iPointIdx)).lpNorm<1>();
//                    else
//                        fDist = (vecXn - MATRIX_X.col(iPointIdx)).norm();  // default L2

                    float fDist = computeDist(vecXn, MATRIX_X.col(iPointIdx));

                    if (fDist <= PARAM_DBSCAN_EPS)
                    {
                        #pragma omp critical
                        {
                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDist));
                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDist));
                        }
                    }
                }
            }
        }
    }

    cout << "Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Store idx and float in the same order
        for (auto const& ifPair: map2D_DBSCAN_Neighbor[n])
        {
            vec2D_DBSCAN_Neighbor[n].push_back(ifPair.first);
            vec2D_DBSCAN_NeighborDist[n].push_back(ifPair.second);
        }

        if ((int)map2D_DBSCAN_Neighbor[n].size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[n] = 1;

            // Only sort if this is the core
            FVector vecNeighborhood = vec2D_DBSCAN_NeighborDist[n];
            nth_element(vecNeighborhood.begin(), vecNeighborhood.begin() + PARAM_DBSCAN_MINPTS - 2, vecNeighborhood.end()); // default is X1 < X2 < ...

            // Store core dist
            vec_CORE_DIST[n] = vecNeighborhood[PARAM_DBSCAN_MINPTS - 2]; // scikit learn includes the point itself, and index start from 0

            // test
    //            cout << "Core dist: " << VECTOR_CORE_DIST(n) << endl;
    //            sort(vecNeighborhood.begin(), vecNeighborhood.end());
    //
    //            cout << "Begin testing " << endl;
    //            for (int i = 0; i < vecNeighborhood.size(); ++i)
    //            {
    //                cout << vecNeighborhood[i].m_fValue << endl;
    //            }
    //
    //            cout << "End testing " << endl;
        }
    }
    cout << "CoreDist time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

/**
Only use for testing - to see Embedding vs Actual Euclidean distance
**/
//void findCoreDist_Symmetric_L2_Embed()
//{
//    float fThresDot = -exp(-PARAM_DBSCAN_EPS * PARAM_DBSCAN_EPS / (2.0 * PARAM_DATA_D * PARAM_DATA_D)); // we use -exp(r^2/2d^2) to simulate distance
//    vec2D_DBSCAN_Neighbor = vector<IVector> (PARAM_DATA_N, IVector());
//    vec2D_DBSCAN_NeighborDist = vector<FVector> (PARAM_DATA_N, FVector());
//
//    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);
//    vec_CORE_DIST = FVector(PARAM_DATA_N);
//
//    vector<unordered_map<int, float>> map2D_DBSCAN_Neighbor(PARAM_DATA_N, unordered_map<int, float>());
//
//    chrono::steady_clock::time_point begin;
//    begin = chrono::steady_clock::now();
//
//    #pragma omp parallel for
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//    {
//        // Get top-k closese/furthest vectors
//        VectorXf vecXn = MATRIX_X_EMBED.col(n);
////        VectorXf vecXn = MATRIX_X.col(n);
//
//        VectorXi vecTopK = MATRIX_TOP_K.col(n); // size 2K: first K is close, last K is far
//
//        boost::dynamic_bitset<> approxNeighbor(PARAM_DATA_N);
//
//        for (int k = 0; k < PARAM_PROJECTION_TOP_K; ++k)
//        {
//            // Closest
//            int Ri = vecTopK(k);
//
//            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
//            {
//                // Compute distance between Xn and Xi
//                int iPointIdx = MATRIX_TOP_MINPTS(i, Ri);
//
//                if (iPointIdx == n)
//                    continue;
//
//                if (!approxNeighbor[iPointIdx]) // cannot find
//                {
//                    approxNeighbor[iPointIdx] = 1;
//
//                    float fDot = -vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)); // take minus to simulate Euclidean distance
//                    if (fDot <= fThresDot)
//                    {
//                        // vecNeighborhood.push_back(IFPair(iPointIdx, fDist));
//                        #pragma omp critical
//                        {
//                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDot));
//                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDot));
//                        }
//                    }
//                }
//            }
//
//            // Far
//            Ri = vecTopK(k + PARAM_PROJECTION_TOP_K);
//
//            for (int i = 0; i < PARAM_DBSCAN_MINPTS; ++i)
//            {
//                // Compute distance between Xn and Xi
//                int iPointIdx = MATRIX_TOP_MINPTS(i + PARAM_DBSCAN_MINPTS, Ri);
//
//                if (iPointIdx == n)
//                    continue;
//
//                if (!approxNeighbor[iPointIdx]) // cannot find
//                {
//                    approxNeighbor[iPointIdx] = 1;
//
//                    float fDot = -vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)); // take minus to simulate Euclidean distance
//                    if (fDot <= fThresDot)
//                    {
//                        #pragma omp critical
//                        {
//                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDot));
//                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDot));
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    cout << "Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
//
//    begin = chrono::steady_clock::now();
//
//    #pragma omp parallel for
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//    {
//        // Store idx and float in the same order
//        for (auto const& ifPair: map2D_DBSCAN_Neighbor[n])
//        {
//            vec2D_DBSCAN_Neighbor[n].push_back(ifPair.first);
//            vec2D_DBSCAN_NeighborDist[n].push_back(ifPair.second);
//        }
//
//        if ((int)map2D_DBSCAN_Neighbor[n].size() >= PARAM_DBSCAN_MINPTS - 1)
//        {
//            bit_CORE_POINTS[n] = 1;
//
//            // Only sort if this is the core
//            FVector vecNeighborhood = vec2D_DBSCAN_NeighborDist[n];
//            nth_element(vecNeighborhood.begin(), vecNeighborhood.begin() + PARAM_DBSCAN_MINPTS - 2, vecNeighborhood.end()); // default is X1 < X2 < ...
//
//            // Store core dist
//            vec_CORE_DIST[n] = vecNeighborhood[PARAM_DBSCAN_MINPTS - 2]; // scikit learn includes the point itself, and index start from 0
//
//            // test
//    //            cout << "Core dist: " << VECTOR_CORE_DIST(n) << endl;
//    //            sort(vecNeighborhood.begin(), vecNeighborhood.end());
//    //
//    //            cout << "Begin testing " << endl;
//    //            for (int i = 0; i < vecNeighborhood.size(); ++i)
//    //            {
//    //                cout << vecNeighborhood[i].m_fValue << endl;
//    //            }
//    //
//    //            cout << "End testing " << endl;
//        }
//    }
//    cout << "CoreDist time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
//
//    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;
//
//}

void formOptics(IVector& p_vecLabels, IVector & p_vecOrder, FVector & p_vecReachDist)
{
    /**
    We form cluster using DBSCAN
    **/
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    int iNumClusters = 0;
    formCluster(iNumClusters, p_vecLabels);

    cout << "Forming cluster time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    vector<IVector> nestedVec_ClusterStructure(iNumClusters, IVector()); // +1 since there is noise (label = -1)
    vector<IVector> nestedVec_ClusterCores(iNumClusters, IVector()); // +1 since there is noise (label = -1)

    // Get each connected components of DBSCAN
//    #pragma omp parallel for : Does not work and cause memory bug (not sure how to fix)
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // NOTE: If send referece p_vecLabels, then this cause memory bug, e.g free() twice
        int iLabel = p_vecLabels[n];

        if (iLabel == -1) // noise
            nestedVec_ClusterStructure[iNumClusters - 1].push_back(n);
        else if (iLabel > -1) // Labeled cluster starts from 0
        {
            nestedVec_ClusterStructure[iLabel].push_back(n);

            if (bit_CORE_POINTS[n])
                nestedVec_ClusterCores[iLabel].push_back(n);
        }
        else
            cout << "There exist some unlabeled points !" << endl;

    }


    /**
    Compute reachable distance for points within the cluster
    **/

    p_vecReachDist = FVector(PARAM_DATA_N, NEG_INF); // Undefined = -2^31

    // Compute reachable distance from each core points to all of its neighborhood, and update the minimum dist
//    #pragma omp parallel for
    for (int c = 0; c < iNumClusters - 1; ++c)  // do not consider c = NumCluster - 1 since it is noise cluster
    {
        /**
        For each connected component, update reachable dist
        **/
        // Xi is core
        for (auto const& Xi : nestedVec_ClusterCores[c])
        {
            float Xi_core_dist = vec_CORE_DIST[Xi];

            // Xj is neighbor of core Xi
            for (int j = 0; j < (int)vec2D_DBSCAN_Neighbor[Xi].size(); ++j)
            {
                int Xj = vec2D_DBSCAN_Neighbor[Xi][j];
                float dist_XiXj = vec2D_DBSCAN_NeighborDist[Xi][j];
                float Xj_reach_dist_from_Xi = max(Xi_core_dist, dist_XiXj);

//                #pragma omp critical
                {
                // Update reach-able dist for Xj
                if (p_vecReachDist[Xj] == NEG_INF) // UNDEFINE = -inf
                    p_vecReachDist[Xj] = Xj_reach_dist_from_Xi;
                else if (p_vecReachDist[Xj] > Xj_reach_dist_from_Xi)
                    p_vecReachDist[Xj] = Xj_reach_dist_from_Xi;
                }
            }
        }
    }

//    cout << "Finish compute reachable-dist" << endl;

    boost::dynamic_bitset<> processSet(PARAM_DATA_N);

    // Simulate OPTICS order by using a PQ
    // This order is different from exact OPTICS due to the approximation from the reachable-dist
    for (int c = 0; c < iNumClusters - 1; ++c)  // do not consider c = NumCluster - 1 since it is noise cluster
    {
        // To match with the order of OPTICS
//        cout << "Process cluster: " << c << " of size " << nestedVec_ClusterStructure[c].size() << endl;

        for (auto const& Xi : nestedVec_ClusterStructure[c])
        {
            if (processSet[Xi])
                continue;

            processSet[Xi] = 1;
            p_vecOrder.push_back(Xi);

            // Only deal with core points since they affect reachable-dist
            // If it is not a core point, then its reachable-dist will be updated later
            if (bit_CORE_POINTS[Xi])
            {
                Min_PQ_Pair seedSet; // One point might be appear several time in the PQ, but we only process the first appearance and check processed

                for (int i = 0; i < (int)vec2D_DBSCAN_Neighbor[Xi].size(); ++i)
                {
                    int Xj = vec2D_DBSCAN_Neighbor[Xi][i];

                    // only update if it is not processed
                    if (processSet[Xj])
                        continue;

                    seedSet.push(IFPair(Xj, p_vecReachDist[Xj]));

                    // set process for point already added into PQ so that next time we would not add it in again
                    processSet[Xj] = 1;
                }

                while (seedSet.size() > 0)
                {
                    int Xj = seedSet.top().m_iIndex;
                    seedSet.pop();

                    processSet[Xj] = 1;
                    p_vecOrder.push_back(Xj);

                    if (bit_CORE_POINTS[Xj])
                    {
                        for (int k = 0; k < (int)vec2D_DBSCAN_Neighbor[Xj].size(); ++k)
                        {
                            int Xk = vec2D_DBSCAN_Neighbor[Xj][k];

                            // only update if it is not processed
                            if (processSet[Xk])
                                continue;

                            seedSet.push(IFPair(Xk, p_vecReachDist[Xk]));

                            // set process for point already added into PQ so that next time we would not add it in again
                            processSet[Xk] = 1;
                        }
                    }
                }
            }
        }
    }

    // Adding noise at the end
    for (int i = 0; i < (int)nestedVec_ClusterStructure[iNumClusters - 1].size(); ++i)
        p_vecOrder.push_back(nestedVec_ClusterStructure[iNumClusters - 1][i]);


}

void formOptics_scikit(IVector & p_vecOrder, FVector & p_vecReachDist)
{
    p_vecReachDist = FVector(PARAM_DATA_N, NEG_INF); // Undefined = 2^31

    boost::dynamic_bitset<> processSet(PARAM_DATA_N);

    for (int Xi = 0; Xi < PARAM_DATA_N; ++Xi)
    {
        if (processSet[Xi])
            continue;

        processSet[Xi] = 1;
        p_vecOrder.push_back(Xi);

        // Only deal with core points since they affect reachable-dist
        // If it is not a core point, then its reachable-dist will be updated later
        if (bit_CORE_POINTS[Xi])
        {
//            unordered_map<int, float> seedSet; // One point might be appear several time in the PQ, but max # element = n * MinPts * 2k
            Min_PQ_Pair seedSet;


            float Xi_core_dist = vec_CORE_DIST[Xi];
            IVector Xi_neighborhood = vec2D_DBSCAN_Neighbor[Xi];

            // Xj is neighbor of core Xi
            for (int j = 0; j < (int)Xi_neighborhood.size(); ++j)
            {
                int Xj = Xi_neighborhood[j];

                // only update if it is not processed
                if (processSet[Xj])
                    continue;

                float dist_XiXj = vec2D_DBSCAN_NeighborDist[Xi][j];
                float Xj_reach_dist_from_Xi = max(Xi_core_dist, dist_XiXj);

                if (p_vecReachDist[Xj] == NEG_INF)
                {
                    p_vecReachDist[Xj] = Xj_reach_dist_from_Xi;
//                    seedSet.insert(make_pair(Xj, p_vecReachDist[Xj]));
                    seedSet.push(IFPair(Xj, p_vecReachDist[Xj]));

                }

                else if (p_vecReachDist[Xj] > Xj_reach_dist_from_Xi)
                {
                    p_vecReachDist[Xj] = Xj_reach_dist_from_Xi;
//                    seedSet[Xj] = p_vecReachDist[Xj];
                    seedSet.push(IFPair(Xj, p_vecReachDist[Xj]));

                }
            }

            while (seedSet.size() > 0)
            {
                // Get minimum value by iterative the seedSet
//                int Xj = seedSet.begin()->first;
//                float fMin = seedSet.begin()->second;
//                for (auto i = seedSet.begin(); i != seedSet.end(); i++)
//                {
//                    if (i->second < fMin)
//                    {
//                        Xj = i->first;
//                        fMin = i->second;
//                    }
//                }
//                seedSet.erase(Xj);

                int Xj = seedSet.top().m_iIndex;
                seedSet.pop();

                if (processSet[Xj])
                    continue;

                processSet[Xj] = 1; // process
                p_vecOrder.push_back(Xj);

                if (bit_CORE_POINTS[Xj])
                {
                    float Xj_core_dist = vec_CORE_DIST[Xj];
                    IVector Xj_neighborhood = vec2D_DBSCAN_Neighbor[Xj];

                    // Xj is neighbor of core Xi
                    for (int k = 0; k < (int)Xj_neighborhood.size(); ++k)
                    {
                        int Xk = Xj_neighborhood[k];

                        // only update if it is not processed
                        if (processSet[Xk])
                            continue;

                        float dist_XjXk = vec2D_DBSCAN_NeighborDist[Xj][k];
                        float Xk_reach_dist_from_Xj = max(Xj_core_dist, dist_XjXk);

                        if (p_vecReachDist[Xk] == NEG_INF)
                        {
                            p_vecReachDist[Xk] = Xk_reach_dist_from_Xj;
//                            seedSet.insert(make_pair(Xk, p_vecReachDist[Xk]));
                            seedSet.push(IFPair(Xk, p_vecReachDist[Xk]));
                        }
                        else if (p_vecReachDist[Xk] > Xk_reach_dist_from_Xj)
                        {
                            p_vecReachDist[Xk] = Xk_reach_dist_from_Xj;
//                            seedSet[Xk] = p_vecReachDist[Xk];
                            seedSet.push(IFPair(Xk, p_vecReachDist[Xk]));
                        }
                    }
                }
            }
        }
    }
}

void fastOptics()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

//    parDbscanIndex_L2(); // only specific for L2 - not affect running time
    parDbscanIndex_Metric();
    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
//    findCoreDist_Asym(); // Faster for multi-threading but loss some accuracy
    findCoreDist();
    cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    IVector vecOrder, vecLabels;
    FVector vecReachDist;
//    formOptics(vecLabels, vecOrder, vecReachDist);

    // We do not need cluster for optics but we still store it to cross check the DBSCAN accuracy given the current setting
    int iNumClusters = 0;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clustering time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    formOptics_scikit(vecOrder, vecReachDist);
    cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + "_L" + int2str(PARAM_DISTANCE) +
                        "_Eps_" + int2str(round(10 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputOptics(vecOrder, vecReachDist, vecLabels, sFileName);
	}
}


void memoryOptics()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    seqDbscanIndex_Metric();
    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    findCoreDist();
    cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    IVector vecOrder, vecLabels;
    FVector vecReachDist;

    // We do not need cluster for optics but we still store it to cross check the DBSCAN accuracy given the current setting
    int iNumClusters = 0;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clustering time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    formOptics_scikit(vecOrder, vecReachDist);
    cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + "_L" + int2str(PARAM_DISTANCE) +
                        "_Eps_" + int2str(round(10 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputOptics(vecOrder, vecReachDist, vecLabels, sFileName);
	}
}
