#include "../include/Header.h"
#include "../include/Utilities.h"
#include "../include/DBSCAN.h"

//#include "google/dense_hash_set"
//#include "google/dense_hash_map"

/**
Finding neighborhood and its distance to the query point using random projection
**/
void findCoreDist()
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

                    if (fDist <= PARAM_DBSCAN_EPS)
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

                    if (fDist <= PARAM_DBSCAN_EPS)
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

    begin = chrono::steady_clock::now();

    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);
    vec_CORE_DIST = FVector(PARAM_DATA_N);

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Store idx and float in the same order
        unordered_map<int, float> mapNeighborhood(vec2D_OPTICS_NeighborDist[n].begin(), vec2D_OPTICS_NeighborDist[n].end());

        vec2D_OPTICS_NeighborDist[n].clear();
        vec2D_OPTICS_NeighborDist[n].insert(vec2D_OPTICS_NeighborDist[n].end(), mapNeighborhood.begin(), mapNeighborhood.end());
        mapNeighborhood.clear();

        if ((int)vec2D_OPTICS_NeighborDist[n].size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[n] = 1;

            // Only sort if this is the core
            FVector vecNeighborhood;
            for (const auto &ifPair : vec2D_OPTICS_NeighborDist[n])
                vecNeighborhood.push_back(ifPair.second);

            nth_element(vecNeighborhood.begin(), vecNeighborhood.begin() + PARAM_DBSCAN_MINPTS - 2, vecNeighborhood.end()); // default is X1 < X2 < ...

            // Store core dist
            vec_CORE_DIST[n] = vecNeighborhood[PARAM_DBSCAN_MINPTS - 2]; // scikit learn includes the point itself, and C++ index start from 0

            // test
//            cout << "Core dist: " << vec_CORE_DIST[n] << endl;
//            sort(vecNeighborhood.begin(), vecNeighborhood.end());
//
//            cout << "Begin testing " << endl;
//            for (int i = 0; i < vecNeighborhood.size(); ++i)
//            {
//                cout << vecNeighborhood[i] << endl;
//            }
//
//            cout << "End testing " << endl;
        }
    }
//    float fSum = 0.0;
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//        fSum += vec2D_OPTICS_NeighborDist[n].size();
//    cout << "Size of data structure: " << fSum << endl;

    cout << "CoreDist time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}


/**
Finding neighborhood and its distance to the query point using sampling
**/
void findCoreDist_sng()
{
    vec2D_OPTICS_NeighborDist = vector< vector< pair<int, float> > > (PARAM_DATA_N, vector< pair<int, float> >());

    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();
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

            if (fDist <= PARAM_DBSCAN_EPS) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
            {
#pragma omp critical
                {
                    vec2D_OPTICS_NeighborDist[n].push_back(make_pair(iPointIdx, fDist)); // duplicate at most twice
                    vec2D_OPTICS_NeighborDist[iPointIdx].push_back(make_pair(n, fDist)); // so vector is much better than map()
                }
            }
        }
    }

    cout << "Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();

    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);
    vec_CORE_DIST = FVector(PARAM_DATA_N);

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Store idx and float in the same order
        unordered_map<int, float> mapNeighborhood(vec2D_OPTICS_NeighborDist[n].begin(), vec2D_OPTICS_NeighborDist[n].end());

        vec2D_OPTICS_NeighborDist[n].clear();
        vec2D_OPTICS_NeighborDist[n].insert(vec2D_OPTICS_NeighborDist[n].end(), mapNeighborhood.begin(), mapNeighborhood.end());
        mapNeighborhood.clear();

        if ((int)vec2D_OPTICS_NeighborDist[n].size() >= PARAM_DBSCAN_MINPTS - 1)
        {
            bit_CORE_POINTS[n] = 1;

            // Only sort if this is the core
            FVector vecNeighborhood;
            for (const auto &ifPair : vec2D_OPTICS_NeighborDist[n])
                vecNeighborhood.push_back(ifPair.second);

            nth_element(vecNeighborhood.begin(), vecNeighborhood.begin() + PARAM_DBSCAN_MINPTS - 2, vecNeighborhood.end()); // default is X1 < X2 < ...

            // Store core dist
            vec_CORE_DIST[n] = vecNeighborhood[PARAM_DBSCAN_MINPTS - 2]; // scikit learn includes the point itself, and C++ index start from 0

            // test
//            cout << "Core dist: " << vec_CORE_DIST[n] << endl;
//            sort(vecNeighborhood.begin(), vecNeighborhood.end());
//
//            cout << "Begin testing " << endl;
//            for (int i = 0; i < vecNeighborhood.size(); ++i)
//            {
//                cout << vecNeighborhood[i] << endl;
//            }
//
//            cout << "End testing " << endl;
        }
    }
//    float fSum = 0.0;
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//        fSum += vec2D_OPTICS_NeighborDist[n].size();
//    cout << "Size of data structure: " << fSum << endl;

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
    We form cluster using dbscan
    **/
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    int iNumClusters = 0;
    formCluster(iNumClusters, p_vecLabels);

    cout << "Forming cluster time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    vector<IVector> nestedVec_ClusterStructure(iNumClusters, IVector()); // +1 since there is noise (label = -1)
    vector<IVector> nestedVec_ClusterCores(iNumClusters, IVector()); // +1 since there is noise (label = -1)

    // Get each connected components of dbscan
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
            vector< pair<int, float>> Xi_neighborhood = vec2D_OPTICS_NeighborDist[Xi];

            for (int j = 0; j < (int)Xi_neighborhood.size(); ++j)
            {
                int Xj = Xi_neighborhood[j].first;
                float dist_XiXj = Xi_neighborhood[j].second;

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

                for (int j = 0; j < (int)vec2D_OPTICS_NeighborDist[Xi].size(); ++j)
                {
                    int Xj = vec2D_OPTICS_NeighborDist[Xi][j].first;

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
                            int Xk = vec2D_OPTICS_NeighborDist[Xj][k].first;

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

    p_vecReachDist = FVector(PARAM_DATA_N, NEG_INF); // NEG_INF = -2^31

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
            vector< pair<int, float> > Xi_neighborhood = vec2D_OPTICS_NeighborDist[Xi];

            // For all: Xj is neighbor of core Xi
            for (int j = 0; j < (int)Xi_neighborhood.size(); ++j)
            {
                int Xj = Xi_neighborhood[j].first;

                // only update if it is not processed
                if (processSet[Xj])
                    continue;

                float dist_XiXj = Xi_neighborhood[j].second;

                float Xj_reach_dist_from_Xi = max(Xi_core_dist, dist_XiXj);

                if (p_vecReachDist[Xj] == NEG_INF)
                {
                    p_vecReachDist[Xj] = Xj_reach_dist_from_Xi;
//                    seedSet.insert(make_pair(Xj, p_vecReachDist[Xj]));
                    seedSet.push(IFPair(Xj, p_vecReachDist[Xj])); // reach from Xi

                }

                else if (p_vecReachDist[Xj] > Xj_reach_dist_from_Xi) // Xj is already reached by some point, but reach from Xi smaller
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
                    vector< pair<int, float> > Xj_neighborhood = vec2D_OPTICS_NeighborDist[Xj];

                    // Xj is neighbor of core Xi
                    for (int k = 0; k < (int)Xj_neighborhood.size(); ++k)
                    {
                        int Xk = Xj_neighborhood[k].first;

                        // only update if it is not processed
                        if (processSet[Xk])
                            continue;

                        float dist_XjXk = Xj_neighborhood[k].second;
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

/**
Fast OPTICS: optimize speed
- We use parallel dbscan index to preprocess and finding neighborhoods
- If m is large, then findCoreDist_Asym is faster for multi-threading
    + findCoreDist_Asym only adds x into B(q) if dist(x, q) < eps
    + findCoreDist addx x into B(q) and q into B(x) if dist(x, q) < eps - not multi-threading friendly
**/
void sOptics()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    parDbscanIndex();

    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
//    findCoreDist_Asym(); // Faster for multi-threading but loss some accuracy
    findCoreDist();
    cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    IVector vecOrder;
    FVector vecReachDist;

//    formOptics(vecLabels, vecOrder, vecReachDist);

    formOptics_scikit(vecOrder, vecReachDist);
    cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + "_L" + int2str(PARAM_DISTANCE) +
                        "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                        "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                        "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                        "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                        "_TopM_" + int2str(PARAM_PROJECTION_TOP_M) +
                        "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputOptics(vecOrder, vecReachDist, sFileName);
	}
}

/**
Sng-based OPTICS: optimize speed
- We use parallel dbscan index to preprocess and finding neighborhoods
- If m is large, then findCoreDist_Asym is faster for multi-threading
    + findCoreDist_Asym only adds x into B(q) if dist(x, q) < eps
    + findCoreDist addx x into B(q) and q into B(x) if dist(x, q) < eps - not multi-threading friendly
**/
void sngOptics()
{
    chrono::steady_clock::time_point begin;
    begin = chrono::steady_clock::now();

    // Find core point
    findCoreDist_sng();
    cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    IVector vecOrder;
    FVector vecReachDist;

//    formOptics(vecLabels, vecOrder, vecReachDist);

    formOptics_scikit(vecOrder, vecReachDist);
    cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = PARAM_OUTPUT_FILE + "_L" + int2str(PARAM_DISTANCE) +
                           "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                           "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                            "_Prob_" + int2str(round(1000 * PARAM_SAMPLING_PROB));

        outputOptics(vecOrder, vecReachDist, sFileName);
    }
}

