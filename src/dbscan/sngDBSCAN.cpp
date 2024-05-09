//
// Created by hphi344 on 10/05/24.
//
#include <omp.h>
#include "../../include/Header.h"
#include "../../include/Utilities.h"
#include "../../include/DBSCAN.h"

/**
This is a multi-thread friendly implementation of sngDBSCAN, much faster than the original sngDBSCAN on large data.

vec2D_DBSCAN_Neighbor: For each sampled point, store its true neighborhood (within radius eps)
bit_CORE_POINTS: Store a bit vector presenting for core point
**/
void findCorePoints_sngDbscan()
{
    vec2D_DBSCAN_Neighbor = vector<vector<int>> (PARAM_DATA_N, vector<int>());
    bit_CORE_POINTS = boost::dynamic_bitset<>(PARAM_DATA_N);

    int iNumSamples = ceil(1.0 * PARAM_DATA_N * PARAM_SAMPLING_PROB);

    chrono::time_point<steady_clock, chrono::nanoseconds> begin;
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
            mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31, 0x9908b0dfUL, 11, 0xffffffffUL, 7, 0x9d2c5680UL, 15, 0xefc60000UL, 18, 1812433253UL> gen(rd()); // mersenne_twister_engine seeded with rd()
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

    cout << "Find core points time = " << chrono::duration_cast<
            duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    cout << "Number of core points: " << bit_CORE_POINTS.count() << endl;

}

/** Multi-thread sngDbscan **/
void sngDbscan()
{
    cout << "Running sngDBSCAN" << endl;

    chrono::time_point<steady_clock, chrono::nanoseconds> begin, start;
    begin = chrono::steady_clock::now();
    start = chrono::steady_clock::now();
    findCorePoints_sngDbscan();
    cout << "Find core points time = " << chrono::duration_cast<
            duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    vector<int> vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "sngDbscan time = " << chrono::duration_cast<duration < int64_t, milli>>(chrono::steady_clock::now() - start).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        basic_string<char> sFileName = PARAM_OUTPUT_FILE + "_L" + int2str(PARAM_DISTANCE) +
                                       "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                                       "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                                       "_Prob_" + int2str(round(1000 * PARAM_SAMPLING_PROB));

        outputDbscan(vecLabels, sFileName);
    }

}