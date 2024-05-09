//
// Created by hphi344 on 10/05/24.
//
#include "../../include/sDBSCAN.h"

/** Multi-thread sDbscan **/
void sDbscan()
{
    chrono::time_point<steady_clock, chrono::nanoseconds> begin;
    begin = chrono::steady_clock::now();

    parDbscanIndex();
    cout << "Build index time = " << chrono::duration_cast<duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
//    findCorePoints_Asym();
    findCorePoints();
    cout << "Find core points time = " << chrono::duration_cast<
            duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    vector<int> vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        basic_string<char> sFileName = PARAM_OUTPUT_FILE + + "_L" + int2str(PARAM_DISTANCE) +
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
    chrono::time_point<steady_clock, chrono::nanoseconds> begin;
    begin = chrono::steady_clock::now();
    seqDbscanIndex();
    cout << "Build index time = " << chrono::duration_cast<duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    findCorePoints();
    cout << "Find core points time = " << chrono::duration_cast<
            duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    int iNumClusters = 0;
    vector<int> vecLabels;
    formCluster(iNumClusters, vecLabels);
    cout << "Form clusters time = " << chrono::duration_cast<duration < int64_t, milli>>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        basic_string<char> sFileName = PARAM_OUTPUT_FILE + + "_L" + int2str(PARAM_DISTANCE) +
                                       "_Eps_" + int2str(round(1000 * PARAM_DBSCAN_EPS)) +
                                       "_MinPts_" + int2str(PARAM_DBSCAN_MINPTS) +
                                       "_NumEmbed_" + int2str(PARAM_KERNEL_EMBED_D) +
                                       "_NumProjection_" + int2str(PARAM_NUM_PROJECTION) +
                                       "_TopM_" + int2str(PARAM_PROJECTION_TOP_M) +
                                       "_TopK_" + int2str(PARAM_PROJECTION_TOP_K);

        outputDbscan(vecLabels, sFileName);
	}

}