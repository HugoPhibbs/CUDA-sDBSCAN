#pragma once

#include "fht.h"

#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <random>

#include <chrono>
#include <iostream> // cin, cout

//#include <boost/multi_array.hpp>
#include <boost/dynamic_bitset.hpp>

#define PI				3.141592653589793238460
#define NEG_INF        -2147483648 // -2^32
#define POS_INF        2147483647 // 2^31-1
#define EPSILON         0.000001 // 2^31-1

using namespace Eigen;
using namespace std;

typedef vector<float> FVector;
typedef vector<int> IVector;

//typedef vector<uint32_t> I32Vector;
//typedef vector<uint64_t> I64Vector;


//typedef boost::multi_array<int, 3> IVector3D;

struct myComp
{

    constexpr bool operator()(
        pair<double, int> const& a,
        pair<double, int> const& b)
    const noexcept
    {
        return a.first > b.first;
    }
};

struct IFPair
{
    int m_iIndex;
    float	m_fValue;

    IFPair()
    {
        m_iIndex = 0;
        m_fValue = 0.0;
    }

    IFPair(int p_iIndex, double p_fValue)
    {
        m_iIndex = p_iIndex;
        m_fValue = p_fValue;
    }

    // Overwrite operation < to get top K largest entries
    bool operator<(const IFPair& p) const
    {
        return m_fValue < p.m_fValue;
    }

    bool operator>(const IFPair& p) const
    {
        return m_fValue > p.m_fValue;
    }
};

typedef priority_queue<IFPair, vector<IFPair>, greater<IFPair>> Min_PQ_Pair;

extern int PARAM_DATA_N; // Number of points (rows) of X
extern int PARAM_DATA_D; // Number of dimensions

extern int PARAM_DISTANCE;
extern int PARAM_NUM_THREADS;

extern float PARAM_DBSCAN_EPS; // radius eps
extern int PARAM_DBSCAN_MINPTS; // MinPts
extern int PARAM_DBSCAN_CLUSTER_NOISE; // assign cluster label to noise

extern int PARAM_PROJECTION_TOP_K; // TopK random vectors closest and furthest to each point
extern int PARAM_PROJECTION_TOP_M; // TopM points closest and furthest to random vector
extern string PARAM_OUTPUT_FILE;

extern int PARAM_KERNEL_EMBED_D;
extern int PARAM_NUM_PROJECTION;
extern int PARAM_INTERNAL_FWHT_PROJECTION;

extern float PARAM_KERNEL_SIGMA;
extern float PARAM_KERNEL_INTERVAL_SAMPLING;

extern float PARAM_SAMPLING_PROB;

extern bool PARAM_INTERNAL_SAVE_OUTPUT;
extern int PARAM_INTERNAL_NUM_ROTATION;

extern MatrixXf MATRIX_X;


// Data structure
extern MatrixXi MATRIX_TOP_K;
extern MatrixXi MATRIX_TOP_M;

extern boost::dynamic_bitset<> bit_CORE_POINTS;
extern vector<IVector> vec2D_DBSCAN_Neighbor;

extern FVector vec_CORE_DIST;
extern vector< vector< pair<int, float> > > vec2D_OPTICS_NeighborDist;

extern MatrixXf MATRIX_R; // We can store its seed to save space
extern boost::dynamic_bitset<> bitHD3_Proj;

extern float PARAM_INTERNAL_TEST_EPS_RANGE;
extern int PARAM_INTERNAL_TEST_UNITS;
extern int PARAM_TEST_REPEAT;

