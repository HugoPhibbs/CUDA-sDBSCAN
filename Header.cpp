#include "Header.h"

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

int PARAM_DATA_N; // Number of points (rows) of X
int PARAM_DATA_D; // Number of dimensions
int PARAM_DISTANCE; // Currently support L1, L2

int PARAM_NUM_THREADS;

float PARAM_DBSCAN_EPS; // radius eps
int PARAM_DBSCAN_MINPTS; // MinPts
int PARAM_DBSCAN_CLUSTER_NOISE; // assign label to noise

int PARAM_PROJECTION_TOP_K; // TopK random vectors closest and furthest to each point
int PARAM_PROJECTION_TOP_M; // TopM points closest and furthest to random vector

string PARAM_OUTPUT_FILE;

int PARAM_KERNEL_EMBED_D;
int PARAM_NUM_PROJECTION;
int PARAM_INTERNAL_FWHT_PROJECTION;

float PARAM_KERNEL_SIGMA;
float PARAM_KERNEL_INTERVAL_SAMPLING;

float PARAM_SAMPLING_PROB;

MatrixXf MATRIX_X;

bool PARAM_INTERNAL_SAVE_OUTPUT = true;
int PARAM_INTERNAL_NUM_ROTATION = 3;

boost::dynamic_bitset<> bit_CORE_POINTS;
vector<IVector> vec2D_DBSCAN_Neighbor;

FVector vec_CORE_DIST;
vector< vector< pair<int, float> > > vec2D_OPTICS_NeighborDist;

MatrixXi MATRIX_TOP_K;
MatrixXi MATRIX_TOP_M;


MatrixXf MATRIX_R; // We can store its seed to save space
boost::dynamic_bitset<> bitHD3_Proj;

float PARAM_INTERNAL_TEST_EPS_RANGE;
int PARAM_INTERNAL_TEST_UNITS;
int PARAM_TEST_REPEAT;
