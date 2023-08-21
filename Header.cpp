#include "Header.h"

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

int PARAM_DATA_N; // Number of points (rows) of X
int PARAM_DATA_D; // Number of dimensions
int PARAM_DISTANCE; // Currently support L1, L2

float PARAM_DBSCAN_EPS; // radius eps
int PARAM_DBSCAN_MINPTS; // MinPts

int PARAM_PROJECTION_TOP_K; // TopK largest and smallest random vector
string PARAM_OUTPUT_FILE;

int PARAM_KERNEL_EMBED_D;
int PARAM_NUM_PROJECTION;
float PARAM_KERNEL_SIGMA;
float PARAM_KERNEL_INTERVAL_SAMPLING;

MatrixXf MATRIX_X;

bool PARAM_INTERNAL_SAVE_OUTPUT = true;
int PARAM_INTERNAL_NUM_ROTATION = 3;

boost::dynamic_bitset<> bit_CORE_POINTS;
vector<IVector> vec2D_DBSCAN_Neighbor;
vector<unordered_set<int>> set2D_DBSCAN_Neighbor;

FVector vec_CORE_DIST;
vector<FVector> vec2D_DBSCAN_NeighborDist; // correlated to vec2D_DBSCAN_Neighbor

MatrixXi MATRIX_TOP_K;
MatrixXi MATRIX_TOP_MINPTS;

