//
// Created by hphi344 on 10/05/24.
//

#include "../../include/gsDBSCAN.h"

/**
 * Performs the gs dbscan algorithm
 *
 * @param X Eigen MatrixXf matrix for the X data points
 * @param D Eigen MatrixXf matrix for the D random vectors
 * @param minPts min number of points as per the DBSCAN algorithm
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 * @param skip_pre_checks boolean flag to skip the pre-checks
 */
void gsDbscan(MatrixXf X, MatrixXf D, int minPts, int k, int m, float eps, bool skip_pre_checks) {
    // TODO implement me!
}

/**
 * Performs the pre-checks for the gs dbscan algorithm
 *
 * @param X Eigen MatrixXf matrix for the X data points
 * @param D Eigen MatrixXf matrix for the D random vectors
 * @param minPts min number of points as per the DBSCAN algorithm
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 */
void preChecks(MatrixXf X, MatrixXf D, int minPts, int k, int m, float eps) {
    // TODO implement me!
}

/**
 * Performs the pre-processing for the gs dbscan algorithm
 *
 * @param D Eigen MatrixXf matrix for the D random vectors
 * @return A, B matrices as per the GS-DBSCAN algorithm. A has size (n, 2*k) and B has size (2*k, m)
 */
void preProcessing(MatrixXf D){
    // TODO implement me!
}

/**
 * Performs random projections between the X dataset and the random vector
 *
 * @param X MatrixXf matrix for the X data points
 * @param D MatrixXf matrix for the D data points
 * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
 * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
 * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
 * @return MatrixXf matrix for the random projections
 */
void randomProjections(MatrixXf X, MatrixXf D, int k, int m, float eps) {
    // TODO implement me!
}

/**
 * Constructs the A and B matrices as per the GS-DBSCAN algorithm
 *
 * A stores the indices of the closest and furthest random vectors per data point.
 *
 * @param X matrix containing the X dataset vectors
 * @param D matrix containing the random vectors
 * @param k k parameter as per the DBSCAN algorithm
 * @param m m parameter as per the DBSCAN algorithm
 */
void constructABMatrices(MatrixXf X, MatrixXf D, int k, int m) {
    // TODO implement me!
}

/**
 * Finds the distances between each of the query points and their candidate neighbourhood vectors
 *
 * @param X matrix containing the X dataset vectors
 * @param A A matrix, see constructABMatrices
 * @param B B matrix, see constructABMatrices
 * @param alpha float for the alpha parameter to tune the batch size
 */
void findDistances(MatrixXf X, MatrixXf A, MatrixXf B, float alpha = 1.2) {
    // TODO implement me!
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
int findDistanceBatchSize(int n, int d, int k, int m, float alpha = 1.2) {
    return 1; // TODO implement me!
}

/**
 * Constructs the cluster graph for the DBSCAN algorithm
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param eps epsilon DBSCAN density param
 * @param k k parameter of the DBSCAN algorithm
 * @param m m parameter of the DBSCAN algorithm
 * @return a vector containing the (flattened) adjacency list of the cluster graph, along with another list V containing the starting index of each query vector in the adjacency list
 */
void constructClusterGraph(MatrixXf distances, float eps, int k, int m) {
    // TODO implement me!
}

/**
 * Assembles the adjacency list for the cluster graph
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param E vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
 * @param V vector containing the starting index of each query vector in the resultant adjacency list (See the G-DBSCAN algorithm)
 * @param A A matrix, see constructABMatrices
 * @param B B matrix, see constructABMatrices
 * @param eps epsilon DBSCAN density param
 * @param blockSize size of each block when calculating the adjacency list - essentially the amount of query vectors to process per block
 */
void assembleAdjacencyList(MatrixXf distances, int E, int V, MatrixXf A, MatrixXf B, float eps, int blockSize = 1024) {
    // TODO implement me!
}

/**
 * Kernel for constructing part of the cluster graph adjacency list for a particular vector
 *
 * @param distances matrix containing the distances between each query vector and it's candidate vectors
 * @param adjacencyList
 * @param V vector containing the degree of each query vector (how many candidate vectors are within eps distance of it)
 * @param E vector containing the starting index of each query vector in the adjacency list
 * @param eps epsilon DBSCAN density param
 */
__global__ void constructAdjacencyListForQueryVector(float *distances, int *adjacencyList, int V, int E, float eps) {
    // TODO implement me!
}