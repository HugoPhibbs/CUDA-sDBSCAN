//
// Created by hphi344 on 10/05/24.
//

#include "Header.h"
#include "Utilities.h"

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#endif //DBSCANCEOS_GSDBSCAN_H

void gsDbscan(MatrixXf X, MatrixXf D, int minPts, int k, int m, float eps, bool skip_pre_check);

void preChecks(MatrixXf X, MatrixXf D, int minPts, int k, int m, float eps);

void preProcessing(MatrixXf D);

void randomProjections(MatrixXf X, MatrixXf D, int k, int m, float eps);

void constructABMatrices(MatrixXf X, MatrixXf D, int k, int m);

void findDistances(MatrixXf X, MatrixXf A, MatrixXf B, float alpha = 1.2);

int findDistanceBatchSize(int n, int d, int k, int m, float alpha = 1.2);

void constructClusterGraph(MatrixXf distances, float eps, int k, int m);

void assembleAdjacencyList(MatrixXf distances, int E, int V, MatrixXf A, MatrixXf B, float eps, int blockSize = 1024);
