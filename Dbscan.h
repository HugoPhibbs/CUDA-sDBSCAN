#pragma once

#include "Header.h"
#include "Utilities.h"



void parDbscanIndex_L2();
//void parDbscanIndex_Sort(); // Only for L2 testing: slower than using PQ

//void parDbscanIndex_Metric();
//void parDbscanIndex_NonMetric();

void parDbscanIndex();
void seqDbscanIndex();

void findCorePoints_Asym();
void findCorePoints();
void findCorePoints_uDbscan(); // based on sampling uDbscan++ - note that formingCluster will be very different from uDbscan++
void findCorePoints_sngDbscan(); // based on sampling

void formCluster(int &, IVector &);
void formCluster_Asym(int &, IVector &);

void sDbscan();
void seq_sDbscan();
void sngDbscan();

void clusterNoise(IVector&, int);







