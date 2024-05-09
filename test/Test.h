#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

#include "../include/Header.h"
#include "../include/Utilities.h"
#include "../include/InputParser.h"
#include "../include/DBSCAN.h"
#include "../include/Optics.h"


void FourierEmbed_L2();
void FourierEmbed_Nonmetric();

void parRandomProjection();
void seqRandomProjection();

void findCorePoints_PrecomputedOptics();
void findCorePoints_maxEps(float);
void findCorePoints_sngDbscan_maxEps(float);

void test_sDbscan(int);
//void test_sDbscan_L2(int); // Test FWHT twice for Random Fourier embedding & random projection

void test_sDbscan_Asym(int);
void test_sOptics_Asym();

void test_uDbscan(int);
void test_sngDbscan(int);
void test_naiveDbscan();

#endif
