#include <iostream>
#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"

#include "Test.h"
#include "Dbscan.h"

#include <time.h> // for time(0) to generate different random number
#include <stdlib.h>
#include <sys/time.h> // for gettimeofday
#include <stdio.h>
#include <unistd.h>

#include <omp.h>

int main(int nargs, char** args)
{
    srand(time(NULL)); // should only be called once for random generator

//    cout << "RAM before loading data" << endl;
//    getRAM();

    /************************************************************************/
	int iType = loadInput(nargs, args);

//    cout << "RAM after loading data" << endl;
//    getRAM();

	/************************************************************************/
	/* Approaches                                             */
	/************************************************************************/

    chrono::steady_clock::time_point begin, end;

    /************************************************************************/
	/* Algorithms                                             */
	/************************************************************************/
	switch (iType)
	{
        case 1:
        {
            begin = chrono::steady_clock::now();

            /**  For testing
            **/
            // L2 embed
//    chrono::steady_clock::time_point begin;
//    begin = chrono::steady_clock::now();
//    L2Embedding();
//    cout << "L2 Embedding time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
//
//    // Random projection
//    begin = chrono::steady_clock::now();
//    parRandomProjection();
////    seqRandomProjection();
//    cout << "Random projection time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
//
//            // Test Embedding for divergence
//            begin = chrono::steady_clock::now();
//            FourierEmbed_Nonmetric();
//            cout << "Chi2/JS Embedding time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;


            fastDbscan(); // speed friendly
//            memoryDbscan(); // memory friendly

            end = chrono::steady_clock::now();
            cout << "DBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;

        }

        case 2:
        {

            begin = chrono::steady_clock::now();

            fastOptics(); // for speed
//            memoryOptics(); // for memory

            end = chrono::steady_clock::now();
            cout << "OPTICS Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

    }
}

