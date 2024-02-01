#include <iostream>
#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"

#include "Test.h"
#include "Dbscan.h"
#include "Optics.h"

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

    // Only for testing

    PARAM_INTERNAL_TEST_UNITS = 10;

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


            sDbscan(); // speed friendly
//            memoryDbscan(); // memory friendly

            end = chrono::steady_clock::now();
            cout << "DBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;

        }

        case 2:
        {

            begin = chrono::steady_clock::now();

            sOptics(); // for speed
//            memoryOptics(); // for memory

            end = chrono::steady_clock::now();
            cout << "OPTICS Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

        case 3:
        {

            begin = chrono::steady_clock::now();

//            sngOptics(); // for speed
//            memoryOptics(); // for memory

            end = chrono::steady_clock::now();
            cout << "sngOPTICS Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

        // Test sDBSCAN
        case 30:
        {

            begin = chrono::steady_clock::now();
            float fBaseEps = PARAM_DBSCAN_EPS;
            for (int i = 0; i < PARAM_TEST_REPEAT; ++i)
            {
                PARAM_DBSCAN_EPS = fBaseEps; // need to reset baseEps
                cout << "Base sDbscan eps: " << PARAM_DBSCAN_EPS << " at time " << i << endl;
                test_sDbscan(i); // speed friendly
            }

            end = chrono::steady_clock::now();
            cout << "sDBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

        // Test DBSCAN L2 FWHT twice
        case 31:
        {

            begin = chrono::steady_clock::now();

            float fBaseEps = PARAM_DBSCAN_EPS;
            for (int i = 0; i < 5; ++i)
            {
                PARAM_DBSCAN_EPS = fBaseEps + i * PARAM_INTERNAL_TEST_EPS_RANGE;
                for (int j = 0; j < 5; ++j)
                {
                    cout << "Dbscan eps: " << PARAM_DBSCAN_EPS << " at time " << j << endl;
                    test_sDbscan_L2(j); // speed friendly
                }
            }

            end = chrono::steady_clock::now();
            cout << "DBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

        // Test DBSCAN L2 FWHT twice + Asymmetric update
        case 32:
        {

            begin = chrono::steady_clock::now();


            float fBaseEps = PARAM_DBSCAN_EPS;
            for (int i = 0; i < 5; ++i)
            {
                PARAM_DBSCAN_EPS = fBaseEps + i * PARAM_INTERNAL_TEST_EPS_RANGE;
                for (int j = 0; j < 5; ++j)
                {
                    cout << "Dbscan eps: " << PARAM_DBSCAN_EPS << " at time " << j << endl;
                    test_sDbscan_Asym(j); // speed friendly
                }
            }


            end = chrono::steady_clock::now();
            cout << "DBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

        // sngDbscan
        case 4:
        {
            begin = chrono::steady_clock::now();

            sngDbscan();

            end = chrono::steady_clock::now();
            cout << "sngDBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

        // Test uniform Dbscan++
        case 41:
        {
            begin = chrono::steady_clock::now();

            float fBaseEps = PARAM_DBSCAN_EPS;
            for (int i = 0; i < 5; ++i)
            {
                PARAM_DBSCAN_EPS = fBaseEps + i * PARAM_INTERNAL_TEST_EPS_RANGE;
                for (int j = 0; j < 5; ++j)
                {
                    cout << "uDbscan eps: " << PARAM_DBSCAN_EPS << " at time " << j << endl;
                    test_uDbscan(j); // speed friendly
                }
            }


            end = chrono::steady_clock::now();
            cout << "Test uDBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }


        // Test sng Dbscan++
        case 42:
        {

            begin = chrono::steady_clock::now();

            float fBaseEps = PARAM_DBSCAN_EPS;

            if (PARAM_SAMPLING_PROB < 1.0)
            {
                // Repeat 5 times
                for (int i = 0; i < PARAM_TEST_REPEAT; ++i)
                {
                    PARAM_DBSCAN_EPS = fBaseEps;
                    cout << "Base sngDbscan eps: " << PARAM_DBSCAN_EPS << " at time " << i << endl;
                    test_sngDbscan(i);
                }

            }
            else
            {
                for (int i = 0; i < 5; ++i)
                {
                    PARAM_DBSCAN_EPS = fBaseEps + i * PARAM_INTERNAL_TEST_EPS_RANGE;
                    cout << "naiveDbscan eps: " << PARAM_DBSCAN_EPS << endl;
                    test_naiveDbscan(); // speed friendly
                }
            }



            end = chrono::steady_clock::now();
            cout << "Test sngDBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }
    }
}

