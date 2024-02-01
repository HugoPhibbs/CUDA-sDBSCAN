#include "InputParser.h"
#include "Header.h"

#include <stdlib.h>     /* atoi */
#include <iostream> // cin, cout
#include <fstream> // fscanf, fopen, ofstream

#include <vector>
#include <string.h> // strcmp

int loadInput(int nargs, char** args)
{
    if (nargs < 6)
        exit(1);

    // NumPoints n
    bool bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numPts") == 0)
        {
            PARAM_DATA_N = atoi(args[i + 1]);
            cout << "Number of rows/points of X: " << PARAM_DATA_N << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Number of rows/points is missing !" << endl;
        exit(1);
    }

    // Dimension
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numDim") == 0)
        {
            PARAM_DATA_D = atoi(args[i + 1]);
            cout << "Number of columns/dimensions: " << PARAM_DATA_D << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        cerr << "Error: Number of columns/dimensions is missing !" << endl;
        exit(1);
    }


    // DBSCAN or OPTICS

    // Algorithm
    int iAlgType = 0;
    bSuccess = false;

    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--alg") == 0)
        {
            if (strcmp(args[i + 1], "sDbscan") == 0)
            {
                iAlgType = 1;
                cout << "Algorithm: sDBSCAN... " << endl;
            }
            else if (strcmp(args[i + 1], "sOptics") == 0)
            {
                iAlgType = 2;
                cout << "Algorithm: sOPTICS... " << endl;
            }
            else if (strcmp(args[i + 1], "sngOptics") == 0)
            {
                iAlgType = 3;
                cout << "Algorithm: sngOPTICS... " << endl;
            }
            else if (strcmp(args[i + 1], "test_sDbscan") == 0)
            {
                iAlgType = 30;
                cout << "Algorithm: Test sDbscan... " << endl;
            }
            else if (strcmp(args[i + 1], "test_sDbscanL2") == 0)
            {
                iAlgType = 31;
                cout << "Algorithm: Test sDbscan L2 with FWHT... " << endl;
            }
            else if (strcmp(args[i + 1], "test_sDbscanAsym") == 0)
            {
                iAlgType = 32;
                cout << "Algorithm: Test asymmetric sDbscan... " << endl;
            }
            else if (strcmp(args[i + 1], "test_sOpticsAsym") == 0)
            {
                iAlgType = 33;
                cout << "Algorithm: Test asymmetric sOptics... " << endl;
            }
            else if (strcmp(args[i + 1], "sngDbscan") == 0)
            {
                iAlgType = 4;
                cout << "Algorithm: sngDbscan... " << endl;
            }
            else if (strcmp(args[i + 1], "test_uDbscan") == 0)
            {
                iAlgType = 41;
                cout << "Algorithm: Test uniform DBSCAN++... " << endl;
            }
            else if (strcmp(args[i + 1], "test_sngDbscan") == 0)
            {
                iAlgType = 42;
                cout << "Algorithm: Test sampling Dbscan... " << endl;
            }
            else
            {
                cerr << "Error: The algorithm is not supported !" << endl;
                exit(1);
            }

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Algorithm is missing !" << endl;
        exit(1);
    }

    // MinPTS
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--minPts") == 0)
        {
            PARAM_DBSCAN_MINPTS = atoi(args[i + 1]);
            cout << "minPts: " << PARAM_DBSCAN_MINPTS << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        cerr << "Error: minPts is missing !" << endl;
        exit(1);
    }

    // Eps
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--eps") == 0)
        {
            PARAM_DBSCAN_EPS = atof(args[i + 1]);
            cout << "Radius eps: " << PARAM_DBSCAN_EPS << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Eps is missing !" << endl;
        exit(1);
    }

    // Only for testing Eps
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--epsRange") == 0)
        {
            PARAM_INTERNAL_TEST_EPS_RANGE = atof(args[i + 1]);
            cout << "Radius eps range: " << PARAM_INTERNAL_TEST_EPS_RANGE << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_INTERNAL_TEST_EPS_RANGE = 0;
    }

    // Clustering noisy points
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--clusterNoise") == 0)
        {
            PARAM_DBSCAN_CLUSTER_NOISE = atoi(args[i + 1]);
            cout << "Cluster noise option: " << PARAM_DBSCAN_CLUSTER_NOISE << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cout << "Default: We do not cluster the noisy points." << endl;
        PARAM_DBSCAN_CLUSTER_NOISE = 0;
    }

    // Sampling probability of uDbscan, kDbscan, sngDbscan
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--p") == 0)
        {
            PARAM_SAMPLING_PROB = atof(args[i + 1]);
            cout << "Sampling probability: " << PARAM_SAMPLING_PROB << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_SAMPLING_PROB = 0.01;
        cout << "Default sampling probability: " << PARAM_SAMPLING_PROB << endl;
    }


    /** NOTE: Reading IO is not safe for OpenMP **/

    // Read the row-wise matrix X, and convert to col-major Eigen matrix
//    cout << "Read row-wise X, it will be converted to col-major Eigen matrix of size D x N..." << endl;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--X") == 0)
        {
            FILE *f = fopen(args[i + 1], "r");
            if (!f)
            {
                cerr << "Error: Data file does not exist !" << endl;
                exit(1);
            }

            MATRIX_X = MatrixXf::Zero(PARAM_DATA_D, PARAM_DATA_N);

            //FVector vecTempX(PARAM_DATA_D * PARAM_DATA_N, 0.0); // this vector<float> has limited length
            // Each line is a vector of D dimensions
            for (int n = 0; n < PARAM_DATA_N; ++n)
            {
                for (int d = 0; d < PARAM_DATA_D; ++d)
                {
                    //fscanf(f, "%f", &vecTempX[n * PARAM_DATA_D + d]);
                    // cout << vecTempX[n + d * PARAM_DATA_N] << " ";
                    // or
                    //                    fscanf(f, "%f", &x);
                    //                    MATRIX_X(d, n) = x;

                    fscanf(f, "%f", &MATRIX_X(d, n));

                }
                // cout << endl;
            }

            // Matrix_X is col-major
//            MATRIX_X = Map<MatrixXf>(vecTempX.data(), PARAM_DATA_D, PARAM_DATA_N);

    //        MATRIX_X.transpose();
    //        cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

            /**
            Print the first col (1 x N)
            Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
            **/
//            cout << MATRIX_X.col(0) << endl << endl;
//            cout << "In memory (col-major):" << endl;
//            for (int n = 0; n < 10; n++)
//                cout << *(MATRIX_X.data() + n) << "  ";
//            cout << endl << endl;

            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        cerr << "Error: Cannot reading the data set !" << endl;
        exit(1);
    }



    // Kernel embedding
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numEmbed") == 0)
        {
            PARAM_KERNEL_EMBED_D = atoi(args[i + 1]);
            cout << "Number of kernel embedded dimensions: " << PARAM_KERNEL_EMBED_D << endl;
            cout << "If using L1 and L2, this must be an even number. Must be pow(2) > D for L2 for FWHT." << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_KERNEL_EMBED_D = PARAM_DATA_D;
        cout << "Default number of kernel embedded dimensions = numDim = " << PARAM_KERNEL_EMBED_D << endl;
    }

    // Random projection
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numProj") == 0)
        {
            PARAM_NUM_PROJECTION = atoi(args[i + 1]);
            cout << "Number of projections: " << PARAM_NUM_PROJECTION << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        int iTemp = ceil(log2(1.0 * PARAM_DATA_D));
        PARAM_NUM_PROJECTION = max(512, 1 << iTemp);
        cout << "Default number of projections: " << PARAM_NUM_PROJECTION << endl;
    }

    // Identify PARAM_INTERNAL_FWHT_PROJECTION to use FWHT
    if (PARAM_NUM_PROJECTION < PARAM_KERNEL_EMBED_D)
        PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_KERNEL_EMBED_D)));
    else
        PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_NUM_PROJECTION)));

    cout << "Internal FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;


    // Top-K close/far random vectors
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--topKProj") == 0)
        {
            PARAM_PROJECTION_TOP_K = atoi(args[i + 1]);
            cout << "Top-k closest/furthest vectors: " << PARAM_PROJECTION_TOP_K << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_PROJECTION_TOP_K = 5;
        cout << "Default top-k closest/furthest vectors: " << PARAM_PROJECTION_TOP_K << endl;
    }

    // m >= MinPts
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--topMProj") == 0)
        {
            PARAM_PROJECTION_TOP_M = atoi(args[i + 1]);
            cout << "Top-m: " << PARAM_PROJECTION_TOP_M << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_PROJECTION_TOP_M = PARAM_DBSCAN_MINPTS;
        cout << "Default top-m = minPts = " << PARAM_PROJECTION_TOP_M << endl;
    }

    // Distance measurement
    PARAM_DISTANCE = 2;
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--dist") == 0)
        {
            if (strcmp(args[i + 1], "Cosine") == 0)
            {
                PARAM_DISTANCE = 0;
                cout << "Cosine distance - no kernel embedding" << endl;
            }
            else if (strcmp(args[i + 1], "L1") == 0)
            {
                PARAM_DISTANCE = 1;
                cout << "L1 distance" << endl;
            }
            else if (strcmp(args[i + 1], "L2") == 0)
            {
                PARAM_DISTANCE = 2;
                cout << "L2 distance" << endl;
            }
            else if (strcmp(args[i + 1], "Chi2") == 0)
            {
                PARAM_DISTANCE = 3;
                cout << "Chi2 distance" << endl;
            }
            else if (strcmp(args[i + 1], "JS") == 0)
            {
                PARAM_DISTANCE = 4;
                cout << "Jensen-Shannon distance" << endl;
            }
            else
            {
                cout << "Default L2 distance" << endl;
                PARAM_DISTANCE = 2;
            }

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cout << "Default L2 distance" << endl;
        PARAM_DISTANCE = 2;
    }

    // Scale sigma of kernel L2 and L1
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--sigma") == 0)
        {
            PARAM_KERNEL_SIGMA = atof(args[i + 1]);
            cout << "Sigma: " << PARAM_KERNEL_SIGMA << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        if (PARAM_DISTANCE == 1)
        {
            PARAM_KERNEL_SIGMA = PARAM_DBSCAN_EPS;
            cout << "Default sigma = eps for L1: " << PARAM_KERNEL_SIGMA << endl;
        }
        else if (PARAM_DISTANCE == 2)
        {
            PARAM_KERNEL_SIGMA = 2 * PARAM_DBSCAN_EPS;
            cout << "Default sigma = 2*eps for L2: " << PARAM_KERNEL_SIGMA << endl;
        }
    }

    // Sampling ratio used on Chi2 and JS - TPAMI 12 (interval_sampling in scikit-learn)
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--samplingRatio") == 0)
        {
            PARAM_KERNEL_INTERVAL_SAMPLING = atof(args[i + 1]);
            cout << "Sampling ratio for divergence distance: " << PARAM_KERNEL_INTERVAL_SAMPLING << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess && ((PARAM_DISTANCE == 3) || (PARAM_DISTANCE == 4)))
    {
        PARAM_KERNEL_INTERVAL_SAMPLING = 0.4;
        cout << "Default sampling ratio for divergence distance: " << PARAM_KERNEL_INTERVAL_SAMPLING << endl;
    }

    // Output
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--output") == 0)
        {
            PARAM_OUTPUT_FILE = args[i + 1];

            if (PARAM_OUTPUT_FILE.empty())
                PARAM_INTERNAL_SAVE_OUTPUT = false;
            else
                PARAM_INTERNAL_SAVE_OUTPUT = true;

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cout << "No output file" << endl;
        PARAM_INTERNAL_SAVE_OUTPUT = false;
    }

    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numRepeat") == 0)
        {
            PARAM_TEST_REPEAT = atoi(args[i + 1]);

            cout << "Number of testing repeats: " << PARAM_TEST_REPEAT << endl;

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_TEST_REPEAT = 1;
        cout << "Default number of repeat is: " << PARAM_TEST_REPEAT << endl;

    }

    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numThreads") == 0)
        {
            PARAM_NUM_THREADS = atoi(args[i + 1]);

            cout << "Number of threads: " << PARAM_NUM_THREADS << endl;

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_NUM_THREADS = 64;
        cout << "Default number of threads is: " << PARAM_NUM_THREADS << endl;

    }

    return iAlgType;
}

