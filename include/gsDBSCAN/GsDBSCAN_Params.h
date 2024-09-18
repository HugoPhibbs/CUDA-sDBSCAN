//
// Created by hphi344 on 12/09/24.
//

#include <string>
#include "../pch.h"
#include <iostream>
#include <sstream>

#ifndef SDBSCAN_GSDBSCAN_PARAMS_H
#define SDBSCAN_GSDBSCAN_PARAMS_H

namespace GsDBSCAN {

    inline bool NEED_TO_NORMALISE_DEFAULT = false;
    inline bool CLUSTER_ON_CPU_DEFAULT = false;
    inline bool TIME_IT_DEFAULT = false;

    inline std::string DISTANCE_METRIC_DEFAULT = "COSINE";

    inline float ALPHA_DEFAULT = 1.2;
    inline float DISTANCE_BATCH_SIZE_DEFAULT = 150;
    inline int CLUSTER_BLOCK_SIZE_DEFAULT = 256;

    inline int MINI_BATCH_SIZE_DEFAULT = 10000;
    inline int A_BATCH_SIZE_DEFAULT = 10000;
    inline int B_BATCH_SIZE_DEFAULT = 128;

    inline int FOURIER_EMBED_DIM_DEFAULT = 1024;
    inline float SIGMA_EMBED_DEFAULT = 1;

    inline bool VERBOSE_DEFAULT = false;
    inline bool USE_BATCH_CLUSTERING_DEFAULT = false;
    inline bool USE_BATCH_AB_MATRICES_DEFAULT = false;

    inline std::string DATASET_DTYPE_DEFAULT = "f32";

    class GsDBSCAN_Params {
    private:

        float adjustEps(float _eps) {
            if (this->distanceMetric == "COSINE") {
                return 1 - _eps; // We use cosine similarity, thus we need to convert the eps to a cosine distance.
            }
            return _eps;
        }


    public:
        std::string dataFilename;
        std::string outputFilename;
        int n;
        int d;
        int D;
        int minPts;
        int k;
        int m;
        float eps;
        float alpha;
        int distancesBatchSize;
        std::string distanceMetric;
        int clusterBlockSize;
        bool timeIt;
        bool clusterOnCpu;
        bool needToNormalise;
        int fourierEmbedDim;
        float sigmaEmbed;
        int ABatchSize;
        int BBatchSize;
        int miniBatchSize;
        bool verbose;
        bool useBatchClustering;
        bool useBatchABMatrices;
        std::string datasetDType;


        GsDBSCAN_Params(std::string dataFilename, std::string outputFilename, int n, int d, int D, int minPts, int k,
                        int m, float eps,
                        const std::string &distanceMetric = DISTANCE_METRIC_DEFAULT,
                        bool clusterOnCpu = CLUSTER_ON_CPU_DEFAULT,
                        bool needToNormalise = NEED_TO_NORMALISE_DEFAULT,
                        float alpha = ALPHA_DEFAULT,
                        int distancesBatchSize = DISTANCE_BATCH_SIZE_DEFAULT,
                        int clusterBlockSize = CLUSTER_BLOCK_SIZE_DEFAULT, bool timeIt = TIME_IT_DEFAULT,
                        int fourierEmbedDim = FOURIER_EMBED_DIM_DEFAULT, float sigmaEmbed = SIGMA_EMBED_DEFAULT,
                        int ABatchSize = A_BATCH_SIZE_DEFAULT,
                        int BBatchSize = B_BATCH_SIZE_DEFAULT, int miniBatchSize = MINI_BATCH_SIZE_DEFAULT,
                        bool verbose = VERBOSE_DEFAULT,
                        bool useBatchClustering = USE_BATCH_CLUSTERING_DEFAULT,
                        bool useBatchABMatrices = USE_BATCH_AB_MATRICES_DEFAULT,
                        std::string datasetDType = DATASET_DTYPE_DEFAULT) {

            this->dataFilename = dataFilename;
            this->outputFilename = outputFilename;
            this->n = n;
            this->d = d;
            this->D = D;
            this->minPts = minPts;
            this->k = k;
            this->m = m;
            this->alpha = alpha;
            this->distancesBatchSize = distancesBatchSize;
            this->distanceMetric = distanceMetric;
            this->eps = adjustEps(eps);
            this->clusterBlockSize = clusterBlockSize;
            this->timeIt = timeIt;
            this->clusterOnCpu = clusterOnCpu;
            this->needToNormalise = needToNormalise;
            this->fourierEmbedDim = fourierEmbedDim;
            this->sigmaEmbed = sigmaEmbed;
            this->ABatchSize = ABatchSize;
            this->BBatchSize = BBatchSize;
            this->miniBatchSize = miniBatchSize;
            this->verbose = verbose;
            this->useBatchClustering = useBatchClustering;
            this->useBatchABMatrices = useBatchABMatrices;

            if (datasetDType != "f16" && datasetDType != "f32") {
                throw std::runtime_error("Invalid dataset dtype. Must be either 'f16' or 'f32'");
            }

            this->datasetDType = datasetDType;
        }

        inline std::string toString() const {
            std::ostringstream oss;

            oss << "\n" << "\n";
            oss << "## PARAMS ##\n" << "\n";
            oss << "Data Filename: " << dataFilename << "\n";
            oss << "Output Filename: " << outputFilename << "\n";
            oss << "n: " << n << "\n";
            oss << "d: " << d << "\n";
            oss << "D: " << D << "\n";
            oss << "minPts: " << minPts << "\n";
            oss << "k: " << k << "\n";
            oss << "m: " << m << "\n";
            oss << "Epsilon (eps) (adjusted): " << eps << "\n";
            oss << "Alpha: " << alpha << "\n";
            oss << "Distances Batch Size: " << distancesBatchSize << "\n";
            oss << "Distance Metric: " << distanceMetric << "\n";
            oss << "Cluster Block Size: " << clusterBlockSize << "\n";
            oss << "Time It: " << (timeIt ? "true" : "false") << "\n";
            oss << "Cluster On CPU: " << (clusterOnCpu ? "true" : "false") << "\n";
            oss << "Need to Normalise: " << (needToNormalise ? "true" : "false") << "\n";
            oss << "Fourier Embed Dimension: " << fourierEmbedDim << "\n";
            oss << "Sigma Embed: " << sigmaEmbed << "\n";
            oss << "A Batch Size: " << ABatchSize << "\n";
            oss << "B Batch Size: " << BBatchSize << "\n";
            oss << "Mini Batch Size: " << miniBatchSize << "\n";
            oss << "Verbose: " << (verbose ? "true" : "false") << "\n";
            oss << "Use Batch Clustering: " << (useBatchClustering ? "true" : "false") << "\n";
            oss << "Use batch creation of A, B matrices: " << (useBatchABMatrices ? "true" : "false") << "\n";
            oss << "Dataset DType: " << datasetDType << "\n";

            return oss.str();
        }
    };

    inline argparse::ArgumentParser &getParser() {
        static argparse::ArgumentParser parser("GsDBSCAN");

        parser.add_argument("--datasetFilename", "-f").required();
        parser.add_argument("--outputFilename", "-o").required();

        parser.add_argument("--n").help("The size of the dataset (number of vectors)").required().scan<'i', int>();
        parser.add_argument("--d").help("The dimension of the dataset").required().scan<'i', int>();

        parser.add_argument("--minPts").help("DBSCAN minPts parameter").required().scan<'i', int>();
        parser.add_argument("--eps").help("DBSCAN eps parameter").required().scan<'f', float>();

        parser.add_argument("--D").help("The number of random vectors to generate").required().scan<'i', int>();
        parser.add_argument("--k").help("S-DBSCAN k parameter").required().scan<'i', int>();
        parser.add_argument("--m").help("S-DBSCAN m parameter").required().scan<'i', int>();

        parser.add_argument("--distanceMetric", "-dm")
                .help("What distance metric to use, either 'L1' 'L2' or 'COSINE'")
                .default_value(DISTANCE_METRIC_DEFAULT);


        parser.add_argument("--alpha", "-a")
                .help("Alpha parameter to tune the distance batch size")
                .scan<'f', float>()
                .default_value(ALPHA_DEFAULT);

        parser.add_argument("--distancesBatchSize", "-dbs")
                .help("Batch size to use when calculating distances")
                .scan<'i', int>()
                .default_value(DISTANCE_BATCH_SIZE_DEFAULT);

        parser.add_argument("--clusterBlockSize", "-cbs")
                .help("Block size to use when clustering on the GPU")
                .scan<'i', int>()
                .default_value(CLUSTER_BLOCK_SIZE_DEFAULT);


        parser.add_argument("--clusterOnCpu", "-cpu")
                .help("Whether the CPU should be used for the clustering step")
                .default_value(CLUSTER_ON_CPU_DEFAULT)
                .implicit_value(true);

        parser.add_argument("--needToNormalize", "-norm").help(
                "Whether the dataset needs to be normalised (i.e. all points on the unit sphere)").default_value(
                NEED_TO_NORMALISE_DEFAULT).implicit_value(true);

        parser.add_argument("--miniBatchSize", "-mbs")
                .help("What batch size to use when mini batching across the entire dataset")
                .scan<'i', int>()
                .default_value(MINI_BATCH_SIZE_DEFAULT);

        parser.add_argument("--ABatchSize", "-abs")
                .help("What batch size to use when creating the A matrix")
                .scan<'i', int>()
                .default_value(A_BATCH_SIZE_DEFAULT);

        parser.add_argument("--BBatchSize", "-bbs")
                .help("What batch size to use when creating the B matrix")
                .scan<'i', int>()
                .default_value(B_BATCH_SIZE_DEFAULT);

        parser.add_argument("--timeIt", "-t")
                .help("Whether the algorithm should be timed or not")
                .default_value(TIME_IT_DEFAULT)
                .implicit_value(true);

        parser.add_argument("--fourierEmbedDim", "-fed")
                .help("The embed dimension for Fourier embeddings to use L1 and L2 norms")
                .scan<'i', int>()
                .default_value(FOURIER_EMBED_DIM_DEFAULT);

        parser.add_argument("--sigmaEmbed", "-se")
                .help("The value of sigma for Fourier embeddings to use L1 and L2 norms")
                .scan<'f', float>()
                .default_value(SIGMA_EMBED_DEFAULT);

        parser.add_argument("--verbose", "-v")
                .help("Whether the algorithm is verbose")
                .default_value(VERBOSE_DEFAULT)
                .implicit_value(true);

        parser.add_argument("--useBatchClustering", "-ubd")
                .help("Whether to use the batch clustering DBSCAN implementation")
                .default_value(false)
                .implicit_value(true);

        parser.add_argument("--useBatchABMatrices", "-ubd")
                .help("Whether to use the batch creation of the AB matrices (unimplemented)")
                .default_value(false)
                .implicit_value(true);

        parser.add_argument("--datasetDType", "-ddt")
                .help("What dtype the dataset is in. Options: 'f16' or 'f32'")
                .default_value(DATASET_DTYPE_DEFAULT);

        return parser;
    }

    inline GsDBSCAN_Params parseArgs(int argc, char *argv[]) {
        argparse::ArgumentParser &parser = getParser();

        try {
            parser.parse_args(argc, argv);
        } catch (const std::runtime_error &err) {
            std::cerr << err.what() << std::endl;
            std::cerr << parser;
            std::exit(1);
        }

        try {
            return GsDBSCAN_Params(
                    parser.get<std::string>("--datasetFilename"),
                    parser.get<std::string>("--outputFilename"),
                    parser.get<int>("--n"),
                    parser.get<int>("--d"),
                    parser.get<int>("--D"),
                    parser.get<int>("--minPts"),
                    parser.get<int>("--k"),
                    parser.get<int>("--m"),
                    parser.get<float>("--eps"),
                    parser.get<std::string>("--distanceMetric"),
                    parser.get<bool>("--clusterOnCpu"),
                    parser.get<bool>("--needToNormalize"),
                    parser.get<float>("--alpha"),
                    parser.get<int>("--distancesBatchSize"),
                    parser.get<int>("--clusterBlockSize"),
                    parser.get<bool>("--timeIt"),
                    parser.get<int>("--fourierEmbedDim"),
                    parser.get<float>("--sigmaEmbed"),
                    parser.get<int>("--ABatchSize"),
                    parser.get<int>("--BBatchSize"),
                    parser.get<int>("--miniBatchSize"),
                    parser.get<bool>("--verbose"),
                    parser.get<bool>("--useBatchClustering"),
                    parser.get<bool>("--useBatchABMatrices"),
                    parser.get<std::string>("--datasetDType")
            );
        } catch (const std::bad_cast &e) {
            std::cerr << "Error: Invalid type in argument conversion. " << e.what() << std::endl;
            std::exit(1);  // Optionally exit with an error code
        } catch (const std::runtime_error &e) {
            std::cerr << "Argument parsing error: " << e.what() << std::endl;
            std::exit(1);
        } catch (...) {
            std::cerr << "Unknown error occurred during argument parsing." << std::endl;
            std::exit(1);
        }
    };
}

#endif //SDBSCAN_GSDBSCAN_PARAMS_H
