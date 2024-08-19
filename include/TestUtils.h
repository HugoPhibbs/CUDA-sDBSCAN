//
// Created by hphi344 on 30/06/24.
//

#ifndef DBSCANCEOS_TESTUTILS_H
#define DBSCANCEOS_TESTUTILS_H

#include <arrayfire.h>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <matx.h>
#include "../include/rapidcsv.h"

namespace testUtils {
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

    Time timeNow();

    int duration(Time start, Time stop);

    void printDuration(Time start, Time stop);

    int durationSecs(Time start, Time stop);

    void printDurationSinceStart(Time start, const std::string& msg = "");

    void printDurationSinceStartSecs(Time start, const std::string& msg = "");

    af::array createMockMnistDataset(int n = 70000, int d = 784);

    std::pair<af::array, af::array> createMockABMatrices(int n = 70000, int k = 2, int m = 2000, int D = 1024);

    af::array createMockRandomVectorsSet(int D = 1024, int d = 784);

    af::array createMockDistances(int n = 70000, int D = 1024);

    template <typename T>
    inline auto createMockDistancesMatX(int n = 70000, int D = 1024) {
        auto distances = matx::random<T>({n, D}, matx::UNIFORM);

        distances.run();

        return distances;
    }

    template <typename T>
    inline auto createMockMnistDatasetMatX(int n = 70000, int d = 784, matx::matxMemorySpace_t space = matx::MATX_DEVICE_MEMORY) {

        if (n > 100000) {
            // Quick and dirty way to avoid running out of memory. I assume n>>d, so main bottleneck is n when doing the random array creation.

            int batchSize = 100000;
            auto mnist_batch = matx::make_tensor<float>({batchSize, d}, space);
            auto mnist_16 = matx::make_tensor<T>({n, d}, space);

            for (int i = 0; i < n; i+=batchSize) {
                (mnist_batch = matx::random<float>({batchSize, d}, matx::UNIFORM)).run();

                (matx::slice(mnist_16, {i, 0}, {i+batchSize, matx::matxEnd}) = matx::as_type<T>(mnist_batch)).run();
            }

            return mnist_16;
        }

        else {
            auto mnist = matx::make_tensor<float>({n, d}, space);
            auto mnist_16 = matx::make_tensor<T>({n, d}, space);

            (mnist = matx::random<float>({n, d}, matx::UNIFORM)).run();
            (mnist_16 = matx::as_type<T>(mnist)).run();
            return mnist_16;
        }
    }

    inline auto createMockAMatrixMatX(int n = 70000, int k = 2, int D = 1024, matx::matxMemorySpace_t space = matx::MATX_DEVICE_MEMORY) {
        auto A = matx::make_tensor<float>({n, 2*k}, space);
        auto A_i = matx::make_tensor<int32_t>({n, 2*k}, space);

        int a = 2 * (D - 1);

        (A = matx::random<float>({n, 2*k}, matx::UNIFORM, 0, a)).run();
        (A_i = matx::as_type<int32_t>(A)).run();

        return A_i;
    }

    inline auto createMockBMatrixMatX(int n = 70000, int m = 2000, int D = 1024, matx::matxMemorySpace_t space = matx::MATX_DEVICE_MEMORY) {
        auto B = matx::make_tensor<float>({2*D, m}, space);
        auto B_i = matx::make_tensor<int32_t >({2*D, m}, space);

        int a = n - 1;

        (B = matx::random<float>({2*D, m}, matx::UNIFORM, 0, a)).run();
        (B_i = matx::as_type<int32_t>(B)).run();

        return B_i;
    }

    bool arraysEqual(const af::array &a, const af::array &b);

    bool arraysApproxEqual(const af::array &a, const af::array &b, double eps=1e-6);

    std::vector<std::vector<float>> readCSV(const std::string &filename);

    af::array csvToArray(const std::string& filename);

    void readFlatCSV(const std::string& filename, float* data, size_t size);
}


#endif //DBSCANCEOS_TESTUTILS_H
