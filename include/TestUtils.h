//
// Created by hphi344 on 30/06/24.
//

#ifndef DBSCANCEOS_TESTUTILS_H
#define DBSCANCEOS_TESTUTILS_H

#endif //DBSCANCEOS_TESTUTILS_H

#include <arrayfire.h>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <matx.h>

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

    inline auto createMockDistancesMatX(int n = 70000, int D = 1024) {
        auto distances = matx::random<float>({n, D}, matx::UNIFORM);

        distances.run();

        return distances;
    }

    inline auto createMockMnistDatasetMatX(int n = 70000, int d = 784) {
        matx::tensor_t<float, 2> mnist({n, d});
        matx::tensor_t<matx::matxFp16, 2> mnist_16({n, d});

        (mnist = matx::random<float>({n, d}, matx::UNIFORM)).run();
        (mnist_16 = matx::as_type<matx::matxFp16>(mnist)).run();

        return mnist_16;
    }

    inline auto createMockAMatrixMatX(int n = 70000, int k = 2, int D = 1024) {
        matx::tensor_t<float, 2> A({n, 2*k});
        matx::tensor_t<int32_t, 2> A_i({n, 2*k});

        int a = 2 * (D - 1);

        (A = matx::random<float>({n, 2*k}, matx::UNIFORM, 0, a)).run();
        (A_i = matx::as_type<int32_t>(A)).run();

        return A_i;
    }

    inline auto createMockBMatrixMatX(int n = 70000, int m = 2000, int D = 1024) {
        matx::tensor_t<float, 2> B({2*D, m});
        matx::tensor_t<int32_t, 2> B_i({2*D, m});

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
