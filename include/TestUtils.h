//
// Created by hphi344 on 30/06/24.
//

#ifndef DBSCANCEOS_TESTUTILS_H
#define DBSCANCEOS_TESTUTILS_H

#endif //DBSCANCEOS_TESTUTILS_H

#include <arrayfire.h>
#include <chrono>
#include <iostream>

namespace testUtils {
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

    Time timeNow();

    int duration(Time start, Time stop);

    void printDuration(Time start, Time stop);

    void printDurationSinceStart(Time start);

    af::array createMockMnistDataset(int n = 70000, int d = 784);

    std::pair<af::array, af::array> createMockABMatrices(int n = 70000, int k = 2, int m = 2000, int D = 1024);

    af::array createMockRandomVectorsSet(int D = 1024, int d = 784);

    af::array createMockDistances(int n = 70000, int D = 1024);

    std::vector<std::vector<float>> readCSV(const std::string &filename);

    af::array csvToArray(const std::string& filename);
}
