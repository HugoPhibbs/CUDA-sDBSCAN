cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs)

set(CMAKE_CXX_STANDARD 17)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.3")
find_package(CUDAToolkit 11.3 REQUIRED)

find_package(ArrayFire REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/lib/eigen-3.4.0
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${gtest_SOURCE_DIR}/include
        ${gtest_SOURCE_DIR}
)

link_directories(
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

add_subdirectory(lib/googletest)

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/fast_copy.c
        src/dbscan/sngDBSCAN.cpp
        include/sngDBSCAN.h
        src/dbscan/DBSCAN.cpp
        src/dbscan/sDBSCAN.cpp
        include/sDBSCAN.h
        src/dbscan/uDBSCAN.cpp
        include/uDBSCAN.h
        src/dbscan/GsDBSCAN.cpp
        include/GsDBSCAN.h
        src/dbscan/preprocessing.cpp
        include/preprocessing.h
        src/fht/fht.c
)

add_executable(run_gs_dbscan_tests
        test/gsDBSCANTest.cpp
        src/dbscan/GsDBSCAN.cpp
)

target_link_libraries(run_gs_dbscan_tests gtest gtest_main ArrayFire::af)
target_link_libraries(${PROJECT_NAME} ArrayFire::af)
