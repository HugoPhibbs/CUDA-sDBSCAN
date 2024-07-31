cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs LANGUAGES CXX CUDA C)
#enable_language(CUDA)
project(sDbscan)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc") # Somehow CLion needs this here (smh)
#SET(CMAKE_C_COMPILER "/usr/bin/g++")
#add_definitions(-DINDE     X_64_BIT)

SET(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.71 REQUIRED NO_MODULE)

# CCCL
include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG v2.4.0
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
find_package(CUDAToolkit 11.8 REQUIRED)

# ArrayFire
find_package(ArrayFire REQUIRED)

# MatX https://github.com/NVIDIA/MatX
find_package(matx CONFIG REQUIRED)

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

set_source_files_properties(
        src/dbscan/GsDBSCAN.cpp
        test/gsDBSCANTest.cpp
        PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/Utilities.cpp
        src/dbscan/sDbscan.cpp
        src/fast_copy.c
        src/dbscan/GsDBSCAN.cpp
        src/fht/fht.c
)

add_executable(run_gs_dbscan_tests
        test/gsDBSCANTest.cpp
        src/dbscan/GsDBSCAN.cpp
        test/TestUtils.cpp
)


target_link_libraries(run_gs_dbscan_tests PRIVATE CCCL::CCCL CUDA::cudart ArrayFire::afcuda matx::matx gtest gtest_main)
target_link_libraries(${PROJECT_NAME} PRIVATE CCCL::CCCL CUDA::cudart ArrayFire::afcuda Eigen3::Eigen matx::matx)