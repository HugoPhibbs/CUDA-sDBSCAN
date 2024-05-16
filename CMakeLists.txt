cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs)

set(CMAKE_CXX_STANDARD 17)


file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp"
)

add_executable(yourProj ${SOURCES} src/main.cpp
        test/Test.cpp
        src/fast_copy.c
        src/dbscan/sngDBSCAN.cpp
        include/sngDBSCAN.h
        src/dbscan/DBSCAN.cpp
        src/dbscan/sDBSCAN.cpp
        include/sDBSCAN.h
        src/dbscan/uDBSCAN.cpp
        include/uDBSCAN.h
        src/dbscan/gsDBSCAN.cpp
        src/fht/fht.c
        include/gsDBSCAN.h
        src/dbscan/preprocessing.cpp
        include/preprocessing.h
)