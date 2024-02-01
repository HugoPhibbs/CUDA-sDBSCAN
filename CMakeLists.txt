cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs)

set(CMAKE_CXX_STANDARD 17)


file(GLOB SOURCES
        /Users/npha145/'Dropbox (Uni of Auckland)'/Working/_Code/C++/DbscanCEOs/*.h
        /Users/npha145/'Dropbox (Uni of Auckland)'/Working/_Code/C++/DbscanCEOs/*.cpp
)

add_executable(yourProj ${SOURCES} main.cpp)