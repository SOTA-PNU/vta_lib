cmake_minimum_required(VERSION 3.10)
project(vtalib)

INCLUDE_DIRECTORIES(include 3rdparty/dlpack/include)
INCLUDE_DIRECTORIES(include 3rdparty/dmlc-core/include)

if(GLOW_WITH_VTASIM)
    INCLUDE_DIRECTORIES(include include/simulator)
    INCLUDE_DIRECTORIES(include include/simulator/vta)
    add_library(vta_runtime lib/simulator/runtime.cc lib/simulator/sim_driver.cc lib/simulator/virtual_memory.cc lib/simulator/sim_tlpp.cc)

    add_subdirectory(tests/simtests)
endif()
