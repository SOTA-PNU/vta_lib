cmake_minimum_required(VERSION 3.10)
project(vtalib)

INCLUDE_DIRECTORIES(include 3rdparty/dlpack/include)
INCLUDE_DIRECTORIES(include 3rdparty/dmlc-core/include)

if(GLOW_WITH_VTASIM)
    add_definitions(-DGLOW_WITH_VTASIM=1)
    INCLUDE_DIRECTORIES(include include/simulator)
    add_library(vta_runtime lib/simulator/runtime.cc lib/simulator/sim_driver.cc lib/simulator/virtual_memory.cc lib/simulator/sim_tlpp.cc)
    add_subdirectory(tests/simtests)
else()
    add_definitions(-DGLOW_WITH_VTASIM=0)
    add_definitions(-DDMLC_USE_FOPEN64=0
            -DVTA_TARGET=zcu102
            -DVTA_HW_VER=0.0.1
            -DVTA_LOG_INP_WIDTH=3
            -DVTA_LOG_WGT_WIDTH=3
            -DVTA_LOG_ACC_WIDTH=5
            -DVTA_LOG_BATCH=0
            -DVTA_LOG_BLOCK=4
            -DVTA_LOG_UOP_BUFF_SIZE=16
            -DVTA_LOG_INP_BUFF_SIZE=16
            -DVTA_LOG_WGT_BUFF_SIZE=19
            -DVTA_LOG_ACC_BUFF_SIZE=18
            -DVTA_LOG_BLOCK_IN=4
            -DVTA_LOG_BLOCK_OUT=4
            -DVTA_LOG_OUT_WIDTH=3
            -DVTA_LOG_OUT_BUFF_SIZE=16
            -DVTA_LOG_BUS_WIDTH=7
            -DVTA_IP_REG_MAP_RANGE=0x1000
            -DVTA_FETCH_ADDR=0xA0000000
            -DVTA_LOAD_ADDR=0xA0001000
            -DVTA_COMPUTE_ADDR=0xA0002000
            -DVTA_STORE_ADDR=0xA0003000
            -DVTA_FETCH_INSN_COUNT_OFFSET=16
            -DVTA_FETCH_INSN_ADDR_OFFSET=24
            -DVTA_LOAD_INP_ADDR_OFFSET=16
            -DVTA_LOAD_WGT_ADDR_OFFSET=24
            -DVTA_COMPUTE_DONE_WR_OFFSET=16
            -DVTA_COMPUTE_DONE_RD_OFFSET=24
            -DVTA_COMPUTE_UOP_ADDR_OFFSET=32
            -DVTA_COMPUTE_BIAS_ADDR_OFFSET=40
            -DVTA_STORE_OUT_ADDR_OFFSET=16
            -DVTA_COHERENT_ACCESSES=true
            -Dvta_EXPORTS)
    add_compile_options(-std=c++14 -faligned-new -O0 -g -Wall -fPIC  -fPIC  -Wno-psabi)
    INCLUDE_DIRECTORIES(include include/zcu102)
    #link_libraries(prebuilt_lib)
    add_library(vta_runtime lib/zcu102/api.cc lib/zcu102/pynq_driver.cc)
    add_subdirectory(tests/zcu102)
endif()

add_library(VTABundle lib/Bundle/VTABundle.cpp lib/Bundle/CPUBundle.cpp)
target_link_libraries(VTABundle PUBLIC
        vta_runtime LLVMSupport pthread glog gflags)

if(NOT HAVE_QUANTIZATIONBASE_LIB)
    target_link_libraries(VTABundle PUBLIC
        QuantizationBase)
else()
    target_link_libraries(VTABundle PUBLIC
            ${HAVE_QUANTIZATIONBASE_LIB})
    message(STATUS "Using Prebuilt QuantizationBase")
endif()


if(NOT HAVE_SUPPORT_LIB)
    target_link_libraries(VTABundle PUBLIC
            Support)
else()
    target_link_libraries(VTABundle PUBLIC
            ${HAVE_SUPPORT_LIB})
    message(STATUS "Using Prebuilt Support")
endif()

target_link_libraries(VTABundle PUBLIC
        Base)

if(NOT GLOW_WITH_VTASIM)
    target_link_libraries(VTABundle PUBLIC cma)
endif()