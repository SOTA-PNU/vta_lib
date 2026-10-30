FILE(GLOB testfiles "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

add_library(VTASimTest
        ${testfiles}
        )

target_link_libraries(VTASimTest vta_runtime)

INCLUDE_DIRECTORIES(include simulator/include)
add_executable(VTABackendSimulatorTest
        VTABackendSimulatorTest.cpp
        )

target_link_libraries(VTABackendSimulatorTest gtest VTASimTest)
set_target_properties( VTABackendSimulatorTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )



# Layout-independent bundle operators (lib/Bundle/CPUBundle_common.cpp).
add_executable(VTABundleOpsTest
        VTABundleOpsTest.cpp
        )
target_link_libraries(VTABundleOpsTest gtest gtest_main VTABundle)
set_target_properties(VTABundleOpsTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )
