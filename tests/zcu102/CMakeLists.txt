FILE(GLOB testfiles "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(object ${testfiles})
    get_filename_component(obj ${object} NAME_WE )
    message(STATUS "zcu102 test ${obj} is added")
    add_executable(${obj} ${object})
    target_link_libraries(${obj}
            vta_runtime cma pthread tinfo glog gflags)
    set_target_properties(${obj} PROPERTIES RUNTIME_OUTPUT_DIRECTORY passed_tests )

    add_definitions(-DGLOW_WITH_ZCU102_TEST=1)

    #if(GLOW_WITH_ZCU102_TEST)
        add_custom_command(
                TARGET ${obj}  POST_BUILD
                COMMAND
                ${obj})
    #endif()
endforeach()