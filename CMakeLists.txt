cmake_minimum_required(VERSION 3.16)

set(TARGET_LIB_NAME nnc)

project(${TARGET_LIB_NAME} C)

set(TARGET_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(src)

set(TARGET_GROUP test)

if(TARGET_GROUP STREQUAL release)
elseif(TARGET_GROUP STREQUAL test)
    add_subdirectory(test)
endif()
