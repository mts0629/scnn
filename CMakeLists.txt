cmake_minimum_required(VERSION 3.16)

set(TARGET_LIB_NAME nnc)

project(${TARGET_LIB_NAME} C)

set(TARGET_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(src)
add_subdirectory(test EXCLUDE_FROM_ALL)
add_subdirectory(example EXCLUDE_FROM_ALL)
