cmake_minimum_required(VERSION 3.16)

set(TARGET_LIB_NAME nnc)

set(TARGET_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

project(${TARGET_LIB_NAME} C)

add_subdirectory(src)
