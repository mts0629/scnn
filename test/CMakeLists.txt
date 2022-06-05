set(TARGET_TEST_RUNNER_NAME test_runner)

file(GLOB SOURCES ./cases/*.c ./runner/*.c)

set(UNITY_ROOT ${PROJECT_SOURCE_DIR}/test/unity)

add_executable(${TARGET_TEST_RUNNER_NAME}
    ${SOURCES}
    ${UNITY_ROOT}/src/unity.c
    ${UNITY_ROOT}/extras/fixture/src/unity_fixture.c
    #${UNITY_ROOT}/extras/memory/src/unity_memory.c
)

add_custom_target(test
    COMMAND ./${TARGET_TEST_RUNNER_NAME}
    DEPENDS ${TARGET_TEST_RUNNER_NAME}
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

target_compile_options(${TARGET_TEST_RUNNER_NAME}
    PUBLIC -Wall -Wextra -Wpedantic -Werror
)

target_include_directories(${TARGET_TEST_RUNNER_NAME}
    PUBLIC ${UNITY_ROOT}/src
    PUBLIC ${UNITY_ROOT}/extras/fixture/src
    #PUBLIC ${UNITY_ROOT}/extras/memory/src
)

target_link_directories(${TARGET_TEST_RUNNER_NAME}
    PUBLIC ${TARGET_LIB_DIR}
)

target_link_libraries(${TARGET_TEST_RUNNER_NAME}
    ${TARGET_LIB_NAME}
)

# using Unity fixtures w/o memory handling
add_definitions(-DUNITY_FIXTURE_NO_EXTRAS)
