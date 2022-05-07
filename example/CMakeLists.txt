add_subdirectory(mnist)

add_custom_target(example
    DEPENDS mnist
)
