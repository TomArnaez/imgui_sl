include(FetchContent)

add_executable(shader_test test.cpp)

target_link_libraries(shader_test PUBLIC slang vulkan_engine)
target_include_directories(shader_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(shader_test PRIVATE cxx_std_23)