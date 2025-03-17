include(FetchContent)

FetchContent_Declare(
    slang
    GIT_REPOSITORY https://github.com/shader-slang/slang.git
    GIT_TAG        master
)

set(SLANG_ENABLE_TESTS OFF)
set(SLANG_ENABLE_EXAMPLES OFF)
set(SLANG_ENABLE_SLANGD OFF)
set(SLANG_LIB_TYPE STATIC)

FetchContent_MakeAvailable(slang)

find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_executable(shader_test test.cpp)

target_link_libraries(shader_test PRIVATE slang Vulkan::Vulkan Vulkan::Headers)
target_include_directories(shader_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(shader_test PRIVATE cxx_std_23)
target_compile_definitions(shader_test PRIVATE 
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)