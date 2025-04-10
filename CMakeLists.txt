cmake_minimum_required(VERSION 3.30.0)
project(imgui_test VERSION 0.1.0 LANGUAGES C CXX)

add_subdirectory(test)

add_library(vulkan_engine
    lib/src/shader_manager.cpp
    lib/src/allocator.cpp
    lib/src/vulkan_core.cpp
)

target_include_directories(vulkan_engine PUBLIC lib/include PRIVATE lib/src)

FetchContent_Declare(
    slang
    GIT_REPOSITORY https://github.com/shader-slang/slang.git
    GIT_TAG        master
)

set(SLANG_LIB_TYPE STATIC)

FetchContent_MakeAvailable(slang)

find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

# Handle shaders

file(GLOB_RECURSE SHADERS "shaders/*.slang")
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${SHADERS} ${SHADER_OUTPUT_DIR}
    DEPENDS ${SHADERS}
)

target_link_libraries(vulkan_engine
    PUBLIC
        Vulkan::Vulkan
        Vulkan::Headers
        spdlog::spdlog
    PRIVATE 
        slang
        glfw
        GPUOpen::VulkanMemoryAllocator
        spdlog::spdlog
)

target_compile_features(vulkan_engine PRIVATE cxx_std_23)
target_compile_definitions(vulkan_engine PUBLIC 
    "$<$<CONFIG:Debug>:APP_USE_VULKAN_DEBUG_REPORT>"
    "VKENGINE_SHADER_DIR=\"${SHADER_OUTPUT_DIR}\""
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    PRIVATE
    VMA_IMPLEMENTATION
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)