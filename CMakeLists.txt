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

set(SLANG_ENABLE_DXIL OFF)
set(SLANG_ENABLE_GFX OFF)
set(SLANG_ENABLE_SLANGD ON)
set(SLANG_ENABLE_SLANGC OFF)
set(SLANG_ENABLE_TESTS OFF)
set(SLANG_ENABLE_RELEASE_DEBUG_INFO OFF)
set(SLANG_ENABLE_SLANG_RHI OFF)
set(SLANG_ENABLE_CUDA OFF)
set(SLANG_ENABLE_OPTIX OFF)
set(SLANG_ENABLE_NVAPI OFF)
set(SLANG_ENABLE_AFTERMATH OFF)
set(SLANG_ENABLE_EXAMPLES OFF)
set(SLANG_LIB_TYPE STATIC)

FetchContent_MakeAvailable(slang)

find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

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
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VMA_IMPLEMENTATION
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)