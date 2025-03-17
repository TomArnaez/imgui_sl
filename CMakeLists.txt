cmake_minimum_required(VERSION 3.30.0)
project(imgui_test VERSION 0.1.0 LANGUAGES C CXX)

add_subdirectory(shaders)

add_executable(imgui_test
    src/main.cpp
)

target_include_directories(imgui_test PRIVATE src)

find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

target_link_libraries(imgui_test 
    PRIVATE 
        imgui::imgui
        glfw
        spdlog::spdlog
        GPUOpen::VulkanMemoryAllocator
        spdlog::spdlog
)

target_compile_features(imgui_test PRIVATE cxx_std_23)
target_compile_definitions(imgui_test PRIVATE 
    "$<$<CONFIG:Debug>:APP_USE_VULKAN_DEBUG_REPORT>"
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VMA_IMPLEMENTATION
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)