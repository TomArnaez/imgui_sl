#include <vk_mem_alloc.h>

#include <vulkan/vulkan.hpp>
#include <vk/vma_context.hpp>
#include <vulkan_error.hpp>

vma_context::vma_context(vk::Device device, vk::PhysicalDevice physical_device, vk::Instance instance) {
    const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;

    VmaVulkanFunctions functions {
        .vkGetInstanceProcAddr = d.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = d.vkGetDeviceProcAddr,
    };

    VmaAllocatorCreateInfo create_info {
        .physicalDevice = physical_device,
        .device = device,
        .pVulkanFunctions = &functions,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_2
    };

    VK_CHECK(vmaCreateAllocator(&create_info, &allocator_));
}

vma_context::~vma_context() {
    vmaDestroyAllocator(allocator_);
}