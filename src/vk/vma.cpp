#include <vulkan/vulkan.hpp>
#include <vulkan_error.hpp>

namespace vma {

allocator::allocator(vk::Instance instance, vk::Device device, vk::PhysicalDevice physical_device) {
    spdlog::trace("Constructing {}", typeid(*this).name());

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
        .vulkanApiVersion = VK_API_VERSION_1_3
    };

    VK_CHECK(vmaCreateAllocator(&create_info, &allocator_));
}

allocator::~allocator() {
    spdlog::trace("Destructing {}", typeid(*this).name());

    vmaDestroyAllocator(allocator_);
}

image allocator::create_image(const vk::ImageCreateInfo& image_info) {
    spdlog::trace("Creating image with extent {0:d}x{1:d}x{2:d}, format: {3:d}",
        image_info.extent.width, image_info.extent.height, image_info.extent.depth, static_cast<uint32_t>(image_info.format));

    VkImageCreateInfo vkImageInfo = static_cast<VkImageCreateInfo>(image_info);
    VmaAllocationCreateInfo allocCreateInfo = {};
    
    VkImage handle;
    VmaAllocation allocation;
    VK_CHECK(vmaCreateImage(allocator_, &vkImageInfo, &allocCreateInfo, &handle, &allocation, nullptr));

    image image = {
        .handle = handle,
        .extent = image_info.extent,
        .format = image_info.format,
        .allocation = allocation
    };

    return image;
}

void allocator::destroy_image(image& image) {
    spdlog::trace("Destroying image with extent: {}x{}x{}", image.extent.width, image.extent.height, image.extent.depth);

    vmaDestroyImage(allocator_, static_cast<VkImage>(image.handle), image.allocation);
}


}