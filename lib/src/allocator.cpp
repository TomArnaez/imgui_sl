#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>
#include <vulkan_error.hpp>
#include <allocator.hpp>

namespace vkengine {

allocator::allocator(const vulkan_core& core) {
    spdlog::trace("Constructing {}", typeid(*this).name());

    const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;

    VmaVulkanFunctions functions {
        .vkGetInstanceProcAddr = d.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = d.vkGetDeviceProcAddr,
    };

    VmaAllocatorCreateInfo create_info {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = core.physical_device(),
        .device = core.device(),
        .pVulkanFunctions = &functions,
        .instance = core.instance(),
        .vulkanApiVersion = VK_API_VERSION_1_3
    };

    VK_CHECK(vmaCreateAllocator(&create_info, &allocator_));
}

allocator::~allocator() {
    spdlog::trace("Destructing {}", typeid(*this).name());

    vmaDestroyAllocator(allocator_);
}

image allocator::create_image(const vk::ImageCreateInfo& image_info) const {
    spdlog::trace("Creating image with extent {0}x{1}x{2}, format: {3}",
        image_info.extent.width,
        image_info.extent.height,
        image_info.extent.depth,
        static_cast<uint32_t>(image_info.format));

    VkImageCreateInfo vkImageInfo = static_cast<VkImageCreateInfo>(image_info);
    VmaAllocationCreateInfo allocCreateInfo = {};

    VkImage handle;
    VmaAllocation allocation;
    VK_CHECK(vmaCreateImage(allocator_, &vkImageInfo, &allocCreateInfo, &handle, &allocation, nullptr));

    return image {
        .handle = handle,
        .extent = image_info.extent,
        .format = image_info.format,
        .allocation = allocation
    };
}

void allocator::destroy_image(image& image) const {
    spdlog::trace("Destroying image with extent {0}x{1}x{2}",
        image.extent.width,
        image.extent.height,
        image.extent.depth);

    vmaDestroyImage(allocator_, static_cast<VkImage>(image.handle), image.allocation);
}

buffer allocator::create_buffer(const vk::BufferCreateInfo& buffer_info) const {
    spdlog::trace("Creating buffer of size {0} bytes", buffer_info.size);

    VkBufferCreateInfo vkBufferInfo = static_cast<VkBufferCreateInfo>(buffer_info);
    VmaAllocationCreateInfo allocCreateInfo = {};

    VkBuffer handle;
    VmaAllocation allocation;
    VK_CHECK(vmaCreateBuffer(allocator_, &vkBufferInfo, &allocCreateInfo, &handle, &allocation, nullptr));

    return buffer {
        .handle = handle,
        .size = buffer_info.size,
        .allocation = allocation
    };
}

void allocator::destroy_buffer(buffer& buffer) const {
    spdlog::trace("Destroying buffer of size {0} bytes", buffer.size);

    vmaDestroyBuffer(allocator_, static_cast<VkBuffer>(buffer.handle), buffer.allocation);
}

}