#include <vulkan/vulkan.hpp>
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

image allocator::create_image(
    const vk::ImageCreateInfo& image_info,
    const VmaAllocationCreateInfo& allocation_create_info
) const {
    spdlog::trace("Creating image with extent {0}x{1}x{2}, format: {3}",
        image_info.extent.width,
        image_info.extent.height,
        image_info.extent.depth,
        static_cast<uint32_t>(image_info.format));

    VkImageCreateInfo vk_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkImage handle;
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;
    VK_CHECK(vmaCreateImage(allocator_, &vk_image_info, &allocation_create_info, &handle, &allocation, &allocation_info));

    return image {
        .handle = handle,
        .extent = image_info.extent,
        .format = image_info.format,
        .allocation = allocation,
        .allocation_info = allocation_info
    };
}

void allocator::destroy_image(image& image) const {
    spdlog::trace("Destroying image with extent {0}x{1}x{2}",
        image.extent.width,
        image.extent.height,
        image.extent.depth);

    vmaDestroyImage(allocator_, static_cast<VkImage>(image.handle), image.allocation);
}

buffer allocator::create_buffer(
    const vk::BufferCreateInfo& buffer_info,
    const VmaAllocationCreateInfo& allocation_create_info
) const {
    spdlog::trace("Creating buffer of size {0} bytes", buffer_info.size);

    VkBufferCreateInfo vk_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);

    VkBuffer handle;
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;
    VK_CHECK(vmaCreateBuffer(allocator_, &vk_buffer_info, &allocation_create_info, &handle, &allocation, &allocation_info));

    return buffer {
        .handle = handle,
        .size = buffer_info.size,
        .allocation = allocation,
        .allocation_info = allocation_info
    };
}

void allocator::destroy_buffer(buffer& buffer) const {
    spdlog::trace("Destroying buffer of size {0} bytes", buffer.size);

    vmaDestroyBuffer(allocator_, static_cast<VkBuffer>(buffer.handle), buffer.allocation);
}

}