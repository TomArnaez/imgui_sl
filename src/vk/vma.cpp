#include <vk_mem_alloc.h>

#include <vulkan/vulkan.hpp>
#include <vk/vma.hpp>
#include <vulkan_error.hpp>

namespace vma {

allocator::allocator(vk::Device device, vk::PhysicalDevice physical_device, vk::Instance instance) {
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

allocator::~allocator() {
    vmaDestroyAllocator(allocator_);
}

staging_buffer allocator::create_staging_buffer(vk::DeviceSize size) {
    vk::BufferCreateInfo buffer_info{};
    buffer_info.size = size;
    buffer_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
    buffer_info.sharingMode = vk::SharingMode::eExclusive;

    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer buffer_c;
    VmaAllocation allocation_c;
    VK_CHECK(vmaCreateBuffer(
        allocator_,
        reinterpret_cast<const VkBufferCreateInfo*>(&buffer_info),
        &alloc_create_info,
        &buffer_c,
        &allocation_c,
        nullptr
    ));

    void* mapping;
    VK_CHECK(vmaMapMemory(allocator_, allocation_c, &mapping));

    return { vk::Buffer(buffer_c), allocation_c, static_cast<std::byte*>(mapping) };
}

void allocator::destroy_staging_buffer(const staging_buffer& buffer) {
    vmaUnmapMemory(allocator_, buffer.allocation);
    vmaDestroyBuffer(allocator_, static_cast<VkBuffer>(buffer.buffer), buffer.allocation);
}

std::pair<vk::Image, VmaAllocation> allocator::create_image(const vk::ImageCreateInfo& image_info) {
    VkImageCreateInfo vkImageInfo = static_cast<VkImageCreateInfo>(image_info);
    VmaAllocationCreateInfo allocCreateInfo = {};
    
    VkImage image;
    VmaAllocation allocation;
    VK_CHECK(vmaCreateImage(allocator_, &vkImageInfo, &allocCreateInfo, &image, &allocation, nullptr));
    
    return { vk::Image(image), allocation };
}

void allocator::destroy_image(vk::Image image, VmaAllocation allocation) {
    vmaDestroyImage(allocator_, static_cast<VkImage>(image), allocation);
}


}