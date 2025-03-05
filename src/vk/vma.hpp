#pragma once

#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>


struct VmaAllocator_T;
typedef struct VmaAllocator_T* VmaAllocator;

struct VmaAllocation_T;
typedef struct VmaAllocation_T* VmaAllocation;

namespace vma {

struct staging_buffer {
    vk::Buffer buffer;
    VmaAllocation allocation;
    std::byte* mapping;
};

class allocator {
    VmaAllocator allocator_;
public:
    allocator(vk::Device device, vk::PhysicalDevice physical_device, vk::Instance instance);
    ~allocator();

    allocator(const allocator&) = delete;
    allocator& operator=(const allocator&) = delete;

    allocator(allocator&&) = delete;
    allocator& operator=(allocator&&) = delete;

    staging_buffer create_staging_buffer(vk::DeviceSize size);
    void destroy_staging_buffer(const staging_buffer& buffer);

    std::pair<vk::Image, VmaAllocation> create_image(const vk::ImageCreateInfo& image_info);
    void destroy_image(vk::Image image, VmaAllocation allocation);
};

}