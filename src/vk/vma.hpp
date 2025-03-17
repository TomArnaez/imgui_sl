#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan_handles.hpp>

// struct VmaAllocator_T;
// typedef struct VmaAllocator_T* VmaAllocator;

// struct VmaAllocation_T;
// typedef struct VmaAllocation_T* VmaAllocation;

namespace vma {

struct image {
    vk::Image handle = VK_NULL_HANDLE;
    vk::Extent3D extent;
    vk::Format format;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

class allocator {
public:
    VmaAllocator allocator_;
    allocator(vk::Instance instance, vk::Device device, vk::PhysicalDevice physical_device);
    ~allocator();

    allocator(const allocator&) = delete;
    allocator& operator=(const allocator&) = delete;
    allocator(allocator&&) = delete;
    allocator& operator=(allocator&&) = delete;

    image create_image(const vk::ImageCreateInfo& image_info);
    void destroy_image(image& image);
};

}