#pragma once

#include <vulkan_core.hpp>

typedef struct VmaAllocator_T* VmaAllocator;
typedef struct VmaAllocation_T* VmaAllocation;

namespace vkengine {

struct image {
    vk::Image handle;
    vk::Extent3D extent;
    vk::Format format;
    VmaAllocation allocation;
};

struct buffer {
    vk::Buffer handle;
    vk::DeviceSize size;
    VmaAllocation allocation;
};

class allocator {
    VmaAllocator allocator_;
public:
    allocator(const vulkan_core& core);
    ~allocator();

    allocator(const allocator&) = delete;
    allocator& operator=(const allocator&) = delete;
    allocator(allocator&&) = delete;
    allocator& operator=(allocator&&) = delete;

    image create_image(const vk::ImageCreateInfo& image_info) const;
    void destroy_image(image& image) const;

    buffer create_buffer(const vk::BufferCreateInfo& buffer_info) const;
    void destroy_buffer(buffer& buffer) const;
};

}