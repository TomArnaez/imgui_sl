#pragma once

#include <vulkan/vulkan_handles.hpp>

struct VmaAllocator_T;
typedef struct VmaAllocator_T* VmaAllocator;

class vma_context {
    VmaAllocator allocator_;
public:
    vma_context(vk::Device device, vk::PhysicalDevice physical_device, vk::Instance instance);
    ~vma_context();
};