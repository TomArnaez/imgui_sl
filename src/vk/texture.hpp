#pragma once

#include <vk/vma.hpp>

#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_enums.hpp>

#include <mdspan> // for std::dextents

class texture {
    vk::DescriptorSet           descriptor_set_;
    std::dextents<size_t, 2>    extents_;
    vk::Format                  format_;
    vk::ImageView               image_view_;
    vk::Image                   image_;
    VmaAllocation               allocation_;
    vk::Sampler                 sampler_;
public:
    texture(
        vma::allocator& allocator,
        vk::Device device,
        vk::CommandPool pool,
        vk::Queue queue,
        std::dextents<size_t, 2> extents,
        vk::Format format,
        std::span<std::byte> data
    );
};