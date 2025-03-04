#pragma once

#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_enums.hpp>

#include <mdspan> // for std::dextents

class texture {
    vk::DescriptorSet           descriptor_set_;
    std::dextents<size_t, 2>    extents_;
    vk::Format                  format_;
    vk::ImageView               image_view_;
    vk::Image                   image_;
    vk::Sampler                 sampler_;
    vk::Buffer                  buffer_;
    vk::DeviceMemory            upload_buffer_memory_;
public:
    texture(vk::Device device, std::dextents<size_t, 2> extents, vk::Format format);
};