#include <vulkan/vulkan.hpp>
#include <vk/texture.hpp>

texture::texture(vk::Device device, std::dextents<size_t, 2> extents, vk::Format format)
    : format_(format), extents_(extents) {

    // Create the Vulkan image
    {
        auto info = vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setFormat(format)
            .setExtent(vk::Extent3D().setHeight(extents.extent(0)).setWidth(extents.extent(1)).setDepth(1))
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
            .setSharingMode(vk::SharingMode::eExclusive);

        image_ = device.createImage(info);
    }
}