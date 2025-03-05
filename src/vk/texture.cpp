
#include <vulkan/vulkan.hpp>
#include <imgui_impl_vulkan.h>

#include <vk/texture.hpp>
#include <vulkan_error.hpp>

texture::texture(
    vma::allocator& allocator,
    vk::Device device,
    vk::CommandPool command_pool,
    vk::Queue queue,
    std::dextents<size_t, 2> extents,
    vk::Format format,
    std::span<std::byte> data)
    : format_(format), extents_(extents) {
    {   
        auto extent = vk::Extent3D()
            .setHeight(extents.extent(0))
            .setWidth(extents.extent(1))
            .setDepth(1);

        auto [image, allocation] = allocator.create_image(
            vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setFormat(format)
                .setExtent(extent)
                .setMipLevels(1)
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setTiling(vk::ImageTiling::eOptimal)
                .setUsage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
                .setSharingMode(vk::SharingMode::eExclusive)
            );

        image_ = image;
        allocation_ = allocation;

        image_view_ = device.createImageView(
            vk::ImageViewCreateInfo()
                .setImage(image_)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(format)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setLevelCount(1)
                        .setLayerCount(1)
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                )
            );

        sampler_ = device.createSampler(
            vk::SamplerCreateInfo()
                .setMagFilter(vk::Filter::eLinear)
                .setMinFilter(vk::Filter::eLinear)
                .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                .setAddressModeU(vk::SamplerAddressMode::eRepeat)
                .setAddressModeV(vk::SamplerAddressMode::eRepeat)
                .setAddressModeW(vk::SamplerAddressMode::eRepeat)
                .setMinLod(0.0f)
                .setMaxLod(1.0f)
                .setMaxAnisotropy(1.0f)
        );

        descriptor_set_ = ImGui_ImplVulkan_AddTexture(
            sampler_,
            image_view_,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vma::staging_buffer staging_buffer = allocator.create_staging_buffer(data.size());
        std::memcpy(staging_buffer.mapping, data.data(), data.size());

        vk::CommandBuffer cmd = device.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo()
                .setCommandPool(command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(1)
        )[0];

        cmd.begin(vk::CommandBufferBeginInfo()
            .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

        // Copy to image
        {
            auto copy_barrier = vk::ImageMemoryBarrier2()
                .setSrcStageMask(vk::PipelineStageFlagBits2::eHost)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
                .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
                .setImage(image_)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setLevelCount(1)
                        .setLayerCount(1)
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                );

            cmd.pipelineBarrier2(
                vk::DependencyInfo()
                    .setImageMemoryBarriers(copy_barrier)
            );

            auto buffer_image_copy_region = vk::BufferImageCopy2()
                .setImageSubresource(
                    vk::ImageSubresourceLayers()
                        .setLayerCount(1)
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    )
                .setImageExtent(extent);

            auto buffer_image_copy_info = vk::CopyBufferToImageInfo2()
                    .setSrcBuffer(staging_buffer.buffer)
                    .setDstImage(image_)
                    .setRegions(buffer_image_copy_region)
                    .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal);

            cmd.copyBufferToImage2(buffer_image_copy_info);

            auto use_barrier = vk::ImageMemoryBarrier2()
                .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
                .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
                .setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
                .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                .setImage(image_)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setLevelCount(1)
                        .setLayerCount(1)
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                );

            cmd.pipelineBarrier2(
                vk::DependencyInfo()
                    .setImageMemoryBarriers(use_barrier)
            );

            cmd.end();

            auto cmd_submit_info = vk::CommandBufferSubmitInfo()
                .setCommandBuffer(cmd);
            auto submit_info = vk::SubmitInfo2()
                .setCommandBufferInfos(cmd_submit_info);
    
            vk::Fence fence = device.createFence(vk::FenceCreateInfo{});
            VK_CHECK(static_cast<VkResult>(queue.submit2(1, &submit_info, fence)));
    
            VK_CHECK(static_cast<VkResult>(device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX)));
            device.destroyFence(fence);
        }

        allocator.destroy_staging_buffer(staging_buffer);
    }
}