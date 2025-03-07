#pragma once

#include <vulkan/vulkan.hpp>
#include <concepts>
#include <vector>
#include <cstdint>

// Concept for both render-target sequences.
template<typename T>
concept render_target_sequence = requires(T t, const typename T::Config& cfg) {
    { T::validate(cfg) } -> std::same_as<bool>;
    { t.acquire_next() } -> std::same_as<vk::Result>;
    { t.submit() } -> std::same_as<vk::Result>;
    { t.current_target() } -> std::same_as<vk::ImageView>;
    { t.extent() } -> std::same_as<vk::Extent2D>;
};

// Tags for dispatching implementations.
struct swapchain_tag {};
struct offscreen_tag {};

// Primary template declaration.
template<typename Tag>
class render_target_sequence_impl;

//---------------------------------------------------------
// Swapchain-Based Implementation (Dynamic Rendering)
//---------------------------------------------------------
template<>
class render_target_sequence_impl<swapchain_tag> {
public:
    using Config = int;  // Placeholder configuration type.

    render_target_sequence_impl(vk::Device device,
                                vk::SurfaceKHR surface,
                                vk::Format format,
                                vk::Extent2D extent)
      : device_(device), surface_(surface), format_(format), extent_(extent) {
        create_swapchain();
        create_image_views();
    }

    static bool validate(const Config& cfg) {
        return true;
    }

    vk::Result acquire_next() {
        vk::Result result = device_.acquireNextImageKHR(
            swapchain_,
            UINT64_MAX,
            image_available_semaphore_,
            nullptr,
            &current_index_
        );
        return result;
    }

    vk::Result submit() {
        return vk::Result::eSuccess;
    }

    vk::ImageView current_target() {
        return image_views_[current_index_];
    }

    vk::Extent2D extent() const {
        return extent_;
    }

private:
    void create_swapchain() {
        swapchain_ = device_.createSwapchainKHR(
            vk::SwapchainCreateInfoKHR()
                .setSurface(surface_)
                .setMinImageCount(2)
                .setImageFormat(format_)
                .setImageExtent(extent_)
                .setImageArrayLayers(1)
                .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        );
    }

    void create_image_views() {
        auto images = device_.getSwapchainImagesKHR(swapchain_);
        for (auto& image : images) {
            vk::ImageViewCreateInfo view_info = vk::ImageViewCreateInfo()
                .setImage(image)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(format_)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                );
            vk::ImageView view = device_.createImageView(view_info);
            image_views_.push_back(view);
        }
    }

    vk::Device device_;
    vk::SurfaceKHR surface_;
    vk::SwapchainKHR swapchain_;
    vk::Format format_;
    vk::Extent2D extent_;
    std::vector<vk::ImageView> image_views_;
    uint32_t current_index_{0};

    vk::Semaphore image_available_semaphore_;
};

//---------------------------------------------------------
// Offscreen Rendering Implementation (Dynamic Rendering)
//---------------------------------------------------------
template<>
class render_target_sequence_impl<offscreen_tag> {
public:
    using Config = int;

    render_target_sequence_impl(vk::Device device,
                                vk::Format format,
                                vk::Extent2D extent)
      : device_(device), format_(format), extent_(extent) {
        create_offscreen_image();
        create_image_view();
    }

    static bool validate(const Config& cfg) {
        return true;
    }

    vk::Result acquire_next() {
        return vk::Result::eSuccess;
    }

    vk::Result submit() {
        return vk::Result::eSuccess;
    }

    vk::ImageView current_target() {
        return image_view_;
    }

    vk::Extent2D extent() const {
        return extent_;
    }

private:
    void create_offscreen_image() {
        vk::ImageCreateInfo image_info = vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setFormat(format_)
            .setExtent(vk::Extent3D(extent_.width, extent_.height, 1))
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
        offscreen_image_ = device_.createImage(image_info);
    }

    void create_image_view() {
        vk::ImageViewCreateInfo view_info = vk::ImageViewCreateInfo()
            .setImage(offscreen_image_)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(format_)
            .setSubresourceRange(
                vk::ImageSubresourceRange()
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setBaseMipLevel(0)
                    .setLevelCount(1)
                    .setBaseArrayLayer(0)
                    .setLayerCount(1)
            );
        image_view_ = device_.createImageView(view_info);
    }

    vk::Device device_;
    vk::Format format_;
    vk::Extent2D extent_;
    vk::Image offscreen_image_;
    vk::ImageView image_view_;
};
