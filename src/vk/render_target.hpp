#pragma once

#include <vulkan/vulkan.hpp>
#include <concepts>
#include <vector>
#include <cstdint>

#include <vk/vma.hpp>
#include <functional>

template<typename T>
concept render_target_sequence = requires(T t, vk::Semaphore sem, const typename T::Config& cfg) {
    { t.acquire_next(sem) } -> std::same_as<vk::Result>;
    { t.submit() } -> std::same_as<vk::Result>;
    { t.current_target() } -> std::same_as<vk::ImageView>;
    { t.extent() } -> std::same_as<vk::Extent2D>;
};

struct swapchain_tag {};
struct offscreen_tag {};

template<typename Tag>
class render_target_sequence_impl;

template<>
class render_target_sequence_impl<swapchain_tag> {
public:
    struct swapchain_config {
        vk::PresentModeKHR present_mode;
        vk::SurfaceFormatKHR surface_format;
        uint32_t min_image_count;
        vk::SurfaceTransformFlagBitsKHR transform;
    };

    render_target_sequence_impl(
        vk::Device device,
        vk::PhysicalDevice physical_device,
        vk::SurfaceKHR surface,
        const swapchain_config& config,
        vk::Extent2D extent
    ) {
        initialise_swapchain(extent);
        initialise_image_views();
    }

    vk::Result acquire_next(vk::Semaphore image_available) {
        auto acquire_info = vk::AcquireNextImageInfoKHR()
            .setSemaphore(image_available)
            .setSwapchain(swapchain_)
            .setTimeout(std::numeric_limits<uint64_t>::max());

        auto acquire_result = device_.acquireNextImage2KHR(acquire_info);

        if (acquire_result.result != vk::Result::eErrorOutOfDateKHR &&
            acquire_result.result != vk::Result::eSuboptimalKHR)
            current_index_ = acquire_result.value;

        return acquire_result.result;
    }

    vk::ImageView current_target() const {
        return views_.at(current_index_);
    }

    vk::Extent2D extent() const {
        return extent_;
    }

    void recreate_swapchain(vk::Extent2D new_extent) {
        // Wait until the device is idle before destroying resources.
        // This is optional if synchronization is carefully handled elsewhere,
        // but often recommended to avoid conflicts.
        device_.waitIdle();

        for (auto& view : views_)
            device_.destroyImageView(view);

        views_.clear();

        device_.destroySwapchainKHR(swapchain_);
        swapchain_ = nullptr;

        extent_ = new_extent;

        create_swapchain_and_image_views(extent_);
    }

private:
    void create_swapchain_and_image_views(vk::Extent2D extent) {
        initialise_swapchain(extent);
        initialise_image_views();
    }

    void initialise_swapchain(vk::Extent2D extent) {
        auto swapchain_create_info = vk::SwapchainCreateInfoKHR()
            .setSurface(surface_)
            .setMinImageCount(swapchain_config_.min_image_count)
            .setImageFormat(swapchain_config_.surface_format.format)
            .setImageExtent(extent)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
            .setImageSharingMode(vk::SharingMode::eExclusive)
            .setPreTransform(swapchain_config_.transform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(swapchain_config_.present_mode);

        swapchain_ = device_.createSwapchainKHR(swapchain_create_info);
    }

    void initialise_image_views() {
        auto images = device_.getSwapchainImagesKHR(swapchain_);
        views_.reserve(images.size());

        for (auto& image: images) {
            auto image_create_info =                     
                vk::ImageViewCreateInfo()
                    .setViewType(vk::ImageViewType::e2D)
                    .setImage(image)
                    .setFormat(swapchain_config_.surface_format.format)
                    .setSubresourceRange(
                        vk::ImageSubresourceRange()
                            .setAspectMask(vk::ImageAspectFlagBits::eColor)
                            .setLayerCount(1)
                            .setLevelCount(1)
                            .setBaseMipLevel(0)
                            .setBaseArrayLayer(0));

            views_.push_back(device_.createImageView(image_create_info));
        }
    }

    vk::Device device_;
    vk::PhysicalDevice phyiscal_device_;
    vk::SurfaceKHR surface_;
    vk::SwapchainKHR swapchain_;
    swapchain_config swapchain_config_;
    vk::Extent2D extent_;
    std::vector<vk::ImageView> views_;
    uint32_t current_index_ = 0;
};

template<>
class render_target_sequence_impl<offscreen_tag> {
public:
    struct offscreen_config {
        vk::Format format;
        vk::ImageUsageFlags usage;
        uint32_t image_count;
        vk::Queue queue;
        uint32_t queue_family_index;
        vk::CommandPool command_pool;
    };

    struct offscreen_frame {
        vk::Image image;
        VmaAllocation allocation;
        vk::ImageView view;
        vk::CommandBuffer command_buffer;
        uint64_t usage_value;
        vk::ImageLayout current_layout = vk::ImageLayout::eUndefined;
    };

    render_target_sequence_impl(
        vk::Device device,
        vma::allocator* vma_allocator,
        const offscreen_config& config,
        vk::Extent2D extent
    ) : device_(device),
        vma_allocator_(vma_allocator),
        config_(config),
        extent_(extent) {
        create_offscreen_images();
        create_timeline_semaphore();

        latest_frame_index_.store(0, std::memory_order_relaxed);
    }

    ~render_target_sequence_impl() {
        cleanup();
    }

    offscreen_frame acquire_next(vk::Semaphore semaphore) {
        current_index_ = (current_index_ + 1) % config_.image_count;

        uint64_t needed_value = frames_[current_index_].usage_value;

        auto semaphore_wait_info = vk::SemaphoreWaitInfo()
            .setSemaphores(timeline_semaphore_)
            .setValues(needed_value);

        device_.waitSemaphores(semaphore_wait_info, std::numeric_limits<uint64_t>::max());

        return frames_[current_index_];
    }

    offscreen_frame get_latest_frame() {
        uint32_t index = latest_frame_index_.load(std::memory_order_acquire);
        return frames_[index];
    }

    vk::Result submit(uint32_t frame_index) {
        frame_counter_++;
        uint64_t signal_value = frame_counter_;

        offscreen_frame& frame = frames_[frame_index];

        frame.usage_value = signal_value;

        // GPU waits for (signal_value - 1) to ensure serialisation.
        uint64_t wait_value = (signal_value > 0) ? (signal_value - 1) : 0;

        auto command_buffer_submit_info = vk::CommandBufferSubmitInfo()
            .setCommandBuffer(frame.command_buffer);

        auto wait_semaphore = vk::SemaphoreSubmitInfo()
            .setSemaphore(timeline_semaphore_)
            .setValue(wait_value)
            .setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        auto signal_semaphore = vk::SemaphoreSubmitInfo()
            .setSemaphore(timeline_semaphore_)
            .setValue(signal_value)
            .setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        auto submit_info = vk::SubmitInfo2()
            .setWaitSemaphoreInfos(wait_semaphore)
            .setSignalSemaphoreInfos(signal_semaphore)
            .setCommandBufferInfos(command_buffer_submit_info);

        config_.queue.submit2(submit_info);

        latest_frame_index_.store(frame_index, std::memory_order_release);

        return vk::Result::eSuccess;
    }
    
private:
    void create_timeline_semaphore() {
        auto semaphore_type_info = vk::SemaphoreTypeCreateInfo()
            .setSemaphoreType(vk::SemaphoreType::eTimeline)
            .setInitialValue(0);

        timeline_semaphore_ = device_.createSemaphore(
            vk::SemaphoreCreateInfo().setPNext(&semaphore_type_info)
        );
    }

    void create_offscreen_images() {
        frames_.resize(config_.image_count);

        auto command_buffers = device_.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo()
                .setCommandPool(config_.command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(config_.image_count)
        );

        for (size_t i = 0; i < config_.image_count; ++i) {
            offscreen_frame& frame = frames_[i];

            frame.command_buffer = command_buffers[i];
            frame.usage_value = 0;

            auto image_create_info = vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D(extent_))
                .setMipLevels(1)
                .setArrayLayers(1)
                .setFormat(config_.format)
                .setTiling(vk::ImageTiling::eOptimal)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setUsage(config_.usage)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setSharingMode(vk::SharingMode::eExclusive);

            auto [image, allocation] = vma_allocator_->create_image(image_create_info);
            frame.image = image;
            frame.allocation = allocation;

            auto view_info = vk::ImageViewCreateInfo()
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(config_.format)
                .setImage(frame.image)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setLayerCount(1)
                        .setLevelCount(1)
                        .setBaseMipLevel(0)
                        .setBaseArrayLayer(0)
                );

            frame.view = device_.createImageView(view_info);
        }
    }

    void cleanup() {
        if (!frames_.empty() && frames_[0].command_buffer) {
            std::vector<vk::CommandBuffer> cbs;
            cbs.reserve(frames_.size());
            for (auto& f : frames_) {
                cbs.push_back(f.command_buffer);
            }
            device_.freeCommandBuffers(config_.command_pool,
                                         static_cast<uint32_t>(cbs.size()),
                                         cbs.data());
        }

        for (auto& frame : frames_) {
            if (frame.view)   device_.destroyImageView(frame.view);
            if (frame.image)  vma_allocator_->destroy_image(frame.image, frame.allocation);
        }

        frames_.clear();
    }

    vk::Device device_;
    vma::allocator* vma_allocator_;
    offscreen_config config_;
    vk::Extent2D extent_;

    std::vector<offscreen_frame> frames_;

    vk::Semaphore timeline_semaphore_;

    uint32_t current_index_ = 0;
    uint32_t frame_counter_ = 0;

    std::atomic<uint32_t> latest_frame_index_{0};
};