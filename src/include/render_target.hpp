#pragma once

#include <vulkan/vulkan.hpp>
#include <concepts>
#include <vector>
#include <cstdint>

#include <vk/vma.hpp>
#include <functional>

class render_target_swapchain {
public:
    struct swapchain_config {
        vk::PresentModeKHR present_mode;
        vk::SurfaceFormatKHR surface_format;
        uint32_t min_image_count;
        vk::SurfaceTransformFlagBitsKHR transform;
    };

    struct swapchain_frame {
        vk::Image image;
        vk::ImageView view;
        vk::CommandBuffer command_buffer;
        vk::Semaphore image_available;
        vk::Semaphore render_finished;
        vk::Fence in_flight_fence;
        vk::ImageLayout current_layout;
    };

    render_target_swapchain(
        vk::Device device,
        vk::PhysicalDevice physical_device,
        vk::SurfaceKHR surface,
        vk::Queue present_queue,
        vk::CommandPool command_pool,
        const swapchain_config& config,
        vk::Extent2D extent
    ) : device_(device), physical_device_(physical_device), surface_(surface),
        present_queue_(present_queue), command_pool_(command_pool),
        config_(config), extent_(extent) {
        create_swapchain();
        create_frames();
    }

    ~render_target_swapchain() {
        cleanup();
    }

    vk::SwapchainKHR swapchain() const {
        return swapchain_;
    }

    swapchain_frame& acquire_next() {
        auto [result, image_index] = device_.acquireNextImageKHR(
            swapchain_,
            UINT64_MAX,
            frames_[current_frame_].image_available,
            nullptr
        );

        if (result == vk::Result::eSuboptimalKHR)
            spdlog::warn("Swapchain suboptimal");

        current_image_index_ = image_index;
        return frames_[current_frame_];
    }

    void present() {
        vk::PresentInfoKHR present_info;
        present_info.setWaitSemaphores(frames_[current_frame_].render_finished)
                    .setSwapchains(swapchain_)
                    .setImageIndices(current_image_index_);
    
        auto result = present_queue_.presentKHR(present_info);

        device_.waitForFences(
            frames_[current_frame_].in_flight_fence,
            VK_TRUE,
            UINT64_MAX
        );
        device_.resetFences(frames_[current_frame_].in_flight_fence);

        current_frame_ = (current_frame_ + 1) % config_.min_image_count;
    }

    void recreate_swapchain(vk::Extent2D new_extent) {
        device_.waitIdle();
        cleanup_frames();
    
        vk::SwapchainKHR old_swapchain = swapchain_;
        extent_ = new_extent;

        create_swapchain(old_swapchain);
        create_frames();

        if (old_swapchain)
            device_.destroySwapchainKHR(old_swapchain);
    }
    vk::Extent2D extent() const { return extent_; }
    const std::vector<swapchain_frame>& frames() const { return frames_; }

private:
    void create_swapchain(vk::SwapchainKHR old_swapchain = nullptr) {
        auto surface_capabilities = physical_device_.getSurfaceCapabilitiesKHR(surface_);

        vk::Extent2D actual_extent = extent_;
        if (surface_capabilities.currentExtent.width != UINT32_MAX) {
            actual_extent = surface_capabilities.currentExtent;
        } else {
            actual_extent.width = std::clamp(actual_extent.width, 
                surface_capabilities.minImageExtent.width, 
                surface_capabilities.maxImageExtent.width);
            actual_extent.height = std::clamp(actual_extent.height, 
                surface_capabilities.minImageExtent.height, 
                surface_capabilities.maxImageExtent.height);
        }
        extent_ = actual_extent;

        vk::SwapchainCreateInfoKHR create_info;
        create_info.setSurface(surface_)
                .setMinImageCount(config_.min_image_count)
                .setImageFormat(config_.surface_format.format)
                .setImageColorSpace(config_.surface_format.colorSpace)
                .setImageExtent(extent_)
                .setImageArrayLayers(1)
                .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                .setImageSharingMode(vk::SharingMode::eExclusive)
                .setPreTransform(config_.transform)
                .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                .setPresentMode(config_.present_mode)
                .setClipped(true)
                .setOldSwapchain(old_swapchain);

        swapchain_ = device_.createSwapchainKHR(create_info);
        images_ = device_.getSwapchainImagesKHR(swapchain_);
    }

    void create_frames() {
        create_image_views();
        create_command_buffers();
        create_sync_objects();

        frames_.resize(images_.size());
        for (size_t i = 0; i < images_.size(); ++i) {
            frames_[i] = {
                images_[i],
                image_views_[i],
                command_buffers_[i],
                image_available_semaphores_[i],
                render_finished_semaphores_[i],
                in_flight_fences_[i],
                vk::ImageLayout::eUndefined
            };
        }
    }

    void create_image_views() {
        image_views_.clear();
        image_views_.reserve(images_.size());
        for (auto image : images_) {
            vk::ImageViewCreateInfo view_info;
            view_info.setImage(image)
                    .setViewType(vk::ImageViewType::e2D)
                    .setFormat(config_.surface_format.format)
                    .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

            image_views_.push_back(device_.createImageView(view_info));
        }
    }

    void create_command_buffers() {
        vk::CommandBufferAllocateInfo alloc_info;
        alloc_info.setCommandPool(command_pool_)
                    .setLevel(vk::CommandBufferLevel::ePrimary)
                    .setCommandBufferCount(static_cast<uint32_t>(images_.size()));

        command_buffers_ = device_.allocateCommandBuffers(alloc_info);
    }

    void create_sync_objects() {
        image_available_semaphores_.resize(images_.size());
        render_finished_semaphores_.resize(images_.size());
        in_flight_fences_.resize(images_.size());

        for (size_t i = 0; i < images_.size(); ++i) {
            image_available_semaphores_[i] = device_.createSemaphore({});
            render_finished_semaphores_[i] = device_.createSemaphore({});
            in_flight_fences_[i] = device_.createFence({vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void cleanup_frames() {
        for (auto& sem : image_available_semaphores_) device_.destroySemaphore(sem);
        for (auto& sem : render_finished_semaphores_) device_.destroySemaphore(sem);
        for (auto& fence : in_flight_fences_) device_.destroyFence(fence);
        for (auto& view : image_views_) device_.destroyImageView(view);
        
        if (!command_buffers_.empty()) device_.freeCommandBuffers(command_pool_, command_buffers_);
    
        image_available_semaphores_.clear();
        render_finished_semaphores_.clear();
        in_flight_fences_.clear();
        image_views_.clear();
        command_buffers_.clear();
        frames_.clear();
    }

    void cleanup_swapchain() {
        device_.destroySwapchainKHR(swapchain_);
        images_.clear();
    }

    void cleanup() {
        device_.waitIdle();
        cleanup_frames();
        cleanup_swapchain();
    }

    vk::Device device_;
    vk::PhysicalDevice physical_device_;
    vk::SurfaceKHR surface_;
    vk::Queue present_queue_;
    vk::CommandPool command_pool_;
    swapchain_config config_;
    vk::Extent2D extent_;

    vk::SwapchainKHR swapchain_;
    std::vector<vk::Image> images_;
    std::vector<vk::ImageView> image_views_;
    std::vector<vk::CommandBuffer> command_buffers_;
    std::vector<vk::Semaphore> image_available_semaphores_;
    std::vector<vk::Semaphore> render_finished_semaphores_;
    std::vector<vk::Fence> in_flight_fences_;

    std::vector<swapchain_frame> frames_;
    uint32_t current_frame_ = 0;
    uint32_t current_image_index_ = 0;
};

class render_target_offscreen {
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
        vma::image image;
        VmaAllocation allocation;
        vk::ImageView view;
        vk::CommandBuffer command_buffer;
        uint64_t usage_value;
        vk::ImageLayout current_layout = vk::ImageLayout::eUndefined;
    };

    render_target_offscreen(
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

    ~render_target_offscreen() {
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

            vma::image image = vma_allocator_->create_image(image_create_info);
            frame.image = image;

            auto view_info = vk::ImageViewCreateInfo()
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(config_.format)
                .setImage(frame.image.handle)
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
            if (frame.image.handle)  vma_allocator_->destroy_image(frame.image);
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