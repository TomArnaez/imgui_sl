#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <spdlog/spdlog.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <cstring>
#include <array>
#include <limits>
#include <cstdlib>

#include <detailed_exception.hpp>
#include <vk/render_target.hpp>
#include <vma/vk_mem_alloc.h>
#include <vk/vulkan_app.hpp>
#include <vulkan_error.hpp>

#include <iostream>

class app_state {       
    std::unique_ptr<vulkan_core> core;

    std::unique_ptr<render_target_swapchain> render_target;
    vk::SurfaceKHR surface_ = VK_NULL_HANDLE;
    vk::SurfaceFormatKHR surface_format_;

    vk::PipelineCache pipeline_cache = VK_NULL_HANDLE;
    vk::DescriptorPool descriptor_pool = VK_NULL_HANDLE;

    GLFWwindow* window_ = nullptr;
    std::string window_title = "DEAR IMGUI";
    vk::Extent2D extent = { 1280, 720 };
    std::vector<const char*> instance_extensions;
    uint32_t min_image_count = 2;

public:
    app_state() {
        init_glfw_window();
        init_vulkan();
        init_render_target();
        init_imgui();

        const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;

        VmaVulkanFunctions functions {
            .vkGetInstanceProcAddr = d.vkGetInstanceProcAddr,
            .vkGetDeviceProcAddr = d.vkGetDeviceProcAddr,
        };

        VmaAllocator allocator;
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorInfo.physicalDevice = core->physical_device();
        allocatorInfo.device = core->device();
        allocatorInfo.instance = core->instance();
        allocatorInfo.pVulkanFunctions = &functions;
        vmaCreateAllocator(&allocatorInfo, &allocator);

        const vk::PhysicalDeviceMemoryProperties memProps = core->physical_device().getMemoryProperties();
        uint32_t hostVisibleTypeIndex = VK_MAX_MEMORY_TYPES;
        bool isDeviceLocal = false;
    
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            const auto flags = memProps.memoryTypes[i].propertyFlags;
            if ((flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
                (flags & vk::MemoryPropertyFlagBits::eHostCoherent) &&
                (flags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                hostVisibleTypeIndex = i;
                break;
            }
        }
    
        if (hostVisibleTypeIndex == VK_MAX_MEMORY_TYPES) {
            std::cerr << "No suitable host-visible memory type found!\n";
        }
    
        std::cout << "Selected memory type " << hostVisibleTypeIndex 
                  << " which is " << (isDeviceLocal ? "device-local" : "non-device-local") << "\n";
    
        // Create VMA memory pool
        VmaPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.memoryTypeIndex = hostVisibleTypeIndex;
        VmaPool pool;
        vmaCreatePool(allocator, &poolCreateInfo, &pool);
    
        // Create buffer through VMA
        vk::BufferCreateInfo bufferCreateInfo({}, 
            1024,  // Buffer size
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::SharingMode::eExclusive);
    
        VmaAllocationCreateInfo allocCreateInfo{};
        allocCreateInfo.pool = pool;
        
        VkBuffer buffer;
        VmaAllocation allocation;
        vmaCreateBuffer(allocator, 
            reinterpret_cast<VkBufferCreateInfo*>(&bufferCreateInfo),
            &allocCreateInfo,
            &buffer,
            &allocation,
            nullptr);
    
        // Verify allocation properties
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(allocator, allocation, &allocInfo);
        
        const vk::MemoryPropertyFlags allocatedFlags = 
            memProps.memoryTypes[allocInfo.memoryType].propertyFlags;
    
        const bool isAllocatedDeviceLocal = 
            static_cast<bool>(allocatedFlags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    
        std::cout << "Allocated buffer is in memory type " << allocInfo.memoryType
                  << " which is " << (isAllocatedDeviceLocal ? "device-local" : "non-device-local") << "\n";
    
        // Cleanup
        vmaDestroyBuffer(allocator, buffer, allocation);
        vmaDestroyPool(allocator, pool);
    }

    void run() {
        bool show_demo_window = true;
        bool show_another_window = false;
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) != 0) {
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

            ImGuiIO& io = ImGui::GetIO(); (void)io;

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::ShowDemoWindow(&show_demo_window);

            {
                static float slider_value = 0.0f;
                static int counter = 0;
        
                ImGui::Begin("Hello, world!");
                ImGui::Text("This is some useful text.");
                ImGui::Checkbox("Demo Window", &show_demo_window);
                ImGui::Checkbox("Another Window", &show_another_window);
                ImGui::SliderFloat("float", &slider_value, 0.0f, 1.0f);
                ImGui::ColorEdit3("clear color", (float*)&clear_color);
                if (ImGui::Button("Button"))
                    counter++;
                ImGui::SameLine();
                ImGui::Text("counter = %d", counter);
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
                ImGui::End();
            }

            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();
            if (!(draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f))
                render_and_present_frame(draw_data);
        }
    }

    ~app_state() {
        cleanup();
    }

private:
    static void glfw_error_callback(int error, const char* description) {
        spdlog::error("GLFW Error {}: {}", error, description);
    }

    void init_glfw_window() {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            throw detailed_exception("GLFW failed to init");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(extent.width, extent.height, window_title.c_str(), nullptr, nullptr);
        if (!window_)
            throw detailed_exception("Failed to create GLFW window");

        if (!glfwVulkanSupported())
            throw detailed_exception("GLFW: Vulkan not supported");

        uint32_t ext_count = 0;
        const char** glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
        for (uint32_t i = 0; i < ext_count; i++)
            instance_extensions.push_back(glfw_exts[i]);
    }

    void init_vulkan() {
        std::vector<const char*> device_extensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
            VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
            VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
            VK_KHR_MULTIVIEW_EXTENSION_NAME,
            VK_KHR_MAINTENANCE2_EXTENSION_NAME
        };
        
        core = std::make_unique<vulkan_core>(instance_extensions, device_extensions);

        std::array<vk::DescriptorPoolSize, 1> pool_sizes = {{
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE)
        }};

        descriptor_pool = core->device().createDescriptorPool(
            vk::DescriptorPoolCreateInfo()
                .setPoolSizes(pool_sizes)
                .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                .setMaxSets(1)
        );

        VkSurfaceKHR raw_surface;
        VK_CHECK(glfwCreateWindowSurface(static_cast<VkInstance>(core->instance()), window_, nullptr, &raw_surface));
        surface_ = raw_surface;
    }

    void init_render_target() {
        vk::Format requested_formats[] = { 
            vk::Format::eB8G8R8A8Unorm,
            vk::Format::eR8G8B8A8Unorm,
            vk::Format::eB8G8R8Unorm,
            vk::Format::eR8G8B8Unorm
        };

        vk::ColorSpaceKHR requested_color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
        surface_format_ = select_surface_format(
            std::vector<vk::Format>(std::begin(requested_formats), std::end(requested_formats)),
            requested_color_space
        );
        
        render_target_swapchain::swapchain_config config = {
            .present_mode = vk::PresentModeKHR::eFifo,
            .surface_format = surface_format_,
            .min_image_count = 2,
            .transform = vk::SurfaceTransformFlagBitsKHR::eIdentity
        };

        render_target = std::make_unique<render_target_swapchain>(
            core->device(),
            core->physical_device(),
            surface_,
            core->graphics_queue(),
            core->graphics_command_pool(),
            config,
            get_window_size());
    }

    vk::SurfaceFormatKHR select_surface_format(const std::vector<vk::Format>& requested_formats, vk::ColorSpaceKHR requested_color_space) {
        auto available_formats = core->physical_device().getSurfaceFormatsKHR(surface_);
        if (available_formats.size() == 1 && available_formats[0].format == vk::Format::eUndefined) {
            return { requested_formats[0], requested_color_space };
        }
        for (const auto& available_format : available_formats) {
            for (auto req_format : requested_formats) {
                if (available_format.format == req_format && available_format.colorSpace == requested_color_space)
                    return available_format;
            }
        }
        return available_formats[0];
    }

    void init_imgui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(window_, true);

        auto pipeline_rendering_create_info = VkPipelineRenderingCreateInfo{};
        pipeline_rendering_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipeline_rendering_create_info.colorAttachmentCount = 1;
        pipeline_rendering_create_info.pColorAttachmentFormats = reinterpret_cast<VkFormat*>(&surface_format_);

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.PipelineRenderingCreateInfo = pipeline_rendering_create_info;
        init_info.UseDynamicRendering = true;
        init_info.Instance = static_cast<VkInstance>(core->instance());
        init_info.PhysicalDevice = static_cast<VkPhysicalDevice>(core->physical_device());
        init_info.Device = static_cast<VkDevice>(core->device());
        init_info.QueueFamily = core->graphics_queue_family();
        init_info.Queue = static_cast<VkQueue>(core->graphics_queue());
        init_info.PipelineCache = pipeline_cache;
        init_info.DescriptorPool = static_cast<VkDescriptorPool>(descriptor_pool);
        init_info.Subpass = 0;
        init_info.MinImageCount = min_image_count;
        init_info.ImageCount = static_cast<uint32_t>(render_target->frames().size());
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = check_vk_result;

        if (!ImGui_ImplVulkan_Init(&init_info))
            throw detailed_exception("Failed to init ImGui with Vulkan");
    }

    void render_and_present_frame(ImDrawData* draw_data) {
        try {
            render_target_swapchain::swapchain_frame& frame = render_target->acquire_next();

            frame.command_buffer.reset();
            frame.command_buffer.begin(
                vk::CommandBufferBeginInfo()
                    .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
            );
            
            auto render_image_memory_barrier = vk::ImageMemoryBarrier()
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setImage(frame.image)
                .setSubresourceRange(vk::ImageSubresourceRange()
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setBaseMipLevel(0)
                    .setLevelCount(1)
                    .setBaseArrayLayer(0)
                    .setLayerCount(1));

            frame.command_buffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTopOfPipe,
                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
                    {},
                    nullptr,
                    nullptr,
                    render_image_memory_barrier
                );

            frame.current_layout = vk::ImageLayout::eColorAttachmentOptimal;

            auto color_attachment = vk::RenderingAttachmentInfo()
                .setImageView(frame.view)
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setClearValue(
                    vk::ClearValue()
                        .setColor(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})));

            auto rendering_info = vk::RenderingInfo()
                .setRenderArea(vk::Rect2D()
                    .setOffset({0, 0})
                    .setExtent(extent))
                .setLayerCount(1)
                .setColorAttachmentCount(1)
                .setPColorAttachments(&color_attachment);

            frame.command_buffer.beginRenderingKHR(rendering_info);
            ImGui_ImplVulkan_RenderDrawData(draw_data, static_cast<VkCommandBuffer>(frame.command_buffer));
            frame.command_buffer.endRenderingKHR();

            auto image_memory_barrier = vk::ImageMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
                .setOldLayout(frame.current_layout)
                .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
                .setImage(frame.image)
                .setSubresourceRange(vk::ImageSubresourceRange()
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setBaseMipLevel(0)
                    .setLevelCount(1)
                    .setBaseArrayLayer(0)
                    .setLayerCount(1));

            frame.command_buffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::PipelineStageFlagBits::eBottomOfPipe,
                {},
                nullptr,
                nullptr,
                image_memory_barrier
            );

            frame.current_layout = vk::ImageLayout::ePresentSrcKHR;

            frame.command_buffer.end();

            vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            core->graphics_queue().submit(
                vk::SubmitInfo()
                    .setCommandBuffers({ frame.command_buffer })
                    .setWaitSemaphores(frame.image_available)
                    .setSignalSemaphores(frame.render_finished)
                    .setWaitDstStageMask(wait_stage),
                frame.in_flight_fence
            );

            render_target->present();
        } catch (std::exception& exception) {
            spdlog::info("Got exception {}, recreating swapchain", exception.what());
            core->device().waitIdle();
            extent = get_window_size();
            if (extent.width == 0 || extent.height == 0) {
                spdlog::info("Window minimized, skipping swapchain recreation");
                return;
            }
            render_target->recreate_swapchain(extent);
        }
    }

    vk::Extent2D get_window_size() {
        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);
        return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

    void cleanup() {
        core->device().waitIdle();

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window_);
        glfwTerminate();
    
        if (descriptor_pool)
            core->device().destroyDescriptorPool(descriptor_pool);

        render_target.reset();
    
        if (surface_)
            core->instance().destroySurfaceKHR(surface_);
    }
};

int main() {
    try {
        app_state app;
        // app.run();
    }
    catch (const std::exception &ex) {
        spdlog::error("Application encountered an error: {}", ex.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
