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
#include <vk/vma.hpp>
#include <vk/texture.hpp>
#include <vulkan_error.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static bool is_extension_available(const std::vector<vk::ExtensionProperties>& properties, const char* extension) {
    for (const auto& prop : properties)
        if (strcmp(prop.extensionName, extension) == 0)
            return true;
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_report_callback(
    VkDebugReportFlagsEXT      flags,
    VkDebugReportObjectTypeEXT object_type,
    uint64_t                   object,
    size_t                     location,
    int32_t                    message_code,
    const char*                p_layer_prefix,
    const char*                p_message,
    void*                      /*p_user_data*/
) {
    spdlog::error("Validation Layer ({}): {}", p_layer_prefix, p_message);
    return VK_FALSE;
}

struct frame_resource {
    vk::CommandPool                   command_pool = VK_NULL_HANDLE;
    std::vector<vk::CommandBuffer>    command_buffers;
    vk::Image                         backbuffer = VK_NULL_HANDLE;
    vk::ImageView                     backbuffer_view = VK_NULL_HANDLE;
    vk::Semaphore                     image_acquired_semaphore, render_completed_semaphore;
    vk::Fence                         render_fence;
};

struct vulkan_state {           
    vk::Instance                        instance = VK_NULL_HANDLE;
    vk::PhysicalDevice                  physical_device = VK_NULL_HANDLE;
    vk::Device                          device = VK_NULL_HANDLE;
    uint32_t                            queue_family = UINT32_MAX;
    vk::Queue                           queue = VK_NULL_HANDLE;

    std::unique_ptr<vma::allocator>       context;


    VkDebugReportCallbackEXT            debug_report = VK_NULL_HANDLE;
    vk::PipelineCache                   pipeline_cache = VK_NULL_HANDLE;
    vk::DescriptorPool                  descriptor_pool = VK_NULL_HANDLE;
    vk::SurfaceKHR                      surface = VK_NULL_HANDLE;
    vk::SurfaceFormatKHR                surface_format = {};
    vk::PresentModeKHR                  present_mode;
    vk::SwapchainKHR                    swapchain = VK_NULL_HANDLE;
    bool                                swapchain_needs_rebuild = false;
    std::vector<frame_resource>         swapchain_frames;
    uint32_t                            image_count = 0;
    uint32_t                            width = 0;
    uint32_t                            height = 0;
    uint32_t                            image_acquired_semaphore_index = 0;
};      

class app_state {       
    vulkan_state vk_state;      
    GLFWwindow* window = nullptr;
    std::string window_title = "DEAR IMGUI";
    std::vector<const char*> instance_extensions;
    uint32_t min_image_count = 2;

public:
    app_state() {
        init_glfw_window();
        init_vulkan();
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
        window = glfwCreateWindow(1280, 720, window_title.c_str(), nullptr, nullptr);
        if (!window)
            throw detailed_exception("Failed to create GLFW window");

        if (!glfwVulkanSupported())
            throw detailed_exception("GLFW: Vulkan not supported");

        uint32_t ext_count = 0;
        const char** glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
        for (uint32_t i = 0; i < ext_count; i++)
            instance_extensions.push_back(glfw_exts[i]);
    }

    void init_vulkan() {
        VULKAN_HPP_DEFAULT_DISPATCHER.init();

        vk::ApplicationInfo app_info("Dear ImGui Vulkan App", 1, "No Engine", 1, VK_API_VERSION_1_1);

        auto available_extensions = vk::enumerateInstanceExtensionProperties(nullptr);
        if (is_extension_available(available_extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        std::vector<const char*> layers;

#ifdef APP_USE_VULKAN_DEBUG_REPORT
        layers.emplace_back("VK_LAYER_KHRONOS_validation");
        instance_extensions.push_back("VK_EXT_debug_report");
#endif

        auto instance_ci = vk::InstanceCreateInfo()
            .setPApplicationInfo(&app_info)
            .setPEnabledExtensionNames(instance_extensions)
            .setPEnabledLayerNames(layers);

        vk_state.instance = vk::createInstance(instance_ci);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_state.instance);

#ifdef APP_USE_VULKAN_DEBUG_REPORT
        vk_state.debug_report = vk_state.instance.createDebugReportCallbackEXT(
            vk::DebugReportCallbackCreateInfoEXT()
                .setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning | vk::DebugReportFlagBitsEXT::ePerformanceWarning)
                .setPfnCallback(debug_report_callback)
            );
#endif

        // --- Select Physical Device ---
        auto physical_devices = vk_state.instance.enumeratePhysicalDevices();
        if (physical_devices.empty())
            throw detailed_exception("No GPUs with Vulkan support found.");

        // Prefer discrete GPUs
        for (const auto& pd : physical_devices) {
            vk::PhysicalDeviceProperties props = pd.getProperties();
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                vk_state.physical_device = pd;
                break;
            }
        }
        if (!vk_state.physical_device)
            vk_state.physical_device = physical_devices[0];

        // --- Select Queue Family ---
        auto queue_properties = vk_state.physical_device.getQueueFamilyProperties();
        for (uint32_t i = 0; i < static_cast<uint32_t>(queue_properties.size()); i++) {
            if (queue_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                vk_state.queue_family = i;
                break;
            }
        }
        if (vk_state.queue_family == UINT32_MAX)
            throw detailed_exception("Failed to find a queue family index with graphics support.");

        // --- Create Device ---
        std::vector<const char*> device_extensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
            VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
            VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
            VK_KHR_MULTIVIEW_EXTENSION_NAME,
            VK_KHR_MAINTENANCE2_EXTENSION_NAME
        };

        float queue_priority = 1.0f;
        vk::DeviceQueueCreateInfo queue_ci({}, vk_state.queue_family, 1, &queue_priority);

        vk::PhysicalDeviceTimelineSemaphoreFeatures timeline_features(true);
        vk::PhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features(true);
        dynamic_rendering_features.pNext = &timeline_features;

        auto device_ci = vk::DeviceCreateInfo()
            .setQueueCreateInfos(queue_ci)
            .setPEnabledExtensionNames(device_extensions)
            .setPNext(&dynamic_rendering_features);

        vk_state.device = vk_state.physical_device.createDevice(device_ci);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_state.device);

        vk_state.queue = vk_state.device.getQueue(vk_state.queue_family, 0);

        vk_state.context = std::make_unique<vma::allocator>(
            vk_state.device,
            vk_state.physical_device,
            vk_state.instance);

        // --- Create Descriptor Pool ---
        std::array<vk::DescriptorPoolSize, 1> pool_sizes = {{
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE)
        }};

        vk_state.descriptor_pool = vk_state.device.createDescriptorPool(
            vk::DescriptorPoolCreateInfo()
                .setPoolSizes(pool_sizes)
                .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                .setMaxSets(1)
        );

        // --- Create Window Surface ---
        VkSurfaceKHR raw_surface;
        VK_CHECK(glfwCreateWindowSurface(static_cast<VkInstance>(vk_state.instance), window, nullptr, &raw_surface));
        vk_state.surface = raw_surface;

        if (!vk_state.physical_device.getSurfaceSupportKHR(vk_state.queue_family, vk_state.surface))
            throw detailed_exception("Error: no WSI support on physical device");

        // Choose a surface format
        vk::Format requested_formats[] = { vk::Format::eB8G8R8A8Unorm,
                                           vk::Format::eR8G8B8A8Unorm,
                                           vk::Format::eB8G8R8Unorm,
                                           vk::Format::eR8G8B8Unorm };
        vk::ColorSpaceKHR requested_color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
        vk_state.surface_format = select_surface_format(
            std::vector<vk::Format>(std::begin(requested_formats), std::end(requested_formats)),
            requested_color_space
        );

        vk_state.present_mode = vk::PresentModeKHR::eFifo;
        IM_ASSERT(min_image_count >= 2);

        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        vk_state.width = static_cast<uint32_t>(fb_width);
        vk_state.height = static_cast<uint32_t>(fb_height);

        recreate_swapchain();

        // --- Setup ImGui ---
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window, true);

        VkPipelineRenderingCreateInfo pipeline_rendering_ci = {};
        pipeline_rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipeline_rendering_ci.colorAttachmentCount = 1;
        pipeline_rendering_ci.pColorAttachmentFormats = reinterpret_cast<VkFormat*>(&vk_state.surface_format.format);

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.PipelineRenderingCreateInfo = pipeline_rendering_ci;
        init_info.UseDynamicRendering = true;
        init_info.Instance = static_cast<VkInstance>(vk_state.instance);
        init_info.PhysicalDevice = static_cast<VkPhysicalDevice>(vk_state.physical_device);
        init_info.Device = static_cast<VkDevice>(vk_state.device);
        init_info.QueueFamily = vk_state.queue_family;
        init_info.Queue = static_cast<VkQueue>(vk_state.queue);
        init_info.PipelineCache = vk_state.pipeline_cache;
        init_info.DescriptorPool = static_cast<VkDescriptorPool>(vk_state.descriptor_pool);
        init_info.Subpass = 0;
        init_info.MinImageCount = min_image_count;
        init_info.ImageCount = vk_state.image_count;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = check_vk_result;

        if (!ImGui_ImplVulkan_Init(&init_info))
            throw detailed_exception("Failed to init ImGui with Vulkan");

        // --- Main loop ---
        bool show_demo_window = true;
        bool show_another_window = false;
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            glfwGetFramebufferSize(window, &fb_width, &fb_height);
            if (fb_width > 0 && fb_height > 0 &&
                (vk_state.swapchain_needs_rebuild || vk_state.width != static_cast<uint32_t>(fb_width) || vk_state.height != static_cast<uint32_t>(fb_height))) {
                
                recreate_swapchain();
                vk_state.image_acquired_semaphore_index = 0;
                vk_state.swapchain_needs_rebuild = false;
            }

            if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

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

        vk_state.device.waitIdle();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    vk::SurfaceFormatKHR select_surface_format(const std::vector<vk::Format>& requested_formats, vk::ColorSpaceKHR requested_color_space) {
        auto available_formats = vk_state.physical_device.getSurfaceFormatsKHR(vk_state.surface);
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

    void recreate_swapchain() {
        auto old_swapchain = vk_state.swapchain;

        if (vk_state.swapchain) {
            vk_state.device.destroySwapchainKHR(vk_state.swapchain, nullptr);
            vk_state.swapchain = VK_NULL_HANDLE;
        }

        vk_state.device.waitIdle();
        for (auto& frame : vk_state.swapchain_frames) {
            if (frame.backbuffer_view)
                vk_state.device.destroyImageView(frame.backbuffer_view, nullptr);
            if (frame.command_pool)
                vk_state.device.destroyCommandPool(frame.command_pool, nullptr);
            if (frame.render_completed_semaphore)
                vk_state.device.destroySemaphore(frame.render_completed_semaphore);
            if (frame.image_acquired_semaphore)
                vk_state.device.destroySemaphore(frame.image_acquired_semaphore);
            if (frame.render_fence)
                vk_state.device.destroyFence(frame.render_fence);
        }
        
        vk_state.swapchain_frames.clear();
        vk::SurfaceCapabilitiesKHR surface_capabilities = vk_state.physical_device.getSurfaceCapabilitiesKHR(vk_state.surface);

        vk::SwapchainCreateInfoKHR swapchain_ci{};
        swapchain_ci.surface = vk_state.surface;
        swapchain_ci.minImageCount = min_image_count;
        swapchain_ci.imageFormat = vk_state.surface_format.format;
        swapchain_ci.imageColorSpace = vk_state.surface_format.colorSpace;
        swapchain_ci.imageArrayLayers = 1;
        swapchain_ci.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        swapchain_ci.imageSharingMode = vk::SharingMode::eExclusive;
        swapchain_ci.preTransform = (surface_capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
                                        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
                                        : surface_capabilities.currentTransform;
        swapchain_ci.presentMode = vk_state.present_mode;
        swapchain_ci.clipped = VK_TRUE;
        swapchain_ci.oldSwapchain = old_swapchain;

        if (min_image_count < surface_capabilities.minImageCount)
            swapchain_ci.minImageCount = surface_capabilities.minImageCount;
        else if (surface_capabilities.maxImageCount != 0 && min_image_count > surface_capabilities.maxImageCount)
            swapchain_ci.minImageCount = surface_capabilities.maxImageCount;

        if (surface_capabilities.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
            swapchain_ci.imageExtent.width = vk_state.width;
            swapchain_ci.imageExtent.height = vk_state.height;
        } else {
            vk_state.width = surface_capabilities.currentExtent.width;
            vk_state.height = surface_capabilities.currentExtent.height;
            swapchain_ci.imageExtent = surface_capabilities.currentExtent;
        }

        vk_state.swapchain = vk_state.device.createSwapchainKHR(swapchain_ci, nullptr);
        auto swapchain_images = vk_state.device.getSwapchainImagesKHR(vk_state.swapchain);
        vk_state.image_count = static_cast<uint32_t>(swapchain_images.size());

        for (auto& swapchain_image : swapchain_images) {
            frame_resource frame;
            frame.backbuffer = swapchain_image;
            vk::ImageViewCreateInfo view_ci({}, swapchain_image, vk::ImageViewType::e2D,
                vk_state.surface_format.format, vk::ComponentMapping(),
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            frame.backbuffer_view = vk_state.device.createImageView(view_ci, nullptr);

            vk::CommandPoolCreateInfo pool_ci({}, vk_state.queue_family);
            frame.command_pool = vk_state.device.createCommandPool(pool_ci, nullptr);

            vk::CommandBufferAllocateInfo cmd_alloc_info(frame.command_pool, vk::CommandBufferLevel::ePrimary, 1);
            frame.command_buffers = vk_state.device.allocateCommandBuffers(cmd_alloc_info);

            vk::SemaphoreCreateInfo sem_ci;
            frame.image_acquired_semaphore = vk_state.device.createSemaphore(vk::SemaphoreCreateInfo());
            frame.render_completed_semaphore = vk_state.device.createSemaphore(vk::SemaphoreCreateInfo());
            frame.render_fence = vk_state.device.createFence(vk::FenceCreateInfo());
            
            vk_state.swapchain_frames.push_back(frame);
        }
    }

    void render_and_present_frame(ImDrawData* draw_data) {
        vk::Semaphore image_acquired_semaphore = vk_state.swapchain_frames[vk_state.image_acquired_semaphore_index].image_acquired_semaphore;
        vk::Fence render_fence = vk_state.swapchain_frames[vk_state.image_acquired_semaphore_index].render_fence;

        auto result_pair = vk_state.device.acquireNextImageKHR(vk_state.swapchain, UINT64_MAX,
                                                                image_acquired_semaphore, VK_NULL_HANDLE);
        vk::Result result = result_pair.result;
        uint32_t frame_index = result_pair.value; 

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
            vk_state.swapchain_needs_rebuild = true;
        if (result != vk::Result::eSuboptimalKHR && result != vk::Result::eSuccess)
            throw detailed_exception("Error during swapchain image acquisition");

        frame_resource& current_frame = vk_state.swapchain_frames[frame_index];
        
        vk_state.device.resetCommandPool(current_frame.command_pool, {});
        vk::CommandBuffer command_buffer = current_frame.command_buffers[0];
        command_buffer.begin(vk::CommandBufferBeginInfo()
            .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
        );

        auto render_image_memory_barrier = vk::ImageMemoryBarrier()
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
            .setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setImage(current_frame.backbuffer)
            .setSubresourceRange(vk::ImageSubresourceRange()
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1));

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {},
            nullptr,
            nullptr,
            render_image_memory_barrier
        );

        vk::RenderingAttachmentInfo color_attachment{};
        color_attachment.imageView = current_frame.backbuffer_view;
        color_attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
        vk::ClearValue clear_value;
        clear_value.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        color_attachment.clearValue = clear_value;

        vk::RenderingInfo rendering_info{};
        rendering_info.renderArea.offset = vk::Offset2D{ 0, 0 };
        rendering_info.renderArea.extent = vk::Extent2D{ vk_state.width, vk_state.height };
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;

        command_buffer.beginRenderingKHR(rendering_info);
        ImGui_ImplVulkan_RenderDrawData(draw_data, static_cast<VkCommandBuffer>(command_buffer));
        command_buffer.endRenderingKHR();

        auto image_memory_barrier = vk::ImageMemoryBarrier()
            .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
            .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setImage(current_frame.backbuffer)
            .setSubresourceRange(vk::ImageSubresourceRange()
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1));

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            {},
            nullptr,
            nullptr,
            image_memory_barrier
        );

        command_buffer.end();

        vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk_state.queue.submit(
            vk::SubmitInfo()
                .setCommandBuffers({ command_buffer })
                .setWaitSemaphores(image_acquired_semaphore)
                .setSignalSemaphores(current_frame.render_completed_semaphore)
                .setWaitDstStageMask(wait_stage),
            render_fence
        );

        auto swapchain_vector = { vk_state.swapchain };
        vk_state.queue.presentKHR(
            vk::PresentInfoKHR()
                .setWaitSemaphores(current_frame.render_completed_semaphore)
                .setSwapchains(swapchain_vector)
                .setImageIndices(frame_index)
        );

        vk_state.device.waitForFences(render_fence, true, UINT64_MAX);
        vk_state.device.resetFences(render_fence);

        vk_state.image_acquired_semaphore_index = (vk_state.image_acquired_semaphore_index + 1) % vk_state.swapchain_frames.size();
    }

    void cleanup() {
        vk_state.device.waitIdle();
    
        // Destroy all frame resources
        for (auto& frame : vk_state.swapchain_frames) {
            if (frame.backbuffer_view)
                vk_state.device.destroyImageView(frame.backbuffer_view);
                frame.backbuffer_view = VK_NULL_HANDLE;
            if (frame.command_pool)
                vk_state.device.destroyCommandPool(frame.command_pool);
                frame.command_pool = VK_NULL_HANDLE;
            if (frame.render_completed_semaphore)
                vk_state.device.destroySemaphore(frame.render_completed_semaphore);
                frame.render_completed_semaphore = VK_NULL_HANDLE;
            if (frame.image_acquired_semaphore)
                vk_state.device.destroySemaphore(frame.image_acquired_semaphore);
                frame.image_acquired_semaphore = VK_NULL_HANDLE;
            if (frame.render_fence)
                vk_state.device.destroyFence(frame.render_fence);
                frame.render_fence = VK_NULL_HANDLE;
        }
        vk_state.swapchain_frames.clear();
    
        // Destroy swapchain
        if (vk_state.swapchain) {
            vk_state.device.destroySwapchainKHR(vk_state.swapchain);
            vk_state.swapchain = VK_NULL_HANDLE;
        }
    
        // Destroy descriptor pool
        if (vk_state.descriptor_pool) {
            vk_state.device.destroyDescriptorPool(vk_state.descriptor_pool);
            vk_state.descriptor_pool = VK_NULL_HANDLE;
        }
    
        // Destroy device
        if (vk_state.device) {
            vk_state.device.destroy();
            vk_state.device = VK_NULL_HANDLE;
        }
    
        // Destroy debug callback
    #ifdef APP_USE_VULKAN_DEBUG_REPORT
        if (vk_state.debug_report) {
            vk_state.instance.destroyDebugReportCallbackEXT(vk_state.debug_report);
            vk_state.debug_report = VK_NULL_HANDLE;
        }
    #endif
    
        // Destroy surface
        if (vk_state.surface) {
            vk_state.instance.destroySurfaceKHR(vk_state.surface);
            vk_state.surface = VK_NULL_HANDLE;
        }
    
        // Destroy instance
        if (vk_state.instance) {
            vk_state.instance.destroy();
            vk_state.instance = VK_NULL_HANDLE;
        }
    }
};

int main() {
    try {
        app_state app;
    }
    catch (const std::exception &ex) {
        spdlog::error("Application encountered an error: {}", ex.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
