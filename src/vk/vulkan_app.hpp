#pragma once

#include <vulkan/vulkan.hpp>
#include <mdspan>
#include <spdlog/spdlog.h>
#include <detailed_exception.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

class vulkan_core {
    VkDebugReportCallbackEXT debug_report_ = VK_NULL_HANDLE;

    vk::Instance        instance_;
    vk::Device          device_;
    vk::PhysicalDevice  physical_device_;

    vk::Queue transfer_queue_;
    uint32_t transfer_queue_family_;

    vk::Queue compute_queue_;
    uint32_t compute_queue_family_;

    vk::Queue graphics_queue_;
    uint32_t graphics_queue_family_;

    vk::CommandPool transfer_pool_;
    vk::CommandPool compute_pool_;
    vk::CommandPool graphics_pool_;

    static bool is_extension_available(const std::vector<vk::ExtensionProperties>& properties, const char* extension) {
        for (const auto& prop : properties)
            if (strcmp(prop.extensionName, extension) == 0)
                return true;
        return false;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report_callback(
        VkDebugReportFlagsEXT      flags,
        VkDebugReportObjectTypeEXT object_type,
        uint64_t                   object,
        size_t                     location,
        int32_t                    message_code,
        const char*                p_layer_prefix,
        const char*                p_message,
        void*                      p_user_data
    ) {
        spdlog::error("Validation Layer ({}): {}", p_layer_prefix, p_message);
        return VK_FALSE;
    }
    
    uint32_t find_dedicated_transfer_family(const std::vector<vk::QueueFamilyProperties>& families) {
        // Look for transfer-only family
        for (uint32_t i = 0; i < families.size(); ++i) {
            const auto& props = families[i];
            if ((props.queueFlags & vk::QueueFlagBits::eTransfer) &&
                !(props.queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))) {
                return i;
            }
        }
        // Fallback to any transfer-capable family
        for (uint32_t i = 0; i < families.size(); ++i) {
            if (families[i].queueFlags & vk::QueueFlagBits::eTransfer) {
                return i;
            }
        }
        throw detailed_exception("No transfer queue family found");
    }
    
    uint32_t find_compute_family(const std::vector<vk::QueueFamilyProperties>& families) {
        // Look for compute-only family
        for (uint32_t i = 0; i < families.size(); ++i) {
            const auto& props = families[i];
            if ((props.queueFlags & vk::QueueFlagBits::eCompute) &&
                !(props.queueFlags & vk::QueueFlagBits::eGraphics)) {
                return i;
            }
        }
        // Fallback to graphics family (which always includes compute)
        for (uint32_t i = 0; i < families.size(); ++i) {
            if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
                return i;
        }
        throw detailed_exception("No compute queue family found");
    }
    
    uint32_t find_graphics_family(const std::vector<vk::QueueFamilyProperties>& families) {
        for (uint32_t i = 0; i < families.size(); ++i) {
            if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
                return i;
        }
        throw detailed_exception("No graphics queue family found");
    }

public:
    vulkan_core(std::vector<const char*> instance_extensions, std::vector<const char*> device_extensions) {
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
    
            instance_ = vk::createInstance(instance_ci);
            VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);

            #ifdef APP_USE_VULKAN_DEBUG_REPORT
            debug_report_ = instance_.createDebugReportCallbackEXT(
            vk::DebugReportCallbackCreateInfoEXT()
                .setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning | vk::DebugReportFlagBitsEXT::ePerformanceWarning)
                .setPfnCallback(debug_report_callback)
            );
            #endif

            auto physical_devices = instance_.enumeratePhysicalDevices();
            if (physical_devices.empty())
                throw detailed_exception("No GPUs with Vulkan support found.");
    
            // Prefer discrete GPUs
            for (const auto& pd : physical_devices) {
                vk::PhysicalDeviceProperties props = pd.getProperties();
                if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                    physical_device_ = pd;
                    break;
                }
            }

            // Else pick the first
            if (!physical_device_)
                physical_device_ = physical_devices[0];

            auto queue_families = physical_device_.getQueueFamilyProperties();

            transfer_queue_family_ = find_dedicated_transfer_family(queue_families);
            compute_queue_family_ = find_compute_family(queue_families);
            graphics_queue_family_ = find_graphics_family(queue_families);

            std::unordered_map<uint32_t, uint32_t> queue_counts;
            queue_counts[transfer_queue_family_]++;
            queue_counts[compute_queue_family_]++;
            queue_counts[graphics_queue_family_]++;

            std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
            std::vector<std::vector<float>> priorities_storage;

            for (const auto& [family, count] : queue_counts) {
                std::vector<float> priorities(count, 1.0f);
                priorities_storage.emplace_back(priorities);
                queue_create_infos.emplace_back(
                    vk::DeviceQueueCreateInfo()
                        .setQueueFamilyIndex(family)
                        .setQueueCount(count)
                        .setPQueuePriorities(priorities_storage.back().data())
                );
            }

            vk::PhysicalDeviceTimelineSemaphoreFeatures timeline_features(true);
            vk::PhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features(true);
            dynamic_rendering_features.pNext = &timeline_features;
    
            auto device_info = vk::DeviceCreateInfo()
                .setQueueCreateInfos(queue_create_infos)
                .setPEnabledExtensionNames(device_extensions)
                .setPNext(&dynamic_rendering_features);

            device_ = physical_device_.createDevice(device_info);
            VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

            std::unordered_map<uint32_t, uint32_t> family_indices;
            transfer_queue_ = device_.getQueue(transfer_queue_family_, family_indices[transfer_queue_family_]++);
            compute_queue_ = device_.getQueue(compute_queue_family_, family_indices[compute_queue_family_]++);
            graphics_queue_ = device_.getQueue(graphics_queue_family_, family_indices[graphics_queue_family_]++);

            transfer_pool_ = device_.createCommandPool(
                vk::CommandPoolCreateInfo()
                    .setQueueFamilyIndex(transfer_queue_family_)
                    .setFlags(vk::CommandPoolCreateFlagBits::eTransient |  // Optimized for short-lived buffers
                              vk::CommandPoolCreateFlagBits::eResetCommandBuffer)  // Allow buffer resets
            );
    
            compute_pool_ = device_.createCommandPool(
                vk::CommandPoolCreateInfo()
                    .setQueueFamilyIndex(compute_queue_family_)
                    .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)  // Allow individual buffer resets
            );
    
            graphics_pool_ = device_.createCommandPool(
                vk::CommandPoolCreateInfo()
                    .setQueueFamilyIndex(graphics_queue_family_)
                    .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)  // Allow individual buffer resets
            );
        }

    ~vulkan_core() {
        device_.destroyCommandPool(transfer_pool_);
        device_.destroyCommandPool(compute_pool_);
        device_.destroyCommandPool(graphics_pool_);
    }

    vk::Instance instance() const {
        return instance_;
    }

    vk::Device device() const {
        return device_;
    }

    vk::PhysicalDevice physical_device() const {
        return physical_device_;
    }

    vk::Queue graphics_queue() const {
        return graphics_queue_;
    }

    uint32_t graphics_queue_family() const {
        return graphics_queue_family_;
    }

    vk::CommandPool graphics_command_pool() const {
        return graphics_pool_;
    }

    vk::Queue compute_queue() const {
        return compute_queue_;
    }

    vk::Queue transfer_queue() const {
        return transfer_queue_;
    }
};